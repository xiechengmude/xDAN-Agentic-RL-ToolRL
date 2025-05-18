"""
BrightData搜索工具实现
"""

import json
import requests
import re
import time
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import sys
import os

# 导入配置
from generator.trace_gen.core.config import SearchToolConfig

# 导入基类的前向声明
BaseSearchTool = None


class BrightDataSearchTool:
    """BrightData搜索工具实现"""
    
    def __init__(self, config: SearchToolConfig):
        # 动态导入基类，避免循环导入
        global BaseSearchTool
        if BaseSearchTool is None:
            from generator.trace_gen.tools.search.search_tools import BaseSearchTool
            
        # 设置基本属性
        self.config = config
        config.name = "brightdata_search"
        config.description = "Do search via BrightData API"
        self.name = config.name
        self.description = config.description
        
        # BrightData特有属性
        self.api_endpoint = config.api_endpoint or "https://api.brightdata.com/request"
        self.api_key = config.api_key
        self.zone = config.zone or "xdan_search_searp"
        self.timeout = config.timeout or 30
        self.format = config.format or "raw"
    
    def search(self, query: str) -> Dict[str, Any]:
        """执行BrightData搜索并返回结果，始终使用真实API，并具有增强的重试机制"""
        if not self.api_key:
            error_msg = "BrightData API密钥未配置，无法执行搜索"
            print(error_msg)
            self._log_api_error(query, "API_KEY_MISSING", error_msg)
            return {
                "result": {
                    "error": error_msg,
                    "status": "error",
                    "query": query,
                    "timestamp": self._get_timestamp()
                }
            }
        
        # 设置重试参数
        max_retries = 5  # 最大重试次数
        retry_count = 0
        base_timeout = self.timeout  # 原始超时时间
        max_timeout = 60  # 最大超时时间
        
        while retry_count <= max_retries:
            current_timeout = min(base_timeout * (2 ** retry_count), max_timeout)  # 指数增长的超时时间
            
            try:
                # 对查询进行URL编码
                encoded_query = requests.utils.quote(query)
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # 根据API端点选择正确的请求格式
                if "serp" in self.api_endpoint:
                    # 新版API端点格式
                    data = {
                        "q": query,  # 直接使用原始查询，不需要URL编码
                        "domain": "google.com",
                        "country": "us",
                        "locale": "en-US"
                    }
                else:
                    # 旧版API端点格式
                    data = {
                        "zone": self.zone,
                        "url": f"https://www.google.com/search?q={encoded_query}",
                        "format": self.format
                    }
                
                retry_suffix = f" (重试 {retry_count}/{max_retries})" if retry_count > 0 else ""
                print(f"发送请求到BrightData API{retry_suffix}: {json.dumps(data)}")
                print(f"API端点: {self.api_endpoint}")
                print(f"Authorization头: Bearer {self.api_key[:10]}...{self.api_key[-10:]}")
                print(f"当前超时设置: {current_timeout}秒")
                
                try:
                    response = requests.post(
                        self.api_endpoint,
                        headers=headers,
                        json=data,
                        timeout=current_timeout
                    )
                    
                    print(f"BrightData API响应状态码: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_msg = f"BrightData API错误响应: {response.text}"
                        print(error_msg)
                        self._log_api_error(query, f"HTTP_{response.status_code}", error_msg, response.text)
                        
                        # 对于某些状态码不进行重试
                        if response.status_code in [400, 401, 403]:
                            return {
                                "result": {
                                    "error": error_msg,
                                    "status": "error",
                                    "query": query,
                                    "timestamp": self._get_timestamp()
                                }
                            }
                    
                    response.raise_for_status()
                    
                    # 处理响应
                    try:
                        # 先尝试解析为JSON
                        json_response = response.json()
                        
                        # 检查是否有错误
                        if "error" in json_response:
                            error_msg = str(json_response.get("error", ""))
                            print(f"BrightData API返回错误: {error_msg}")
                            # 如果是可重试的错误，重试
                            if "rate limit" in error_msg.lower() or "timeout" in error_msg.lower():
                                raise Exception(f"API限流或超时: {error_msg}")
                            return {"results": []}
                        
                        # 检查是否有搜索结果
                        if "organic" in json_response:
                            # 新版API格式
                            results = []
                            for item in json_response.get("organic", [])[:10]:
                                results.append({
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "url": item.get("url", "")
                                })
                            return {"results": results}
                        else:
                            # 其他格式，使用格式化函数
                            return self._format_response(json_response)
                    except ValueError:
                        # 如果不是JSON，尝试解析HTML
                        if "text/html" in response.headers.get("Content-Type", ""):
                            return self._parse_html_response(response.text, query)
                        else:
                            # 如果无法解析，返回原始文本的简单处理
                            return {
                                "results": [
                                    {
                                        "title": "API响应",
                                        "snippet": response.text[:200] + "...",
                                        "url": f"https://www.google.com/search?q={encoded_query}"
                                    }
                                ]
                            }
                except requests.exceptions.Timeout as e:
                    # 超时错误处理
                    error_msg = f"BrightData API超时 ({current_timeout}秒): {str(e)}"
                    print(error_msg)
                    self._log_api_error(query, "TIMEOUT", error_msg)
                    
                    # 如果还有重试次数，继续重试
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count  # 指数退避策略
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "result": {
                                "error": f"超过最大重试次数 ({max_retries}): {error_msg}",
                                "status": "error",
                                "query": query,
                                "timestamp": self._get_timestamp()
                            }
                        }
                        
                except requests.exceptions.RequestException as e:
                    # 其他请求异常处理
                    error_msg = f"BrightData API请求异常: {str(e)}"
                    print(error_msg)
                    response_text = ""
                    if hasattr(e, 'response') and e.response is not None:
                        response_text = e.response.text
                        print(f"API错误响应内容: {response_text}")
                    
                    self._log_api_error(query, "REQUEST_EXCEPTION", error_msg, response_text)
                    
                    # 如果还有重试次数，继续重试
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count  # 指数退避策略
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "result": {
                                "error": f"超过最大重试次数 ({max_retries}): {error_msg}",
                                "status": "error",
                                "query": query,
                                "timestamp": self._get_timestamp()
                            }
                        }
            except Exception as e:
                error_msg = f"BrightData搜索其他错误: {str(e)}"
                print(error_msg)
                self._log_api_error(query, "GENERAL_EXCEPTION", error_msg)
                
                # 如果还有重试次数，继续重试
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # 指数退避策略
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "result": {
                            "error": f"超过最大重试次数 ({max_retries}): {error_msg}",
                            "status": "error",
                            "query": query,
                            "timestamp": self._get_timestamp()
                        }
                    }
    
    def to_dict(self) -> Dict[str, Any]:
        """BrightData工具的字典表示"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"}
                }
            }
        }
    
    def _parse_html_response(self, html_content: str, query: str) -> Dict[str, Any]:
        """解析HTML响应并提取搜索结果"""
        results = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取搜索结果
            search_results = soup.select("div.g")
            
            for result in search_results[:10]:  # 限制结果数量
                title_elem = result.select_one("h3")
                link_elem = result.select_one("a")
                snippet_elem = result.select_one("div.VwiC3b")
                
                if title_elem and link_elem:
                    title = title_elem.get_text()
                    url = link_elem.get("href")
                    
                    # 确保URL是完整的
                    if url and url.startswith("/url?"):
                        url_match = re.search(r"url=([^&]+)", url)
                        if url_match:
                            url = url_match.group(1)
                    
                    snippet = ""
                    if snippet_elem:
                        snippet = snippet_elem.get_text()
                    
                    results.append({
                        "title": title,
                        "snippets": snippet,
                        "url": url
                    })
            
            # 如果没有找到结果，尝试其他选择器
            if not results:
                alternative_results = soup.select("div.tF2Cxc")
                for result in alternative_results[:10]:
                    title_elem = result.select_one("h3")
                    link_elem = result.select_one("a")
                    snippet_elem = result.select_one("div.VwiC3b, div.IsZvec")
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text()
                        url = link_elem.get("href")
                        
                        snippet = ""
                        if snippet_elem:
                            snippet = snippet_elem.get_text()
                        
                        results.append({
                            "title": title,
                            "snippets": snippet,
                            "url": url
                        })
        except Exception as e:
            print(f"解析HTML出错: {str(e)}")
        
        # 如果仍然没有结果，返回模拟数据
        if not results:
            return self._mock_search(query)
        
        return {"results": results}
    
    def _format_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """格式化API响应（用于非raw格式）"""
        # 如果BrightData返回的是结构化JSON，则在这里处理
        results = []
        
        # 根据实际API响应结构进行调整
        if "results" in raw_response:
            for item in raw_response["results"]:
                results.append({
                    "title": item.get("title", ""),
                    "snippets": item.get("snippet", ""),
                    "url": item.get("url", "")
                })
        
        return {"results": results}
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _log_api_error(self, query: str, error_type: str, error_message: str, response_text: str = "") -> None:
        """记录API错误到日志文件"""
        try:
            import os
            from datetime import datetime
            
            # 确保日志目录存在
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 创建日志文件名
            log_file = os.path.join(log_dir, f"brightdata_api_errors_{datetime.now().strftime('%Y%m%d')}.log")
            
            # 写入错误日志
            with open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "timestamp": self._get_timestamp(),
                    "query": query,
                    "error_type": error_type,
                    "error_message": error_message,
                    "response_text": response_text,
                    "api_endpoint": self.api_endpoint,
                    "zone": self.zone
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"记录API错误日志失败: {str(e)}")
    
    def _mock_search(self, query: str) -> Dict[str, Any]:
        """已弃用的模拟搜索结果方法，保留仅供参考"""
        print("警告: _mock_search方法已弃用，系统配置为始终使用真实API")
        error_msg = "系统配置为始终使用真实API，不再支持模拟数据"
        return {
            "result": {
                "error": error_msg,
                "status": "error",
                "query": query,
                "timestamp": self._get_timestamp()
            }
        }
