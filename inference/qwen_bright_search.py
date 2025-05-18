import os
import re
import json
import time
import argparse
import requests
import urllib.parse
import logging
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 加载.env文件中的环境变量
load_dotenv()

# 配置日志系统
def setup_logger(log_level=logging.INFO):
    """设置日志系统"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('qwen_search')

# 默认创建一个日志记录器
logger = setup_logger()

# 配置常量
API_URL = "http://60.50.228.238:62735/v1"
MODEL = "Qwen2.5-7B-Instruct_ZeroSearch"
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# 获取BrightData API密钥
BRIGHT_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
BRIGHT_API_ENDPOINT = os.getenv("BRIGHTDATA_API_ENDPOINT", "https://api.brightdata.com/request")
BRIGHT_ZONE = os.getenv("BRIGHTDATA_ZONE", "xdan_search_searp")
BRIGHT_FORMAT = os.getenv("BRIGHTDATA_FORMAT", "raw")
BRIGHT_TIMEOUT = int(os.getenv("BRIGHTDATA_TIMEOUT", "30"))

# 定义系统提示 - 增强版提示，强调多轮搜索和拆分思考
SYSTEM_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search>, and it "
    "will return the top searched results between <information> and </information>. \n\n"
    "IMPORTANT: For complex questions, you should break down the problem into smaller sub-questions "
    "and search for each part separately. DO NOT try to search for everything at once. "
    "Instead, search for one piece of information, analyze the results, then search again "
    "with a refined query based on what you've learned. This multi-turn search approach "
    "will lead to much better answers. \n\n"
    "You can and should search as many times as needed to gather complete information. "
    "Only after you have gathered all necessary information through multiple searches, "
    "provide the final answer inside <answer> and </answer> without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)

# 创建一个带有重试机制的会话
def create_retry_session(retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 生成错误响应（当API不可用时）
def generate_error_response(query: str, error_msg: str) -> str:
    thinking = f"""我需要分析用户的问题："{query}"。
但是搜索服务出现了问题：{error_msg}
我将尝试直接回答这个问题，但可能无法提供最新或最准确的信息。"""
    
    answer = f"""抱歉，搜索服务暂时不可用：{error_msg}。我无法提供关于"{query}"的最新信息。请稍后再试或重新表述您的问题。"""
    
    return f"<think>{thinking}</think>\n<answer>{answer}</answer>"

# API调用函数
def call_api(prompt: str, max_tokens: int = 16384, temperature: float = TEMPERATURE) -> str:
    """调用远程API获取模型响应"""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{SYSTEM_PROMPT} Question: {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    session = create_retry_session()
    
    try:
        logger.info(f"正在调用API: {API_URL}/chat/completions")
        logger.info(f"请求模型: {MODEL}, 温度: {temperature}")
        response = session.post(f"{API_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        logger.info(f"API响应状态码: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        
        # 只打印响应的关键部分
        if 'choices' in result and len(result['choices']) > 0:
            logger.info("成功获取API响应")
            
        # 从正确的位置获取响应内容
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_content = e.response.json()
                logger.error(f"错误详情: {json.dumps(error_content, ensure_ascii=False, indent=2)}")
            except:
                logger.error(f"响应内容: {e.response.text}")
        
        # 在API错误时返回错误响应
        error_msg = str(e)
        logger.error(f"返回错误响应: {error_msg}")
        return generate_error_response(prompt, error_msg)
    except Exception as e:
        logger.error(f"其他错误: {e}")
        
        # 在其他错误时返回错误响应
        error_msg = str(e)
        logger.error(f"返回错误响应: {error_msg}")
        return generate_error_response(prompt, error_msg)

# 从模型输出中提取思考内容
def parse_thinking(output: str) -> Optional[str]:
    if "<think>" in output and "</think>" in output:
        return output.split("<think>")[1].split("</think>")[0].strip()
    return None

# 从模型输出中提取搜索查询
def parse_search_queries(output: str) -> List[str]:
    queries = []
    start_pos = 0
    while True:
        start_tag = output.find("<search>", start_pos)
        if start_tag == -1:
            break
        end_tag = output.find("</search>", start_tag)
        if end_tag == -1:
            break
        query = output[start_tag + len("<search>"):end_tag].strip()
        queries.append(query)
        start_pos = end_tag + len("</search>")
    return queries

# 从模型输出中提取最终答案
def parse_answer(output: str) -> Optional[str]:
    if "<answer>" in output and "</answer>" in output:
        return output.split("<answer>")[1].split("</answer>")[0].strip()
    return None

# 使用BrightData执行真实搜索
def execute_bright_search(query: str) -> str:
    """
    使用BrightData API执行真实搜索
    
    参数:
        query: 搜索查询
        
    返回:
        搜索结果文本
    """
    if not BRIGHT_API_KEY:
        logger.warning("警告: BrightData API密钥未配置，将使用模拟搜索结果")
        return execute_mock_search(query)
    
    try:
        logger.info(f"使用BrightData执行搜索: {query}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BRIGHT_API_KEY}"
        }
        
        # 对查询进行URL编码
        encoded_query = requests.utils.quote(query)
        
        # 根据API端点选择正确的请求格式
        if "serp" in BRIGHT_API_ENDPOINT:
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
                "zone": BRIGHT_ZONE,
                "url": f"https://www.google.com/search?q={encoded_query}",
                "format": "raw"
            }
        
        logger.info(f"发送请求到BrightData API: {json.dumps(data, ensure_ascii=False)}")
        
        response = requests.post(
            BRIGHT_API_ENDPOINT,
            headers=headers,
            json=data,
            timeout=30
        )
        
        logger.info(f"BrightData API响应状态码: {response.status_code}")
        response.raise_for_status()
        
        # 处理响应
        try:
            # 先尝试解析为JSON
            json_response = response.json()
            
            # 检查是否有错误
            if "error" in json_response:
                error_msg = str(json_response.get("error", ""))
                logger.error(f"BrightData API返回错误: {error_msg}")
                return f"搜索错误: {error_msg}"
            
            # 检查是否有搜索结果
            if "organic" in json_response:
                # 新版API格式
                results = []
                for item in json_response.get("organic", [])[:10]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    url = item.get("url", "")
                    results.append(f"标题: {title}\n摘要: {snippet}\n链接: {url}\n")
                return "\n".join(results)
            else:
                # 其他格式，简单返回JSON字符串
                return json.dumps(json_response, ensure_ascii=False, indent=2)
        except ValueError:
            # 如果不是JSON，返回文本
            return response.text[:1000]  # 限制长度
    except Exception as e:
        logger.error(f"BrightData搜索错误: {e}")
        return f"搜索过程中发生错误: {str(e)}"

# 执行搜索
def execute_search(query: str) -> str:
    """
    执行搜索，使用BrightData API
    
    参数:
        query: 搜索查询
        
    返回:
        搜索结果文本
    """
    return execute_bright_search(query)

# 主要推理函数
def run_search_chat(query: str, max_search_rounds: int = 6, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, List[str], str, List[Dict[str, str]]]:
    """
    使用远程API进行搜索增强对话推理
    
    参数:
        query: 用户查询
        max_search_rounds: 最大搜索轮数
        conversation_history: 对话历史
        
    返回:
        完整的模型输出、搜索查询列表、最终答案和更新后的对话历史
    """
    # 初始化对话历史
    if conversation_history is None:
        conversation_history = []
    
    current_round = 0
    final_output = ""
    all_search_queries = []
    final_answer = ""
    
    # 初始提示
    current_prompt = query
    
    # 如果有对话历史，添加到提示中
    if conversation_history:
        history_text = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                if "answer" in msg:
                    history_text += f"Assistant: {msg['answer']}\n"
        if history_text:
            current_prompt = f"{history_text}\nUser: {query}"
    
    while current_round < max_search_rounds:
        logger.info(f"\n===== 第 {current_round + 1} 轮 =====")
        
        # 调用API获取响应
        generated_text = call_api(current_prompt)
        logger.info(f"模型输出:\n{generated_text}")
        
        # 解析思考、搜索查询和答案
        thinking = parse_thinking(generated_text)
        if thinking:
            logger.info(f"\n思考过程:\n{thinking}")
        
        search_queries = parse_search_queries(generated_text)
        if search_queries:
            logger.info(f"\n搜索查询:\n{search_queries}")
            all_search_queries.extend(search_queries)
        
        answer = parse_answer(generated_text)
        if answer:
            logger.info(f"\n最终答案:\n{answer}")
            final_answer = answer
        
        # 如果没有搜索查询或已有最终答案，结束循环
        if not search_queries or answer:
            final_output = generated_text
            # 添加到对话历史
            conversation_history.append({
                "role": "user",
                "content": query
            })
            conversation_history.append({
                "role": "assistant",
                "content": generated_text,
                "thinking": thinking,
                "answer": answer
            })
            break
        
        # 执行搜索并准备下一轮提示
        search_results = []
        for query in search_queries:
            result = execute_search(query)
            search_results.append(f"<information>\n{result}\n</information>")
        
        # 更新提示，包含原始查询和搜索结果
        current_prompt = f"{query}\n\n{''.join(search_results)}"
        
        current_round += 1
        logger.info(f"完成第 {current_round} 轮搜索")
    
    return final_output, all_search_queries, final_answer, conversation_history

# 多轮对话函数
def run_multi_turn_conversation(max_turns: int = 5, max_search_rounds: int = 3):
    """
    运行多轮对话
    
    参数:
        max_turns: 最大对话轮数
        max_search_rounds: 每轮对话的最大搜索轮数
    """
    conversation_history = []
    
    logger.info("\n===== 开始多轮对话 =====")
    logger.info("输入'退出'结束对话")
    
    for turn in range(max_turns):
        user_query = input("\n用户: ")
        if user_query.lower() in ['退出', 'exit', 'quit']:
            break
        
        _, search_queries, answer, conversation_history = run_search_chat(
            user_query, 
            max_search_rounds=max_search_rounds,
            conversation_history=conversation_history
        )
        
        logger.info(f"\n助手: {answer}")
    
    logger.info("\n===== 对话结束 =====")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5搜索增强测试（使用BrightData API）')
    parser.add_argument('--query', type=str, default="请告诉我关于量子计算的最新进展",
                        help='用户查询')
    parser.add_argument('--rounds', type=int, default=3,
                        help='最大搜索轮数')

    parser.add_argument('--model', type=str, default=MODEL,
                        help='使用的模型名称')
    parser.add_argument('--api_url', type=str, default=API_URL,
                        help='API URL')
    parser.add_argument('--interactive', action='store_true',
                        help='启用交互式对话模式')

    parser.add_argument('--bright_api_key', type=str, default=None,
                        help='BrightData API密钥（直接指定，优先级高于.env文件）')
    parser.add_argument('--bright_api_endpoint', type=str, default=None,
                        help='BrightData API端点（直接指定，优先级高于.env文件）')
    parser.add_argument('--bright_zone', type=str, default=None,
                        help='BrightData区域（直接指定，优先级高于.env文件）')
    parser.add_argument('--log_level', type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='日志级别')
    parser.add_argument('--log_to_file', type=str, help='将日志输出到指定文件')
    
    return parser.parse_args()

# 主函数
def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    # 如果指定了日志文件，添加文件处理器
    if args.log_to_file:
        file_handler = logging.FileHandler(args.log_to_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"日志将同时输出到文件: {args.log_to_file}")
    
    global MODEL, API_URL, BRIGHT_API_KEY, BRIGHT_API_ENDPOINT, BRIGHT_ZONE
    
    if args.model:
        MODEL = args.model
    
    API_URL = args.api_url
    
    # 命令行参数指定的BrightData配置优先级高于.env文件
    if args.bright_api_key:
        BRIGHT_API_KEY = args.bright_api_key
    if args.bright_api_endpoint:
        BRIGHT_API_ENDPOINT = args.bright_api_endpoint
    if args.bright_zone:
        BRIGHT_ZONE = args.bright_zone
    
    logger.info("=== 启动Qwen2.5搜索增强测试（BrightData版）===")
    logger.info(f"模型: {MODEL}")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"真实搜索: 启用")
    logger.info(f"BrightData API密钥: {'已配置' if BRIGHT_API_KEY else '未配置'}")
    logger.info(f"最大搜索轮数: {args.rounds}")
    logger.info(f"日志级别: {args.log_level}")
    
    print(f"\n===== Qwen2.5搜索增强测试（BrightData版） =====")
    print(f"模型: {MODEL}")
    print(f"API URL: {API_URL}")
    print(f"真实搜索: 启用")
    print(f"BrightData API密钥: {'已配置' if BRIGHT_API_KEY else '未配置'}")
    print(f"最大搜索轮数: {args.rounds}")
    print(f"日志级别: {args.log_level}")
    
    # 交互式模式
    if args.interactive:
        run_multi_turn_conversation(max_search_rounds=args.rounds)
        return
    
    print(f"用户查询: {args.query}")
    print("================================\n")
    
    # 运行搜索增强对话
    final_output, search_queries, final_answer, _ = run_search_chat(args.query, args.rounds)
    
    # 输出结果摘要
    print("\n===== 结果摘要 =====")
    print(f"总搜索次数: {len(search_queries)}")
    if search_queries:
        print(f"搜索查询: {', '.join(search_queries)}")
    if final_answer:
        print(f"最终答案: {final_answer}")
    else:
        print("未获得最终答案")
    print("====================\n")

if __name__ == "__main__":
    main()
