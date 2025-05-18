#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastMCP 客户端测试脚本

此脚本使用 FastMCP 客户端连接到 MCP 服务器并获取可用的工具列表。
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 尝试导入 fastmcp
try:
    from fastmcp import Client
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    logger.error("fastmcp 未安装，请先安装 fastmcp: pip install fastmcp")

async def list_mcp_tools():
    """使用 FastMCP 客户端获取 MCP 工具列表"""
    if not HAS_FASTMCP:
        logger.error("fastmcp 未安装，无法使用 FastMCP 客户端")
        return None
    
    # MCP 服务器 URL
    mcp_url = os.getenv("MCP_URL", "http://34.124.211.73:7223/mcp")
    logger.info(f"连接到 MCP 服务器: {mcp_url}")
    
    try:
        # 创建 FastMCP 客户端
        client = Client(mcp_url)
        
        # 连接到服务器并获取工具列表
        async with client:
            logger.info(f"客户端已连接: {client.is_connected()}")
            
            # 获取工具列表
            tools = await client.list_tools()
            logger.info(f"获取到 {len(tools)} 个工具")
            
            # 获取原始 MCP 工具定义
            try:
                tools_mcp = await client.list_tools_mcp()
                logger.info(f"获取到原始 MCP 工具定义")
            except Exception as e:
                logger.warning(f"获取原始 MCP 工具定义失败: {e}")
                tools_mcp = None
            
            # 打印工具信息
            print("\n" + "=" * 80)
            print(f"MCP 服务器工具列表 (共 {len(tools)} 个工具)")
            print("=" * 80)
            
            for i, tool in enumerate(tools, 1):
                print(f"\n{i}. 名称: {tool.name}")
                print(f"   描述: {tool.description.split(chr(10))[0] if chr(10) in tool.description else tool.description}")
                
                # 打印参数信息
                if hasattr(tool, 'parameters'):
                    try:
                        params = tool.parameters
                        if isinstance(params, dict) and 'properties' in params:
                            print(f"   参数: {json.dumps(params['properties'], ensure_ascii=False, indent=2)}")
                    except Exception as e:
                        print(f"   参数: 无法解析 - {e}")
            
            # 保存工具列表到文件
            output_file = os.path.join(os.path.dirname(__file__), "available_mcp_tools.json")
            
            # 如果有原始 MCP 工具定义，使用原始定义
            if tools_mcp and hasattr(tools_mcp, 'tools'):
                logger.info(f"使用原始 MCP 工具定义保存")
                # 尝试将原始工具定义转换为字典
                try:
                    tools_dict = []
                    for tool in tools_mcp.tools:
                        tool_dict = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                            }
                        }
                        
                        # 添加参数信息
                        if hasattr(tool, 'parameters') and tool.parameters:
                            tool_dict["function"]["parameters"] = tool.parameters
                            
                        # 添加其他元数据
                        if hasattr(tool, 'tags') and tool.tags:
                            tool_dict["function"]["tags"] = tool.tags
                            
                        tools_dict.append(tool_dict)
                        
                    # 保存详细工具定义
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(tools_dict, f, ensure_ascii=False, indent=2)
                    logger.info(f"工具列表已保存到 {output_file}")
                except Exception as e:
                    logger.error(f"保存原始 MCP 工具定义失败: {e}")
                    # 如果失败，尝试保存原始对象的字符串表示
                    try:
                        with open(output_file + ".raw", "w", encoding="utf-8") as f:
                            f.write(str(tools_mcp))
                        logger.info(f"原始 MCP 工具定义已保存到 {output_file}.raw")
                    except Exception as e2:
                        logger.error(f"保存原始 MCP 工具定义字符串失败: {e2}")
            else:
                # 使用普通工具列表
                logger.info(f"使用普通工具列表保存")
                with open(output_file, "w", encoding="utf-8") as f:
                    tools_dict = []
                    for tool in tools:
                        tool_dict = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description
                            }
                        }
                        if hasattr(tool, 'parameters'):
                            tool_dict["function"]["parameters"] = tool.parameters
                        tools_dict.append(tool_dict)
                    json.dump(tools_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"工具列表已保存到 {output_file}")
            
            return tools
    except Exception as e:
        logger.error(f"获取工具列表失败: {e}")
        return None

async def test_call_tool(tool_name, arguments=None):
    """测试调用 MCP 工具"""
    if not HAS_FASTMCP:
        logger.error("fastmcp 未安装，无法使用 FastMCP 客户端")
        return None
    
    # MCP 服务器 URL
    mcp_url = os.getenv("MCP_URL", "http://34.124.211.73:7223/mcp")
    logger.info(f"连接到 MCP 服务器: {mcp_url}")
    
    try:
        # 创建 FastMCP 客户端
        client = Client(mcp_url)
        
        # 连接到服务器并调用工具
        async with client:
            logger.info(f"客户端已连接: {client.is_connected()}")
            
            # 调用工具
            logger.info(f"调用工具: {tool_name}, 参数: {arguments}")
            result = await client.call_tool(tool_name, arguments)
            
            # 打印结果
            print("\n" + "=" * 80)
            print(f"工具调用结果")
            print("=" * 80)
            
            for i, content in enumerate(result, 1):
                if hasattr(content, 'text'):
                    print(f"\n{i}. 文本内容: {content.text}")
                elif hasattr(content, 'image'):
                    print(f"\n{i}. 图片内容: {content.image}")
                else:
                    print(f"\n{i}. 内容: {content}")
            
            return result
    except Exception as e:
        logger.error(f"调用工具失败: {e}")
        return None

async def main():
    """主函数"""
    # 获取工具列表
    tools = await list_mcp_tools()
    
    # 如果获取到工具列表，可以选择一个工具进行测试
    if tools and len(tools) > 0:
        # 获取第一个工具的名称
        first_tool = tools[0]
        logger.info(f"准备测试第一个工具: {first_tool.name}")
        
        # 构建测试参数 (这里需要根据实际工具的参数进行调整)
        test_args = {}
        
        # 测试调用工具
        await test_call_tool(first_tool.name, test_args)

if __name__ == "__main__":
    # 使用 .venv 环境运行
    import sys
    venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".venv")
    if os.path.exists(venv_path):
        python_path = os.path.join(venv_path, "bin", "python3")
        if os.path.exists(python_path):
            if sys.executable != python_path:
                logger.info(f"切换到 .venv 环境: {python_path}")
                os.execl(python_path, python_path, *sys.argv)
    
    # 运行主函数
    asyncio.run(main())
