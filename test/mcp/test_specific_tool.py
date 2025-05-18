#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试特定MCP工具的脚本

此脚本用于测试特定的MCP工具，并获取其参数信息。
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

# 导入 fastmcp
from fastmcp import Client

async def test_tool(tool_name: str, args: Dict[str, Any] = None):
    """测试特定的MCP工具"""
    # MCP 服务器 URL
    mcp_url = os.getenv("MCP_URL", "http://34.124.211.73:7223/mcp")
    logger.info(f"连接到 MCP 服务器: {mcp_url}")
    
    try:
        # 创建 FastMCP 客户端
        client = Client(mcp_url)
        
        # 连接到服务器
        async with client:
            logger.info(f"客户端已连接: {client.is_connected()}")
            
            # 获取工具列表
            tools = await client.list_tools()
            logger.info(f"获取到 {len(tools)} 个工具")
            
            # 查找特定工具
            target_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    target_tool = tool
                    break
            
            if not target_tool:
                logger.error(f"未找到工具: {tool_name}")
                return
            
            logger.info(f"找到工具: {tool_name}")
            logger.info(f"工具描述: {target_tool.description}")
            
            # 尝试获取工具参数信息
            if hasattr(target_tool, 'parameters'):
                logger.info(f"工具参数: {json.dumps(target_tool.parameters, ensure_ascii=False, indent=2)}")
            else:
                logger.info("工具没有参数信息")
            
            # 尝试调用工具
            if args is None:
                args = {}
            
            logger.info(f"调用工具 {tool_name} 参数: {json.dumps(args, ensure_ascii=False)}")
            try:
                result = await client.call_tool(tool_name, args)
                
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
    except Exception as e:
        logger.error(f"连接服务器失败: {e}")
        return None

async def main():
    """主函数"""
    # 要测试的工具名称
    tool_name = "yahooFinanceMcp_get_stock_info"
    
    # 测试参数
    args = {
        "symbol": "AAPL"  # 苹果公司股票代码
    }
    
    # 测试工具
    await test_tool(tool_name, args)
    
    # 测试另一个工具
    tool_name2 = "miraSearchMcp_search_input"
    args2 = {
        "query": "最新的人工智能发展"
    }
    
    await test_tool(tool_name2, args2)

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
