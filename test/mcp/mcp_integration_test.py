#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP 集成测试脚本

此脚本用于测试 xDAN 模型与 MCP 服务的集成，验证模型是否能够正确生成工具调用请求，
并处理工具调用的响应。
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# 尝试导入 langchain_mcp_adapters
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置
API_URL = os.getenv("API_URL", "http://14.103.133.112:8006/v1")
MODEL = os.getenv("MODEL", "xDAN-R1-Thinking-ToolRL-step105-0506")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# 全局变量
mcp_client = None

# 系统提示
SYSTEM_PROMPT = """
你是一个智能金融分析助手，可以帮助用户分析股票和金融数据。
你可以使用各种工具来获取信息，然后根据这些信息进行分析和决策。

请按照以下步骤处理用户请求：
1. 分析用户需求，确定需要获取哪些信息
2. 规划工作流程，决定使用哪些工具以及调用顺序
3. 执行工具调用，获取必要信息
4. 根据获取的信息进行分析和推理
5. 如有必要，进行后续工具调用以获取更多信息
6. 最终提供全面的分析结果和建议

在你的回复中，请使用以下格式：

<think>
在这里写下你的思考过程，包括对用户需求的理解、工作流规划、分析推理等
</think>

如果你需要调用工具，请使用以下格式：

<tool_call>
{"name": "工具名称", "parameters": {"参数名": "参数值"}}
</tool_call>

如果你要给用户最终回复，请使用以下格式：

<response>
在这里写下你对用户的最终回复
</response>
"""

def load_tools() -> List[Dict[str, Any]]:
    """加载 MCP 工具定义"""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_tools.json"), "r", encoding="utf-8") as f:
            tools_data = json.load(f)

        # 转换为我们需要的格式
        tools = []
        for tool_data in tools_data:
            if "function" in tool_data:
                function_data = tool_data["function"]
                tools.append({
                    "name": function_data["name"],
                    "description": function_data["description"],
                    "parameters": function_data["parameters"]
                })

        logger.info(f"加载了 {len(tools)} 个 MCP 工具")
        return tools
    except Exception as e:
        logger.error(f"加载工具定义失败: {e}")
        return []

def format_tools(tools: List[Dict[str, Any]]) -> str:
    """格式化工具列表，用于系统提示"""
    lines = []
    for idx, t in enumerate(tools, 1):
        props = json.dumps(t["parameters"]["properties"] if "properties" in t["parameters"] else {}, ensure_ascii=False)
        description = t['description'].split('\n')[0] if '\n' in t['description'] else t['description']
        lines.append(f"{idx}. 名称: {t['name']}\n   描述: {description}\n   参数: {props}")
    return "\n".join(lines)

def parse_tool_calls(output: str) -> List[Dict[str, Any]]:
    """解析模型输出中的工具调用"""
    calls = []
    if "<tool_call>" in output and "</tool_call>" in output:
        # 尝试提取所有工具调用
        parts = output.split("<tool_call>")
        for i in range(1, len(parts)):
            if "</tool_call>" in parts[i]:
                body = parts[i].split("</tool_call>")[0].strip()
                try:
                    call = json.loads(body)
                    # 确保参数字段名称一致
                    if "arguments" in call and "parameters" not in call:
                        call["parameters"] = call.pop("arguments")
                    calls.append(call)
                except json.JSONDecodeError as e:
                    logger.error(f"解析工具调用失败: {e}, 原始文本: {body}")
    return calls

def parse_thinking(output: str) -> str:
    """提取模型输出中的思考内容"""
    if "<think>" in output and "</think>" in output:
        return output.split("<think>")[1].split("</think>")[0].strip()
    return ""

def parse_response(output: str) -> str:
    """提取模型输出中的响应内容"""
    if "<response>" in output and "</response>" in output:
        return output.split("<response>")[1].split("</response>")[0].strip()
    return ""

def init_mcp_client():
    """初始化 MCP 客户端"""
    global mcp_client
    
    if not HAS_MCP_CLIENT:
        logger.warning("langchain_mcp_adapters 未安装，无法使用 MultiServerMCPClient")
        return None
    
    try:
        # MCP 服务配置
        mcp_url = os.getenv("MCP_URL", "http://34.87.170.99:7223/mcp")
        server_name = os.getenv("MCP_SERVER_NAME", "xDAN_tools")
        
        # 创建 MCP 客户端配置
        config = {
            server_name: {
                "url": mcp_url,
                "transport": "sse"
            }
        }
        
        # 初始化 MCP 客户端
        mcp_client = MultiServerMCPClient(config)
        logger.info(f"已初始化 MCP 客户端: {server_name} -> {mcp_url}")
        
        # 获取并打印可用工具列表
        try:
            tools = mcp_client.get_tools()
            logger.info(f"共有 {len(tools)} 个可用工具:")
            for i, tool in enumerate(tools, 1):
                logger.info(f"{i}. {tool.name}: {tool.description}")
            return mcp_client
        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")
            return None
    except Exception as e:
        logger.error(f"初始化 MCP 客户端失败: {e}")
        return None

async def execute_tool_call(mcp_client, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """通过 MCP 服务执行工具调用"""
    logger.info(f"执行工具调用: {tool_name}, 参数: {parameters}")
    
    if not mcp_client:
        logger.error("MCP 客户端未初始化")
        return {
            "status": "error",
            "message": "MCP 客户端未初始化"
        }
    
    try:
        # 获取工具列表
        tools = mcp_client.get_tools()
        
        # 查找对应的工具
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool:
            logger.info(f"使用 MCP 客户端调用工具: {tool_name}")
            result = await tool.ainvoke(parameters)
            logger.info(f"MCP 客户端响应: {result}")
            return result
        else:
            logger.error(f"未找到工具: {tool_name}")
            return {
                "status": "error",
                "message": f"未找到工具: {tool_name}"
            }
    except Exception as e:
        logger.error(f"MCP 客户端调用错误: {e}")
        return {
            "status": "error",
            "message": f"工具 {tool_name} 执行失败: {str(e)}"
        }

def call_api(prompt: str, max_tokens: int = 16384, temperature: float = TEMPERATURE) -> str:
    """调用远程API获取模型响应"""
    headers = {
        "Content-Type": "application/json"
    }

    # 提取系统消息和用户消息
    system_message = ""
    user_message = prompt

    if "<|im_start|>system" in prompt and "<|im_end|>" in prompt:
        parts = prompt.split("<|im_end|>")
        if len(parts) >= 2:
            system_part = parts[0].split("<|im_start|>system")[1].strip()
            system_message = system_part

            user_part = parts[1].split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
            user_message = user_part

    # 构建消息数组
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        logger.info(f"调用API: {API_URL}")
        response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"API调用错误: {e}")
        return ""

async def run_multi_round_chat(mcp_client, tools: List[Dict[str, Any]], user_query: str, max_rounds: int = 5) -> str:
    """
    运行多轮对话，模型可以自主决定工具调用顺序

    Args:
        mcp_client: MCP 客户端
        tools: 可用工具列表
        user_query: 用户查询
        max_rounds: 最大对话轮数

    Returns:
        最终的对话结果
    """
    # 初始化对话历史
    conversation_history = []
    
    # 添加用户查询
    conversation_history.append({"role": "user", "content": user_query})
    
    # 记录所有响应
    all_responses = []
    final_response = ""
    
    # 多轮对话循环
    for round_idx in range(max_rounds):
        logger.info(f"对话轮次 {round_idx + 1}/{max_rounds}")
        
        # 构建提示
        user_prompt = "对话历史\n"
        
        for message in conversation_history:
            if message["role"] == "user":
                user_prompt += f"<user> {message['content']} </user>\n"
            elif message["role"] == "assistant":
                user_prompt += f"{message['content']}\n"
            elif message["role"] == "tool":
                tool_name = message.get("name", "unknown_tool")
                tool_result = message.get("content", "{}")
                user_prompt += f"<obs> 工具调用 {tool_name} 的执行结果: {tool_result} </obs>\n"
        
        # 构建提示词
        system_part = "<|im_start|>system\n" + SYSTEM_PROMPT + "\n\n可用工具列表:\n" + format_tools(tools) + "<|im_end|>"
        user_part = "<|im_start|>user\n" + user_prompt.strip() + "<|im_end|>"
        assistant_part = "<|im_start|>assistant\n"
        
        # 合并所有部分
        prompt = system_part + "\n" + user_part + "\n" + assistant_part
        
        # 调用API获取响应
        model_output = call_api(prompt)
        logger.info(f"模型输出: {model_output[:200]}...")
        
        # 解析模型输出
        thinking = parse_thinking(model_output)
        tool_calls = parse_tool_calls(model_output)
        response = parse_response(model_output)
        
        # 记录模型响应
        conversation_history.append({"role": "assistant", "content": model_output})
        
        # 如果没有工具调用或者达到最大轮数，结束对话
        if not tool_calls or round_idx == max_rounds - 1:
            final_response = response or model_output
            break
        
        # 执行工具调用
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            parameters = tool_call.get("parameters", {})
            
            # 异步执行工具调用
            result = await execute_tool_call(mcp_client, tool_name, parameters)
            
            # 添加工具响应到对话历史
            conversation_history.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)
            })
            
            # 记录工具调用结果
            logger.info(f"工具 {tool_name} 执行结果: {json.dumps(result, ensure_ascii=False)[:200]}...")
    
    return final_response

async def main():
    """主函数"""
    # 初始化 MCP 客户端
    client = init_mcp_client()
    
    if not client and HAS_MCP_CLIENT:
        logger.error("MCP 客户端初始化失败，请检查 MCP 服务器配置")
        return
    
    # 加载工具定义
    tools = load_tools()
    
    # 测试场景
    test_scenarios = [
        {
            "name": "股票信息查询",
            "query": "请查询平安银行(000001.SZ)的基本信息。"
        },
        {
            "name": "股票日线数据查询",
            "query": "请查询平安银行(000001.SZ)最近一周的日线数据。"
        },
        {
            "name": "财务指标查询",
            "query": "请查询平安银行(000001.SZ)的主要财务指标。"
        }
    ]
    
    # 运行测试场景
    for scenario in test_scenarios:
        logger.info(f"开始测试场景: {scenario['name']}")
        logger.info(f"用户查询: {scenario['query']}")
        
        result = await run_multi_round_chat(client, tools, scenario['query'])
        
        logger.info(f"最终结果: {result[:200]}...")
        logger.info("=" * 80)

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
