#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多轮次工作流测试脚本

此脚本用于测试 xDAN 模型的多轮次对话能力，模型可以自我思考并动态编排工作流，
实现多次工具调用来完成复杂任务。
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import requests
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置
API_URL = "http://14.103.133.112:8006/v1"
MODEL = "xDAN-R1-Thinking-ToolRL-step105-0506"
TEMPERATURE = 0.7

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

<tool_call>
{"name": "工具名称", "parameters": {"参数名": "参数值"}}
</tool_call>

<response>
在这里提供对用户的最终回复
</response>

请确保你的分析全面且深入，充分利用可用的工具获取必要的信息。
"""

# 从 mcp_tools.json 加载工具定义
def load_tools() -> List[Dict[str, Any]]:
    """加载 MCP 工具定义"""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mcp_tools.json"), "r", encoding="utf-8") as f:
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

# 格式化工具列表
def format_tools(tools: List[Dict[str, Any]]) -> str:
    """格式化工具列表，用于系统提示"""
    lines = []
    for idx, t in enumerate(tools, 1):
        props = json.dumps(t["parameters"]["properties"] if "properties" in t["parameters"] else {}, ensure_ascii=False)
        lines.append(f"{idx}. 名称: {t['name']}\n   描述: {t['description'].split('\\n')[0]}\n   参数: {props}")
    return "\n".join(lines)

# 从模型输出中解析工具调用
def parse_tool_calls(output: str) -> List[Dict[str, Any]]:
    """解析模型输出中的工具调用"""
    calls = []
    if "<tool_call>" in output:
        body = output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        for line in body.splitlines():
            try:
                call = json.loads(line)
                # 确保参数字段名称一致
                if "arguments" in call and "parameters" not in call:
                    call["parameters"] = call.pop("arguments")
                calls.append(call)
            except json.JSONDecodeError:
                continue
    return calls

# 从模型输出中提取思考内容
def parse_thinking(output: str) -> Optional[str]:
    """提取模型输出中的思考内容"""
    if "<think>" in output and "</think>" in output:
        return output.split("<think>")[1].split("</think>")[0].strip()
    return None

# 从模型输出中提取响应内容
def parse_response(output: str) -> Optional[str]:
    """提取模型输出中的响应内容"""
    if "<response>" in output and "</response>" in output:
        return output.split("<response>")[1].split("</response>")[0].strip()
    return None

# 模拟执行工具调用
async def execute_tool_call(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    模拟执行工具调用
    
    在实际应用中，这里应该调用真实的 MCP 服务执行工具
    """
    logger.info(f"执行工具调用: {tool_name}, 参数: {parameters}")
    
    # 模拟 Tushare 相关工具的响应
    if tool_name == "finance_tushare_token_status":
        # 模拟 Tushare Token 状态
        return {
            "status": "valid",
            "expire_date": "2026-05-07",
            "calls_used": 3500,
            "calls_limit": 10000,
            "vip_level": 1
        }
    
    elif tool_name == "finance_tushare_stock_info":
        ts_code = parameters.get("ts_code", "000001.SZ")
        name = parameters.get("name", "平安银行")
        
        # 模拟股票基本信息
        return {
            "data": [
                {
                    "ts_code": ts_code or "000001.SZ",
                    "name": name or "平安银行",
                    "area": "深圳",
                    "industry": "银行",
                    "list_date": "19910403",
                    "market": "主板"
                }
            ]
        }
    
    elif tool_name == "finance_tushare_search_stocks":
        keyword = parameters.get("keyword", "平安")
        
        # 模拟股票搜索结果
        return {
            "data": [
                {
                    "ts_code": "000001.SZ",
                    "name": "平安银行",
                    "area": "深圳",
                    "industry": "银行",
                    "list_date": "19910403",
                    "market": "主板"
                },
                {
                    "ts_code": "601318.SH",
                    "name": "中国平安",
                    "area": "深圳",
                    "industry": "保险",
                    "list_date": "20070301",
                    "market": "主板"
                },
                {
                    "ts_code": "000627.SZ",
                    "name": "天茂实业",
                    "area": "湖北",
                    "industry": "保险",
                    "list_date": "19960712",
                    "market": "主板"
                }
            ]
        }
    
    elif tool_name == "finance_tushare_income_statement":
        ts_code = parameters.get("ts_code", "000001.SZ")
        
        # 模拟利润表数据
        return {
            "data": [
                {
                    "ts_code": ts_code,
                    "ann_date": "20241231",
                    "f_ann_date": "20250331",
                    "end_date": "20241231",
                    "report_type": "1",
                    "comp_type": "1",
                    "total_revenue": 1650000000.0,
                    "revenue": 1620000000.0,
                    "int_income": 1200000000.0,
                    "n_income": 850000000.0,
                    "n_income_attr_p": 845000000.0
                },
                {
                    "ts_code": ts_code,
                    "ann_date": "20230930",
                    "f_ann_date": "20231030",
                    "end_date": "20230930",
                    "report_type": "1",
                    "comp_type": "1",
                    "total_revenue": 1520000000.0,
                    "revenue": 1490000000.0,
                    "int_income": 1100000000.0,
                    "n_income": 780000000.0,
                    "n_income_attr_p": 775000000.0
                }
            ]
        }
    
    elif tool_name == "finance_tushare_daily":
        ts_code = parameters.get("ts_code", "000001.SZ")
        
        # 模拟日线行情数据
        today = datetime.now()
        dates = [(today.replace(day=today.day-i)).strftime("%Y%m%d") for i in range(30)]
        opens = [13.25 + (i * 0.1 - 1.5) for i in range(30)]
        highs = [13.45 + (i * 0.1 - 1.5) for i in range(30)]
        lows = [13.05 + (i * 0.1 - 1.5) for i in range(30)]
        closes = [13.35 + (i * 0.1 - 1.5) for i in range(30)]
        vols = [250000 + (i * 10000) for i in range(30)]
        
        return {
            "data": [
                {
                    "ts_code": ts_code,
                    "trade_date": dates[i],
                    "open": opens[i],
                    "high": highs[i],
                    "low": lows[i],
                    "close": closes[i],
                    "vol": vols[i],
                    "amount": vols[i] * closes[i],
                    "change": closes[i] - closes[i-1] if i > 0 else 0.1,
                    "pct_chg": (closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0.8
                }
                for i in range(30)
            ]
        }
    
    # 默认返回空结果
    return {"status": "success", "message": f"工具 {tool_name} 执行成功，但没有模拟数据"}

# API调用函数
def call_api(prompt: str, max_tokens: int = 16384, temperature: float = TEMPERATURE) -> str:
    """调用远程API获取模型响应"""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        logger.info(f"调用API: {API_URL}")
        response = requests.post(f"{API_URL}/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "")
    except Exception as e:
        logger.error(f"API调用错误: {e}")
        return ""

# 多轮对话函数
async def run_multi_round_chat(tools: List[Dict[str, Any]], user_query: str, max_rounds: int = 5) -> str:
    """
    运行多轮对话，模型可以自主决定工具调用顺序
    
    Args:
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
        user_prompt = "**对话历史**\n"
        
        for message in conversation_history:
            if message["role"] == "user":
                user_prompt += f"<user> {message['content']} </user>\n"
            elif message["role"] == "assistant":
                user_prompt += f"{message['content']}\n"
            elif message["role"] == "tool":
                tool_name = message.get("name", "unknown_tool")
                tool_result = message.get("content", "{}")
                user_prompt += f"<obs> 工具调用 {tool_name} 的执行结果: {tool_result} </obs>\n"
        
        # 添加系统提示
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n可用工具列表:\n{format_tools(tools)}<|im_end|>\n<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
        
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
            
            # 执行工具调用
            result = await execute_tool_call(tool_name, parameters)
            
            # 添加工具响应到对话历史
            conversation_history.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)
            })
            
            # 记录工具调用结果
            logger.info(f"工具 {tool_name} 执行结果: {json.dumps(result, ensure_ascii=False)[:200]}...")
    
    return final_response

# 主函数
async def main():
    """主函数"""
    # 加载工具
    tools = load_tools()
    
    # 测试场景
    test_scenarios = [
        {
            "name": "A股分析",
            "query": "帮我分析平安银行(000001.SZ)的股票，包括基本面、技术面和行业对比，并给出投资建议。"
        },
        {
            "name": "行业对比",
            "query": "请对比分析中国平安(601318.SH)和招商银行(600036.SH)这两家公司，从财务状况、市场表现和未来前景等方面进行评估。"
        },
        {
            "name": "投资组合建议",
            "query": "我想投资中国金融行业，预算10万人民币，请推荐一个包含3-5只股票的投资组合，并说明理由。"
        }
    ]
    
    # 运行测试场景
    for scenario in test_scenarios:
        logger.info(f"开始测试场景: {scenario['name']}")
        logger.info(f"用户查询: {scenario['query']}")
        
        result = await run_multi_round_chat(tools, scenario['query'])
        
        logger.info(f"最终结果: {result[:200]}...")
        logger.info("=" * 80)

# 入口点
if __name__ == "__main__":
    # 使用 .venv 环境运行
    import sys
    import os
    venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".venv", "lib", "python3.11", "site-packages")
    sys.path.insert(0, venv_path)
    
    asyncio.run(main())
