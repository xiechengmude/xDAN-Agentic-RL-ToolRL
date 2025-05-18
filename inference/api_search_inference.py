import json
import requests
import time
import argparse
import sys
import random
from typing import List, Dict, Any, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置
API_URL = "http://60.50.228.238:62735/v1"
MODEL = "Qwen2.5-7B-Instruct_ZeroSearch"
TEMPERATURE = 0.7

# 定义系统提示
SYSTEM_PROMPT = (
    "你是一个强大的助手，能够利用搜索工具来回答用户的问题。你应该思考如何通过多轮搜索获取最准确、最全面的信息。\n\n"
    "**可用工具**\n"
    "在你的回答中，你可以使用以下工具：\n{tools}\n\n"
    "**每轮步骤**\n"
    "1. **思考：** 分析用户问题，确定需要搜索的关键信息。\n"
    "2. **决定搜索策略：** 确定最佳搜索查询，可能需要多轮搜索以获取完整信息。\n"
    "3. **适当回应：** 基于搜索结果提供全面而准确的回答。\n\n"
    "**输出格式**\n```plaintext\n"
    "<think> 你的思考和推理过程 </think>\n"
    "<tool_call>\n"
    '{{"name": "工具名称", "parameters": {{"参数名称": "参数内容", "... ...": "... ..."}}}}\n'
    '{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}\n'
    "...\n"
    "</tool_call>\n"
    "<response> AI的最终回应 </response>\n"
    "```\n\n"
    "**重要说明**\n"
    "1. 你必须始终包含`<think>`字段来概述你的推理。提供至少一个`<tool_call>`或`<response>`。"
    "决定是使用`<tool_call>`（可能多次），`<response>`，或两者都用。\n"
    '2. 你可以在`<tool_call>`字段中同时调用多个工具。每个工具调用应该是一个JSON对象，包含"name"'
    '字段和包含参数字典的"parameters"字段。如果不需要参数，请将"parameters"字段留为空字典。\n'
    "3. 参考历史对话记录，包括用户的查询、之前的`<tool_call>`、`<response>`以及任何标记为`<obs>`的工具反馈"
    "（如果存在）。\n"
    "4. 对于复杂问题，考虑进行多轮搜索，每轮搜索基于前一轮的结果优化查询。\n"
)

# 工具格式化函数，用于系统提示中的工具列表
def format_tools(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, t in enumerate(tools, 1):
        props = json.dumps(t["parameters"]["properties"], ensure_ascii=False)
        lines.append(f"{idx}. 名称: {t['name']}\n描述: {t['description']}\n参数: {props}")
    return "\n".join(lines)

# 从模型输出中解析工具调用
def parse_tool_calls(output: str) -> List[Dict[str, Any]]:
    calls = []
    if "<tool_call>" in output:
        body = output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        for line in body.splitlines():
            try:
                line = line.strip()
                if not line:
                    continue
                call = json.loads(line)
                calls.append(call)
            except json.JSONDecodeError:
                continue
    return calls

# 从模型输出中提取思考内容
def parse_thinking(output: str) -> Optional[str]:
    if "<think>" in output and "</think>" in output:
        return output.split("<think>")[1].split("</think>")[0].strip()
    return None

# 从模型输出中提取响应内容
def parse_response(output: str) -> Optional[str]:
    if "<response>" in output and "</response>" in output:
        return output.split("<response>")[1].split("</response>")[0].strip()
    return None

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
        print(f"正在调用API: {API_URL}/chat/completions")
        print(f"请求负载: {json.dumps(payload, ensure_ascii=False)}")
        response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
        print(f"API响应状态码: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print(f"API响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
        # 从正确的位置获取响应内容
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_content = e.response.json()
                print(f"错误详情: {json.dumps(error_content, ensure_ascii=False, indent=2)}")
            except:
                print(f"响应内容: {e.response.text}")
        return ""
    except Exception as e:
        print(f"其他错误: {e}")
        return ""

# 模拟搜索工具的执行
def execute_search_tool(query: str) -> str:
    """
    模拟执行搜索工具，返回搜索结果
    在实际应用中，这里应该调用真实的搜索API
    """
    print(f"执行搜索: {query}")
    # 这里可以替换为实际的搜索API调用
    # 为了演示，我们返回一些模拟的搜索结果
    time.sleep(1)  # 模拟网络延迟
    return f"搜索结果: 关于'{query}'的信息如下...[这里是搜索结果的详细内容]"

# 执行工具调用
def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, str]:
    """
    执行工具调用并返回结果
    
    参数:
        tool_call: 解析后的工具调用
        
    返回:
        包含工具名称和执行结果的字典
    """
    tool_name = tool_call.get("name", "")
    parameters = tool_call.get("parameters", {})
    
    if tool_name == "search":
        query = parameters.get("query", "")
        result = execute_search_tool(query)
        return {"name": tool_name, "content": result}
    
    # 可以添加其他工具的处理逻辑
    
    return {"name": tool_name, "content": "不支持的工具调用"}

# 主要推理函数
def run_search_chat(tools: List[Dict[str, Any]], messages: List[Dict[str, str]], 
             max_search_rounds: int = 3) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    使用远程API进行搜索增强对话推理
    
    参数:
        tools: 可用工具列表
        messages: 对话历史
        max_search_rounds: 最大搜索轮数
        
    返回:
        完整的模型输出、解析后的工具调用列表和更新后的对话历史
    """
    history = messages.copy()
    current_round = 0
    final_output = ""
    all_tool_calls = []
    
    while current_round < max_search_rounds:
        USER_PROMPT = "**对话历史记录**\n"

        for message in history:
            if message["role"] == "system":
                continue
            elif message["role"] == "user":
                USER_PROMPT += f"<user> {message['content'].strip()} </user>\n"
            elif message["role"] == "tool":
                tool_name = message["name"].strip()
                tool_result = message["content"].strip()
                USER_PROMPT += f"<obs> 你已经进行了工具调用 {tool_name}。执行结果: {tool_result} </obs>\n"
            elif message["role"] == "assistant":
                USER_PROMPT += f"\n{message['content'].strip()}\n"

        USER_PROMPT = USER_PROMPT.strip()

        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT.format(tools=format_tools(tools))}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"

        # 调用API获取响应
        generated_text = call_api(prompt)
        final_output = generated_text
        
        # 解析工具调用和响应
        tool_calls = parse_tool_calls(generated_text)
        all_tool_calls.extend(tool_calls)
        response_text = parse_response(generated_text)
        
        # 添加助手回应到历史
        history.append({"role": "assistant", "content": generated_text})
        
        # 如果没有工具调用或已达到最大轮数，结束循环
        if not tool_calls or current_round >= max_search_rounds - 1:
            break
        
        # 执行工具调用并添加结果到历史
        for tool_call in tool_calls:
            tool_result = execute_tool_call(tool_call)
            history.append({"role": "tool", "name": tool_result["name"], "content": tool_result["content"]})
        
        current_round += 1
        print(f"完成第 {current_round} 轮搜索")
    
    return final_output, all_tool_calls, history

# 主函数
def main(query: str, max_search_rounds: int = 3):
    """
    主函数，处理用户查询
    
    参数:
        query: 用户查询
        max_search_rounds: 最大搜索轮数
    """
    # 初始化对话历史
    history = [{"role": "user", "content": query}]
    
    # 定义可用工具
    tools = [
        {
            "name": "search", 
            "description": "搜索知识库获取信息。", 
            "parameters": {"properties": {"query": {"type": "string", "description": "搜索查询"}}}
        }
    ]
    
    # 运行搜索增强对话
    output, tool_calls, updated_history = run_search_chat(tools, history, max_search_rounds)
    
    # 输出结果
    print("\n===== 最终结果 =====")
    print("模型输出:\n", output)
    
    thinking = parse_thinking(output)
    if thinking:
        print("\n思考过程:\n", thinking)
    
    response = parse_response(output)
    if response:
        print("\n最终回应:\n", response)
    
    if tool_calls:
        print("\n工具调用:\n", json.dumps(tool_calls, indent=2, ensure_ascii=False))
    
    print("\n===== 对话历史 =====")
    for i, msg in enumerate(updated_history):
        role = msg["role"]
        if role == "user":
            print(f"用户: {msg['content']}")
        elif role == "assistant":
            print(f"助手: [输出省略]")
        elif role == "tool":
            print(f"工具 ({msg['name']}): {msg['content'][:50]}...")
    
    return response, tool_calls, updated_history

# 示例用法
if __name__ == "__main__":
    # 用户查询示例
    user_query = "请告诉我关于量子计算的最新进展"
    main(user_query, max_search_rounds=3)
