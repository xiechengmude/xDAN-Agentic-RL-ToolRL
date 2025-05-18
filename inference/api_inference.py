import json
import requests
from typing import List, Dict, Any, Tuple, Optional

# 配置
API_URL = "http://14.103.133.112:8002/v1"
MODEL = "xDAN-R2-Thinking-32b-Agent-RL-step225"
TEMPERATURE = 0.7

# 定义系统提示
SYSTEM_PROMPT = (
    "You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.\n\n"
    "**Available Tools**\n"
    "In your response, you can use the following tools:\n{tools}\n\n"
    "**Steps for Each Turn**\n"
    "1. **Think:** Recall relevant context and analyze the current user goal.\n"
    "2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.\n"
    "3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.\n\n"
    "**Output Format**\n```plaintext\n"
    "<think> Your thoughts and reasoning </think>\n"
    "<tool_call>\n"
    '{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}\n'
    '{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}\n'
    "...\n"
    "</tool_call>\n"
    "<response> AI's final response </response>\n"
    "```\n\n"
    "**Important Notes**\n"
    "1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. "
    "Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.\n"
    '2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" '
    'field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.\n'
    "3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback "
    "noted as `<obs>` (if exists).\n"
)

# 工具格式化函数，用于系统提示中的工具列表，应该与MCP兼容
def format_tools(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, t in enumerate(tools, 1):
        props = json.dumps(t["parameters"]["properties"])
        lines.append(f"{idx}. Name: {t['name']}\nDescription: {t['description']}\nParameters: {props}")
    return "\n".join(lines)

# 从模型输出中解析工具调用
def parse_tool_calls(output: str) -> List[Dict[str, Any]]:
    calls = []
    if "<tool_call>" in output:
        body = output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        for line in body.splitlines():
            try:
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

# 主要推理函数
def run_chat(tools: List[Dict[str, Any]], messages: List[Dict[str, str]], 
             turn_type: str = "multi_turn") -> Tuple[str, List[Dict[str, Any]]]:
    """
    使用远程API进行对话推理
    
    参数:
        tools: 可用工具列表
        messages: 对话历史
        turn_type: 对话类型，"multi_turn"或其他
        
    返回:
        完整的模型输出和解析后的工具调用列表
    """
    USER_PROMPT = "**Dialogue Records History**\n"

    for msg_idx, message in enumerate(messages):
        if message["role"] == "system":
            continue

        elif message["role"] == "user":
            if turn_type == "multi_turn":
                USER_PROMPT += f"<user> {message['content'].strip()}\nUse the one or more necessary tool calls to complete the task. You could "
                USER_PROMPT += "perform tool calls for multiple rounds so you can try and error. Please make a comprehensive plan about how to "
                USER_PROMPT += "achieve the goal step by step, and begin to call the tool step by step. If no tools apply or required parameters "
                USER_PROMPT += "are missing, please also directly state this in your response without tool calls. </user>\n"
            else:
                USER_PROMPT += f"<user> {message['content'].strip()}\nIf there's no appropriate tools to apply or required parameters are missing, "
                USER_PROMPT += "please directly inform me in your response without any tool call, or call the tool with the name as 'None'. Otherwise, "
                USER_PROMPT += "you should use one or more necessary tool calls to complete the given task in this turn. </user>\n"

        elif message["role"] == "tool":
            tool_name = message["name"].strip()
            tool_result = message["content"].strip()
            USER_PROMPT += f"<obs> You have made the tool call {tool_name}. Execution returns: {tool_result} </obs>\n"
            if msg_idx == len(messages) - 1:
                USER_PROMPT += "\n<user> If you think you have completed the current task, or the task cannot be finished, please respond directly without "
                USER_PROMPT += "additional tool calls. If you encounter an error during tool execution or the task remains unfinished, retry with the one or "
                USER_PROMPT += "more necessary tool calls according to your thought and plan until completion. Based on the tool execution feedback, reflect on "
                USER_PROMPT += "if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on. </user>\n"

        elif message["role"] == "assistant":
            # 在历史记录中使思考部分更简短和简洁
            content = message["content"].strip()
            message["content"] = content
            USER_PROMPT += f"\n{message['content'].strip()}\n"

    USER_PROMPT = USER_PROMPT.strip()

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT.format(tools=format_tools(tools))}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"

    # 调用API获取响应
    generated_text = call_api(prompt)
    calls = parse_tool_calls(generated_text)
    return generated_text, calls

# 示例用法
if __name__ == "__main__":
    history = [{"role": "user", "content": "What's the weather today in Helsinki?"}]

    # 示例可用工具描述，用于DeepResearch用例
    tools = [
        {
            "name": "search", 
            "description": "Search over a knowledge base.", 
            "parameters": {"properties": {"query": {"type": "string"}}}
        }, 
        {
            "name": "summarize", 
            "description": "Summarize a document or webpage given its URI.", 
            "parameters": {"properties": {"uri": {"type": "string"}}}
        }
    ]

    output, tool_calls = run_chat(tools, history)
    print("模型输出:\n", output)
    if tool_calls:
        print("解析的工具调用:\n", json.dumps(tool_calls, indent=2, ensure_ascii=False))

    # 现在可以执行实际的工具调用并将结果发送回模型（追加到历史记录）
    # 用于多轮交互
    # ......
