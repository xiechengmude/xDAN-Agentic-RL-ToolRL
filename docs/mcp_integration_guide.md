# xDAN 模型与 MCP 服务集成指南

本指南旨在帮助开发者测试 xDAN 模型是否兼容 MCP (Model Control Protocol) 协议，以及如何将 xDAN 模型与 MCP 服务进行集成。

## 目录

- [背景介绍](#背景介绍)
- [环境准备](#环境准备)
- [MCP 协议概述](#mcp-协议概述)
- [xDAN 模型的工具调用格式](#xdan-模型的工具调用格式)
- [集成步骤](#集成步骤)
- [测试方法](#测试方法)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

## 背景介绍

MCP (Model Control Protocol) 是一种用于大语言模型与外部工具交互的协议。通过 MCP，模型可以调用外部工具来获取信息或执行操作，从而增强模型的能力。

xDAN 模型是一种支持工具调用的大语言模型，可以通过特定的格式生成工具调用请求，并处理工具调用的响应。本指南将介绍如何测试 xDAN 模型是否兼容 MCP 协议，以及如何将 xDAN 模型与 MCP 服务进行集成。

## 环境准备

### 依赖安装

首先，确保您已经安装了必要的依赖：

```bash
pip install langchain-mcp-adapters requests python-dotenv
```

### 环境变量配置

创建一个 `.env` 文件，并配置以下环境变量：

```
MCP_URL=http://your-mcp-server-url/mcp
MCP_SERVER_NAME=your_server_name
API_URL=http://your-model-api-url/v1
```

## MCP 协议概述

MCP 协议定义了模型与工具之间的交互方式。主要包括以下几个部分：

1. **工具定义**：描述工具的名称、功能和参数。
2. **工具调用**：模型生成的工具调用请求，包含工具名称和参数。
3. **工具响应**：工具执行后返回的结果。

MCP 服务器通常会提供一个 API 端点，用于接收工具调用请求并返回工具执行结果。

## xDAN 模型的工具调用格式

xDAN 模型使用特定的格式生成工具调用请求。具体格式如下：

```
<think>
模型的思考过程，包括对用户需求的理解、工作流规划、分析推理等
</think>

<tool_call>
{"name": "工具名称", "parameters": {"参数名": "参数值"}}
</tool_call>

<response>
模型对用户的最终回复
</response>
```

其中：
- `<think>...</think>` 标签包含模型的思考过程。
- `<tool_call>...</tool_call>` 标签包含 JSON 格式的工具调用请求，包括工具名称和参数。
- `<response>...</response>` 标签包含模型对用户的最终回复。

## 集成步骤

### 1. 初始化 MCP 客户端

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

def init_mcp_client():
    """初始化 MCP 客户端"""
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
    print(f"已初始化 MCP 客户端: {server_name} -> {mcp_url}")
    
    return mcp_client
```

### 2. 解析工具调用

```python
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
                    print(f"解析工具调用失败: {e}, 原始文本: {body}")
    return calls
```

### 3. 执行工具调用

```python
async def execute_tool_call(mcp_client, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """通过 MCP 服务执行工具调用"""
    print(f"执行工具调用: {tool_name}, 参数: {parameters}")
    
    try:
        # 获取工具列表
        tools = mcp_client.get_tools()
        
        # 查找对应的工具
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool:
            print(f"使用 MCP 客户端调用工具: {tool_name}")
            result = await tool.ainvoke(parameters)
            print(f"MCP 客户端响应: {result}")
            return result
        else:
            print(f"未找到工具: {tool_name}")
            return {
                "status": "error",
                "message": f"未找到工具: {tool_name}"
            }
    except Exception as e:
        print(f"MCP 客户端调用错误: {e}")
        return {
            "status": "error",
            "message": f"工具 {tool_name} 执行失败: {str(e)}"
        }
```

### 4. 多轮对话实现

```python
async def run_multi_round_chat(mcp_client, tools: List[Dict[str, Any]], user_query: str, max_rounds: int = 5) -> str:
    """运行多轮对话，模型可以自主决定工具调用顺序"""
    # 初始化对话历史
    conversation_history = []
    
    # 添加用户查询
    conversation_history.append({"role": "user", "content": user_query})
    
    # 记录所有响应
    all_responses = []
    final_response = ""
    
    # 多轮对话循环
    for round_idx in range(max_rounds):
        print(f"对话轮次 {round_idx + 1}/{max_rounds}")
        
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
        print(f"模型输出: {model_output[:200]}...")
        
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
            print(f"工具 {tool_name} 执行结果: {json.dumps(result, ensure_ascii=False)[:200]}...")
    
    return final_response
```

### 5. 主函数

```python
async def main():
    """主函数"""
    # 初始化 MCP 客户端
    mcp_client = init_mcp_client()
    
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
        print(f"开始测试场景: {scenario['name']}")
        print(f"用户查询: {scenario['query']}")
        
        result = await run_multi_round_chat(mcp_client, tools, scenario['query'])
        
        print(f"最终结果: {result[:200]}...")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
```

## 测试方法

### 1. 基本测试

运行以下命令，测试模型是否能够正确生成工具调用请求，并处理工具调用的响应：

```bash
python test_mcp_integration.py
```

### 2. 单工具测试

测试模型是否能够正确调用单个工具：

```python
# 测试场景
test_scenarios = [
    {
        "name": "股票信息查询",
        "query": "请查询平安银行(000001.SZ)的基本信息。"
    }
]
```

### 3. 多工具测试

测试模型是否能够正确调用多个工具，并根据工具调用的结果进行后续的工具调用：

```python
# 测试场景
test_scenarios = [
    {
        "name": "股票分析",
        "query": "请分析平安银行(000001.SZ)的股票，包括基本面、技术面和行业对比，并给出投资建议。"
    }
]
```

### 4. 错误处理测试

测试模型是否能够正确处理工具调用失败的情况：

```python
# 测试场景
test_scenarios = [
    {
        "name": "错误处理",
        "query": "请查询不存在的股票(999999.XX)的基本信息。"
    }
]
```

## 常见问题

### 1. 工具调用格式不正确

**问题**：模型生成的工具调用格式不符合 MCP 协议的要求。

**解决方案**：
- 检查系统提示中是否正确描述了工具调用的格式。
- 确保模型理解并遵循 `<tool_call>...</tool_call>` 的格式。
- 使用更明确的示例来指导模型生成正确的工具调用格式。

### 2. 工具调用失败

**问题**：模型调用工具时出现错误。

**解决方案**：
- 检查 MCP 服务器是否正常运行。
- 确认工具名称和参数是否正确。
- 查看 MCP 服务器的日志，了解具体的错误原因。

### 3. 模型无法理解工具响应

**问题**：模型无法正确理解和处理工具调用的响应。

**解决方案**：
- 确保工具响应的格式符合模型的预期。
- 在系统提示中添加更多关于如何处理工具响应的指导。
- 使用更明确的示例来指导模型处理工具响应。

## 参考资料

- [MCP 协议规范](https://github.com/langchain-ai/langchain-mcp/blob/main/mcp-spec.md)
- [LangChain MCP 适配器](https://github.com/langchain-ai/langchain-mcp-adapters)
- [xDAN 模型文档](https://github.com/xiechengmude/xDAN-Agentic-RL-ToolRL)
