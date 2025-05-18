#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试使用系统提示词调用xDAN-R1-Thinking-ToolRL模型
"""

import os
import json
import requests
import time
import argparse
from datetime import datetime

# API配置
VLLM_URL = "http://14.103.133.112:8003/v1/chat/completions"
MODEL_NAME = "xDAN-R1-Thinking-ToolRL-step60-0506"

def load_system_prompt(lang="zh"):
    """加载系统提示词"""
    # 尝试从文件加载
    try:
        with open("data/prompt/system_prompt.md", "r", encoding="utf-8") as f:
            content = f.read()
            
        if lang == "zh":
            # 提取中文系统提示词
            start_marker = "```\n您是一个能够利用工具调用解决用户任务"
            end_marker = "```\n\n## 部署建议"
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                return content[start_idx+3:end_idx].strip()
        else:
            # 提取英文系统提示词
            start_marker = "```\nYou are a helpful multi-turn dialogue assistant"
            end_marker = "```\n\n## 中文版本系统提示词"
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                return content[start_idx+3:end_idx].strip()
    except Exception as e:
        print(f"加载系统提示词文件失败: {e}")
    
    # 如果加载失败，返回默认系统提示词
    if lang == "zh":
        return """您是一个能够利用工具调用解决用户任务并提供结构化聊天响应的多轮对话助手。

**输出格式**
<think>您的思考和推理过程</think>
<tool_call>
{"name": "工具名称", "parameters": {"参数名": "参数内容", "...": "..."}}
</tool_call>
<response>AI的最终回应</response>

**重要说明**
1. 您必须始终包含`<think>`字段来概述您的推理过程
2. 根据需要使用`<tool_call>`或`<response>`，或两者都用
3. 每个工具调用应该是一个包含"name"字段和"parameters"字段的JSON对象"""
    else:
        return """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Output Format**
<think> Your thoughts and reasoning </think>
<tool_call>
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "...": "..."}}
</tool_call>
<response> AI's final response </response>

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning
2. Provide at least one of `<tool_call>` or `<response>`
3. Each tool call should be a JSON object with a "name" field and a "parameters" field"""

def call_model(user_query, system_prompt, available_tools=None):
    """调用模型API"""
    # 如果有可用工具，添加到系统提示词中
    full_system_prompt = system_prompt
    if available_tools:
        tools_text = "\n\n**可用工具**\n"
        for i, tool in enumerate(available_tools, 1):
            tools_text += f"{i}. 名称：{tool['name']}\n"
            tools_text += f"   描述：{tool['description']}\n"
            tools_text += f"   参数：{json.dumps(tool['parameters'], ensure_ascii=False)}\n"
        full_system_prompt = full_system_prompt.replace("**输出格式**", f"{tools_text}\n**输出格式**")
    
    # 构建请求
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        print(f"发送请求到 {VLLM_URL}...")
        print(f"系统提示词长度: {len(full_system_prompt)} 字符")
        start_time = time.time()
        response = requests.post(VLLM_URL, headers=headers, json=data, timeout=60)
        end_time = time.time()
        print(f"请求耗时: {end_time - start_time:.2f} 秒")
        print(f"收到响应，状态码: {response.status_code}")
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"响应长度: {len(content)} 字符")
            return content
        else:
            print(f"API返回格式异常: {result}")
            return ""
    except Exception as e:
        print(f"API调用失败: {e}")
        print(f"错误详情: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"响应内容: {response.text[:500]}...")
        return ""

def extract_sections(response):
    """从响应中提取思考、工具调用和回复部分"""
    sections = {
        "think": "",
        "tool_call": [],
        "response": ""
    }
    
    # 提取思考部分
    if "<think>" in response and "</think>" in response:
        think_text = response.split("<think>")[1].split("</think>")[0].strip()
        sections["think"] = think_text
    
    # 提取工具调用部分
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        try:
            # 处理可能的多行或单行JSON
            if "\n" in tool_call_text:
                for line in tool_call_text.split("\n"):
                    if line.strip():
                        try:
                            sections["tool_call"].append(json.loads(line.strip()))
                        except:
                            print(f"无法解析工具调用行: {line}")
            else:
                # 处理单个JSON对象
                try:
                    sections["tool_call"].append(json.loads(tool_call_text))
                except:
                    print(f"无法解析工具调用: {tool_call_text}")
        except Exception as e:
            print(f"提取工具调用时出错: {e}")
    
    # 提取回复部分
    if "<response>" in response and "</response>" in response:
        response_text = response.split("<response>")[1].split("</response>")[0].strip()
        sections["response"] = response_text
    
    return sections

def analyze_response(response):
    """分析模型响应"""
    print("\n===== 响应分析 =====")
    
    # 提取各部分
    sections = extract_sections(response)
    
    # 分析思考部分
    if sections["think"]:
        print("\n--- 思考部分 ---")
        print(sections["think"])
        print(f"思考长度: {len(sections['think'])} 字符")
    else:
        print("\n--- 思考部分 ---")
        print("未找到思考部分")
    
    # 分析工具调用部分
    if sections["tool_call"]:
        print("\n--- 工具调用部分 ---")
        for i, tool in enumerate(sections["tool_call"], 1):
            print(f"工具 {i}:")
            print(f"  名称: {tool.get('name', '未指定')}")
            print(f"  参数: {json.dumps(tool.get('parameters', {}), ensure_ascii=False, indent=2)}")
        print(f"工具调用数量: {len(sections['tool_call'])}")
    else:
        print("\n--- 工具调用部分 ---")
        print("未找到工具调用部分")
    
    # 分析回复部分
    if sections["response"]:
        print("\n--- 回复部分 ---")
        print(sections["response"])
        print(f"回复长度: {len(sections['response'])} 字符")
    else:
        print("\n--- 回复部分 ---")
        print("未找到回复部分")
    
    # 格式分析
    format_correct = True
    format_issues = []
    
    if not sections["think"]:
        format_correct = False
        format_issues.append("缺少思考部分 (<think>)")
    
    if not sections["tool_call"] and not sections["response"]:
        format_correct = False
        format_issues.append("缺少工具调用和回复部分 (至少需要一个)")
    
    print("\n--- 格式分析 ---")
    if format_correct:
        print("✅ 格式正确")
    else:
        print("❌ 格式有问题:")
        for issue in format_issues:
            print(f"  - {issue}")
    
    return sections

def main():
    parser = argparse.ArgumentParser(description="测试使用系统提示词调用xDAN-R1-Thinking-ToolRL模型")
    parser.add_argument("--query", type=str, default="请帮我查询北京今天的天气。", help="用户查询")
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="系统提示词语言")
    parser.add_argument("--save", action="store_true", help="是否保存结果")
    args = parser.parse_args()
    
    # 加载系统提示词
    system_prompt = load_system_prompt(args.lang)
    print(f"\n===== 使用{args.lang}语言系统提示词 =====")
    
    # 定义可用工具
    available_tools = [
        {
            "name": "天气查询",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "城市": {"description": "要查询天气的城市名称", "type": "string"},
                "日期": {"description": "要查询的日期，如'今天'、'明天'等", "type": "string", "default": "今天"}
            }
        },
        {
            "name": "计算器",
            "description": "执行数学计算",
            "parameters": {
                "表达式": {"description": "要计算的数学表达式", "type": "string"}
            }
        }
    ]
    
    # 调用模型
    print(f"\n===== 用户查询 =====\n{args.query}")
    response = call_model(args.query, system_prompt, available_tools)
    
    if response:
        print("\n===== 模型响应 =====")
        print(response)
        
        # 分析响应
        sections = analyze_response(response)
        
        # 保存结果
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = "results"
            os.makedirs(result_dir, exist_ok=True)
            
            result = {
                "timestamp": timestamp,
                "query": args.query,
                "system_prompt": system_prompt,
                "available_tools": available_tools,
                "response": response,
                "sections": {
                    "think": sections["think"],
                    "tool_call": sections["tool_call"],
                    "response": sections["response"]
                }
            }
            
            result_file = os.path.join(result_dir, f"test_result_{timestamp}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
