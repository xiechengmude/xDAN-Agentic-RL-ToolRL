#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用OpenAI兼容格式测试xDAN-R1-Thinking-ToolRL模型
使用训练系统自带的原版系统提示词
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

# 原版系统提示词（从训练数据中提取）
ORIGINAL_SYSTEM_PROMPT = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
1. Name: checkWeather
Description: Check the weather for a specific location and date
Parameters: {"location": {"description": "The city or location to check weather for", "type": "string", "default": ""}, "date": {"description": "The date to check weather for (e.g., today, tomorrow, 2023-05-20)", "type": "string", "default": "today"}}
2. Name: calculator
Description: Perform mathematical calculations
Parameters: {"expression": {"description": "The mathematical expression to evaluate", "type": "string", "default": ""}}
3. Name: searchPapers
Description: Search for academic papers on a specific topic
Parameters: {"query": {"description": "The search query", "type": "string", "default": ""}, "sort_by": {"description": "How to sort the results (relevance, date)", "type": "string", "default": "relevance"}, "limit": {"description": "Maximum number of results to return", "type": "integer", "default": 5}}
4. Name: summarizeText
Description: Summarize a given text
Parameters: {"text": {"description": "The text to summarize", "type": "string", "default": ""}, "max_length": {"description": "Maximum length of the summary in words", "type": "integer", "default": 100}}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists)."""

# 测试用例
TEST_CASES = [
    {
        "name": "天气查询",
        "query": "请帮我查询北京今天的天气。",
        "expected_tool": {
            "name": "checkWeather",
            "parameters": {"location": "北京", "date": "today"}
        }
    },
    {
        "name": "数学计算",
        "query": "计算15乘以27的结果。",
        "expected_tool": {
            "name": "calculator",
            "parameters": {"expression": "15 * 27"}
        }
    },
    {
        "name": "复合查询",
        "query": "我需要计算15乘以27的结果，并且查询一下上海明天的天气。",
        "expected_tools": [
            {
                "name": "calculator",
                "parameters": {"expression": "15 * 27"}
            },
            {
                "name": "checkWeather",
                "parameters": {"location": "上海", "date": "tomorrow"}
            }
        ]
    },
    {
        "name": "论文搜索",
        "query": "请帮我搜索关于机器学习的最新论文，按日期排序，限制10篇。",
        "expected_tool": {
            "name": "searchPapers",
            "parameters": {"query": "机器学习", "sort_by": "date", "limit": 10}
        }
    },
    {
        "name": "知识问答",
        "query": "请解释一下什么是强化学习，并用简单的例子说明。",
        "expected_format": r"^<think>.*?</think>\n<response>.*?</response>$"
    }
]

def call_openai_api(query, system_prompt=ORIGINAL_SYSTEM_PROMPT, temperature=0.7, max_tokens=1024):
    """使用OpenAI兼容格式调用API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    try:
        print(f"发送请求到 {VLLM_URL}...")
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

def analyze_response(response, test_case):
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
        
        # 评估工具调用是否符合预期
        if "expected_tool" in test_case:
            expected_tool = test_case["expected_tool"]
            print("\n--- 工具调用评估 ---")
            print(f"预期工具: {expected_tool['name']}")
            print(f"预期参数: {json.dumps(expected_tool['parameters'], ensure_ascii=False, indent=2)}")
            
            # 查找匹配的工具调用
            found_match = False
            for tool in sections["tool_call"]:
                if tool.get("name") == expected_tool["name"] or \
                   (tool.get("name") == "checkWeather" and expected_tool["name"] == "天气查询") or \
                   (tool.get("name") == "天气查询" and expected_tool["name"] == "checkWeather") or \
                   (tool.get("name") == "calculator" and expected_tool["name"] == "计算器") or \
                   (tool.get("name") == "计算器" and expected_tool["name"] == "calculator"):
                    
                    # 检查参数
                    params_match = True
                    for key, value in expected_tool["parameters"].items():
                        # 处理参数名称的变体
                        if key == "location" and "城市" in tool.get("parameters", {}):
                            if tool["parameters"]["城市"] != value:
                                params_match = False
                                break
                        elif key == "date" and "日期" in tool.get("parameters", {}):
                            date_value = tool["parameters"]["日期"]
                            if value == "today" and date_value not in ["今天", "today"]:
                                params_match = False
                                break
                            elif value == "tomorrow" and date_value not in ["明天", "tomorrow"]:
                                params_match = False
                                break
                            elif date_value != value:
                                params_match = False
                                break
                        elif key == "expression" and "表达式" in tool.get("parameters", {}):
                            if tool["parameters"]["表达式"] != value and tool["parameters"]["表达式"].replace(" ", "") != value.replace(" ", ""):
                                params_match = False
                                break
                        elif key in tool.get("parameters", {}) and tool["parameters"][key] != value:
                            params_match = False
                            break
                        elif key not in tool.get("parameters", {}):
                            # 检查是否有等效的参数名
                            found_equiv = False
                            for param_key in tool.get("parameters", {}):
                                if (key == "location" and param_key in ["城市", "地点"]) or \
                                   (key == "date" and param_key in ["日期", "时间"]) or \
                                   (key == "expression" and param_key in ["表达式", "计算式"]):
                                    found_equiv = True
                                    break
                            if not found_equiv:
                                params_match = False
                                break
                    
                    if params_match:
                        found_match = True
                        break
            
            if found_match:
                print("✅ 工具调用符合预期")
            else:
                print("❌ 工具调用不符合预期")
        
        elif "expected_tools" in test_case:
            expected_tools = test_case["expected_tools"]
            print("\n--- 工具调用评估 ---")
            print(f"预期工具数量: {len(expected_tools)}")
            
            # 检查是否所有预期的工具都被调用
            found_matches = 0
            for expected_tool in expected_tools:
                print(f"预期工具: {expected_tool['name']}")
                print(f"预期参数: {json.dumps(expected_tool['parameters'], ensure_ascii=False, indent=2)}")
                
                # 查找匹配的工具调用
                found_match = False
                for tool in sections["tool_call"]:
                    if tool.get("name") == expected_tool["name"] or \
                       (tool.get("name") == "checkWeather" and expected_tool["name"] == "天气查询") or \
                       (tool.get("name") == "天气查询" and expected_tool["name"] == "checkWeather") or \
                       (tool.get("name") == "calculator" and expected_tool["name"] == "计算器") or \
                       (tool.get("name") == "计算器" and expected_tool["name"] == "calculator"):
                        
                        # 检查参数
                        params_match = True
                        for key, value in expected_tool["parameters"].items():
                            # 处理参数名称的变体
                            if key == "location" and "城市" in tool.get("parameters", {}):
                                if tool["parameters"]["城市"] != value:
                                    params_match = False
                                    break
                            elif key == "date" and "日期" in tool.get("parameters", {}):
                                date_value = tool["parameters"]["日期"]
                                if value == "today" and date_value not in ["今天", "today"]:
                                    params_match = False
                                    break
                                elif value == "tomorrow" and date_value not in ["明天", "tomorrow"]:
                                    params_match = False
                                    break
                                elif date_value != value:
                                    params_match = False
                                    break
                            elif key == "expression" and "表达式" in tool.get("parameters", {}):
                                if tool["parameters"]["表达式"] != value and tool["parameters"]["表达式"].replace(" ", "") != value.replace(" ", ""):
                                    params_match = False
                                    break
                            elif key in tool.get("parameters", {}) and tool["parameters"][key] != value:
                                params_match = False
                                break
                            elif key not in tool.get("parameters", {}):
                                # 检查是否有等效的参数名
                                found_equiv = False
                                for param_key in tool.get("parameters", {}):
                                    if (key == "location" and param_key in ["城市", "地点"]) or \
                                       (key == "date" and param_key in ["日期", "时间"]) or \
                                       (key == "expression" and param_key in ["表达式", "计算式"]):
                                        found_equiv = True
                                        break
                                if not found_equiv:
                                    params_match = False
                                    break
                        
                        if params_match:
                            found_match = True
                            found_matches += 1
                            break
                
                if found_match:
                    print(f"✅ 工具 {expected_tool['name']} 调用符合预期")
                else:
                    print(f"❌ 工具 {expected_tool['name']} 调用不符合预期")
            
            if found_matches == len(expected_tools):
                print("✅ 所有工具调用都符合预期")
            else:
                print(f"❌ 只有 {found_matches}/{len(expected_tools)} 个工具调用符合预期")
    else:
        print("\n--- 工具调用部分 ---")
        print("未找到工具调用部分")
        
        if "expected_tool" in test_case or "expected_tools" in test_case:
            print("\n--- 工具调用评估 ---")
            print("❌ 预期有工具调用，但未找到")
    
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

def run_test(test_case, save_results=False):
    """运行单个测试用例"""
    print(f"\n\n{'='*50}")
    print(f"测试用例: {test_case['name']}")
    print(f"{'='*50}")
    print(f"查询: {test_case['query']}")
    
    # 调用API
    response = call_openai_api(test_case['query'])
    
    if response:
        print("\n----- 原始响应 -----")
        print(response)
        
        # 分析响应
        sections = analyze_response(response, test_case)
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = "results"
            os.makedirs(result_dir, exist_ok=True)
            
            result = {
                "timestamp": timestamp,
                "test_case": test_case,
                "response": response,
                "sections": {
                    "think": sections["think"],
                    "tool_call": sections["tool_call"],
                    "response": sections["response"]
                }
            }
            
            result_file = os.path.join(result_dir, f"test_result_{test_case['name']}_{timestamp}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果已保存到: {result_file}")
    
    return response

def main():
    parser = argparse.ArgumentParser(description="使用OpenAI兼容格式测试xDAN-R1-Thinking-ToolRL模型")
    parser.add_argument("--test", type=str, default="all", help="要运行的测试用例名称，默认运行所有测试")
    parser.add_argument("--save", action="store_true", help="是否保存测试结果")
    args = parser.parse_args()
    
    # 运行测试
    if args.test == "all":
        print(f"运行所有 {len(TEST_CASES)} 个测试用例")
        for test_case in TEST_CASES:
            run_test(test_case, args.save)
    else:
        # 查找匹配的测试用例
        found = False
        for test_case in TEST_CASES:
            if test_case["name"] == args.test:
                run_test(test_case, args.save)
                found = True
                break
        
        if not found:
            print(f"未找到名为 '{args.test}' 的测试用例")
            print("可用的测试用例:")
            for test_case in TEST_CASES:
                print(f"- {test_case['name']}")

if __name__ == "__main__":
    main()
