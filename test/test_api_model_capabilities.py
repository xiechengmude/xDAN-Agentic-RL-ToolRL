#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试通过 vllm 部署的 OpenAI 兼容格式的 xDAN-Agentic-RL-ToolRL 模型的工具调用能力
"""

import os
import json
import argparse
import re
import time
from collections import Counter
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API 配置
VLLM_URL = os.getenv("VLLM_URL", "http://14.103.133.112:8003/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "xDAN-R1-Thinking-ToolRL-step60-0506")  # 使用正确的模型名称

# 验证API是否可用
MODELS_URL = "http://14.103.133.112:8003/v1/models"

# 测试用例类型
TEST_CASES = {
    "simple_tool": [
        {
            "instruction": "请帮我查询北京今天的天气。",
            "expected_tool": {
                "name": "check_weather",
                "parameters": {"location": "北京", "date": "today"}
            }
        },
        {
            "instruction": "计算15乘以27的结果。",
            "expected_tool": {
                "name": "calculator",
                "parameters": {"expression": "15 * 27"}
            }
        }
    ],
    "complex_tool": [
        {
            "instruction": "请帮我查找关于机器学习的最新论文，并总结其中三篇的主要观点。",
            "expected_tools": [
                {
                    "name": "search_papers",
                    "parameters": {"query": "机器学习", "sort_by": "date", "limit": 10}
                },
                {
                    "name": "summarize_text",
                    "parameters": {"text": "{paper_content}", "max_length": 200}
                }
            ]
        }
    ],
    "format_test": [
        {
            "instruction": "解释什么是强化学习。",
            "expected_format": r"^<think>.*?</think>\n<response>.*?</response>$"
        }
    ]
}

def match_score(list1, list2):
    """计算两个列表的相似度，考虑元素频率但忽略顺序"""
    if list1 == list2:
        return 1.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)
    
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    
    return intersection / max_possible if max_possible > 0 else 0.0

def compute_tool_call_score(gt_tools, pd_tools):
    """计算工具调用的匹配分数"""
    if not pd_tools:
        print("模型未生成工具调用")
        return 0.0
        
    if gt_tools == pd_tools:
        print("工具调用完全匹配")
        return 1.0
    
    # 打印工具调用信息便于分析
    print(f"期望的工具调用: {json.dumps(gt_tools, ensure_ascii=False, indent=2)}")
    print(f"实际的工具调用: {json.dumps(pd_tools, ensure_ascii=False, indent=2)}")
    
    # 工具名称匹配分数
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    name_score = match_score(list(gt_names), list(pd_names))
    print(f"工具名称匹配分数: {name_score:.2f}")
    
    # 如果工具名称完全不匹配，但功能相似，给予一定分数
    if name_score == 0:
        # 检查常见的功能等效工具
        weather_tools = ["check_weather", "天气查询", "weather", "get_weather"]
        calc_tools = ["calculator", "计算器", "calculate", "compute"]
        
        for gt_tool in gt_tools:
            gt_name = gt_tool["name"]
            for pd_tool in pd_tools:
                pd_name = pd_tool["name"]
                
                # 天气工具等效性检查
                if (gt_name in weather_tools and pd_name in weather_tools) or \
                   (gt_name in calc_tools and pd_name in calc_tools):
                    name_score = 0.8
                    print(f"检测到功能等效工具: {gt_name} ≈ {pd_name}, 给予部分分数")
    
    # 参数匹配分数
    param_score = 0.0
    param_max_score = 0.0
    
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        param_max_score += len(gt_params)  # 每个参数最多1分
        
        # 寻找名称相同或功能等效的工具
        for pd_tool in pd_tools:
            pd_name = pd_tool["name"]
            
            # 如果工具名称相同或功能等效
            if pd_name == gt_name or \
               (gt_name in weather_tools and pd_name in weather_tools) or \
               (gt_name in calc_tools and pd_name in calc_tools):
                
                pd_params = pd_tool["parameters"]
                
                # 检查参数名称匹配
                param_keys_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
                print(f"参数名称匹配分数: {param_keys_score:.2f}")
                
                # 检查参数值匹配
                for k, v in gt_params.items():
                    # 处理参数名称的变体
                    matching_keys = [k]
                    if k.lower() == "location" or k.lower() == "city":
                        matching_keys.extend(["城市", "地点", "位置"])
                    elif k.lower() == "date":
                        matching_keys.extend(["日期", "时间"])
                    elif k.lower() == "expression":
                        matching_keys.extend(["表达式", "计算式"])
                    
                    # 检查所有可能的参数名称
                    for mk in matching_keys:
                        if mk in pd_params:
                            # 检查参数值匹配
                            pd_v = pd_params[mk]
                            if pd_v == v or \
                               (v == "today" and pd_v in ["今天", "今日", "today"]) or \
                               (v == "北京" and pd_v in ["北京", "Beijing", "beijing"]):
                                param_score += 1.0
                                print(f"参数匹配: {mk}={pd_v}")
                                break
    
    # 如果没有参数要求，给予满分
    if param_max_score == 0:
        param_score_normalized = 1.0
    else:
        param_score_normalized = param_score / param_max_score
    
    print(f"参数匹配分数: {param_score}/{param_max_score} = {param_score_normalized:.2f}")
    
    # 综合分数，名称占比重要
    final_score = name_score * 0.6 + param_score_normalized * 0.4
    print(f"最终分数: {final_score:.2f} = 名称({name_score:.2f})*0.6 + 参数({param_score_normalized:.2f})*0.4")
    
    return final_score

def extract_tool_calls(response):
    """从模型响应中提取工具调用"""
    try:
        if "<tool_call>" in response and "</tool_call>" in response:
            tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            # 处理可能的多行或单行 JSON
            if "\n" in tool_call_text:
                tool_calls = []
                for line in tool_call_text.split("\n"):
                    if line.strip():
                        try:
                            tool_calls.append(json.loads(line.strip()))
                        except:
                            print(f"无法解析工具调用行: {line}")
                return tool_calls
            else:
                # 处理单个JSON对象
                try:
                    return [json.loads(tool_call_text)]
                except:
                    print(f"无法解析工具调用: {tool_call_text}")
        return []
    except Exception as e:
        print(f"提取工具调用时出错: {e}")
        return []

def extract_thinking(response):
    """从模型响应中提取思考过程"""
    try:
        if "<think>" in response and "</think>" in response:
            return response.split("<think>")[1].split("</think>")[0].strip()
        return ""
    except:
        return ""

def check_format(response, expected_format):
    """检查响应格式是否符合预期"""
    return bool(re.search(expected_format, response, re.DOTALL))

def generate_response_api(instruction, api_url=None, model_name=None, max_tokens=1024):
    """使用API生成响应"""
    api_url = api_url or VLLM_URL
    model_name = model_name or MODEL_NAME
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 添加系统提示，要求模型使用特定格式
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "请使用以下格式回答用户问题：\n<think>你的思考过程</think>\n如果需要工具调用：\n<tool_call>\n{\"name\": \"工具名称\", \"parameters\": {\"参数名\": \"参数值\"}}\n</tool_call>\n<response>你的回答</response>"},
            {"role": "user", "content": instruction}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        print(f"发送请求到 {api_url}...")
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
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

def test_model_api(api_url=None, model_name=None, test_type="all"):
    """测试API模型的工具调用能力"""
    api_url = api_url or VLLM_URL
    model_name = model_name or MODEL_NAME
    
    print(f"测试API模型: {model_name}")
    print(f"API地址: {api_url}")
    
    results = {
        "simple_tool": {"total": 0, "correct": 0, "scores": []},
        "complex_tool": {"total": 0, "correct": 0, "scores": []},
        "format_test": {"total": 0, "correct": 0, "scores": []},
    }
    
    test_types = [test_type] if test_type != "all" else list(TEST_CASES.keys())
    
    for test_type in test_types:
        print(f"\n测试类型: {test_type}")
        for i, test_case in enumerate(TEST_CASES[test_type]):
            print(f"\n测试用例 {i+1}: {test_case['instruction']}")
            
            # 添加延迟避免API限流
            if i > 0:
                time.sleep(1)
                
            response = generate_response_api(test_case['instruction'], api_url, model_name)
            print(f"模型响应:\n{response}")
            
            if not response:
                print("未获取到有效响应，跳过此测试用例")
                continue
                
            results[test_type]["total"] += 1
            
            if test_type == "simple_tool":
                expected_tool = [test_case["expected_tool"]]
                extracted_tools = extract_tool_calls(response)
                score = compute_tool_call_score(expected_tool, extracted_tools)
                results[test_type]["scores"].append(score)
                if score > 0.8:  # 设置阈值
                    results[test_type]["correct"] += 1
                print(f"工具调用评分: {score:.2f}")
                
            elif test_type == "complex_tool":
                expected_tools = test_case["expected_tools"]
                extracted_tools = extract_tool_calls(response)
                score = compute_tool_call_score(expected_tools, extracted_tools)
                results[test_type]["scores"].append(score)
                if score > 0.7:  # 复杂工具调用的阈值稍低
                    results[test_type]["correct"] += 1
                print(f"工具调用评分: {score:.2f}")
                
            elif test_type == "format_test":
                expected_format = test_case["expected_format"]
                format_correct = check_format(response, expected_format)
                score = 1.0 if format_correct else 0.0
                results[test_type]["scores"].append(score)
                if format_correct:
                    results[test_type]["correct"] += 1
                print(f"格式正确: {format_correct}")
                
            # 评估思考过程
            thinking = extract_thinking(response)
            think_length = len(thinking.split())
            print(f"思考过程长度: {think_length} 词")
    
    # 打印总结果
    print("\n===== 测试结果汇总 =====")
    for test_type in test_types:
        if results[test_type]["total"] == 0:
            print(f"{test_type}: 无有效测试结果")
            continue
            
        correct = results[test_type]["correct"]
        total = results[test_type]["total"]
        avg_score = sum(results[test_type]["scores"]) / total if total > 0 else 0
        print(f"{test_type}: 正确率 {correct}/{total} ({correct/total*100:.1f}%), 平均分数: {avg_score:.2f}")

def check_api_availability(models_url=MODELS_URL):
    """检查API是否可用"""
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        models_data = response.json()
        print(f"API可用，发现以下模型:")
        for model in models_data.get("data", []):
            print(f"- {model.get('id')}")
        return True, models_data
    except Exception as e:
        print(f"API不可用: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="测试通过API部署的xDAN-Agentic-RL-ToolRL模型的工具调用能力")
    parser.add_argument("--api_url", type=str, default=VLLM_URL, help="API地址")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="模型名称")
    parser.add_argument("--test_type", type=str, default="all", choices=["all", "simple_tool", "complex_tool", "format_test"], help="测试类型")
    args = parser.parse_args()
    
    # 首先检查API是否可用
    api_available, models_data = check_api_availability()
    if not api_available:
        print("测试终止: API不可用")
        return
    
    # 检查指定的模型是否存在
    available_models = [model.get('id') for model in models_data.get("data", [])]
    if args.model_name not in available_models:
        print(f"警告: 指定的模型 '{args.model_name}' 不在可用模型列表中")
        print(f"可用模型: {available_models}")
        if available_models:
            print(f"自动使用可用模型: {available_models[0]}")
            args.model_name = available_models[0]
        else:
            print("测试终止: 无可用模型")
            return
    
    test_model_api(args.api_url, args.model_name, args.test_type)

if __name__ == "__main__":
    main()
