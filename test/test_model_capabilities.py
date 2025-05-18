#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试xDAN-Agentic-RL-ToolRL模型的工具调用能力
"""

import os
import json
import argparse
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    if gt_tools == pd_tools:
        return 1.0
    
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))
    
    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        
        local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # 寻找最匹配的工具
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            
            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # 计算参数值的匹配度
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    return score / local_max_possible

def extract_tool_calls(response):
    """从模型响应中提取工具调用"""
    try:
        if "<tool_call>" in response and "</tool_call>" in response:
            tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            tool_calls = [json.loads(line) for line in tool_call_text.split("\n") if line.strip()]
            return tool_calls
        return []
    except:
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

def generate_response(model, tokenizer, instruction, device="cuda"):
    """使用模型生成响应"""
    prompt = f"用户: {instruction}\n助手: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def test_model(model_path, test_type="all"):
    """测试模型的工具调用能力"""
    print(f"加载模型: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
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
            
            response = generate_response(model, tokenizer, test_case['instruction'], device)
            print(f"模型响应:\n{response}")
            
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
        correct = results[test_type]["correct"]
        total = results[test_type]["total"]
        avg_score = sum(results[test_type]["scores"]) / total if total > 0 else 0
        print(f"{test_type}: 正确率 {correct}/{total} ({correct/total*100:.1f}%), 平均分数: {avg_score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="测试xDAN-Agentic-RL-ToolRL模型的工具调用能力")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_type", type=str, default="all", choices=["all", "simple_tool", "complex_tool", "format_test"], help="测试类型")
    args = parser.parse_args()
    
    test_model(args.model_path, args.test_type)

if __name__ == "__main__":
    main()
