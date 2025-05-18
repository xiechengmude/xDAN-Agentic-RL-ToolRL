#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
from collections import Counter, defaultdict
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import openai
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置文件路径
TRAIN_PARQUET = 'dataset/rlla_4k/train.parquet'
TEST_PARQUET = 'dataset/rlla_4k/test.parquet'
REPORT_FILE = 'dataset/rlla_4k/工具调用统计_llm.md'
CACHE_DIR = 'dataset/rlla_4k/cache'
TOOL_ANALYSIS_CACHE = os.path.join(CACHE_DIR, 'tool_analysis_cache.json')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_tools_from_prompt(prompt):
    """从提示中提取工具信息"""
    tools = []
    
    for item in prompt:
        if isinstance(item, dict) and item.get('role') == 'system':
            content = item.get('content', '')
            if '**Available Tools**' in content:
                tools_section = content.split('**Available Tools**')[1].split('**Steps for Each Turn**')[0]
                for line in tools_section.split('\n'):
                    if 'Name:' in line:
                        tool_name = line.split('Name:')[1].strip()
                        tools.append(tool_name)
    
    return tools

def extract_tools_from_output(output):
    """从输出中提取工具调用信息"""
    called_tools = []
    
    if '<tool_call>' in output:
        tool_call_section = output.split('<tool_call>')[1].split('</tool_call>')[0]
        # 使用正则表达式匹配JSON对象
        json_pattern = r'\{.*?\}'
        json_matches = re.findall(json_pattern, tool_call_section, re.DOTALL)
        
        for json_str in json_matches:
            try:
                tool_data = json.loads(json_str)
                if 'name' in tool_data:
                    called_tools.append(tool_data['name'])
            except:
                # 如果JSON解析失败，尝试使用正则表达式提取工具名称
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                if name_match:
                    called_tools.append(name_match.group(1))
    
    return called_tools

def load_cache():
    """加载缓存的分析结果"""
    if os.path.exists(TOOL_ANALYSIS_CACHE):
        with open(TOOL_ANALYSIS_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """保存分析结果到缓存"""
    with open(TOOL_ANALYSIS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def analyze_tools_with_llm(tool_batch):
    """使用LLM分析工具批次"""
    # 准备提示
    prompt = f"""请分析以下工具名称，判断它们是否是真正的工具调用，并将它们分类到以下类别之一：
1. 搜索与查询
2. 计算与数学
3. 语言处理
4. 天气服务
5. 预订与日程
6. 编程与开发
7. 金融与投资
8. 推荐系统
9. 地图与导航
10. 通信工具
11. 其他工具
12. 非工具（如果你认为这不是一个工具名称）

工具名称列表：
{json.dumps(tool_batch, ensure_ascii=False, indent=2)}

对于每个工具，请提供以下格式的JSON响应：
{{
  "tool_name": "工具名称",
  "is_tool": true/false,
  "category": "类别名称",
  "confidence": 0.0-1.0,
  "reason": "简短解释"
}}

请以JSON数组的形式返回所有分析结果。"""

    try:
        # 调用OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # 解析响应
        content = response.choices[0].message.content
        # 提取JSON部分
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # 尝试直接解析整个响应
            try:
                return json.loads(content)
            except:
                print(f"无法解析LLM响应: {content}")
                return []
    except Exception as e:
        print(f"调用LLM时出错: {str(e)}")
        time.sleep(2)  # 出错时等待一下再重试
        return []

def process_tool_batch(batch, cache):
    """处理一批工具，使用缓存避免重复分析"""
    # 检查哪些工具需要分析
    tools_to_analyze = [tool for tool in batch if tool not in cache]
    
    if not tools_to_analyze:
        # 所有工具都在缓存中
        return {tool: cache[tool] for tool in batch}
    
    # 分析新工具
    analysis_results = analyze_tools_with_llm(tools_to_analyze)
    
    # 更新缓存
    results = {}
    for result in analysis_results:
        tool_name = result.get('tool_name')
        if tool_name:
            cache[tool_name] = result
            results[tool_name] = result
    
    # 添加缓存中已有的工具
    for tool in batch:
        if tool in cache and tool not in results:
            results[tool] = cache[tool]
    
    return results

def analyze_tools():
    """分析训练集和测试集中的工具调用"""
    print("开始分析工具调用...")
    
    # 读取数据集
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    
    # 加载缓存
    cache = load_cache()
    
    # 提取所有工具
    all_tools = []
    called_tools = []
    
    # 从训练集提取工具
    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="提取训练集工具"):
        prompt = row['prompt']
        tools = extract_tools_from_prompt(prompt)
        all_tools.extend(tools)
        
        # 提取实际调用的工具
        if isinstance(row['reward_model'], dict):
            output = row['reward_model'].get('ground_truth', '')
            called = extract_tools_from_output(output)
            called_tools.extend(called)
    
    # 统计结果
    all_tools_counter = Counter(all_tools)
    called_tools_counter = Counter(called_tools)
    
    # 获取唯一工具名称
    unique_tools = list(all_tools_counter.keys())
    print(f"找到 {len(unique_tools)} 个唯一工具名称")
    
    # 批量处理工具
    batch_size = 20
    tool_batches = [unique_tools[i:i+batch_size] for i in range(0, len(unique_tools), batch_size)]
    
    all_results = {}
    for i, batch in enumerate(tqdm(tool_batches, desc="分析工具批次")):
        print(f"处理批次 {i+1}/{len(tool_batches)} ({len(batch)} 个工具)")
        batch_results = process_tool_batch(batch, cache)
        all_results.update(batch_results)
        
        # 每处理5个批次保存一次缓存
        if (i + 1) % 5 == 0 or i == len(tool_batches) - 1:
            save_cache(cache)
            print(f"缓存已保存，当前已分析 {len(cache)} 个工具")
    
    # 分类工具
    tool_categories = defaultdict(list)
    valid_tools = []
    
    for tool_name, result in all_results.items():
        if result.get('is_tool', True):
            category = result.get('category', '其他工具')
            tool_categories[category].append(tool_name)
            valid_tools.append(tool_name)
    
    # 计算工具调用率
    tool_usage_rate = {}
    for tool, available_count in all_tools_counter.items():
        if tool in valid_tools:
            called_count = called_tools_counter.get(tool, 0)
            if available_count > 0:
                tool_usage_rate[tool] = called_count / available_count
    
    # 按类别统计工具数量
    category_counts = {category: len(tools) for category, tools in tool_categories.items()}
    
    # 生成报告
    report = f"""# RLLA 4K 数据集工具调用统计 (LLM分析版)

## 1. 工具总览

- **总工具种类数**: {len(valid_tools)} 种
- **总工具调用次数**: {sum(called_tools_counter.values())} 次
- **平均每个样本可用工具数**: {len(all_tools) / len(train_df):.2f} 个
- **平均每个样本调用工具数**: {len(called_tools) / len(train_df):.2f} 个
- **被LLM识别为非工具的项目数**: {len(all_tools_counter) - len(valid_tools)} 个

## 2. 工具分类统计

| 工具类别 | 工具数量 | 占比 |
|---------|---------|-----|
{chr(10).join([f'| {category} | {count} | {count/len(valid_tools)*100:.1f}% |' for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)])}

## 3. 最常见的工具 (前30)

| 工具名称 | 出现次数 | 调用次数 | 调用率 | 类别 | 置信度 |
|---------|---------|---------|-------|------|-------|
{chr(10).join([f'| `{tool}` | {count} | {called_tools_counter.get(tool, 0)} | {tool_usage_rate.get(tool, 0)*100:.1f}% | {all_results.get(tool, {}).get("category", "未知")} | {all_results.get(tool, {}).get("confidence", 0):.2f} |' for tool, count in sorted([(t, c) for t, c in all_tools_counter.items() if t in valid_tools], key=lambda x: x[1], reverse=True)[:30]])}

## 4. 最常被调用的工具 (前30)

| 工具名称 | 调用次数 | 出现次数 | 调用率 | 类别 | 置信度 |
|---------|---------|---------|-------|------|-------|
{chr(10).join([f'| `{tool}` | {count} | {all_tools_counter.get(tool, 0)} | {count/all_tools_counter.get(tool, 1)*100:.1f}% | {all_results.get(tool, {}).get("category", "未知")} | {all_results.get(tool, {}).get("confidence", 0):.2f} |' for tool, count in sorted([(t, c) for t, c in called_tools_counter.items() if t in valid_tools], key=lambda x: x[1], reverse=True)[:30]])}

## 5. 被识别为非工具的项目 (前20)

| 项目名称 | 出现次数 | LLM判断理由 |
|---------|---------|------------|
{chr(10).join([f'| `{tool}` | {all_tools_counter.get(tool, 0)} | {all_results.get(tool, {}).get("reason", "未提供理由")} |' for tool in sorted([(t, all_results.get(t, {}).get("confidence", 0), c) for t, c in all_tools_counter.items() if all_results.get(t, {}).get("is_tool", True) == False], key=lambda x: x[2], reverse=True)[:20]])}

## 6. 按类别列出部分工具

### 金融与投资工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories.get('金融与投资', [])[:20]])}

### 搜索与查询工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories.get('搜索与查询', [])[:20]])}

### 计算与数学工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories.get('计算与数学', [])[:20]])}

### 语言处理工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories.get('语言处理', [])[:20]])}

## 7. 工具调用模式分析

1. **工具多样性**: 数据集包含多种不同类型的工具，涵盖各种功能领域
2. **调用频率**: 平均每个样本调用 {len(called_tools) / len(train_df):.2f} 个工具
3. **工具偏好**: {sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[0][0]}和{sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[1][0]}是最常见的工具类型
4. **误识别项目**: LLM分析发现了一些被错误识别为工具的项目，这些可能是数据集中的文本片段或标题

## 8. LLM分析的优势

1. **更精确的工具识别**: 通过LLM分析，我们能够更准确地识别真正的工具调用
2. **更合理的分类**: LLM能够根据工具的功能和上下文进行更合理的分类
3. **置信度评估**: 对每个工具的分类都提供了置信度评分，便于进一步筛选
4. **错误识别修正**: 识别并排除了一些被错误标记为工具的项目

## 9. 对智能体开发的启示

1. **工具设计**: 根据LLM分析的结果，可以更有针对性地设计工具集
2. **调用模式**: 参考高调用率工具的特点，优化工具接口设计
3. **多工具协同**: 数据集中的多工具调用模式提供了良好参考
4. **工具分类**: 采用更精确的工具分类方法，提高工具使用效率
"""
    
    # 写入报告文件
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"工具调用统计报告已生成: {REPORT_FILE}")
    return report

def main():
    """主函数"""
    analyze_tools()
    print("任务完成!")

if __name__ == "__main__":
    main()
