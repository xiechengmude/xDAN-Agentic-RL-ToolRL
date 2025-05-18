#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
from collections import Counter
import os
from tqdm import tqdm
import random

# 设置文件路径
TRAIN_PARQUET = 'dataset/rlla_4k/train.parquet'
EXAMPLES_FILE = 'dataset/rlla_4k/工具调用示例.md'

def extract_tools_from_prompt(prompt):
    """从提示中提取工具信息"""
    tools = []
    tool_descriptions = []
    
    for item in prompt:
        if isinstance(item, dict) and item.get('role') == 'system':
            content = item.get('content', '')
            if '**Available Tools**' in content:
                tools_section = content.split('**Available Tools**')[1].split('**Steps for Each Turn**')[0]
                current_tool = None
                current_description = []
                
                for line in tools_section.split('\n'):
                    if 'Name:' in line:
                        # 如果已经有一个工具，保存它
                        if current_tool:
                            tool_descriptions.append({
                                'name': current_tool,
                                'description': '\n'.join(current_description)
                            })
                            current_description = []
                        
                        current_tool = line.split('Name:')[1].strip()
                        tools.append(current_tool)
                    elif current_tool and line.strip():
                        current_description.append(line.strip())
                
                # 保存最后一个工具
                if current_tool:
                    tool_descriptions.append({
                        'name': current_tool,
                        'description': '\n'.join(current_description)
                    })
    
    return tools, tool_descriptions

def extract_tools_from_output(output):
    """从输出中提取工具调用信息"""
    called_tools = []
    tool_calls = []
    
    if '<tool_call>' in output:
        tool_call_sections = output.split('<tool_call>')
        for i in range(1, len(tool_call_sections)):
            if '</tool_call>' in tool_call_sections[i]:
                tool_call_content = tool_call_sections[i].split('</tool_call>')[0]
                # 使用正则表达式匹配JSON对象
                json_pattern = r'\{.*?\}'
                json_matches = re.findall(json_pattern, tool_call_content, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        tool_data = json.loads(json_str)
                        if 'name' in tool_data:
                            called_tools.append(tool_data['name'])
                            tool_calls.append({
                                'name': tool_data['name'],
                                'call': json_str,
                                'full_section': tool_call_content
                            })
                    except:
                        # 如果JSON解析失败，尝试使用正则表达式提取工具名称
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                        if name_match:
                            called_tools.append(name_match.group(1))
                            tool_calls.append({
                                'name': name_match.group(1),
                                'call': json_str,
                                'full_section': tool_call_content
                            })
    
    return called_tools, tool_calls

def extract_examples():
    """从训练集中提取工具调用示例"""
    print("开始提取工具调用示例...")
    
    # 读取数据集
    train_df = pd.read_parquet(TRAIN_PARQUET)
    
    # 存储所有工具调用示例
    all_examples = []
    
    # 从训练集提取工具调用示例
    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="提取工具调用示例"):
        prompt = row['prompt']
        tools, tool_descriptions = extract_tools_from_prompt(prompt)
        
        # 提取实际调用的工具
        if isinstance(row['reward_model'], dict):
            output = row['reward_model'].get('ground_truth', '')
            called_tools, tool_calls = extract_tools_from_output(output)
            
            # 如果有工具调用，保存示例
            if tool_calls:
                # 获取用户查询
                user_query = ""
                for item in prompt:
                    if isinstance(item, dict) and item.get('role') == 'user':
                        user_query = item.get('content', '')
                        break
                
                # 保存示例
                for call in tool_calls:
                    # 查找对应的工具描述
                    tool_desc = ""
                    for desc in tool_descriptions:
                        if desc['name'] == call['name']:
                            tool_desc = desc['description']
                            break
                    
                    all_examples.append({
                        'user_query': user_query,
                        'tool_name': call['name'],
                        'tool_description': tool_desc,
                        'tool_call': call['call'],
                        'full_section': call['full_section'],
                        'available_tools': tools
                    })
    
    # 按工具类型分类示例
    examples_by_category = {}
    
    # 定义工具类别
    categories = {
        '搜索与查询': ['search', 'find', 'query', 'lookup', 'get'],
        '计算与数学': ['calculate', 'compute', 'math', 'sum', 'average', 'mean', 'median', 'mode', 'std', 'var', 'min', 'max'],
        '语言处理': ['translate', 'language', 'text', 'nlp', 'sentiment', 'summarize', 'grammar'],
        '天气服务': ['weather', 'forecast', 'temperature', 'climate'],
        '预订与日程': ['book', 'reserve', 'appointment', 'schedule', 'calendar'],
        '编程与开发': ['code', 'program', 'function', 'api', 'debug'],
        '金融与投资': ['invest', 'stock', 'finance', 'money', 'currency', 'price', 'cost', 'budget', 'loan', 'mortgage', 'interest', 'tax'],
        '推荐系统': ['recommend', 'suggest', 'rating', 'review', 'rank'],
        '地图与导航': ['map', 'location', 'direction', 'distance', 'route', 'navigation'],
        '通信工具': ['email', 'message', 'send', 'contact', 'call', 'phone']
    }
    
    # 对示例进行分类
    for example in all_examples:
        tool_name = example['tool_name'].lower()
        category = '其他工具'
        
        # 检查工具名称是否匹配任何类别
        for cat, keywords in categories.items():
            if any(keyword in tool_name for keyword in keywords):
                category = cat
                break
        
        if category not in examples_by_category:
            examples_by_category[category] = []
        
        examples_by_category[category].append(example)
    
    # 从每个类别中选择代表性示例
    representative_examples = []
    categories_to_sample = list(examples_by_category.keys())
    
    # 确保至少包含一些主要类别
    must_include = ['搜索与查询', '计算与数学', '金融与投资', '语言处理']
    for category in must_include:
        if category in examples_by_category and examples_by_category[category]:
            # 从该类别中随机选择一个示例
            example = random.choice(examples_by_category[category])
            representative_examples.append(example)
            # 从待采样类别中移除
            if category in categories_to_sample:
                categories_to_sample.remove(category)
    
    # 从剩余类别中随机选择，直到达到10个示例
    while len(representative_examples) < 10 and categories_to_sample:
        category = random.choice(categories_to_sample)
        if examples_by_category[category]:
            example = random.choice(examples_by_category[category])
            representative_examples.append(example)
        categories_to_sample.remove(category)
    
    # 如果还不够10个，从所有示例中随机选择
    all_remaining = [ex for cat in examples_by_category.values() for ex in cat]
    while len(representative_examples) < 10 and all_remaining:
        idx = random.randint(0, len(all_remaining) - 1)
        representative_examples.append(all_remaining.pop(idx))
    
    # 生成报告
    report = "# RLLA 4K 数据集工具调用示例\n\n"
    report += "以下是从数据集中提取的10个经典代表性工具调用示例：\n\n"
    
    for i, example in enumerate(representative_examples[:10], 1):
        report += f"## 示例 {i}: `{example['tool_name']}`\n\n"
        report += f"### 用户查询\n\n```\n{example['user_query']}\n```\n\n"
        report += f"### 工具描述\n\n```\n{example['tool_description']}\n```\n\n"
        report += f"### 工具调用\n\n```json\n{example['tool_call']}\n```\n\n"
        report += f"### 可用工具列表\n\n"
        for tool in example['available_tools'][:5]:  # 只显示前5个工具
            report += f"- `{tool}`\n"
        if len(example['available_tools']) > 5:
            report += f"- ... 等共 {len(example['available_tools'])} 个工具\n"
        report += "\n---\n\n"
    
    # 写入报告文件
    with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"工具调用示例已生成: {EXAMPLES_FILE}")
    return report

def main():
    """主函数"""
    extract_examples()
    print("任务完成!")

if __name__ == "__main__":
    main()
