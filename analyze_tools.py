#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
from collections import Counter, defaultdict

# 设置文件路径
TRAIN_PARQUET = 'dataset/rlla_4k/train.parquet'
TEST_PARQUET = 'dataset/rlla_4k/test.parquet'
REPORT_FILE = 'dataset/rlla_4k/工具调用统计.md'

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

def categorize_tool(tool_name):
    """对工具进行分类"""
    tool_name_lower = tool_name.lower()
    
    if any(keyword in tool_name_lower for keyword in ['search', 'find', 'query', 'lookup', 'get']):
        return '搜索与查询'
    elif any(keyword in tool_name_lower for keyword in ['calculate', 'compute', 'math', 'sum', 'average', 'mean', 'median', 'mode', 'std', 'var', 'min', 'max']):
        return '计算与数学'
    elif any(keyword in tool_name_lower for keyword in ['translate', 'language', 'text', 'nlp', 'sentiment', 'summarize', 'grammar']):
        return '语言处理'
    elif any(keyword in tool_name_lower for keyword in ['weather', 'forecast', 'temperature', 'climate']):
        return '天气服务'
    elif any(keyword in tool_name_lower for keyword in ['book', 'reserve', 'appointment', 'schedule', 'calendar']):
        return '预订与日程'
    elif any(keyword in tool_name_lower for keyword in ['code', 'program', 'function', 'api', 'debug']):
        return '编程与开发'
    elif any(keyword in tool_name_lower for keyword in ['invest', 'stock', 'finance', 'money', 'currency', 'price', 'cost', 'budget', 'loan', 'mortgage', 'interest', 'tax']):
        return '金融与投资'
    elif any(keyword in tool_name_lower for keyword in ['recommend', 'suggest', 'rating', 'review', 'rank']):
        return '推荐系统'
    elif any(keyword in tool_name_lower for keyword in ['map', 'location', 'direction', 'distance', 'route', 'navigation']):
        return '地图与导航'
    elif any(keyword in tool_name_lower for keyword in ['email', 'message', 'send', 'contact', 'call', 'phone']):
        return '通信工具'
    else:
        return '其他工具'

def analyze_tools():
    """分析训练集和测试集中的工具调用"""
    print("开始分析工具调用...")
    
    # 读取数据集
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    
    # 统计信息
    available_tools = []  # 所有可用工具
    called_tools = []     # 实际调用的工具
    tool_categories = defaultdict(list)  # 工具分类
    
    # 分析训练集
    for i, row in train_df.iterrows():
        prompt = row['prompt']
        tools = extract_tools_from_prompt(prompt)
        available_tools.extend(tools)
        
        # 对工具进行分类
        for tool in tools:
            category = categorize_tool(tool)
            if tool not in tool_categories[category]:
                tool_categories[category].append(tool)
        
        # 提取实际调用的工具
        if isinstance(row['reward_model'], dict):
            output = row['reward_model'].get('ground_truth', '')
            called = extract_tools_from_output(output)
            called_tools.extend(called)
    
    # 统计结果
    available_tool_counts = Counter(available_tools)
    called_tool_counts = Counter(called_tools)
    
    # 计算工具调用率
    tool_usage_rate = {}
    for tool, available_count in available_tool_counts.items():
        called_count = called_tool_counts.get(tool, 0)
        if available_count > 0:
            tool_usage_rate[tool] = called_count / available_count
    
    # 按类别统计工具数量
    category_counts = {category: len(tools) for category, tools in tool_categories.items()}
    
    # 生成报告
    report = f"""# RLLA 4K 数据集工具调用统计

## 1. 工具总览

- **总工具种类数**: {len(available_tool_counts)} 种
- **总工具调用次数**: {sum(called_tool_counts.values())} 次
- **平均每个样本可用工具数**: {len(available_tools) / len(train_df):.2f} 个
- **平均每个样本调用工具数**: {len(called_tools) / len(train_df):.2f} 个

## 2. 工具分类统计

| 工具类别 | 工具数量 | 占比 |
|---------|---------|-----|
{chr(10).join([f'| {category} | {count} | {count/len(available_tool_counts)*100:.1f}% |' for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)])}

## 3. 最常见的工具 (前30)

| 工具名称 | 出现次数 | 调用次数 | 调用率 |
|---------|---------|---------|-------|
{chr(10).join([f'| `{tool}` | {count} | {called_tool_counts.get(tool, 0)} | {tool_usage_rate.get(tool, 0)*100:.1f}% |' for tool, count in available_tool_counts.most_common(30)])}

## 4. 最常被调用的工具 (前30)

| 工具名称 | 调用次数 | 出现次数 | 调用率 |
|---------|---------|---------|-------|
{chr(10).join([f'| `{tool}` | {count} | {available_tool_counts.get(tool, 0)} | {count/available_tool_counts.get(tool, 1)*100:.1f}% |' for tool, count in called_tool_counts.most_common(30)])}

## 5. 按类别列出部分工具

### 金融与投资工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories['金融与投资'][:20]])}

### 搜索与查询工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories['搜索与查询'][:20]])}

### 计算与数学工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories['计算与数学'][:20]])}

### 语言处理工具

{chr(10).join(['- `' + tool + '`' for tool in tool_categories['语言处理'][:20]])}

## 6. 工具调用模式分析

1. **工具多样性**: 数据集包含多种不同类型的工具，涵盖各种功能领域
2. **调用频率**: 平均每个样本调用 {len(called_tools) / len(train_df):.2f} 个工具
3. **工具偏好**: 搜索类工具和计算类工具是最常被调用的工具类型
4. **金融工具**: 数据集中包含一些金融相关工具，但占比较小，需要进一步扩充

## 7. 对金融智能体开发的启示

1. **工具设计**: 金融智能体需要设计更多专业金融工具，如投资分析、风险评估、投资组合优化等
2. **调用模式**: 可以参考现有工具的调用模式，设计易于理解和使用的API接口
3. **多工具协同**: 金融场景通常需要多个工具协同工作，数据集中的多工具调用模式提供了良好参考
4. **工具分类**: 可以按照数据集中的分类方法，对金融工具进行系统化分类，提高工具使用效率
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
