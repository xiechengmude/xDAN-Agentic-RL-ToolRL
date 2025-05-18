#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import random
from collections import Counter, defaultdict

# 设置文件路径
TRAIN_PARQUET = 'dataset/rlla_4k/train.parquet'
TEST_PARQUET = 'dataset/rlla_4k/test.parquet'
REPORT_FILE = 'dataset/rlla_4k/数据集特点与场景分析.md'

def analyze_datasets():
    """分析训练集和测试集的特点和场景"""
    print("开始分析数据集...")
    
    # 读取数据集
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    
    # 基本统计信息
    train_size = len(train_df)
    test_size = len(test_df)
    
    # 分析工具使用情况
    train_tools = []
    train_user_queries = []
    train_scenarios = defaultdict(int)
    train_tool_counts = []
    
    # 分析训练集
    for i, row in train_df.iterrows():
        prompt = row['prompt']
        tools_in_sample = []
        user_query = ""
        
        # 提取工具和用户查询
        for item in prompt:
            if isinstance(item, dict):
                if item.get('role') == 'system':
                    content = item.get('content', '')
                    if '**Available Tools**' in content:
                        tools_section = content.split('**Available Tools**')[1].split('**Steps for Each Turn**')[0]
                        tool_names = []
                        for line in tools_section.split('\n'):
                            if 'Name:' in line:
                                tool_name = line.split('Name:')[1].strip()
                                tool_names.append(tool_name)
                        tools_in_sample.extend(tool_names)
                        train_tool_counts.append(len(tool_names))
                elif item.get('role') == 'user':
                    user_query = item.get('content', '')
                    train_user_queries.append(user_query)
                    
                    # 简单场景分类
                    if 'weather' in user_query.lower() or 'temperature' in user_query.lower():
                        train_scenarios['天气查询'] += 1
                    elif 'search' in user_query.lower() or 'find' in user_query.lower():
                        train_scenarios['搜索查询'] += 1
                    elif 'book' in user_query.lower() or 'reservation' in user_query.lower():
                        train_scenarios['预订服务'] += 1
                    elif 'calculate' in user_query.lower() or 'math' in user_query.lower():
                        train_scenarios['计算问题'] += 1
                    elif 'code' in user_query.lower() or 'program' in user_query.lower():
                        train_scenarios['编程任务'] += 1
                    elif 'translate' in user_query.lower() or 'language' in user_query.lower():
                        train_scenarios['翻译任务'] += 1
                    elif 'schedule' in user_query.lower() or 'calendar' in user_query.lower():
                        train_scenarios['日程安排'] += 1
                    elif 'recommend' in user_query.lower() or 'suggest' in user_query.lower():
                        train_scenarios['推荐服务'] += 1
                    else:
                        train_scenarios['其他'] += 1
        
        train_tools.extend(tools_in_sample)
    
    # 分析输出格式
    train_output_formats = {'think': 0, 'tool_call': 0, 'response': 0}
    for i, row in train_df.iterrows():
        reward_model = row['reward_model']
        if isinstance(reward_model, dict):
            output = reward_model.get('ground_truth', '')
            if '<think>' in str(output):
                train_output_formats['think'] += 1
            if '<tool_call>' in str(output):
                train_output_formats['tool_call'] += 1
            if '<response>' in str(output):
                train_output_formats['response'] += 1
    
    # 统计结果
    train_tool_freq = Counter(train_tools)
    avg_tools_per_sample = sum(train_tool_counts) / len(train_tool_counts) if train_tool_counts else 0
    
    # 随机选择样本进行展示
    random_idx = random.randint(0, train_size - 1)
    sample_row = train_df.iloc[random_idx]
    sample_prompt = sample_row['prompt']
    sample_user_query = ""
    sample_tools = ""
    
    for item in sample_prompt:
        if isinstance(item, dict):
            if item.get('role') == 'user':
                sample_user_query = item.get('content', '')
            elif item.get('role') == 'system':
                content = item.get('content', '')
                if '**Available Tools**' in content:
                    sample_tools = content.split('**Available Tools**')[1].split('**Steps for Each Turn**')[0]
    
    sample_output = ""
    if isinstance(sample_row['reward_model'], dict):
        sample_output = sample_row['reward_model'].get('ground_truth', '')
    
    # 生成报告
    report = f"""# RLLA 4K 数据集特点与场景分析

## 1. 数据集规模

- **训练集**: {train_size} 条样本
- **测试集**: {test_size} 条样本
- **比例**: 训练集:测试集 = {train_size/test_size:.1f}:1

## 2. 工具使用分析

- **平均每个样本工具数量**: {avg_tools_per_sample:.2f} 个
- **最常见的工具** (前10):
{chr(10).join(['  - ' + f'`{tool}`: {count} 次' for tool, count in train_tool_freq.most_common(10)])}

## 3. 场景分布

{chr(10).join(['- **' + f'{scenario}**: {count} 条样本 ({count/train_size*100:.1f}%)' for scenario, count in sorted(train_scenarios.items(), key=lambda x: x[1], reverse=True)])}

## 4. 输出格式分析

- **包含思考过程 (`<think>`)**: {train_output_formats['think']} 条样本 ({train_output_formats['think']/train_size*100:.1f}%)
- **包含工具调用 (`<tool_call>`)**: {train_output_formats['tool_call']} 条样本 ({train_output_formats['tool_call']/train_size*100:.1f}%)
- **包含直接回应 (`<response>`)**: {train_output_formats['response']} 条样本 ({train_output_formats['response']/train_size*100:.1f}%)

## 5. 数据集特点总结

1. **工具多样性**: 数据集包含多种不同类型的工具，涵盖各种功能领域
2. **场景丰富性**: 包含多种日常使用场景，如查询、预订、计算、编程等
3. **结构化输出**: 所有样本都要求模型输出思考过程，大部分需要工具调用
4. **真实交互模拟**: 模拟真实用户与AI助手的交互场景，包含复杂查询和多步骤任务

## 6. 样本展示

### 用户查询示例:

```
{sample_user_query[:500] + '...' if len(sample_user_query) > 500 else sample_user_query}
```

### 可用工具示例:

```
{sample_tools[:500] + '...' if len(sample_tools) > 500 else sample_tools}
```

### 标准输出示例:

```
{sample_output[:500] + '...' if len(sample_output) > 500 else sample_output}
```

## 7. 训练目标与应用场景

这个数据集设计用于训练模型:

1. **理解复杂指令**: 从用户查询中提取关键信息和意图
2. **工具选择能力**: 根据任务需求选择合适的工具
3. **参数提取能力**: 从用户查询中提取正确的参数
4. **结构化输出**: 按照指定格式输出思考过程和最终回应
5. **多步骤规划**: 处理需要多个工具调用的复杂任务

### 应用场景:

1. **个人助手**: 帮助用户完成日常任务，如查询信息、预订服务等
2. **专业工具**: 在特定领域提供专业服务，如编程辅助、数据分析等
3. **信息检索**: 通过API调用获取和整合各种信息源的数据
4. **自动化流程**: 代替用户执行多步骤的复杂流程
"""
    
    # 写入报告文件
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已生成: {REPORT_FILE}")
    return report

def main():
    """主函数"""
    analyze_datasets()
    print("任务完成!")

if __name__ == "__main__":
    main()
