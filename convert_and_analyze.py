#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# 设置输出文件路径
INPUT_PARQUET = 'dataset/rlla_4k/train.parquet'
OUTPUT_JSON = 'dataset/rlla_4k/train.json'
REPORT_FILE = 'dataset/rlla_4k/数据集分析报告.md'

# 自定义JSON编码器，处理NumPy数组和其他非JSON可序列化对象
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

def convert_parquet_to_json():
    """将parquet文件转换为JSON格式"""
    print(f"开始转换 {INPUT_PARQUET} 到 {OUTPUT_JSON}...")
    
    # 读取parquet文件
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 转换为Python原生对象
    result = []
    for _, row in df.iterrows():
        item = {}
        for col in df.columns:
            item[col] = row[col]
        result.append(item)
    
    # 写入JSON文件
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    
    print(f"转换完成! 共转换 {len(result)} 条记录")
    return df

def analyze_dataset(df):
    """分析数据集并生成报告"""
    print("开始分析数据集...")
    
    # 基本统计信息
    total_samples = len(df)
    data_sources = df['data_source'].value_counts().to_dict()
    abilities = df['ability'].value_counts().to_dict()
    
    # 分析提示内容
    prompt_types = []
    prompt_lengths = []
    system_instructions = []
    tool_counts = []
    
    # 分析奖励模型
    reward_model_keys = []
    output_formats = []
    
    # 分析样本
    for i, row in df.iterrows():
        # 分析提示
        prompt = row['prompt']
        if isinstance(prompt, str):
            try:
                prompt_data = json.loads(prompt)
                prompt_types.append('json')
            except:
                prompt_data = prompt
                prompt_types.append('string')
        else:
            prompt_data = prompt
            prompt_types.append('list/dict')
        
        prompt_lengths.append(len(str(prompt_data)))
        
        # 尝试提取系统指令和工具数量
        if isinstance(prompt_data, list):
            for item in prompt_data:
                if isinstance(item, dict) and item.get('role') == 'system':
                    content = item.get('content', '')
                    system_instructions.append(content[:100])  # 只取前100个字符
                    
                    # 提取工具数量
                    tool_count = content.count('Name:')
                    tool_counts.append(tool_count)
                    break
        
        # 分析奖励模型
        reward_model = row['reward_model']
        if isinstance(reward_model, dict):
            reward_model_keys.extend(list(reward_model.keys()))
            
            # 分析输出格式
            output = reward_model.get('ground_truth', '')
            if '<think>' in str(output):
                output_formats.append('think')
            if '<tool_call>' in str(output):
                output_formats.append('tool_call')
            if '<response>' in str(output):
                output_formats.append('response')
    
    # 统计结果
    prompt_type_counts = Counter(prompt_types)
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    reward_model_key_counts = Counter(reward_model_keys)
    output_format_counts = Counter(output_formats)
    avg_tool_count = sum(tool_counts) / len(tool_counts) if tool_counts else 0
    
    # 生成报告
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    avg_prompt_length_str = f"{avg_prompt_length:.2f}"
    avg_tool_count_str = f"{avg_tool_count:.2f}"
    
    report = f"""# RLLA 4K 数据集分析报告

## 1. 基本信息

- **数据集大小**: {total_samples} 条样本
- **数据源分布**: {str(data_sources)}
- **能力分布**: {str(abilities)}
- **生成时间**: {now_time}

## 2. 数据结构

数据集包含以下列:
- data_source: 数据来源
- prompt: 提示内容
- ability: 能力标签
- reward_model: 奖励模型信息
- extra_info: 额外信息

## 3. 提示内容分析

- **提示类型分布**: {str(dict(prompt_type_counts))}
- **平均提示长度**: {avg_prompt_length_str} 字符
- **平均工具数量**: {avg_tool_count_str} 个工具/样本

## 4. 奖励模型分析

- **奖励模型字段分布**: {str(dict(reward_model_key_counts))}
- **输出格式分布**: {str(dict(output_format_counts))}

## 5. 数据结构抽象

```
{{
  "data_source": "数据来源标识",
  "prompt": [
    {{
      "role": "system",
      "content": "系统指令，包含可用工具描述和输出格式要求"
    }},
    {{
      "role": "user",
      "content": "用户查询或请求"
    }}
  ],
  "ability": "能力标签",
  "reward_model": {{
    "ground_truth": "标准答案，通常包含<think>、<tool_call>或<response>标签",
    "style": "回答风格标签"
  }},
  "extra_info": {{
    "instruction": "详细指令",
    "output": "模型输出",
    "split": "数据集分割标识"
  }}
}}
```

## 6. 数据集特点

1. **工具调用训练**: 数据集专为训练模型使用工具调用能力而设计
2. **结构化输出**: 训练目标是让模型生成符合特定格式的输出
3. **多轮对话**: 支持多轮对话训练，包含对话历史记录
4. **工具多样性**: 包含各种不同的工具API，涵盖查询、检索、验证等多种功能

## 7. 训练目标

这个数据集设计用于训练模型:
1. 理解用户查询并判断是否需要调用工具
2. 正确选择合适的工具并提供正确的参数
3. 按照指定格式输出思考过程和最终回应
4. 处理多轮对话中的上下文信息
"""
    
    # 写入报告文件
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已生成: {REPORT_FILE}")
    return report

def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # 转换数据
    df = convert_parquet_to_json()
    
    # 分析数据
    analyze_dataset(df)
    
    print("任务完成!")

if __name__ == "__main__":
    main()
