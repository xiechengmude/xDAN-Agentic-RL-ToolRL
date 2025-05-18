# xDAN-Agentic-RL-ToolRL 项目架构文档

## 1. 项目概述

xDAN-Agentic-RL-ToolRL是一个基于强化学习的智能体工具调用训练框架，构建于veRL框架之上，旨在通过精细化的奖励机制提升大语言模型使用工具的能力。该项目支持PPO(近端策略优化)和GRPO(群组相对策略优化)算法，特别关注工具调用场景的训练需求。

## 2. 系统架构

### 2.1 核心组件架构

```
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │     │                       │
│    模型组件(Models)    │     │    训练器(Trainer)    │     │    工作器(Workers)    │
│                       │     │                       │     │                       │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘
        │                               │                               │
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │     │                       │
│   数据协议(DataProto)  │◄────┤   工具集(Utils)      │◄────┤   控制器(Controller)  │
│                       │     │                       │     │                       │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

### 2.2 训练流水线架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│  数据加载   │───►│  模型生成   │───►│  奖励计算   │───►│  策略优化   │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ▲                                                        │
       │                                                        │
       └────────────────────────────────────────────────────────┘
```

## 3. 目录结构

```
xDAN-Agentic-RL-ToolRL/
├── assets/                 # 资源文件
├── benchmarks/             # 基准测试
├── data/                   # 数据文件
│   └── prompt/             # 提示词模板
├── dataset/                # 训练数据集
│   ├── rlla_4k/            # 处理后的RLLA 4K数据集
│   └── rlla_4k_raw/        # 原始RLLA 4K数据集
├── docs/                   # 文档
│   ├── Train/              # 训练相关文档
│   ├── data/               # 数据相关文档
│   └── 深度搜索/            # 深度搜索智能体文档
├── examples/               # 样例代码
│   ├── grpo_trainer/       # GRPO训练器示例
│   ├── ppo_trainer/        # PPO训练器示例
│   └── sglang_multiturn/   # SGLang多轮对话示例
├── inference/              # 推理相关代码
│   └── tools/              # 工具实现
├── test/                   # 测试代码
├── train_scripts/          # 训练脚本
│   ├── Deepretrieve/       # 深度检索训练
│   ├── deepspeed/          # DeepSpeed配置
│   ├── openrlhf/           # OpenRLHF集成
│   ├── rllm/               # RLLM训练脚本
│   ├── scripts/            # 通用训练脚本
│   └── swift/              # Swift训练脚本
├── trace_gen/              # 轨迹生成器
└── verl/                   # 核心功能包
    ├── models/             # 模型定义
    │   ├── llama/          # LLaMA模型实现
    │   └── transformers/   # Transformers集成
    ├── single_controller/  # 单控制器实现
    ├── third_party/        # 第三方依赖
    │   └── vllm/           # VLLM集成
    ├── trainer/            # 训练器
    │   ├── config/         # 训练配置
    │   └── ppo/            # PPO算法实现
    ├── utils/              # 工具函数
    │   ├── dataset/        # 数据集工具
    │   ├── logger/         # 日志工具
    │   └── reward_score/   # 奖励计算模块
    └── workers/            # 工作器组件
        ├── actor/          # Actor实现
        ├── critic/         # Critic实现
        ├── reward_model/   # 奖励模型
        │   └── megatron/   # Megatron奖励模型
        └── rollout/        # Rollout工作器
            ├── naive/      # 简单Rollout实现
            └── vllm_rollout/ # VLLM加速Rollout
```

## 4. 核心模块详解

### 4.1 模型组件 (models)

- **支持的模型**: LLaMA系列、Qwen2系列等主流LLM
- **核心文件**: 
  - `registry.py`: 模型注册机制
  - `transformers/*.py`: 对接Transformers库的模型实现
  - `llama/megatron/*.py`: 基于Megatron的LLaMA实现

### 4.2 训练器 (trainer)

- **核心文件**:
  - `main_ppo.py`: PPO训练主循环
  - `main_eval.py`: 模型评估
  - `ppo/core_algos.py`: PPO核心算法实现
  - `ppo/ray_trainer.py`: 基于Ray的分布式训练

- **关键类**:
  - `RewardManager`: 奖励信号管理，负责计算和分发奖励

### 4.3 工作器 (workers)

- **Actor**: 负责生成动作(模型输出)
- **Critic**: 负责估计价值函数
- **reward_model**: 奖励模型，计算奖励信号
- **rollout**: 执行环境交互，收集经验

### 4.4 奖励系统 (reward_score)

- **模块路径**: `verl/utils/reward_score/`
- **支持的任务**:
  - `rlla.py`: RLLA工具调用奖励计算
  - `gsm8k.py`: GSM8K数学推理奖励
  - `math.py`: 一般数学问题奖励
  - `multiply.py`: 乘法运算奖励

- **奖励组件**:
  - **格式评分**: 评估输出格式正确性
  - **正确性评分**: 评估工具调用准确性
  - **长度评分**: 评估思考过程充分性

### 4.5 工具集 (utils)

- **dataset**: 数据集处理工具
- **logger**: 日志记录工具
- **reward_score**: 奖励计算模块
- **debug**: 调试工具

## 5. 训练流程

### 5.1 GRPO训练流程

1. **数据准备**: 从`dataset/rlla_4k`加载处理好的数据
2. **模型初始化**: 加载预训练模型权重
3. **训练循环**:
   - Actor生成响应
   - 奖励模型计算奖励信号
   - 根据相对排名计算策略梯度
   - 更新模型参数
4. **评估与保存**: 定期评估模型性能并保存检查点

### 5.2 奖励计算流程

1. **预处理**: 提取模型输出和标准答案
2. **格式评分**: 使用正则表达式检查标签格式
3. **正确性评分**: 计算工具名称、参数名和参数值的匹配度
4. **长度评分**: 计算思考过程长度比例
5. **综合计算**: 综合三个维度得分得到最终奖励

## 6. 配置系统

### 6.1 环境变量配置

框架支持通过环境变量配置奖励函数行为:

| 环境变量 | 功能描述 |
|---------|---------|
| WITHLENGTH | 启用长度奖励 |
| SCHEDULELENGTH | 启用动态长度奖励 |
| CORRECTMAX1 | 使用相等最大值 |
| MAX1STEP30MAX3 | 使用两阶段尺度 |
| SCHEDULEREWARD | 使用平滑动态尺度 |
| REFINEDREWARD | 使用细粒度奖励 |
| INTERMEDIATEREWARD | 使用中间奖励 |
| COARSEREWARD | 使用粗粒度奖励 |

### 6.2 训练脚本配置

通过`train_grpo.sh`和`train_ppo.sh`脚本设置基本训练参数:
- 基础模型路径
- 实验名称
- 数据集路径
- 批处理大小
- 学习率等

## 7. 应用领域

### 7.1 通用工具调用智能体

支持通用工具调用场景，如网页搜索、数据分析、API调用等。

### 7.2 领域特化智能体

#### 7.2.1 金融投资智能体
- 支持市场数据、金融分析、投资组合管理、风险评估等金融工具
- 提供合规性、投资逻辑等特定奖励函数

#### 7.2.2 深度搜索智能体
- 支持多模态信息整合、时间敏感信息追踪等高级搜索能力
- 提供复杂推理链、结果评估等特化功能

## 8. 扩展与定制

### 8.1 添加新领域智能体

1. 定义领域特定工具集
2. 设计领域特定奖励函数
3. 准备领域训练数据
4. 配置训练脚本

### 8.2 自定义奖励函数

1. 在`verl/utils/reward_score/`添加新的奖励计算模块
2. 实现奖励计算函数
3. 在训练脚本中引用该奖励函数

## 9. 项目优势与特点

1. **精细化奖励设计**: 多维度、可配置的奖励机制
2. **灵活模型支持**: 支持多种主流大语言模型
3. **分布式训练**: 基于Ray的高效分布式训练
4. **多算法支持**: 支持PPO和GRPO等先进强化学习算法
5. **领域适配能力**: 可扩展到金融、搜索等专业领域

## 10. 数据集

项目使用RLLA 4K数据集进行训练，该数据集由以下部分组成:
- 2K ToolACE数据
- 1K Hammer (Masked)数据
- 1K xLAM数据

SFT数据包含从Deepseek-R1提取的思考内容，而RL数据在思考字段中仅包含占位符。 

def compute_multi_turn_reward(
    conversation_history,  # 完整对话历史
    gt_conversation,       # 标准答案对话历史
    max_reward=3.0,
    min_reward=-3.0
):
    """计算多轮工具调用的奖励"""
    # 1. 基础奖励计算
    base_reward = 0.0
    
    # 2. 工具选择评分 - 每轮工具选择是否正确
    tool_selection_reward = evaluate_tool_selection(conversation_history, gt_conversation)
    
    # 3. 信息利用评分 - 评估模型如何利用前轮工具返回信息
    info_utilization_reward = evaluate_info_utilization(conversation_history)
    
    # 4. 推理链评分 - 评估整个工具调用链是否合理
    reasoning_chain_reward = evaluate_reasoning_chain(conversation_history)
    
    # 5. 问题解决评分 - 评估是否最终解决了用户问题
    problem_solving_reward = evaluate_problem_solving(conversation_history)
    
    # 综合计算最终奖励
    final_reward = base_reward + tool_selection_reward + info_utilization_reward + \
                  reasoning_chain_reward + problem_solving_reward
    
    # 归一化到指定范围
    return normalize_reward(final_reward, max_reward, min_reward) 