#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <step_number>"
    echo "例如: $0 225"
    exit 1
fi

# 获取步骤数
STEP=$1

# 定义路径
BASE_PATH_GPU005="/data/vayu/train/xDAN-Agentic-RL-ToolRL/checkpoints/xDAN-R2-Thinking-AgentRL/qwen-xdan-l2-qwen3-32b-reasoning"
BASE_PATH_GPU006_007="/data/vayu/train/checkpoints/xDAN-R2-Thinking-AgentRL/qwen-xdan-l2-qwen3-32b-reasoning"
DEST_PATH="/data/vayu/train/xDAN-Agentic-RL-ToolRL/checkpoints/xDAN-R2-Thinking-AgentRL/qwen-xdan-l2-qwen3-32b-reasoning"
DEST_SERVER="vayu@gpu004"

# 创建目标目录
echo "创建目标目录..."
ssh ${DEST_SERVER} "mkdir -p ${DEST_PATH}/global_step_${STEP}/actor"

# 从GPU005复制到GPU004
echo "从GPU005复制到GPU004..."
scp /data/vayu/train/xDAN-Agentic-RL-ToolRL/checkpoints/xDAN-R2-Thinking-AgentRL/qwen-xdan-l2-qwen3-32b-reasoning/global_step_${STEP}/actor/* ${DEST_SERVER}:${DEST_PATH}/global_step_${STEP}/actor/

# 从GPU006复制到GPU004
echo "从GPU006复制到GPU004..."
ssh vayu@gpu006 "scp ${BASE_PATH_GPU006_007}/global_step_${STEP}/actor/* ${DEST_SERVER}:${DEST_PATH}/global_step_${STEP}/actor/"

# 从GPU007复制到GPU004
echo "从GPU007复制到GPU004..."
ssh vayu@gpu007 "scp ${BASE_PATH_GPU006_007}/global_step_${STEP}/actor/* ${DEST_SERVER}:${DEST_PATH}/global_step_${STEP}/actor/"

echo "完成！所有文件已复制到GPU004的${DEST_PATH}/global_step_${STEP}/actor/"
