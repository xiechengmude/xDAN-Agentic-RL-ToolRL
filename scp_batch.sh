#!/bin/bash

# 检查至少提供了一个参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file_or_directory_to_transfer> <target_directory_on_remote_server> [optional: remote_hosts...]"
    exit 1
fi

FILE_OR_DIR_TO_TRANSFER=$1
TARGET_DIRECTORY=$2
USER="vayu" # 定义远程主机上的用户

# 如果提供了远程主机参数，则使用它们；否则，使用默认主机列表
if [ "$#" -gt 2 ]; then
    shift 2 # 移除前两个参数
    REMOTE_HOSTS=("$@") # 剩余的所有参数都是远程主机
else
    # 定义默认远程主机列表
    REMOTE_HOSTS=("gpu005" "gpu006" "gpu007" "gpu008" "gpu009")

fi

# 循环遍历每个远程主机并执行SCP命令传输文件或目录
for HOST in "${REMOTE_HOSTS[@]}"; do
    echo "Starting transfer of $FILE_OR_DIR_TO_TRANSFER to $USER@$HOST:$TARGET_DIRECTORY"
    scp -r "$FILE_OR_DIR_TO_TRANSFER" "$USER@$HOST:$TARGET_DIRECTORY" && echo "Transfer of $FILE_OR_DIR_TO_TRANSFER to $HOST completed."
done

echo "All transfers completed."