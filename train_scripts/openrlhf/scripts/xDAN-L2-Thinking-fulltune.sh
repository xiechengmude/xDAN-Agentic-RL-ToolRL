#!/bin/bash

# 指定IB网卡地址和配置
export NCCL_SOCKET_IFNAME=ibs13  # 根据ip addr命令输出，活跃的IB接口是ibs13
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0  # 根据ibstat输出，使用mlx5_0作为HCA
export NCCL_DEBUG=INFO  # 启用调试信息，帮助排查问题
export MASTER_ADDR=10.110.10.5  # 使用ibs13接口的IP地址

set -x

deepspeed --hostfile /data/vayu/train/config/hostfile_openrlhf --num_gpus 8 --num_nodes 3 --module openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset /data/vayu/train/datasets/xDAN-Thinking-Large-alpaca/alpaca_0.9M.json \
   --input_key instruction \
   --output_key output \
   --train_batch_size 48 \
   --micro_train_batch_size 1 \
   --max_samples 5000000 \
   --pretrain /data/vayu/train/models/xDAN-L2-Qwen25-32b-Base\
   --save_path /data/vayu/train/models/ckpts/xDAN-R2-Thinking-Large-0404-sft \
   --save_steps 10000 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --adam_offload \
   --adam_betas 0.95 0.99 \
   --lr_warmup_ratio 0.03 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --save_hf_ckpt \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620

    # --wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620
    # --packing_samples