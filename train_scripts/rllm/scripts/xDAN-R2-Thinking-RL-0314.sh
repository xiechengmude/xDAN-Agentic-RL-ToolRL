#!/bin/bash

DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
MODEL_CPK_NAME="xDAN-R2-RL-32B-Alignment-Thinking-0314"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-R2-Thinking-Base"
SAVE_PATH="/data/vayu/train/models/thinking/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# ray start --head \
#     --num-gpus 8 \
#     --resources='{"head_node": 1}' \
#     --node-ip-address 10.11.50.33 \
#     --port=6379 \
#     --temp-dir /data/vayu/train/ray \
#     --object-store-memory=100000000000

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-Vision-RL-Zero", "env_vars": {"MASTER_ADDR": "10.11.50.32", "MASTER_PORT": "24999"}}' \
   -- python3 -m openrlhf.cli.train_sft \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --max_len 32768 \
   --dataset $DATASET \
   --input_key prompt \
   --output_key answer \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --save_steps 1000 \
   --save_hf_ckpt \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --adam_offload \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --packing_samples \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing