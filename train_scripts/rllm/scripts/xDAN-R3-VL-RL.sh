#!/bin/bash
set -x
# wget https://raw.githubusercontent.com/TideDra/lmm-r1/refs/heads/main/examples/data/mathlv345_8k_chatml.json
DATASET="examples/data/mathlv345_8k_chatml_1p.json"
#DATASET="/data/vayu/train/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
PRETRAIN_MODEL="/data/vayu/train/models/xDAN-L3-VL-72b-RL-Base"

MODEL_CPK_NAME="xDAN-V2-RL-72B-0308"
SAVE_PATH="/data/vayu/train/models/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}/logs"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/vayu/train/xDAN-Vision-RL-Zero", "env_vars": {"MASTER_ADDR": "10.11.50.36", "MASTER_PORT": "24999"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url examples/scripts/math_verifier.py \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --vllm_gpu_memory_utilization 0.85 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 64 \
   --temperature 1 \
   --n_samples_per_prompt 8 \
   --max_epochs 2 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 10000 \
   --generate_max_len 3000 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
   --adam_offload \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.02 \
   --prompt_data $DATASET \
   --input_key prompt \
   --label_key answer \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 4 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620

   # for visual dataset