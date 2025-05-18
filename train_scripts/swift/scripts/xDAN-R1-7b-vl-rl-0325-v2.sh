#!/bin/bash
{
# Two GPUs are left for vLLM inference acceleration.
# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 8 * 60GiB

MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/vayu/train/models/Qwen2.5-VL-7B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-ModelScope/chartqa_digit_r1v_format \
    --max_completion_length 1536 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-7 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 24 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 4 \
    --tensor_parallel_size 2 \
    --report_to tensorboard wandb
} > Task_SFT_Finetune_xDAN-R1-7b-Thinking-VL-0325-$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S').log 2>&1 &