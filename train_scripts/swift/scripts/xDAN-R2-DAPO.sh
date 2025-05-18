export GLOO_SOCKET_IFNAME=ens14f1
export NCCL_SOCKET_IFNAME=ibs13
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=10.110.10.2
export MASTER_PORT=29500
export NPROC_PER_NODE=4

swift rlhf \
    --rlhf_type grpo \
    --model /data/vayu/train/models/xDAN-R2-Thinking-0401 \
    --model_type qwen2_5 \
    --reward_funcs accuracy soft_overlong \
    --max_completion_length 4096 \
    --soft_cache_length 819 \
    --seed 42 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --overlong_filter true \
    --max_resample_times 3 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.6 \
    --tensor_parallel_size 4 \
    --num_infer_workers 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 1.0 \
    --deepspeed zero3 \
    --async_generator false \
    --log_completions true \
    --num_iterations 2 \
    --report_to  wandb \
    --beta 0.0 