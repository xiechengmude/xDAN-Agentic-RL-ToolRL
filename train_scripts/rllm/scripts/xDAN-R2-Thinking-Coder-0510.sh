#!/bin/bash
set -x

conda activate rllm
export NCCL_SOCKET_IFNAME=ibs13
export GLOO_SOCKET_IFNAME=ens14f1

# 设置内存限制
#ulimit -v 20971520

#export RAY_WORKER_MEMORY_LIMIT="101943040" # 100GB内存限制
#export RAY_WORKER_TIMEOUT="30" # 30秒超时
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    # MODEL_PATH="/data/vayu/train/models/xDAN-R1-Thinking-32b"
    MODEL_PATH="/data/vayu/train/models/DeepSeek-R1-Distill-Qwen-32B"
fi

# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/vayu/train/rllm/rllm/data/deepcoder_train.parquet \
    data.val_files=/data/vayu/train/rllm/rllm/data/test_livecodebench.parquet \
    data.train_batch_size=24 \
    data.val_batch_size=48 \
    data.max_prompt_length=1536 \
    data.max_response_length=12288 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size=12 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.compute_reward=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.n_val=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepcoder' \
    trainer.experiment_name='xDAN-R1-Thinking-32b-16k-grpo+-code-0412' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=3 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 "${@:1}"