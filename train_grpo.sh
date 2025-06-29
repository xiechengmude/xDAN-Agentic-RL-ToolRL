#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export ROLLOUT_TP_SIZE=2
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
# export VLLM_ATTENTION_BACKEND=XFORMERS

# All the env variables below are set to 0 by default
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

#export CUDA_HOME=$CONDA_PREFIX
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=ibs13
export GLOO_SOCKET_IFNAME=ibs13

export DATA_DIR="./dataset/rlla_4k"
#export DATA_DIR="./dataset/Eurus-2-RL-Data"
#export BASE_MODEL="/data/vayu/train/models/xDAN-R2-Thinking-0401"
#export BASE_MODEL="/data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct"
export BASE_MODEL="/data/vayu/train/models/xDAN-L2-Qwen3-14b-Instruct" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="qwen-xdan-l2-qwen3-14b-reasoning" # e.g., "grpo-qwen2.5-3b"
bash ./examples/grpo_trainer/run_grpo.sh
