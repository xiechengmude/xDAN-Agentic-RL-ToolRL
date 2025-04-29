# export CUDA_VISIBLE_DEVICES=0,1
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

export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="PATH/TO/BASE_MODEL" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="PATH/TO/SAVE_DIR" # e.g., "ppo-qwen2.5-3b"
bash ./examples/ppo_trainer/run_ppo.sh