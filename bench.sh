#!/bin/bash

# Run all STEP generations simultaneously with separate log files
DATA_PATH=/data/vayu/train/am/bench

echo "Starting all generations at $(date)"

# Function to run generation for a specific STEP
run_generation() {
    local STEP=$1
    local MODEL_PATH=/data/vayu/train/models/xDAN-Qwen3-Code-Math-Step-$STEP
    echo "Starting generation for STEP=$STEP at $(date)" >> generation_step_${STEP}.log
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$DATA_PATH/beyondAIME-ojbench-merged.parquet \
        data.prompt_key=prompt \
        data.batch_size=2048 \
        data.n_samples=64 \
        data.output_path=$DATA_PATH/beyondAIME-ojbench-64-$STEP.parquet \
        model.path=$MODEL_PATH \
        rollout.temperature=0.6 \
        rollout.top_p=0.95 \
        rollout.prompt_length=16384 \
        rollout.response_length=16384 \
        rollout.tensor_model_parallel_size=4 \
        rollout.gpu_memory_utilization=0.8 \
        rollout.max_num_batched_tokens=65536 \
        > generation_step_${STEP}.log 2>&1
    echo "Completed generation for STEP=$STEP at $(date)" >> generation_step_${STEP}.log
}

# Start all generations in background
for STEP in 0 60 80 140 160 180; do
    run_generation $STEP &
    sleep 60
done

# Wait for all background jobs to complete
wait

echo "All generations completed at $(date)"
