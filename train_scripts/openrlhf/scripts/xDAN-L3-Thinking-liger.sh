set -x

deepspeed  --num_gpus 8 --num_nodes 1 --module openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset /data/vayu/train/datasets/xDAN-Thinking-Large-alpaca/alpaca_0.9M.json \
   --input_key instruction \
   --output_key output \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --max_samples 90000 \
   --pretrain /data/vayu/train/models/xDAN-L3-Qwen25-72b-Base \
   --save_path /data/vayu/train/models/ckpts/xDAN-R3-72b-Thinking-Large-0405-sft-ring-e1-sample90k \
   --save_steps 1000 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --adam_offload \
   --adam_betas 0.95 0.99 \
   --lr_warmup_ratio 0.03 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --ring_attn_size 8 \
   --ring_head_stride 2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --save_hf_ckpt \
   --use_wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620

    # --wandb 1b2653c58df0ccf5b38f3ffa1bf21b78d48fd620
    # --packing_samples