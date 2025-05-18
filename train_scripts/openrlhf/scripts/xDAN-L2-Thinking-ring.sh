set -x

deepspeed  --num_gpus 8 --num_nodes 1 --module openrlhf.cli.train_sft \
   --max_len 16384 \
   --dataset /data/vayu/train/datasets/xDAN-Thinking-Large-alpaca/alpaca_0.9M.json \
   --input_key instruction \
   --output_key output \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --max_samples 100000 \
   --pretrain /data/vayu/train/models/xDAN-L2-Qwen25-32b-Base \
   --save_path /data/vayu/train/models/ckpts/xDAN-R2-Thinking-Large-0404-sft-ring-100k-e2 \
   --save_steps 5000 \
   --logging_steps 1 \
   --eval_steps 5000 \
   --zero_stage 3 \
   --adam_offload \
   --adam_betas 0.95 0.99 \
   --lr_warmup_ratio 0.03 \
   --max_epochs 2 \
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
