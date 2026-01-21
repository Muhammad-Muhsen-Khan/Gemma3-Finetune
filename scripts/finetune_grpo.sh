#!/bin/bash

MODEL_NAME="/workspace/output/checkpoint-2976"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_API_KEY="804f99947d014002648b0e99ae3c09633161e7a0"
export WANDB_PROJECT="gemma"

deepspeed src/train/train_grpo.py \
    --loss_type "grpo" \
    --optim adamw_bnb_8bit \
    --max_completion_length 256 \
    --max_prompt_length 256 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /root/Gemma3-Finetune/data/train_snomed_prediction_rl.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --freeze_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --output_dir /workspace/output/snomed_prediction_grpo \
    --num_train_epochs 5 \
    --num_generations 4 \
    --per_device_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 64