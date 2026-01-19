#!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_API_KEY="804f99947d014002648b0e99ae3c09633161e7a0"
export WANDB_PROJECT="gemma"

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/train_snomed_prediction_sft.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --freeze_projector False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --output_dir /outputs/snomed_4999_models \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 10 \
    --dataloader_num_workers 64