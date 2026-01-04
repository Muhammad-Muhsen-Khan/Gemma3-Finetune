#!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`

export PYTHONPATH=src:$PYTHONPATH
export WANDB_API_KEY="804f99947d014002648b0e99ae3c09633161e7a0"
export WANDB_PROJECT="gemma"

deepspeed src/train/train_sft.py \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/train_md_symptoms_sft.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --freeze_projector False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --output_dir output/test_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
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
    --dataloader_num_workers 64 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10