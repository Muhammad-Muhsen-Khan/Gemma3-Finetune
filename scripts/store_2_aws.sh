#!/bin/bash

EPOCH_NUM=1
CHECKPOINT_NUM=$((EPOCH_NUM * 7))
MODEL_PATH="/root/Gemma3-Finetune/output/test/checkpoint-$CHECKPOINT_NUM"
aws s3 cp --recursive --exclude "*.pt" $MODEL_PATH \
    s3://avey-migration-bucket/tasnim/gemma-poc-6/epoch-$EPOCH_NUM/checkpoint-$CHECKPOINT_NUM/

if [ $? -eq 0 ]; then
    echo "Checkpoint $CHECKPOINT_NUM stored successfully"
    rm -rf $MODEL_PATH
else
    echo "Checkpoint $CHECKPOINT_NUM storage failed"
fi