#!/bin/bash

# Small-scale training test với LoRA
# Chạy 50 steps để verify training loop works

MODEL_NAME="ViT-B-32"
PRETRAINED="laion2b_s34b_b79k"
BATCH_SIZE=4
WORKERS=4
EPOCHS=1
LR=1e-5

# Đường dẫn
DATA_PATH="D:/Projects/doan/metadata.csv"
OUTPUT_DIR="D:/Projects/doan/checkpoints/test_lora"
LOG_DIR="D:/Projects/doan/logs/test_lora"

# Training với single GPU - test mode
python -m open_clip_train.main \
    --save-frequency 1 \
    --train-data $DATA_PATH \
    --csv-separator "," \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 10 \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wd 0.1 \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --model $MODEL_NAME \
    --pretrained $PRETRAINED \
    --logs $LOG_DIR \
    --name "test_lora_training" \
    --report-to tensorboard \
    --local-loss \
    --gather-with-grad \
    --lock-image \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-target-modules "in_proj,out_proj" \
    --save-lora-only
