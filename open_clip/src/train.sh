#!/bin/bash

# Cấu hình
MODEL_NAME="ViT-B-32"
PRETRAINED="D:\Projects\doan\logs\AllData_finetune4\All-data-finetune4\checkpoints\epoch_10.pt"
BATCH_SIZE=8
WORKERS=14
EPOCHS=20
LR=1e-4

# Đường dẫn
DATA_PATH="D:/Projects/kaggle/train.csv"
VAL_DATA_PATH="D:/Projects/kaggle/val.csv"
OUTPUT_DIR="D:/Projects/doan/checkpoints/AllData_finetune5_from_ckpt4"
LOG_DIR="D:/Projects/doan/logs/AllData_finetune5_from_ckpt4"

# Tạo thư mục output
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Training không LoRA với single GPU
python -m open_clip_train.main \
    --save-frequency 2 \
    --train-data $DATA_PATH \
    --val-data $VAL_DATA_PATH \
    --csv-separator ";" \
    --csv-img-key image_filename \
    --csv-caption-key caption \
    --warmup 500 \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wd 0.1 \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --model $MODEL_NAME \
    --resume $PRETRAINED \
    --logs $LOG_DIR \
    --name "All-data-finetune5_from_ckpt4" \
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
    #--pretrained $PRETRAINED \
# Nếu có nhiều GPU, sử dụng:
# torchrun --nproc_per_node=4 -m training.main \
#     [... các tham số như trên ...]




# LoRA Training
# python -m open_clip_train.main \
#     --save-frequency 1 \
#     --train-data $DATA_PATH \
#     --csv-separator "," \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --warmup 10 \
#     --batch-size 4 \
#     --lr 1e-5 \
#     --wd 0.1 \
#     --epochs 1 \
#     --workers $WORKERS \
#     --model $MODEL_NAME \
#     --pretrained laion2b_s34b_b79k \
#     --logs "D:/Projects/doan/logs/test_lora" \
#     --name "test_lora_training" \
#     --report-to tensorboard \
#     --local-loss \
#     --gather-with-grad \
#     --lock-image \
#     --use-lora \
#     --lora-r 16 \
#     --lora-alpha 16 \
#     --lora-dropout 0.1 \
#     --save-lora-only