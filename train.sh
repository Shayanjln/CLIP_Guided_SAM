#!/bin/bash
# =============================================================================
# Training script for CLIP-Guided SAM (supervised, multi-GPU, no SLURM)
# Usage: bash train.sh
# Set DATASET below to one of: coco | pascal | ade20k | camouflaged
# =============================================================================

# ---- Dataset selection -------------------------------------------------------
DATASET="coco"   # <-- change this to switch datasets

case $DATASET in
  coco)
    SPLIT="1_512"
    LABELED_JSON="COCO_samples_${SPLIT}_labeled.json"
    VAL_JSON="COCO_samples_val.json"
    SAM_CKPT="sam_vit_b_01ec64.pth"
    ADE=false; PASCAL=false; CROPS=false; CAMOUFLAGED=false; USE_MAE=false
    ;;
  pascal)
    SPLIT="1_16"
    LABELED_JSON="Pascal_samples_${SPLIT}_labeled.json"
    VAL_JSON="Pascal_samples_val.json"
    SAM_CKPT="sam_vit_b_01ec64.pth"
    ADE=false; PASCAL=true; CROPS=false; CAMOUFLAGED=false; USE_MAE=false
    ;;
  ade20k)
    SPLIT="1_4"
    LABELED_JSON="ADE_samples_${SPLIT}_labeled.json"
    VAL_JSON="ADE_samples_val.json"
    SAM_CKPT="sam_vit_b_01ec64.pth"
    ADE=true; PASCAL=false; CROPS=false; CAMOUFLAGED=false; USE_MAE=false
    ;;
  camouflaged)
    SPLIT="1_1"
    LABELED_JSON=""
    VAL_JSON=""
    SAM_CKPT="sam_vit_b_01ec64.pth"
    ADE=false; PASCAL=false; CROPS=false; CAMOUFLAGED=true; USE_MAE=true
    ;;
  *)
    echo "Unknown DATASET: $DATASET. Choose one of: coco | pascal | ade20k | camouflaged"
    exit 1
    ;;
esac

# Manual Example
# SPLIT="1_16"
# LABELED_JSON="../../SAM_CLIP_Script_Training/Pascal_samples_${SPLIT}_labeled.json"
# VAL_JSON="../../SAM_CLIP_Script_Training/Pascal_samples_val.json"
# SAM_CKPT="sam_vit_b_01ec64.pth"
# ADE=false; PASCAL=true; CROPS=false; CAMOUFLAGED=false; USE_MAE=false


# ---- Training hyperparameters -----------------------------------------------
BATCH_SIZE=1
EPOCHS=40
TOTAL_EPOCHS=40
LR_IE=5e-5
LR_MD=5e-5
LR_PE=5e-5
LR_CLIP=1e-7

# ---- Checkpoint resume (set RESUME_CKPT="" to train from scratch) ------------
RESUME_CKPT=None

# ---- Infrastructure ---------------------------------------------------------
NUM_GPUS=2
MASTER_PORT=12353
SAVE_PATH="."
VERSION=1
LOG_DIR="./logs"
NUM_WORKERS=4

# =============================================================================
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  train.py \
  --labeled_json $LABELED_JSON \
  --val_json $VAL_JSON \
  --sam_ckpt $SAM_CKPT \
  --split $SPLIT \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --total_epochs $TOTAL_EPOCHS \
  --lr_IE $LR_IE \
  --lr_MD $LR_MD \
  --lr_PE $LR_PE \
  --lr_clip $LR_CLIP \
  --save_path $SAVE_PATH \
  --version $VERSION \
  --log_dir $LOG_DIR \
  --num_workers $NUM_WORKERS \
  --ade20k $ADE \
  --pascal $PASCAL \
  --crops $CROPS \
  --camouflaged $CAMOUFLAGED \
  --use_mae $USE_MAE \
  --resume_checkpoint $RESUME_CKPT \
  --text_vis true \
  --finetune_clip true \
  --mask_prompts true \
  --finetune_PE true \
  --clip_crop_size 512 \
  --iou_loss true \
  --weight_decay 0.001 \
  --SAM_IE Parallel_Text_Vis \
  --clip_type CS-ViT-B/16 \
  --num_gpus $NUM_GPUS