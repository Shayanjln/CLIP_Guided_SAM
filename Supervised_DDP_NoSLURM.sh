#!/bin/bash

# Number of GPUs to use (adjust based on availability)
NUM_GPUS=2

# Free port for communication between processes
MASTER_PORT=12353

CAMOUFLAGED=False
USE_MAE=False


# SPLIT=1_512
# LABELED_JSON="COCO_samples_${SPLIT}_labeled.json"
# UNLABELED_JSON="COCO_samples_${SPLIT}_unlabeled_uniformly_reduced_100.json"
# VAL_JSON="COCO_samples_val.json"
# #SUPERVISED_CHECKPOINT="trained_supervised_sam_${SPLIT}.pth"
# SAM_CKPT="sam_vit_h_4b8939.pth"
# CLIP_CHECKPOINT="s"
# ADE=false
# PASCAL=false

# SPLIT=1_16
# LABELED_JSON="Pascal_samples_${SPLIT}_labeled.json"
# #UNLABELED_JSON="COCO_samples_${SPLIT}_unlabeled_uniformly_reduced_100.json"
# VAL_JSON="Pascal_samples_val.json"
# SAM_CKPT="sam_vit_b_01ec64.pth"
# #SAM_CKPT="sam_vit_h_4b8939.pth"
# CLIP_CHECKPOINT="sssupervised_best_clip_IoU__v5_Supervised_PascalVOC_CLIP_1e-05_batchsize1.pth"
# PASCAL=true
# ADE=false
# CROPS=false


# SPLIT=1_4
# LABELED_JSON="ADE_samples_${SPLIT}_labeled.json"
# #UNLABELED_JSON="COCO_samples_${SPLIT}_unlabeled_uniformly_reduced_100.json"
# VAL_JSON="ADE_samples_val.json"
# SUPERVISED_CHECKPOINT="trained_supervised_sam_${SPLIT}.pth"
# SAM_CKPT="sam_vit_b_01ec64.pth"
# CLIP_CHECKPOINT="s"
# ADE=true
# PASCAL=false

###Camouflaged
# SPLIT=1_1
# LABELED_JSON="s"
# VAL_JSON="s"
# SAM_CKPT="sam_vit_b_01ec64.pth"
# CLIP_CHECKPOINT="s"
# ADE=false
# PASCAL=false
# CAMOUFLAGED=True
# USE_MAE=True

#Crops
SPLIT=1_1
LABELED_JSON="crop_jsons/crops_train.json"
VAL_JSON="crop_jsons/crops_test.json"
SAM_CKPT="sam_vit_b_01ec64.pth"
CLIP_CHECKPOINT="s"
ADE=false
PASCAL=false
CAMOUFLAGED=false
CROPS=true
USE_MAE=false

RESUME_EPOCH=20
RESUME_CKPT="./ssssupervised_best_IoU__v30_split_1_1_Supervised_Crops_IEParaTextVisMDVanilla_lr5e-05_batchsize1.pth"
# Other parameters

BATCH_SIZE=1
EPOCHS=40
LR_IE=5e-5
LR_MD=5e-5
LR_PE=5e-5
LR_CLIP=1e-7
SAVE_PATH="."
VERSION=35
LOG_DIR="./logs"
NUM_WORKERS=4


# Run the script using torchrun
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  SupervisedTrainingScript_DDP_noSLURM.py \
  --labeled_json $LABELED_JSON \
  --val_json $VAL_JSON \
  --sam_ckpt $SAM_CKPT \
  --split $SPLIT \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr_IE $LR_IE \
  --lr_MD $LR_MD \
  --save_path $SAVE_PATH \
  --version $VERSION \
  --log_dir $LOG_DIR \
  --num_workers $NUM_WORKERS \
  --skip_init_eval false \
  --points_from_gt_train false \
  --points_from_gt_eval false \
  --simple_loss false \
  --total_epochs 40 \
  --finetune_clip true \
  --use_scheduler false \
  --clip_crop_size 512 \
  --include_backgrounds false \
  --lr_clip $LR_CLIP \
  --lr_PE $LR_PE \
  --mask_prompts true \
  --prompt_ensemble false \
  --sim_func_surgery false \
  --norm_dim 1 \
  --iou_loss true \
  --structure_loss false \
  --enhanced_structure_loss false \
  --weight_decay 0.001 \
  --resume_checkpoint $RESUME_CKPT \
  --ignore_wandb false \
  --sam_cross false \
  --lr_cr 5e-5 \
  --finetune_PE true \
  --parallel_sim false \
  --finetune_neck false \
  --lr_neck 1e-5 \
  --text_vis true \
  --text_vis_cross false \
  --clip_decoder_head false \
  --lr_clip_dec_head 5e-5 \
  --out_size_512 false \
  --clip_logscale false \
  --load_clip false \
  --clip_ckpt $CLIP_CHECKPOINT \
  --ade20k $ADE \
  --pascal $PASCAL \
  --crops $CROPS \
  --camouflaged $CAMOUFLAGED \
  --use_mae $USE_MAE \
  --bce_weight 1 \
  --dice_weight 1 \
  --iou_weight 1 \
  --no_text false \
  --SAM_IE Parallel_Text_Vis \
  --clip_type CS-ViT-B/16 \
  --wandb_extension 'v2.5' \
  --use_soft_prompts false \
  --soft_prompt_sigmoid false \
  --num_gpus $NUM_GPUS \
  --local_rank 1
   

#cp supervised*.pth /workspace/