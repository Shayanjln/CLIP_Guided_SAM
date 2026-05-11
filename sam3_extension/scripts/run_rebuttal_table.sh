#!/usr/bin/env bash
# sam3_extension/scripts/run_rebuttal_table.sh
# -----------------------------------------------------------------------------
# Runs the four rows of the rebuttal ablation back-to-back and dumps a final
# summary. Each row trains for ~30 epochs on PASCAL VOC 1/16 (~92 images) so
# total wall-clock should fit in a normal rebuttal window on a single A100/4090.
#
# Usage:
#   bash sam3_extension/scripts/run_rebuttal_table.sh /path/to/voc_1_16
# -----------------------------------------------------------------------------

set -euo pipefail

DATA_ROOT="${1:-./data/pascal_voc_1_16}"
SAVE_DIR="${SAVE_DIR:-./checkpoints_sam3_rebuttal}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$SAVE_DIR"
LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$LOG_DIR"

run_row () {
    local row="$1"
    echo "=================================================="
    echo "Running ablation $row"
    echo "=================================================="
    python -m sam3_extension.scripts.run_pascalvoc_1_16 \
        --ablation "$row" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --save_dir "$SAVE_DIR" \
        --data_root "$DATA_ROOT" \
        2>&1 | tee "$LOG_DIR/$row.log"
}

run_row row2   # regular adapters only, frozen CLIP, no semantic injection
run_row row3   # semantic adapters, frozen CLIP (no co-adaptation)
run_row row4   # semantic adapters + CLIP last 4 (full framework)
run_row row5   # row4 + train-test prompt alignment (semi-auto train)

echo
echo "DONE. Logs in $LOG_DIR"
echo "Summary:"
for row in row2 row3 row4 row5; do
    iou=$(grep "FINISHED" "$LOG_DIR/$row.log" | awk '{print $5}')
    echo "  $row : best mIoU = $iou"
done