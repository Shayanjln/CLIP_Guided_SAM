#!/usr/bin/env bash
# sam3_extension/scripts/run_rebuttals_flexible.sh
# Flexible runner for SAM3 Pascal VOC ablations.
#
# Examples from inside sam3_extension:
#   bash scripts/run_rebuttals_flexible.sh row3
#   bash scripts/run_rebuttals_flexible.sh zs_interactive zs_interactive_gt ft_interactive row2 row3
#   EPOCHS=5 VAL_MAX_BATCHES=0 NUM_WORKERS=4 NOTIFY=1 bash scripts/run_rebuttals_flexible.sh row3 row4

# row2: regular adapters, CLIP-train / CLIP-eval
# row3: semantic adapters, frozen CLIP, CLIP-train / CLIP-eval
# row4: semantic adapters + CLIP co-adaptation, CLIP-train / CLIP-eval
# row5: same as row4, but GT-train / CLIP-eval

set -euo pipefail

TRAIN_JSON="${TRAIN_JSON:-../../SAM_CLIP_Script_Training/Pascal_samples_1_16_labeled.json}"
VAL_JSON="${VAL_JSON:-../../SAM_CLIP_Script_Training/Pascal_samples_val.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXT_DIR/.." && pwd)"

RAW_SAVE_DIR="${SAVE_DIR:-$EXT_DIR/checkpoints_sam3_rebuttal}"
# If SAVE_DIR is relative, resolve it relative to sam3_extension, not repo root.
if [[ "$RAW_SAVE_DIR" != /* ]]; then
  RAW_SAVE_DIR="$EXT_DIR/$RAW_SAVE_DIR"
fi

SAVE_DIR="$(mkdir -p "$RAW_SAVE_DIR" && cd "$RAW_SAVE_DIR" && pwd)"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
LR_ADAPTERS="${LR_ADAPTERS:-5e-5}"
LR_INTERACTIVE_PROMPT="${LR_INTERACTIVE_PROMPT:-5e-5}"
LR_INTERACTIVE_MASK="${LR_INTERACTIVE_MASK:-5e-5}"
LR_CLIP="${LR_CLIP:-1e-7}"
NUM_POINTS="${NUM_POINTS:-5}"
IMG_SIZE="${IMG_SIZE:-1024}"
OUT_SIZE="${OUT_SIZE:-256}"
SAM3_IMAGE_SIZE="${SAM3_IMAGE_SIZE:-1008}"
CLIP_CROP_SIZE="${CLIP_CROP_SIZE:-512}"
SEED="${SEED:-42}"
NOTIFY="${NOTIFY:-0}"
RUN_TAG="${RUN_TAG:-}"
EVAL_EVERY="${EVAL_EVERY:-1}"

if [[ "$#" -gt 0 ]]; then
  ABLATIONS=("$@")
else
  ABLATIONS=(row2 row3 row4 row5)
fi

mkdir -p "$SAVE_DIR"
MASTER_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="$SAVE_DIR/batch_logs"
mkdir -p "$MASTER_LOG_DIR"
MASTER_LOG="$MASTER_LOG_DIR/batch_${MASTER_TS}.log"

{
  echo "=================================================="
  echo "SAM3 Pascal VOC ablation batch"
  echo "Started: $(date)"
  echo "Ablations: ${ABLATIONS[*]}"
  echo "TRAIN_JSON=$TRAIN_JSON"
  echo "VAL_JSON=$VAL_JSON"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "EPOCHS=$EPOCHS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "VAL_MAX_BATCHES=$VAL_MAX_BATCHES"
  echo "LR_ADAPTERS=$LR_ADAPTERS"
  echo "LR_INTERACTIVE_PROMPT=$LR_INTERACTIVE_PROMPT"
  echo "LR_INTERACTIVE_MASK=$LR_INTERACTIVE_MASK"
  echo "LR_CLIP=$LR_CLIP"
  echo "RUN_TAG=$RUN_TAG"
  echo "EVAL_EVERY=$EVAL_EVERY"
  echo "=================================================="
} | tee -a "$MASTER_LOG"

run_one () {
  local ablation="$1"
  local ts="$(date +%Y%m%d_%H%M%S)"
  local val_label="val${VAL_MAX_BATCHES}"
  [[ "$VAL_MAX_BATCHES" == "0" ]] && val_label="fullval"

  local run_name
  if [[ -n "$RUN_TAG" ]]; then
    run_name="${ts}_${RUN_TAG}_${ablation}_ep${EPOCHS}_bs${BATCH_SIZE}_${val_label}"
  else
    run_name="${ts}_${ablation}_ep${EPOCHS}_bs${BATCH_SIZE}_${val_label}"
  fi

  {
    echo ""
    echo "=================================================="
    echo "Running ablation: $ablation"
    echo "Run name: $run_name"
    echo "Started: $(date)"
    echo "=================================================="
  } | tee -a "$MASTER_LOG"

  local notify_flag=()
  if [[ "$NOTIFY" == "1" ]]; then
    notify_flag=(--notify)
  fi

  local cmd=(
    python -m scripts.run_pascalvoc_1_16
    --ablation "$ablation"
    --train_json "$TRAIN_JSON"
    --val_json "$VAL_JSON"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --save_dir "$SAVE_DIR"
    --val_max_batches "$VAL_MAX_BATCHES"
    --lr_adapters "$LR_ADAPTERS"
    --lr_interactive_prompt "$LR_INTERACTIVE_PROMPT"
    --lr_interactive_mask "$LR_INTERACTIVE_MASK"
    --lr_clip "$LR_CLIP"
    --num_points "$NUM_POINTS"
    --img_size "$IMG_SIZE"
    --out_size "$OUT_SIZE"
    --sam3_image_size "$SAM3_IMAGE_SIZE"
    --clip_crop_size "$CLIP_CROP_SIZE"
    --seed "$SEED"
    --run_name "$run_name"
    --eval_every "$EVAL_EVERY"
    "${notify_flag[@]}"
  )

  echo "[command] PYTHONPATH=..:. ${cmd[*]}" | tee -a "$MASTER_LOG"
  local start="$(date +%s)"

  if PYTHONPATH=..:. "${cmd[@]}"; then
    local status="SUCCESS"
  else
    local status="FAILED"
  fi

  local end="$(date +%s)"
  local elapsed=$((end - start))
  echo "Finished ablation: $ablation | status=$status | elapsed=${elapsed}s" | tee -a "$MASTER_LOG"

  local result_json="$SAVE_DIR/runs/$run_name/result.json"
  if [[ -f "$result_json" ]]; then
    echo "[result_json] $result_json" | tee -a "$MASTER_LOG"
    python - <<PY | tee -a "$MASTER_LOG"
import json
path = "$result_json"
with open(path) as f:
    r = json.load(f)
print(json.dumps({
    "ablation": r.get("ablation"),
    "miou": r.get("miou"),
    "best_miou": r.get("best_miou"),
    "loss": r.get("loss"),
    "elapsed_hhmmss": r.get("elapsed_hhmmss"),
    "best_path": r.get("best_path"),
    "status": r.get("status", "ok"),
}, indent=2))
PY
  else
    echo "[warn] result_json not found: $result_json" | tee -a "$MASTER_LOG"
  fi

  [[ "$status" == "SUCCESS" ]]
}

for ablation in "${ABLATIONS[@]}"; do
  run_one "$ablation"
done

{
  echo ""
  echo "=================================================="
  echo "DONE. Finished: $(date)"
  echo "Master log: $MASTER_LOG"
  echo "Summary:"
} | tee -a "$MASTER_LOG"

for ablation in "${ABLATIONS[@]}"; do
  latest_result="$(ls -t "$SAVE_DIR"/runs/*"_${ablation}_"*/result.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_result" ]]; then
    python - <<PY | tee -a "$MASTER_LOG"
import json
path = "$latest_result"
with open(path) as f:
    r = json.load(f)
score = r.get("best_miou", r.get("miou", None))
print(f"  {r.get('ablation')}: score={score} elapsed={r.get('elapsed_hhmmss')} result={path}")
PY
  else
    echo "  $ablation: no result found" | tee -a "$MASTER_LOG"
  fi
done

if [[ "$NOTIFY" == "1" ]]; then
  printf '\a'
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    command -v notify-send >/dev/null 2>&1 && \
      notify-send "SAM3 batch finished" "Ablations: ${ABLATIONS[*]}" >/dev/null 2>&1 || true
  fi
fi
