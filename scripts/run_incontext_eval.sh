#!/bin/bash
# Reconstruction-quality evaluation across the in-context conditioning ablation
# (arm1 adaLN+in-context, arm2 cls_cross, arm3 adaLN-single+in-context).
#
#   1. generate      — scripts/eval_generation.py, two arms per GPU round at a time
#   2. quality       — scripts/quality_metrics.py  (FID, FRD)
#   3. distance      — scripts/distance_metrics.py (CheXWorld representation distance)
#
# Usage: bash scripts/run_incontext_eval.sh [num_images] [batch_size]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_IMAGES="${1:-3403}"
BATCH_SIZE="${2:-32}"
OUTPUT_ROOT="${REPO_ROOT}/output"
LOG_DIR="${OUTPUT_ROOT}/eval_logs_incontext"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  arm1_adaln:ablation_incontext_arm1_adaln
  arm2_cls_cross:ablation_incontext_arm2_cls_cross
  arm3_adaln_single:ablation_incontext_arm3_adaln_single
)
CKPT_STEP="stepstep=0100000.ckpt"

echo "=========================================="
echo "In-context ablation reconstruction evaluation"
echo "  images    : $NUM_IMAGES"
echo "  batch size: $BATCH_SIZE"
echo "=========================================="

echo ""
echo "[1/3] Generating (2 GPUs, two arms at a time)..."
gpu=0
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"; dir="${entry##*:}"
  ckpt="${OUTPUT_ROOT}/${dir}/checkpoints/${CKPT_STEP}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"
  if [ ! -f "$ckpt" ]; then
    echo "  ⚠️  skipping ${name}: checkpoint not found at ${ckpt}"; continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_DIR}/eval_generation.py" \
    --model_type rct --ckpt_path "$ckpt" --save_path "$save_path" \
    --num_images "$NUM_IMAGES" --batch_size "$BATCH_SIZE" --gpu 0 \
    > "${LOG_DIR}/gen_${name}.log" 2>&1 &
  echo "  ${name} -> GPU${gpu}, PID $!"
  gpu=$(((gpu + 1) % 2))
  [ "$gpu" -eq 0 ] && wait   # two arms in flight at a time
done
wait
echo "  ✓ generation finished"

echo ""
echo "[2/3] Computing quality metrics (FID/FRD)..."
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"; dir="${entry##*:}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"
  [ -d "${save_path}/gen" ] || continue
  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "${save_path}/real" --gen_dir "${save_path}/gen" \
    --out "${save_path}/metrics.txt" \
    2>&1 | tee "${LOG_DIR}/metrics_${name}.log"
done

echo ""
echo "[3/3] Computing CheXWorld distance metrics..."
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"; dir="${entry##*:}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"
  [ -d "${save_path}/gen" ] || continue
  python "${SCRIPT_DIR}/distance_metrics.py" \
    --real_dir "${save_path}/real" --gen_dir "${save_path}/gen" \
    --out "${save_path}/distance_metrics.txt" --gpu 0 \
    2>&1 | tee "${LOG_DIR}/distance_${name}.log"
done

echo ""
echo "✅ Done. Results under ${OUTPUT_ROOT}/ablation_incontext_arm*/fid_eval/{metrics,distance_metrics}.txt"
