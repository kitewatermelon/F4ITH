#!/bin/bash
# Full reconstruction-quality evaluation across the RCT ChexWorld ablations
# (exp1-exp5; exp6 has no checkpoint -- OOM'd during training).
#
#   1. generate — scripts/eval_generation.py, one experiment per GPU at a time
#   2. metrics  — scripts/quality_metrics.py (FID, FRD)
#
# Usage: bash scripts/run_eval_parallel.sh [num_images] [batch_size]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_IMAGES="${1:-3403}"
BATCH_SIZE="${2:-32}"
OUTPUT_ROOT="${REPO_ROOT}/output"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"

mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  exp1_baseline:ablation_exp1_baseline_chexworld
  exp2_no_conv:ablation_exp2_no_conv_chexworld
  exp3_no_skip:ablation_exp3_no_skip_chexworld
  exp4_no_conv_no_skip:ablation_exp4_no_conv_no_skip_chexworld
  exp5_h512_no_conv:ablation_exp5_h512_no_conv_chexworld
)
CKPT_STEP="stepstep=0100000.ckpt"

echo "=========================================="
echo "RCT ablation reconstruction evaluation"
echo "  images    : $NUM_IMAGES"
echo "  batch size: $BATCH_SIZE"
echo "=========================================="

echo ""
echo "[1/2] Generating (2 GPUs, one experiment at a time)..."
gpu=0
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry##*:}"
  ckpt="${OUTPUT_ROOT}/${dir}/checkpoints/${CKPT_STEP}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"

  if [ ! -f "$ckpt" ]; then
    echo "  ⚠️  skipping ${name}: checkpoint not found at ${ckpt}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_DIR}/eval_generation.py" \
    --model_type rct --ckpt_path "$ckpt" --save_path "$save_path" \
    --num_images "$NUM_IMAGES" --batch_size "$BATCH_SIZE" --gpu 0 \
    > "${LOG_DIR}/gen_${name}.log" 2>&1 &
  echo "  ${name} -> GPU${gpu}, PID $!"

  gpu=$(((gpu + 1) % 2))
  if [ "$gpu" -eq 0 ]; then
    wait   # 2 experiments in flight at a time
  fi
done
wait
echo "  ✓ generation finished"

echo ""
echo "[2/2] Computing metrics..."
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry##*:}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"
  [ -d "${save_path}/gen" ] || continue

  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "${save_path}/real" --gen_dir "${save_path}/gen" \
    --out "${save_path}/metrics.txt" \
    2>&1 | tee "${LOG_DIR}/metrics_${name}.log"
done

echo ""
echo "✅ Done. Results under ${OUTPUT_ROOT}/<experiment>/fid_eval/metrics.txt"
