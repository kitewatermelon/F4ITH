#!/bin/bash
# FID-only evaluation for the three t_embed_mode 128px ChexWorld ablations
# (token / adaln / adaln_single) at step 100k.
#
# Reuses the already-built RCDM real reference set as the reconstruction
# conditioning source, so only <save_path>/gen/ is generated. FID is computed
# against that same shared real dir.
#
# Usage: bash scripts/run_t_emb_fid.sh [num_images] [batch_size]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_IMAGES="${1:-3403}"
BATCH_SIZE="${2:-32}"
CFG="${3:-1.5}"
OUTPUT_ROOT="${REPO_ROOT}/output"
REAL_DIR="${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/real"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"
CKPT_STEP="stepstep=0100000.ckpt"

mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  token:ablation_t_emb_token_128_chexworld
  adaln:ablation_t_emb_adaln_128_chexworld
  adaln_single:ablation_t_emb_adaln_single_128_chexworld
)

echo "=========================================="
echo "t_embed_mode ablation FID evaluation"
echo "  images    : $NUM_IMAGES"
echo "  batch size: $BATCH_SIZE"
echo "  cfg       : $CFG"
echo "  real dir  : $REAL_DIR"
echo "=========================================="

echo ""
echo "[1/2] Generating gen/ (2 GPUs, one experiment at a time)..."
gpu=0
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry##*:}"
  ckpt="${OUTPUT_ROOT}/${dir}/checkpoints/${CKPT_STEP}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"

  if [ ! -f "$ckpt" ]; then
    echo "  skipping ${name}: checkpoint not found at ${ckpt}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_DIR}/eval_generation.py" \
    --model_type rct --ckpt_path "$ckpt" --save_path "$save_path" \
    --real_from_dir "$REAL_DIR" --cfg "$CFG" \
    --num_images "$NUM_IMAGES" --batch_size "$BATCH_SIZE" --gpu 0 \
    > "${LOG_DIR}/gen_t_emb_${name}.log" 2>&1 &
  echo "  ${name} -> GPU${gpu}, PID $!"

  gpu=$(((gpu + 1) % 2))
  if [ "$gpu" -eq 0 ]; then
    wait   # 2 experiments in flight at a time
  fi
done
wait
echo "  generation finished"

echo ""
echo "[2/2] Computing metrics (FID, FRD)..."
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry##*:}"
  save_path="${OUTPUT_ROOT}/${dir}/fid_eval"
  [ -d "${save_path}/gen" ] || continue

  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "$REAL_DIR" --gen_dir "${save_path}/gen" \
    --out "${save_path}/metrics.txt" \
    2>&1 | tee "${LOG_DIR}/metrics_t_emb_${name}.log"
done

echo ""
echo "Done. Results:"
for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry##*:}"
  f="${OUTPUT_ROOT}/${dir}/fid_eval/metrics.txt"
  echo "=== ${name} ==="
  [ -f "$f" ] && cat "$f"
done
