#!/bin/bash
# Final eval for the token 224/patch14 300k model:
#   1. generate reconstructions (EMA2, cfg 1.5, Heun 50, seed-42 sampling) on the
#      MIMIC test set at 224 (builds its own 224 real reference set)
#   2. quality metrics (FID/FRD)
#   3. distance metrics (CheXWorld representation space, native 224)
#
# Usage: bash scripts/eval_token_224_final.sh [ckpt] [num_images]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${REPO_ROOT}/output/rct_token_224_p14_chexworld"
CKPT="${1:-${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt}"
NUM_IMAGES="${2:-3403}"
SAVE_PATH="${MODEL_DIR}/fid_eval"
REAL_DIR="${SAVE_PATH}/real"
GEN_DIR="${SAVE_PATH}/gen"
LOG_DIR="${REPO_ROOT}/output/eval_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "token 224/p14 300k final eval"
echo "  ckpt   : $CKPT"
echo "  cfg 1.5 / ema2 / heun 50 / 224px / n=$NUM_IMAGES"
echo "=========================================="

echo ""; echo "[1/3] Generating (builds 224 real + gen)..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$SAVE_PATH" \
  --cfg 1.5 --ema ema2 --num_images "$NUM_IMAGES" --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_token_224_final.log"

echo ""; echo "[2/3] Quality metrics (FID/FRD)..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "$REAL_DIR" --gen_dir "$GEN_DIR" \
  --out "${SAVE_PATH}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_token_224_final.log"

echo ""; echo "[3/3] Distance metrics (CheXWorld, native 224)..."
python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "$REAL_DIR" --gen_dir "$GEN_DIR" --gpu 0 \
  --out "${SAVE_PATH}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_token_224_final.log"

echo ""; echo "Done."
echo "=== quality ==="; cat "${SAVE_PATH}/metrics.txt"
echo "=== distance ==="; cat "${SAVE_PATH}/distance_metrics.txt"
