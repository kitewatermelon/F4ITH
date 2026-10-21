#!/bin/bash
# Generic final eval for an RCT model dir:
#   1. generate reconstructions (EMA/cfg from args) at the model's resolution
#      (builds its own real reference set)
#   2. quality metrics (FID/FRD)
#   3. distance metrics (CheXWorld representation space, native)
#
# Usage: bash scripts/eval_rct_final.sh <model_dir> [cfg] [ema] [num_images]
#   e.g. bash scripts/eval_rct_final.sh output/rct_pooled_224_chexworld 1.5 ema2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${1:?usage: eval_rct_final.sh <model_dir> [cfg] [ema] [num_images]}"
CFG="${2:-1.5}"
EMA="${3:-ema2}"
NUM_IMAGES="${4:-3403}"

CKPT="${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt"
SAVE_PATH="${MODEL_DIR}/fid_eval"
REAL_DIR="${SAVE_PATH}/real"
GEN_DIR="${SAVE_PATH}/gen"
LOG_DIR="${REPO_ROOT}/output/eval_logs"
TAG="$(basename "$MODEL_DIR")"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "final eval: $TAG"
echo "  ckpt: $CKPT"
echo "  cfg $CFG / $EMA / heun 50 / n=$NUM_IMAGES"
echo "=========================================="

echo ""; echo "[1/3] Generating (builds real + gen)..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$SAVE_PATH" \
  --cfg "$CFG" --ema "$EMA" --num_images "$NUM_IMAGES" --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_${TAG}_final.log"

echo ""; echo "[2/3] Quality metrics..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "$REAL_DIR" --gen_dir "$GEN_DIR" --out "${SAVE_PATH}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_${TAG}_final.log"

echo ""; echo "[3/3] Distance metrics..."
python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "$REAL_DIR" --gen_dir "$GEN_DIR" --gpu 0 --out "${SAVE_PATH}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_${TAG}_final.log"

echo ""; echo "Done: $TAG"
echo "=== quality ==="; cat "${SAVE_PATH}/metrics.txt"
echo "=== distance ==="; cat "${SAVE_PATH}/distance_metrics.txt"
