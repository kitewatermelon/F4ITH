#!/bin/bash
# Cross-dataset eval for the two 224 models (spatial token, pooled cls_only) on
# NIH (test) and CheXpert (valid). MIMIC was already done separately.
#   generate (EMA2 / cfg 1.5) -> quality (FID/FRD) -> distance.
# Results under <model_dir>/fid_eval_<dataset>/.
#
# Usage: bash scripts/run_cross_dataset_eval.sh [num_images]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/output"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"
NUM_IMAGES="${1:-3403}"          # NIH caps at this; CheXpert has only ~202
mkdir -p "$LOG_DIR"

MODELS=(
  "spatial:${OUTPUT_ROOT}/rct_token_224_p14_chexworld"
  "pooled:${OUTPUT_ROOT}/rct_pooled_224_chexworld"
)
DATASETS=(nih chexpert)

# name:model_dir:dataset
CONFIGS=()
for m in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    CONFIGS+=("${m%%:*}_${ds}:${m##*:}:${ds}")
  done
done

echo "=========================================="
echo "cross-dataset eval (cfg 1.5 / ema2, n<=$NUM_IMAGES)"
echo "  configs: ${CONFIGS[*]%%:*}"
echo "=========================================="

echo ""; echo "[1/2] Generating (2 GPUs, 2 at a time)..."
gpu=0
for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name model_dir ds <<< "$entry"
  ckpt="${model_dir}/checkpoints/stepstep=0300000.ckpt"
  save_path="${model_dir}/fid_eval_${ds}"
  CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_DIR}/eval_generation.py" \
    --model_type rct --ckpt_path "$ckpt" --save_path "$save_path" \
    --dataset "$ds" --cfg 1.5 --ema ema2 --num_images "$NUM_IMAGES" --batch_size 32 --gpu 0 \
    > "${LOG_DIR}/gen_${name}.log" 2>&1 &
  echo "  ${name} -> GPU${gpu}, PID $!"
  gpu=$(((gpu + 1) % 2))
  [ "$gpu" -eq 0 ] && wait
done
wait
echo "  generation finished"

echo ""; echo "[2/2] Metrics (quality + distance)..."
for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name model_dir ds <<< "$entry"
  save_path="${model_dir}/fid_eval_${ds}"
  [ -d "${save_path}/gen" ] || continue
  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "${save_path}/real" --gen_dir "${save_path}/gen" \
    --out "${save_path}/metrics.txt" > "${LOG_DIR}/metrics_${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
    --real_dir "${save_path}/real" --gen_dir "${save_path}/gen" --gpu 0 \
    --out "${save_path}/distance_metrics.txt" > "${LOG_DIR}/distance_${name}.log" 2>&1
  echo "  ${name}: $(grep '^FID' "${save_path}/metrics.txt" 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' "${save_path}/distance_metrics.txt" 2>/dev/null)"
done

echo ""; echo "Done. Results:"
for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name model_dir ds <<< "$entry"
  sp="${model_dir}/fid_eval_${ds}"
  echo "=== ${name} ==="; cat "${sp}/metrics.txt" 2>/dev/null; echo "---"; cat "${sp}/distance_metrics.txt" 2>/dev/null
done
