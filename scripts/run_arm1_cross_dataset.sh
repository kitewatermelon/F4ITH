#!/bin/bash
# Cross-dataset eval for the arm1 in-context 224 model (cls_only + adaLN +
# in-context, mean-pool) on NIH (test) and CheXpert (valid). MIMIC was already
# done at <model>/fid_eval/. Mirrors run_cross_dataset_eval.sh (cfg 1.5 / ema2,
# generate -> quality -> distance). Results under <model>/fid_eval_<dataset>/.
#
# Usage: bash scripts/run_arm1_cross_dataset.sh [num_images]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output"; LOG="${OUT}/eval_logs_incontext"
MODEL="${OUT}/arm1_incontext_224_p14_chexworld"
CK="${MODEL}/checkpoints/stepstep=0300000.ckpt"
N="${1:-30000}"          # NIH caps at 25,596; CheXpert has ~202
mkdir -p "$LOG"

echo "[arm1-xds] NIH: sharded generation across 2 GPUs at $(date)"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "${MODEL}/fid_eval_nih" --dataset nih --shard 0/2 --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_arm1_nih_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "${MODEL}/fid_eval_nih" --dataset nih --shard 1/2 --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_arm1_nih_s1.log" 2>&1 &
wait
echo "[arm1-xds] NIH gen done ($(ls ${MODEL}/fid_eval_nih/gen 2>/dev/null | wc -l) imgs) at $(date)"

echo "[arm1-xds] CheXpert generation"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "${MODEL}/fid_eval_chexpert" --dataset chexpert --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_arm1_chexpert.log" 2>&1
echo "[arm1-xds] all generation done at $(date); computing metrics"

for entry in "arm1_nih:nih" "arm1_chexpert:chexpert"; do
  IFS=':' read -r name ds <<< "$entry"
  sp="${MODEL}/fid_eval_${ds}"
  [ -d "${sp}/gen" ] || { echo "[arm1-xds] skip ${name} (no gen)"; continue; }
  echo "[arm1-xds] metrics: ${name}"
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --out "${sp}/metrics.txt" > "${LOG}/metrics_${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_${name}.log" 2>&1
  echo "  ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt 2>/dev/null)"
done
echo "[arm1-xds] DONE at $(date)"
