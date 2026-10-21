#!/bin/bash
# Finish the cross-dataset eval after the original runner was stopped:
#   - spatial_nih gen is already (nearly) done; spatial_chexpert gen done.
#   - pooled_nih: generate SHARDED across both GPUs (2x faster than 1-GPU).
#   - pooled_chexpert: generate (tiny).
#   - then quality + distance metrics for ALL 4 configs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output"; LOG="${OUT}/eval_logs"
SPATIAL="${OUT}/rct_token_224_p14_chexworld"; POOLED="${OUT}/rct_pooled_224_chexworld"
CK_S="${SPATIAL}/checkpoints/stepstep=0300000.ckpt"; CK_P="${POOLED}/checkpoints/stepstep=0300000.ckpt"
N=30000

echo "[remain] waiting for spatial_nih gen to finish..."
until [ "$(ls ${SPATIAL}/fid_eval_nih/gen 2>/dev/null | wc -l)" -ge 25596 ]; do sleep 30; done
echo "[remain] spatial_nih gen done at $(date)"

echo "[remain] pooled_nih: sharded generation across 2 GPUs"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK_P" \
  --save_path "${POOLED}/fid_eval_nih" --dataset nih --shard 0/2 --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_pooled_nih_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK_P" \
  --save_path "${POOLED}/fid_eval_nih" --dataset nih --shard 1/2 --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_pooled_nih_s1.log" 2>&1 &
wait
echo "[remain] pooled_nih gen done ($(ls ${POOLED}/fid_eval_nih/gen | wc -l) imgs) at $(date)"

echo "[remain] pooled_chexpert generation"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK_P" \
  --save_path "${POOLED}/fid_eval_chexpert" --dataset chexpert --cfg 1.5 --ema ema2 \
  --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_pooled_chexpert.log" 2>&1
echo "[remain] all generation done at $(date); computing metrics"

# name : model_dir : dataset
for entry in \
  "spatial_nih:${SPATIAL}:nih" "spatial_chexpert:${SPATIAL}:chexpert" \
  "pooled_nih:${POOLED}:nih"  "pooled_chexpert:${POOLED}:chexpert"; do
  IFS=':' read -r name mdir ds <<< "$entry"
  sp="${mdir}/fid_eval_${ds}"
  [ -d "${sp}/gen" ] || { echo "[remain] skip ${name} (no gen)"; continue; }
  echo "[remain] metrics: ${name}"
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --out "${sp}/metrics.txt" > "${LOG}/metrics_${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_${name}.log" 2>&1
  echo "  ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt 2>/dev/null)"
done
echo "[remain] DONE at $(date)"
