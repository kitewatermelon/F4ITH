#!/bin/bash
# Re-run arm1 (in-context, 224/p14/300k) cross-dataset eval at cfg 2.5 / ema2 to
# match the MIMIC CFG sweep pick. Reuses the already-built NIH/CheXpert real sets
# as conditioning (--real_from_dir). MIMIC@2.5 already exists under cfg_sweep/.
# Results in <model>/fid_eval_<ds>_cfg2.5/ (cfg1.5 dirs left intact).
#
# Usage: bash scripts/run_arm1_cross_dataset_cfg25.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output"; LOG="${OUT}/logs/cross_arm1_cfg25"
MODEL="${OUT}/arm1_incontext_224_p14_chexworld"
CK="${MODEL}/checkpoints/stepstep=0300000.ckpt"
CFG=2.5
mkdir -p "$LOG"

echo "[xds2.5] NIH: sharded generation across 2 GPUs at $(date)"
NIH_REAL="${MODEL}/fid_eval_nih/real"; NIH_SP="${MODEL}/fid_eval_nih_cfg2.5"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "$NIH_SP" --real_from_dir "$NIH_REAL" --shard 0/2 --cfg "$CFG" --ema ema2 \
  --num_images 30000 --batch_size 32 --gpu 0 > "${LOG}/gen_nih_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "$NIH_SP" --real_from_dir "$NIH_REAL" --shard 1/2 --cfg "$CFG" --ema ema2 \
  --num_images 30000 --batch_size 32 --gpu 0 > "${LOG}/gen_nih_s1.log" 2>&1 &
wait
echo "[xds2.5] NIH gen done ($(ls ${NIH_SP}/gen 2>/dev/null | wc -l) imgs) at $(date)"

echo "[xds2.5] CheXpert generation"
CX_REAL="${MODEL}/fid_eval_chexpert/real"; CX_SP="${MODEL}/fid_eval_chexpert_cfg2.5"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
  --save_path "$CX_SP" --real_from_dir "$CX_REAL" --cfg "$CFG" --ema ema2 \
  --num_images 30000 --batch_size 32 --gpu 0 > "${LOG}/gen_chexpert.log" 2>&1
echo "[xds2.5] all gen done at $(date); computing metrics"

for entry in "nih:${NIH_SP}" "chexpert:${CX_SP}"; do
  IFS=':' read -r ds sp <<< "$entry"
  [ -d "${sp}/gen" ] || { echo "[xds2.5] skip ${ds} (no gen)"; continue; }
  real="${MODEL}/fid_eval_${ds}/real"
  echo "[xds2.5] metrics: ${ds}"
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "$real" --gen_dir "${sp}/gen" \
    --out "${sp}/metrics.txt" > "${LOG}/metrics_${ds}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "$real" --gen_dir "${sp}/gen" \
    --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_${ds}.log" 2>&1
  echo "  ${ds}: $(grep '^FID' ${sp}/metrics.txt) | $(grep '^CosSim_mean' ${sp}/distance_metrics.txt) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt)"
done
echo "[xds2.5] DONE at $(date)"
