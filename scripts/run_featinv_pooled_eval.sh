#!/bin/bash
# Evaluate the FeatInv POOLED CheXWorld baseline at the paper's pooled setting:
#   guidance scale (CFG) = 8.0, control strength = 1.5, default prompt,
#   50-step DDIM, 256px gen, metrics @224.
# This mirrors run_featinv_cfg1.75_reeval.sh (the unpooled re-eval) but uses the
# pooled ckpt + --featinv_pool so the ControlNet hint is the broadcast mean-pooled
# descriptor. Conditioning is REUSED from the same cfg9 real/ dirs via
# --real_from_dir, so pooled vs. unpooled is an exact A/B (identical inputs).
# New outputs go to *_pooled/ and do not touch existing dirs. Datasets need not
# be mounted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
FI="${REPO}/baseline/FeatInv"
CKPT="${FI}/output/control_featinv_chexworld_pooled_fin.ckpt"
CFG_YAML="${FI}/code/models/cldm_featinv_chexworld.yaml"
LOG="${REPO}/output/eval_logs"; mkdir -p "$LOG"
CFGSCALE=8.0
CTRL=1.5

# name : source real dir (cfg9 baseline)
run_ds() {
  local name="$1" realdir="$2"
  local sp="${FI}/output/fid_eval_${name}_pooled"
  [ "$name" = "mimic" ] && sp="${FI}/output/fid_eval_pooled"
  mkdir -p "$sp"
  echo "[fi-pool] === ${name}: gen (cfg ${CFGSCALE}, ctrl ${CTRL}) -> ${sp}/gen ==="
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/eval_generation.py" --model_type featinv \
    --ckpt_path "$CKPT" --featinv_config "$CFG_YAML" --real_from_dir "$realdir" \
    --save_path "$sp" --cfg_scale "$CFGSCALE" --featinv_pool --featinv_control_strength "$CTRL" \
    --num_images 30000 --batch_size 16 --gpu 0 --shard 0/2 \
    > "${LOG}/gen_featinv_${name}_pooled_s0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "${REPO}/scripts/eval_generation.py" --model_type featinv \
    --ckpt_path "$CKPT" --featinv_config "$CFG_YAML" --real_from_dir "$realdir" \
    --save_path "$sp" --cfg_scale "$CFGSCALE" --featinv_pool --featinv_control_strength "$CTRL" \
    --num_images 30000 --batch_size 16 --gpu 0 --shard 1/2 \
    > "${LOG}/gen_featinv_${name}_pooled_s1.log" 2>&1 &
  wait
  local n=$(ls "${sp}/gen" 2>/dev/null | wc -l)
  echo "[fi-pool] ${name} gen done (${n} imgs) at $(date); metrics @224"
  python "${REPO}/scripts/quality_metrics.py" --real_dir "$realdir" --gen_dir "${sp}/gen" \
    --resize 224 --out "${sp}/metrics.txt" > "${LOG}/metrics_featinv_${name}_pooled.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/distance_metrics.py" --real_dir "$realdir" --gen_dir "${sp}/gen" \
    --gpu 0 --resize 224 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_featinv_${name}_pooled.log" 2>&1
  echo "[fi-pool] ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep -i cos ${sp}/distance_metrics.txt 2>/dev/null | head -1)"
}

echo "[fi-pool] START $(date)"
run_ds mimic    "${FI}/output/fid_eval/real"
run_ds chexpert "${FI}/output/fid_eval_chexpert/real"
run_ds nih      "${FI}/output/fid_eval_nih/real"
echo "[fi-pool] ALL DONE $(date)"
