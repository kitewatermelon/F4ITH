#!/bin/bash
# Re-evaluate the FeatInv (unpooled) baseline at the ORIGINAL-AUTHOR setting:
#   guidance scale (CFG) = 1.75, control strength = 1.0 (default), default prompt,
#   50-step DDIM, 256px gen, metrics @224.
# The prior baseline used CFG 9.0 (eval_generation.py default), which the FeatInv
# paper (TMLR 12/2025, Supp. A.2 / Figs 9-11) says is a pooled-only regime that
# distorts unpooled reconstructions. Unpooled recommended: CFG 1.75, control 1.0.
#
# Conditioning is REUSED from the cfg9 real/ dirs via --real_from_dir, so this is
# an exact A/B (identical inputs; only CFG differs). Datasets need not be mounted.
# cfg9 results are preserved; new outputs go to *_cfg1.75/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
FI="${REPO}/baseline/FeatInv"
CKPT="${FI}/output/control_featinv_chexworld_fin.ckpt"
CFG_YAML="${FI}/code/models/cldm_featinv_chexworld.yaml"
LOG="${REPO}/output/eval_logs"; mkdir -p "$LOG"
CFGSCALE=1.75

# name : source real dir (cfg9 baseline)
run_ds() {
  local name="$1" realdir="$2"
  local sp="${FI}/output/fid_eval_${name}_cfg1.75"
  [ "$name" = "mimic" ] && sp="${FI}/output/fid_eval_cfg1.75"
  mkdir -p "$sp"
  echo "[fi175] === ${name}: gen (cfg ${CFGSCALE}) -> ${sp}/gen ==="
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/eval_generation.py" --model_type featinv \
    --ckpt_path "$CKPT" --featinv_config "$CFG_YAML" --real_from_dir "$realdir" \
    --save_path "$sp" --cfg_scale "$CFGSCALE" --num_images 30000 --batch_size 16 \
    --gpu 0 --shard 0/2 > "${LOG}/gen_featinv_${name}_cfg175_s0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "${REPO}/scripts/eval_generation.py" --model_type featinv \
    --ckpt_path "$CKPT" --featinv_config "$CFG_YAML" --real_from_dir "$realdir" \
    --save_path "$sp" --cfg_scale "$CFGSCALE" --num_images 30000 --batch_size 16 \
    --gpu 0 --shard 1/2 > "${LOG}/gen_featinv_${name}_cfg175_s1.log" 2>&1 &
  wait
  local n=$(ls "${sp}/gen" 2>/dev/null | wc -l)
  echo "[fi175] ${name} gen done (${n} imgs) at $(date); metrics @224"
  python "${REPO}/scripts/quality_metrics.py" --real_dir "$realdir" --gen_dir "${sp}/gen" \
    --resize 224 --out "${sp}/metrics.txt" > "${LOG}/metrics_featinv_${name}_cfg175.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/distance_metrics.py" --real_dir "$realdir" --gen_dir "${sp}/gen" \
    --gpu 0 --resize 224 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_featinv_${name}_cfg175.log" 2>&1
  echo "[fi175] ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep -i cos ${sp}/distance_metrics.txt 2>/dev/null | head -1)"
}

echo "[fi175] START $(date)"
run_ds mimic    "${FI}/output/fid_eval/real"
run_ds chexpert "${FI}/output/fid_eval_chexpert/real"
run_ds nih      "${FI}/output/fid_eval_nih/real"
echo "[fi175] ALL DONE $(date)"
