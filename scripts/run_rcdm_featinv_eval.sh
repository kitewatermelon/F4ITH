#!/bin/bash
# Cross-dataset eval for the RCDM and FeatInv baselines, queued to start after
# the RCT cross-dataset eval finishes. Both are slow diffusion samplers, so
# every generation is sharded 0/2 + 1/2 across the two GPUs.
#
#   RCDM   : 200-step DDIM (was 1000; too slow), 128px, pooled CheXWorld feat.
#            datasets: mimic (redo @200) -> chexpert -> nih (full 25,596)
#            results in output/rcdm_chexworld_cxr/fid_eval_<ds>/ (old 1000-step
#            fid_eval/ kept for reference).
#   FeatInv: 50-step DDIM ControlNet, 256px. MIMIC already done; add chexpert, nih.
#            distance metrics use --resize 224 (CheXWorld native).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output"; LOG="${OUT}/eval_logs"
N=30000
RCDM_CKPT="${OUT}/rcdm_chexworld_cxr/ema_0.9999_300000.pt"
FEATINV_CKPT="${REPO_ROOT}/baseline/FeatInv/output/control_featinv_chexworld_fin.ckpt"
FEATINV_CFG="${REPO_ROOT}/baseline/FeatInv/code/models/cldm_featinv_chexworld.yaml"

# gate: wait for the RCT cross-dataset eval to finish
GATE="${OUT}/rct_pooled_224_chexworld/fid_eval_chexpert/distance_metrics.txt"
echo "[rf] waiting for RCT cross-dataset eval to finish (gate: $GATE)"
until [ -f "$GATE" ]; do sleep 120; done
echo "[rf] gate cleared at $(date); starting RCDM/FeatInv eval"

# shard_gen_and_metrics <name> <model_type> <ckpt> <save_path> <dataset> <batch> <dist_resize_arg> <extra...>
run_one() {
  local name="$1" mtype="$2" ckpt="$3" sp="$4" ds="$5" bs="$6" distarg="$7"; shift 7
  local extra=("$@")
  echo "[rf] === ${name}: sharded gen ==="
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type "$mtype" --ckpt_path "$ckpt" \
    --save_path "$sp" --dataset "$ds" --shard 0/2 --num_images "$N" --batch_size "$bs" --gpu 0 "${extra[@]}" \
    > "${LOG}/gen_${name}_s0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type "$mtype" --ckpt_path "$ckpt" \
    --save_path "$sp" --dataset "$ds" --shard 1/2 --num_images "$N" --batch_size "$bs" --gpu 0 "${extra[@]}" \
    > "${LOG}/gen_${name}_s1.log" 2>&1 &
  wait
  echo "[rf] ${name} gen done ($(ls ${sp}/gen 2>/dev/null | wc -l) imgs) at $(date); metrics"
  # $distarg carries the measurement resize (FeatInv: --resize 224, since it
  # generates at 256 but conditioning + FID/distance are evaluated at 224).
  # Applied to BOTH quality and distance so FID/FRD are at 224 too.
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    $distarg --out "${sp}/metrics.txt" > "${LOG}/metrics_${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --gpu 0 $distarg --out "${sp}/distance_metrics.txt" > "${LOG}/distance_${name}.log" 2>&1
  echo "[rf] ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt 2>/dev/null)"
}

# ---- RCDM (200-step DDIM, 128px) ----
# One model at 128px, so each dataset's 128 real is built exactly once (no
# cross-model duplication to share away).
for ds in mimic chexpert nih; do
  run_one "rcdm_${ds}" rcdm "$RCDM_CKPT" "${OUT}/rcdm_chexworld_cxr/fid_eval_${ds}" "$ds" 32 "" \
    --timestep_respacing ddim200
done

# ---- FeatInv (50-step DDIM ControlNet, 256px); MIMIC already done ----
for ds in chexpert nih; do
  run_one "featinv_${ds}" featinv "$FEATINV_CKPT" "${REPO_ROOT}/baseline/FeatInv/output/fid_eval_${ds}" "$ds" 16 "--resize 224" \
    --featinv_config "$FEATINV_CFG"
done

echo "[rf] ALL DONE at $(date)"
