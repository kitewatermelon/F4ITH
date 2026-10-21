#!/bin/bash
# CFG sweep for the arm1 in-context model (cls_only + adaLN-Zero + JiT in-context,
# mean-pool, 224px/p14, 300k). Grid: cfg {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5} @
# ema2 only. Conditioning = the same 3403 MIMIC-test real set already cached at
# <model>/fid_eval/real. Each config generates (sharded across 2 GPUs) then gets
# both quality (FID/FRD) and distance/retrieval metrics.
#
# cfg=1.5/ema2 is skipped: it is exactly the existing <model>/fid_eval result and
# is folded into the final report.
#
# Usage: bash scripts/run_arm1_cfg_sweep.sh [num_images]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output"
MODEL="${OUT}/arm1_incontext_224_p14_chexworld"
CK="${MODEL}/checkpoints/stepstep=0300000.ckpt"
REAL="${MODEL}/fid_eval/real"
SWEEP="${MODEL}/cfg_sweep"
LOG="${OUT}/logs/cfg_sweep_arm1"
N="${1:-3403}"
mkdir -p "$SWEEP" "$LOG"

CFGS=(1 2 2.5 3 3.5 4 4.5 5)   # 1.5 reused from fid_eval

echo "=========================================="
echo "arm1 CFG sweep (ema2, n=${N})"
echo "  ckpt : $CK"
echo "  real : $REAL"
echo "  cfgs : ${CFGS[*]}  (+1.5 from fid_eval)"
echo "=========================================="

for cfg in "${CFGS[@]}"; do
  tag="cfg${cfg}_ema2"
  sp="${SWEEP}/${tag}"
  if [ -f "${sp}/metrics.txt" ] && [ -f "${sp}/distance_metrics.txt" ]; then
    echo "[sweep] ${tag} already done — skipping"
    continue
  fi
  echo ""
  echo "[sweep] ${tag}: sharded generation across 2 GPUs at $(date)"
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
    --save_path "$sp" --real_from_dir "$REAL" --shard 0/2 --cfg "$cfg" --ema ema2 \
    --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_${tag}_s0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CK" \
    --save_path "$sp" --real_from_dir "$REAL" --shard 1/2 --cfg "$cfg" --ema ema2 \
    --num_images "$N" --batch_size 32 --gpu 0 > "${LOG}/gen_${tag}_s1.log" 2>&1 &
  wait
  ng=$(ls "${sp}/gen" 2>/dev/null | wc -l)
  echo "[sweep] ${tag}: gen done (${ng} imgs) at $(date); computing metrics"
  [ "$ng" -eq 0 ] && { echo "[sweep] ${tag}: no gen, skipping metrics"; continue; }

  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "$REAL" --gen_dir "${sp}/gen" \
    --out "${sp}/metrics.txt" > "${LOG}/metrics_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "$REAL" --gen_dir "${sp}/gen" \
    --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_${tag}.log" 2>&1
  echo "  ${tag}: $(grep '^FID' ${sp}/metrics.txt) | $(grep '^CosSim_mean' ${sp}/distance_metrics.txt) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt)"
done

echo ""
echo "=========================================="
echo "arm1 CFG sweep — summary (ema2)"
echo "=========================================="
printf "%-6s %10s %10s %10s %14s\n" cfg FID FRD CosSim Rank_g2c_cos
for cfg in 1 1.5 2 2.5 3 3.5 4 4.5 5; do
  if [ "$cfg" = "1.5" ]; then m="${MODEL}/fid_eval/metrics.txt"; d="${MODEL}/fid_eval/distance_metrics.txt";
  else m="${SWEEP}/cfg${cfg}_ema2/metrics.txt"; d="${SWEEP}/cfg${cfg}_ema2/distance_metrics.txt"; fi
  [ -f "$m" ] || { printf "%-6s %10s\n" "$cfg" "(pending)"; continue; }
  fid=$(grep '^FID' "$m" | awk '{print $3}')
  frd=$(grep '^FRD' "$m" | awk '{print $3}')
  cos=$(grep '^CosSim_mean' "$d" | awk '{print $3}')
  rk=$(grep '^Rank_gen2cond_cosine' "$d" | awk '{print $3}')
  printf "%-6s %10s %10s %10s %14s\n" "$cfg" "$fid" "$frd" "$cos" "$rk"
done | tee "${SWEEP}/summary.txt"
echo ""
echo "[sweep] DONE at $(date) -> ${SWEEP}/summary.txt"
