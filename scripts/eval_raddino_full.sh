#!/bin/bash
# Full eval pipeline for rct_token_224_p14_raddino model:
#   1. MIMIC: generation + quality + distance + seg/cls agreement
#   2. NIH: cross-dataset generation + quality + distance + seg/cls agreement
#   3. CheXpert: cross-dataset generation + quality + distance + seg/cls agreement
#   4. FeatInv MIMIC quality metrics (missing)
#
# Usage: bash scripts/eval_raddino_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${REPO_ROOT}/output/rct_token_224_p14_raddino"
CKPT="${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt"
LOG_DIR="${REPO_ROOT}/output/eval_logs"
mkdir -p "$LOG_DIR"

CFG=1.5
EMA=ema2

echo "=========================================="
echo "Full eval: rct_token_224_p14_raddino"
echo "  ckpt: $CKPT"
echo "  cfg=$CFG ema=$EMA heun50"
echo "  start: $(date)"
echo "=========================================="

# ---- 0. FeatInv MIMIC quality metrics (missing) ----
FEATINV_MIMIC="${REPO_ROOT}/baseline/FeatInv/output/fid_eval"
if [ ! -f "${FEATINV_MIMIC}/metrics.txt" ] && [ -d "${FEATINV_MIMIC}/real" ] && [ -d "${FEATINV_MIMIC}/gen" ]; then
  echo ""; echo "[0/7] FeatInv MIMIC quality metrics (was missing)..."
  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "${FEATINV_MIMIC}/real" --gen_dir "${FEATINV_MIMIC}/gen" \
    --out "${FEATINV_MIMIC}/metrics.txt" \
    2>&1 | tee "${LOG_DIR}/metrics_featinv_mimic.log"
fi

# ---- 1. MIMIC generation ----
MIMIC_SP="${MODEL_DIR}/fid_eval"
echo ""; echo "[1/7] MIMIC generation (3403 images)..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$MIMIC_SP" \
  --dataset mimic --cfg "$CFG" --ema "$EMA" --num_images 3403 --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_raddino_mimic.log"

# ---- 2. MIMIC quality + distance ----
echo ""; echo "[2/7] MIMIC quality metrics..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${MIMIC_SP}/real" --gen_dir "${MIMIC_SP}/gen" \
  --out "${MIMIC_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_raddino_mimic.log"

echo ""; echo "[2b/7] MIMIC distance metrics..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${MIMIC_SP}/real" --gen_dir "${MIMIC_SP}/gen" --gpu 0 \
  --ssl_model raddino \
  --out "${MIMIC_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_raddino_mimic.log"

# ---- 3. NIH cross-dataset generation (sharded across 2 GPUs) ----
NIH_SP="${MODEL_DIR}/fid_eval_nih"
echo ""; echo "[3/7] NIH cross-dataset generation (sharded, 2 GPUs)..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$NIH_SP" \
  --dataset nih --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  --shard 0/2 \
  > "${LOG_DIR}/gen_raddino_nih_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$NIH_SP" \
  --dataset nih --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  --shard 1/2 \
  > "${LOG_DIR}/gen_raddino_nih_s1.log" 2>&1 &
wait
echo "NIH gen done: $(ls ${NIH_SP}/gen 2>/dev/null | wc -l) images"

# ---- 4. NIH quality + distance ----
echo ""; echo "[4/7] NIH quality + distance metrics..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${NIH_SP}/real" --gen_dir "${NIH_SP}/gen" \
  --out "${NIH_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_raddino_nih.log"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${NIH_SP}/real" --gen_dir "${NIH_SP}/gen" --gpu 0 \
  --ssl_model raddino \
  --out "${NIH_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_raddino_nih.log"

# ---- 5. CheXpert cross-dataset generation ----
CX_SP="${MODEL_DIR}/fid_eval_chexpert"
echo ""; echo "[5/7] CheXpert cross-dataset generation..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$CX_SP" \
  --dataset chexpert --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_raddino_chexpert.log"

# ---- 6. CheXpert quality + distance ----
echo ""; echo "[6/7] CheXpert quality + distance metrics..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${CX_SP}/real" --gen_dir "${CX_SP}/gen" \
  --out "${CX_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_raddino_chexpert.log"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${CX_SP}/real" --gen_dir "${CX_SP}/gen" --gpu 0 \
  --ssl_model raddino \
  --out "${CX_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_raddino_chexpert.log"

# ---- 7. Seg + Cls agreement (all datasets) ----
echo ""; echo "[7/7] Seg + Cls agreement for all datasets..."
for ds_dir in "$MIMIC_SP" "$NIH_SP" "$CX_SP"; do
  if [ -d "${ds_dir}/gen" ]; then
    echo "  seg/cls agreement: $ds_dir"
    python "${SCRIPT_DIR}/eval_seg_agreement.py" --root "$ds_dir" --resize 224 \
      --out-name seg_agreement_r224 --n-overlays 6 \
      2>&1 | tee -a "${LOG_DIR}/agreement_raddino.log"
    python "${SCRIPT_DIR}/eval_cls_agreement.py" --root "$ds_dir" --resize 224 \
      --out-name cls_agreement_r224 \
      2>&1 | tee -a "${LOG_DIR}/agreement_raddino.log"
  fi
done

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="

# Print summary
echo ""
echo "=== MIMIC Quality ==="
cat "${MIMIC_SP}/metrics.txt"
echo "=== MIMIC Distance ==="
cat "${MIMIC_SP}/distance_metrics.txt"
echo "=== NIH Quality ==="
cat "${NIH_SP}/metrics.txt"
echo "=== NIH Distance ==="
cat "${NIH_SP}/distance_metrics.txt"
echo "=== CheXpert Quality ==="
cat "${CX_SP}/metrics.txt"
echo "=== CheXpert Distance ==="
cat "${CX_SP}/distance_metrics.txt"
