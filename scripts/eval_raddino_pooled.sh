#!/bin/bash
# Eval pipeline for rct_pooled_224_raddino (cls_only / adaLN conditioning).
# Checkpoint lives in output/main_result/ (trained earlier, step 300000).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/output/eval_logs"
mkdir -p "$LOG_DIR"

MODEL_NAME="rct_pooled_224_raddino"
MODEL_DIR="${REPO_ROOT}/output/main_result/${MODEL_NAME}"
CKPT="${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt"
CFG=1.5
EMA=ema2

echo "=========================================="
echo "Evaluating: ${MODEL_NAME}"
echo "  ckpt: $CKPT"
echo "  start: $(date)"
echo "=========================================="

MIMIC_SP="${MODEL_DIR}/fid_eval"
NIH_SP="${MODEL_DIR}/fid_eval_nih"
CX_SP="${MODEL_DIR}/fid_eval_chexpert"

echo "[1] MIMIC generation..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$MIMIC_SP" \
  --dataset mimic --cfg "$CFG" --ema "$EMA" --num_images 3403 --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_${MODEL_NAME}_mimic.log"

echo "[2] MIMIC quality + distance..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${MIMIC_SP}/real" --gen_dir "${MIMIC_SP}/gen" \
  --out "${MIMIC_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_${MODEL_NAME}_mimic.log"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${MIMIC_SP}/real" --gen_dir "${MIMIC_SP}/gen" --gpu 0 \
  --ssl_model chexworld \
  --out "${MIMIC_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_${MODEL_NAME}_mimic.log"

echo "[3] NIH generation (sharded)..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$NIH_SP" \
  --dataset nih --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  --shard 0/2 > "${LOG_DIR}/gen_${MODEL_NAME}_nih_s0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$NIH_SP" \
  --dataset nih --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  --shard 1/2 > "${LOG_DIR}/gen_${MODEL_NAME}_nih_s1.log" 2>&1 &
wait

echo "[4] NIH quality + distance..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${NIH_SP}/real" --gen_dir "${NIH_SP}/gen" \
  --out "${NIH_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_${MODEL_NAME}_nih.log"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${NIH_SP}/real" --gen_dir "${NIH_SP}/gen" --gpu 0 \
  --ssl_model chexworld \
  --out "${NIH_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_${MODEL_NAME}_nih.log"

echo "[5] CheXpert generation..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" \
  --model_type rct --ckpt_path "$CKPT" --save_path "$CX_SP" \
  --dataset chexpert --cfg "$CFG" --ema "$EMA" --num_images 30000 --batch_size 32 --gpu 0 \
  2>&1 | tee "${LOG_DIR}/gen_${MODEL_NAME}_chexpert.log"

echo "[6] CheXpert quality + distance..."
python "${SCRIPT_DIR}/quality_metrics.py" \
  --real_dir "${CX_SP}/real" --gen_dir "${CX_SP}/gen" \
  --out "${CX_SP}/metrics.txt" \
  2>&1 | tee "${LOG_DIR}/metrics_${MODEL_NAME}_chexpert.log"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" \
  --real_dir "${CX_SP}/real" --gen_dir "${CX_SP}/gen" --gpu 0 \
  --ssl_model chexworld \
  --out "${CX_SP}/distance_metrics.txt" \
  2>&1 | tee "${LOG_DIR}/distance_${MODEL_NAME}_chexpert.log"

echo "[7] Seg + Cls agreement..."
for ds_dir in "$MIMIC_SP" "$NIH_SP" "$CX_SP"; do
  if [ -d "${ds_dir}/gen" ]; then
    python "${SCRIPT_DIR}/eval_seg_agreement.py" --root "$ds_dir" --resize 224 \
      --out-name seg_agreement_r224 --n-overlays 6 \
      2>&1 | tee -a "${LOG_DIR}/agreement_${MODEL_NAME}.log"
    python "${SCRIPT_DIR}/eval_cls_agreement.py" --root "$ds_dir" --resize 224 \
      --out-name cls_agreement_r224 \
      2>&1 | tee -a "${LOG_DIR}/agreement_${MODEL_NAME}.log"
  fi
done

echo ""
echo "=== ${MODEL_NAME} DONE at $(date) ==="
echo "--- MIMIC quality ---" && cat "${MIMIC_SP}/metrics.txt"
echo "--- MIMIC distance ---" && cat "${MIMIC_SP}/distance_metrics.txt"
echo "--- NIH quality ---" && cat "${NIH_SP}/metrics.txt"
echo "--- NIH distance ---" && cat "${NIH_SP}/distance_metrics.txt"
echo "--- CheXpert quality ---" && cat "${CX_SP}/metrics.txt"
echo "--- CheXpert distance ---" && cat "${CX_SP}/distance_metrics.txt"
