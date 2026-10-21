#!/bin/bash
# CFG x EMA sweep for the t_embed_mode=token 128px ChexWorld model at step 100k.
# Grid: cfg {1, 1.5, 3, 5} x ema {ema1, ema2} = 8 configs.
# Conditioning = shared RCDM real set (128 native, 3403). Metric = FRD only.
#
# Usage: bash scripts/run_token_cfg_ema_sweep.sh [num_images] [batch_size]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_IMAGES="${1:-3403}"
BATCH_SIZE="${2:-32}"
OUTPUT_ROOT="${REPO_ROOT}/output"
CKPT="${OUTPUT_ROOT}/ablation_t_emb_token_128_chexworld/checkpoints/stepstep=0100000.ckpt"
REAL_DIR="${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/real"
SWEEP_ROOT="${OUTPUT_ROOT}/token_cfg_ema_sweep"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"
mkdir -p "$SWEEP_ROOT" "$LOG_DIR"

CFGS=(1 1.5 3 5)
EMAS=(ema1 ema2)

# build the flat config list: "cfg:ema"
CONFIGS=()
for ema in "${EMAS[@]}"; do
  for cfg in "${CFGS[@]}"; do
    CONFIGS+=("${cfg}:${ema}")
  done
done

echo "=========================================="
echo "token CFG x EMA sweep (FRD only)"
echo "  images : $NUM_IMAGES   batch: $BATCH_SIZE"
echo "  configs: ${CONFIGS[*]}"
echo "=========================================="

echo ""
echo "[1/2] Generating (2 GPUs, 2 configs at a time)..."
gpu=0
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; ema="${entry##*:}"
  tag="cfg${cfg}_${ema}"
  save_path="${SWEEP_ROOT}/${tag}"

  CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_DIR}/eval_generation.py" \
    --model_type rct --ckpt_path "$CKPT" --save_path "$save_path" \
    --real_from_dir "$REAL_DIR" --cfg "$cfg" --ema "$ema" \
    --num_images "$NUM_IMAGES" --batch_size "$BATCH_SIZE" --gpu 0 \
    > "${LOG_DIR}/gen_token_${tag}.log" 2>&1 &
  echo "  ${tag} -> GPU${gpu}, PID $!"

  gpu=$(((gpu + 1) % 2))
  [ "$gpu" -eq 0 ] && wait
done
wait
echo "  generation finished"

echo ""
echo "[2/2] Computing FRD..."
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; ema="${entry##*:}"
  tag="cfg${cfg}_${ema}"
  save_path="${SWEEP_ROOT}/${tag}"
  [ -d "${save_path}/gen" ] || continue

  python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "$REAL_DIR" --gen_dir "${save_path}/gen" \
    --metrics frd --out "${save_path}/metrics_frd.txt" \
    2>&1 | tee "${LOG_DIR}/frd_token_${tag}.log"
done

echo ""
echo "Done. FRD sweep results:"
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; ema="${entry##*:}"
  f="${SWEEP_ROOT}/cfg${cfg}_${ema}/metrics_frd.txt"
  [ -f "$f" ] && echo "  cfg=${cfg} ${ema}: $(grep '^FRD' "$f")"
done
