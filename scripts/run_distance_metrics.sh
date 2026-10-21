#!/bin/bash
# CheXWorld representation-space distance metrics (scripts/distance_metrics.py)
# across the t_embed_mode ablations, RCDM, and FeatInv.
#
# The RCT ablations were generated with --real_from_dir pointing at the shared
# RCDM real set, so they have no own real/ dir -- we pass that shared real dir
# as --real_dir. FeatInv resizes both sets to 224 for the encoder.
#
# Usage: bash scripts/run_distance_metrics.sh [gpu]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU="${1:-1}"

OUTPUT_ROOT="${REPO_ROOT}/output"
SHARED_REAL="${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/real"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"
mkdir -p "$LOG_DIR"

# name : real_dir : gen_dir : resize(0=native)
SETS=(
  "token:${SHARED_REAL}:${OUTPUT_ROOT}/ablation_t_emb_token_128_chexworld/fid_eval/gen:0"
  "adaln:${SHARED_REAL}:${OUTPUT_ROOT}/ablation_t_emb_adaln_128_chexworld/fid_eval/gen:0"
  "adaln_single:${SHARED_REAL}:${OUTPUT_ROOT}/ablation_t_emb_adaln_single_128_chexworld/fid_eval/gen:0"
  "rcdm:${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/real:${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/gen:0"
  "featinv:${REPO_ROOT}/baseline/FeatInv/output/fid_eval/real:${REPO_ROOT}/baseline/FeatInv/output/fid_eval/gen:224"
)

for entry in "${SETS[@]}"; do
  IFS=':' read -r name real_dir gen_dir resize <<< "$entry"
  if [ ! -d "$gen_dir" ]; then
    echo "  skipping ${name}: gen dir not found ($gen_dir)"; continue
  fi
  resize_arg=""
  [ "$resize" != "0" ] && resize_arg="--resize ${resize}"
  out="$(dirname "$gen_dir")/distance_metrics.txt"

  echo "=========================================="
  echo "[${name}] real=${real_dir}"
  echo "         gen=${gen_dir} ${resize_arg}"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES="$GPU" python "${SCRIPT_DIR}/distance_metrics.py" \
    --real_dir "$real_dir" --gen_dir "$gen_dir" $resize_arg --gpu 0 --out "$out" \
    2>&1 | tee "${LOG_DIR}/distance_${name}.log"
  echo ""
done

echo "Done. Results:"
for entry in "${SETS[@]}"; do
  IFS=':' read -r name real_dir gen_dir resize <<< "$entry"
  f="$(dirname "$gen_dir")/distance_metrics.txt"
  echo "=== ${name} ==="; [ -f "$f" ] && cat "$f"
done
