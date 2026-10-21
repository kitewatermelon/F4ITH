#!/bin/bash
# Downstream segmentation-agreement re-measurement for the FeatInv baseline at the
# ORIGINAL-AUTHOR setting (cfg 1.75, control strength 1.0), to match the cfg1.75
# quality/distance re-eval. Uses the already-generated cfg1.75 gen/ images.
#
# The cfg1.75 dirs hold gen/ only (real was reused via --real_from_dir from the
# cfg9 baseline), so we symlink the cfg9 real/ into each cfg1.75 root -- filenames
# (global indices) match exactly, which is what eval_seg_agreement.py asserts.
#
# Gated to start AFTER the cfg1.75 quality/distance re-eval finishes (shared GPU),
# by waiting on the nih distance_metrics.txt it produces last.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
FI="${REPO}/baseline/FeatInv/output"
LOG="${REPO}/output/eval_logs"; mkdir -p "$LOG"

# name : cfg1.75 gen root : matching cfg9 real dir
declare -A REALSRC=(
  [mimic]="${FI}/fid_eval/real"
  [chexpert]="${FI}/fid_eval_chexpert/real"
  [nih]="${FI}/fid_eval_nih/real"
)
declare -A ROOT=(
  [mimic]="${FI}/fid_eval_cfg1.75"
  [chexpert]="${FI}/fid_eval_chexpert_cfg1.75"
  [nih]="${FI}/fid_eval_nih_cfg1.75"
)

# gate: cfg1.75 re-eval must have finished nih (its last distance_metrics.txt)
GATE="${FI}/fid_eval_nih_cfg1.75/distance_metrics.txt"
echo "[fi175-seg] waiting for cfg1.75 re-eval to finish (gate: $GATE)"
until [ -f "$GATE" ]; do sleep 120; done
echo "[fi175-seg] gate cleared at $(date); starting seg agreement"

for name in mimic chexpert nih; do
  root="${ROOT[$name]}"; realsrc="${REALSRC[$name]}"
  # link cfg9 real/ into cfg1.75 root as real/ (skip if already present)
  if [ ! -e "${root}/real" ]; then ln -s "$realsrc" "${root}/real"; fi
  echo "[fi175-seg] === ${name}: seg agreement @224 -> ${root}/seg_agreement_r224 ==="
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/eval_seg_agreement.py" \
    --root "$root" --resize 224 --out-name seg_agreement_r224 \
    > "${LOG}/seg_featinv_${name}_cfg175.log" 2>&1
  echo "[fi175-seg] ${name}: $(grep -i 'dice' ${root}/seg_agreement_r224/summary.csv 2>/dev/null | head -1)"
done
echo "[fi175-seg] ALL DONE $(date)"
