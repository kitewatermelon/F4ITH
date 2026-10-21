#!/bin/bash
# Downstream segmentation-agreement re-measurement for the FeatInv POOLED baseline
# (cfg 8.0, control strength 1.5), matching run_featinv_pooled_eval.sh. Uses the
# already-generated pooled gen/ images.
#
# The pooled dirs hold gen/ only (real was reused via --real_from_dir from the
# cfg9 baseline), so we symlink the cfg9 real/ into each pooled root -- filenames
# (global indices) match exactly, which is what eval_seg_agreement.py asserts.
#
# Gated to start AFTER the pooled quality/distance eval finishes (shared GPU), by
# waiting on the nih distance_metrics.txt it produces last.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
FI="${REPO}/baseline/FeatInv/output"
LOG="${REPO}/output/eval_logs"; mkdir -p "$LOG"

# name : matching cfg9 real dir
declare -A REALSRC=(
  [mimic]="${FI}/fid_eval/real"
  [chexpert]="${FI}/fid_eval_chexpert/real"
  [nih]="${FI}/fid_eval_nih/real"
)
declare -A ROOT=(
  [mimic]="${FI}/fid_eval_pooled"
  [chexpert]="${FI}/fid_eval_chexpert_pooled"
  [nih]="${FI}/fid_eval_nih_pooled"
)

# gate: pooled eval must have finished nih (its last distance_metrics.txt)
GATE="${FI}/fid_eval_nih_pooled/distance_metrics.txt"
echo "[fi-pool-seg] waiting for pooled eval to finish (gate: $GATE)"
until [ -f "$GATE" ]; do sleep 120; done
echo "[fi-pool-seg] gate cleared at $(date); starting seg agreement"

for name in mimic chexpert nih; do
  root="${ROOT[$name]}"; realsrc="${REALSRC[$name]}"
  # link cfg9 real/ into pooled root as real/ (skip if already present)
  if [ ! -e "${root}/real" ]; then ln -s "$realsrc" "${root}/real"; fi
  echo "[fi-pool-seg] === ${name}: seg agreement @224 -> ${root}/seg_agreement_r224 ==="
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/eval_seg_agreement.py" \
    --root "$root" --resize 224 --out-name seg_agreement_r224 --n-overlays 6 \
    > "${LOG}/seg_featinv_${name}_pooled.log" 2>&1
  echo "[fi-pool-seg] ${name} seg: $(grep -i 'MACRO_MEAN' ${root}/seg_agreement_r224/summary.csv 2>/dev/null | head -1)"
  echo "[fi-pool-seg] === ${name}: cls agreement @224 -> ${root}/cls_agreement_r224 ==="
  CUDA_VISIBLE_DEVICES=0 python "${REPO}/scripts/eval_cls_agreement.py" \
    --root "$root" --resize 224 --out-name cls_agreement_r224 \
    > "${LOG}/cls_featinv_${name}_pooled.log" 2>&1
  echo "[fi-pool-seg] ${name} cls done"
done
echo "[fi-pool-seg] ALL DONE $(date)"
