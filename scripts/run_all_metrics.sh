#!/usr/bin/env bash
# Run seg + cls agreement for all models x datasets, all resized to 224.
set -euo pipefail
cd /home/yspark/RCT

declare -A MODELS=(
  [rct_token]="output/main_result/rct_token_224_p14_chexworld"
  [rct_pooled]="output/rct_pooled_224_chexworld"
  [rcdm]="output/main_result/rcdm_chexworld_cxr"
  [featinv]="baseline/FeatInv/output"
)
# small datasets first for fast comparison, NIH last (heavy)
DATASETS=(fid_eval fid_eval_chexpert fid_eval_nih)
ORDER=(rct_token rct_pooled rcdm featinv)

run_one() {
  local root="$1" name="$2"
  echo "########## $name  ($root) ##########"
  python scripts/eval_seg_agreement.py --root "$root" --resize 224 \
      --out-name seg_agreement_r224 --n-overlays 6
  python scripts/eval_cls_agreement.py --root "$root" --resize 224 \
      --out-name cls_agreement_r224
}

for ds in "${DATASETS[@]}"; do
  for m in "${ORDER[@]}"; do
    d="${MODELS[$m]}/$ds"
    [ -d "$d/gen" ] || { echo "skip missing $d"; continue; }
    run_one "$d" "$m/$ds"
  done
done
echo "ALL_DONE"
