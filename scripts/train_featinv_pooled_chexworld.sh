#!/bin/bash
# Train the FeatInv CheXWorld POOLED variant (Table-1 baseline).
# Identical setup to the unpooled run (same config, miniSD init, lr 1e-5, bf16,
# 96,660 steps = 3 epochs, 2 GPUs) -- only FEATINV_POOL=1 changes the ControlNet
# hint from spatial patch tokens to the broadcast mean-pooled descriptor.
# Writes output/control_featinv_chexworld_pooled_fin.ckpt (does NOT touch the
# unpooled fin ckpt). The unpooled run took ~1d18h; budget similarly.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../baseline/FeatInv/code"

export FEATINV_POOL=1
export MIMIC_DIR="${MIMIC_DIR:-/mnt/nvme1/mimic-cxr-jpg}"
export NIH_DIR="${NIH_DIR:-/mnt/nvme1/nih-chest-x-ray-14}"
export CHEXPERT_DIR="${CHEXPERT_DIR:-/mnt/nvme1/chexpert}"
export FEATINV_DEVICES="${FEATINV_DEVICES:-2}"
export FEATINV_BATCH_SIZE="${FEATINV_BATCH_SIZE:-8}"
# all other hyperparameters inherit tutorial_train_featinv.py defaults (= unpooled)

# preflight: datasets must be mounted (raw CXR are read during training)
for d in "$MIMIC_DIR" "$NIH_DIR" "$CHEXPERT_DIR"; do
  [ -d "$d" ] || { echo "[ERROR] dataset dir missing: $d (mount /mnt/nvme1 first)"; exit 1; }
done
[ -f ../output/control_featinv_chexworld_init.ckpt ] || {
  echo "[ERROR] miniSD-merged init ckpt missing: baseline/FeatInv/output/control_featinv_chexworld_init.ckpt"; exit 1; }

echo "[featinv-pooled] START $(date)"
python tutorial_train_featinv.py
echo "[featinv-pooled] DONE $(date) -> baseline/FeatInv/output/control_featinv_chexworld_pooled_fin.ckpt"
