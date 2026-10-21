#!/bin/bash
# CrossAttnJiT-B/16 on MIMIC-CXR, conditioned on RAD-DINO via per-block
# cross-attention (dense) + adaLN (pooled).
# Hyperparameters are a starting point carried over from the S/16 run, not
# tuned for the larger model.
# Usage: bash scripts/train_rct_b16.sh [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir       /mnt/nvme1/mimic-cxr-jpg \
  --output_dir     "${REPO_ROOT}/output/rct_b16_raddino_cxr" \
  --model_variant  B_16 \
  --img_size       224 \
  --batch_size     128 \
  --num_workers    12 \
  --num_gpus       -1 \
  --lr             5e-5 \
  --weight_decay   0.0 \
  --max_steps      300000 \
  --warmup_steps   10000 \
  --cond_drop_prob 0.1 \
  --cfg            1.5 \
  --ode_steps      50 \
  --ode_method     heun \
  --interval_min   0.0 \
  --interval_max   1.0 \
  --save_every     10000 \
  --sample_every   10000 \
  --num_samples    8 \
  --ema_decay1     0.9999 \
  --ema_decay2     0.9996 \
  --P_mean         -0.8 \
  --P_std          0.8 \
  --noise_scale    1.0 \
  --t_eps          0.05 \
  --log_every      50 \
  --wandb_project  rct-b16-raddino-cxr \
  --compile \
  "$@"
