#!/bin/bash
# Resolution ablation, arm 1/3: img_size=224, patch_size=14.
# num_patches = (224/14)^2 = 256 -- deliberately matched across all three
# arms (128/patch8, 224/patch14, 256/patch16) so sequence length (and thus
# transformer compute/memory) is held fixed and only image resolution varies.
# t_embed_mode=adaln (baseline), hidden_size=512, depth=13, long_skip ON.
#
# batch_size=96 (not 128): a quick GPU memory probe at 256 tokens suggests
# batch_size=128/GPU may be tight against 24GB cards (the same config at
# 196 tokens/patch16 already uses ~19.2/24.5GB). Override with `-- --batch_size 128`
# if you've confirmed it fits.
# Usage: bash scripts/ablation_res_224_p14.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/ablation_res_224_p14_chexworld" \
  --model_variant      S_16 \
  --hidden_size        512 \
  --num_heads          8 \
  --img_size           224 \
  --patch_size         14 \
  --batch_size         96 \
  --num_workers        12 \
  --num_gpus           -1 \
  --lr                 5e-5 \
  --weight_decay       0.0 \
  --max_steps          100000 \
  --warmup_steps       10000 \
  --cond_drop_prob     0.1 \
  --cfg                1.5 \
  --ode_steps          50 \
  --ode_method         heun \
  --interval_min       0.0 \
  --interval_max       1.0 \
  --save_every         10000 \
  --sample_every       10000 \
  --num_samples        8 \
  --ema_decay1         0.9999 \
  --ema_decay2         0.9996 \
  --P_mean             -0.8 \
  --P_std              0.8 \
  --noise_scale        1.0 \
  --t_eps              0.05 \
  --log_every          50 \
  --conditioning_mode  patch_only \
  --backbone           chexworld \
  --t_embed_mode       adaln \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      ablation-res-token256-chexworld \
  --wandb_run_name     "res_224_p14" \
  --compile \
  "$@"
