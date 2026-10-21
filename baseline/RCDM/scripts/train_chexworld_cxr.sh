#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# guided_diffusion_rcdm is not pip-installed; make it importable for torchrun's workers.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8"

torchrun --nproc_per_node=2 "${SCRIPT_DIR}/train_chexworld_cxr.py" \
  ${MODEL_FLAGS} ${TRAIN_FLAGS} \
  --feat_cond --type_model chexworld \
  --out_dir "${REPO_ROOT}/../../output/rcdm_chexworld_cxr" \
  "$@"
