#!/bin/bash
# Sequential runner for the fixed-token-count resolution ablation
# (224/patch14, 128/patch8, 256/patch16 -- all num_patches=256).
# Each arm uses --num_gpus -1 (both GPUs via DDP), so the three runs cannot
# be parallelized on this machine -- they run one after another.
# Usage: bash scripts/run_res_ablation.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/output/ablation_logs"
mkdir -p "$LOG_DIR"

for spec in \
  "224_p14:scripts/ablation_res_224_p14.sh" \
  "128_p8:scripts/ablation_res_128_p8.sh"
do
  name="${spec%%:*}"
  script="${spec#*:}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name ($script)"
  bash "${REPO_ROOT}/${script}" "$@" > "${LOG_DIR}/res_${name}.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $name"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name (exit $status)"
  fi
done
