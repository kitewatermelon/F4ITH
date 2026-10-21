#!/bin/bash
# Sequential runner for the 128px t_embed_mode ablation (adaln / token / adaln_single).
# Each arm uses --num_gpus -1 (both GPUs via DDP), so the three runs cannot
# be parallelized on this machine -- they run one after another.
# Usage: bash scripts/run_t_emb_ablation_128.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/output/ablation_logs"
mkdir -p "$LOG_DIR"

for spec in \
  "adaln:scripts/ablation_t_emb_adaln_128.sh" \
  "token:scripts/ablation_t_emb_token_128.sh" \
  "adaln_single:scripts/ablation_t_emb_adaln_single_128.sh"
do
  name="${spec%%:*}"
  script="${spec#*:}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name ($script)"
  bash "${REPO_ROOT}/${script}" > "${LOG_DIR}/t_emb_${name}_128.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $name"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name (exit $status)"
  fi
done
