#!/bin/bash
# Compute FID/FRD for all CFG sweep gen dirs (images already generated).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG="${REPO_ROOT}/output/logs/frd_sweep"
mkdir -p "$LOG"

run_metrics() {
  local label="$1" real="$2" gen="$3" out="$4"
  if [ -f "$out" ]; then
    echo "[frd] ${label} already done — skipping"
    return
  fi
  [ -d "$gen" ] || { echo "[frd] ${label} no gen dir — skipping"; return; }
  echo "[frd] ${label} at $(date)"
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "$real" --gen_dir "$gen" \
    --out "$out" > "${LOG}/${label}.log" 2>&1
  echo "  $(grep '^FID\|^FRD' "$out" 2>/dev/null | tr '\n' ' ')"
}

for model in pooled token; do
  if [ "$model" = "pooled" ]; then dir="${REPO_ROOT}/output/main_result/rct_pooled_224_chexworld_v2"; fi
  if [ "$model" = "token" ]; then dir="${REPO_ROOT}/output/main_result/rct_token_224_p14_chexworld_v2"; fi

  for ds in mimic nih chexpert; do
    if [ "$ds" = "mimic" ]; then
      sweep="${dir}/cfg_sweep"
      real="${dir}/fid_eval/real"
    else
      sweep="${dir}/cfg_sweep_${ds}"
      real="${dir}/fid_eval_${ds}/real"
    fi

    for cfg in 1 2 2.5 3; do
      sp="${sweep}/cfg${cfg}_ema2"
      run_metrics "${model}_${ds}_cfg${cfg}" "$real" "${sp}/gen" "${sp}/metrics.txt"
    done
  done
done

# Summary
echo ""
echo "=========================================="
echo "FRD/FID sweep summary"
echo "=========================================="
for model in pooled token; do
  if [ "$model" = "pooled" ]; then dir="${REPO_ROOT}/output/main_result/rct_pooled_224_chexworld_v2"; fi
  if [ "$model" = "token" ]; then dir="${REPO_ROOT}/output/main_result/rct_token_224_p14_chexworld_v2"; fi
  for ds in mimic nih chexpert; do
    echo "--- ${model} / ${ds} ---"
    printf "%-6s %12s %12s\n" cfg FID FRD
    for cfg in 1 1.5 2 2.5 3; do
      if [ "$cfg" = "1.5" ]; then
        if [ "$ds" = "mimic" ]; then m="${dir}/fid_eval/metrics.txt"
        else m="${dir}/fid_eval_${ds}/metrics.txt"; fi
      else
        if [ "$ds" = "mimic" ]; then m="${dir}/cfg_sweep/cfg${cfg}_ema2/metrics.txt"
        else m="${dir}/cfg_sweep_${ds}/cfg${cfg}_ema2/metrics.txt"; fi
      fi
      fid=$(grep '^FID' "$m" 2>/dev/null | awk '{print $3}')
      frd=$(grep '^FRD' "$m" 2>/dev/null | awk '{print $3}')
      printf "%-6s %12s %12s\n" "$cfg" "${fid:-(pending)}" "${frd:-(pending)}"
    done
  done
done
echo "[frd] ALL DONE at $(date)"
