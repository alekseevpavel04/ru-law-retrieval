#!/usr/bin/env bash
# E3: extra seeds for the best small and base configurations (chosen by dev in E1/E2).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for cfg in configs/train/e1_small_llm_hn.yaml configs/train/e2_base_llm_hn.yaml; do
  for seed in 43 44; do
    echo "=== $cfg seed $seed $(date +%H:%M:%S)"
    $PY -m rlr train "$cfg" --seed $seed 2>&1 | grep -E "dev best|Traceback|Error"
  done
done
$PY -m rlr analysis train
echo "grid done"
