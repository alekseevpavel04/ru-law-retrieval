#!/usr/bin/env bash
# Ablation training runs (E1, E2, E4) and the extra seeds of their winners. Finished runs are skipped.
#
# E1  e5-small x {article titles, LLM questions, both} x {in-batch, + hard negative}
# E2  the two best data setups on e5-base
# E4  learning curve: 25% and 50% of the LLM questions
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
mkdir -p logs

train() {  # full output to a log, only the dev numbers on screen, a failure is loud
  local log="logs/train_$1${2:+_s$2}.log"
  echo "=== $1 ${2:+seed $2} $(date +%H:%M:%S)"
  if ! $PY -m rlr train "configs/train/$1.yaml" ${2:+--seed "$2"} > "$log" 2>&1; then
    echo "FAILED: $1 ${2:+seed $2}, see $log"; return 1
  fi
  grep -E "dev base|dev best" "$log" || true
}
for cfg in e1_small_titles_inb e1_small_titles_hn e1_small_llm_inb e1_small_llm_hn \
           e1_small_both_inb e1_small_both_hn e2_base_llm_hn e2_base_both_hn \
           e4_small_llm_hn_f25 e4_small_llm_hn_f50; do
  train "$cfg"
done
for cfg in e1_small_llm_hn e2_base_llm_hn; do
  for seed in 43 44; do train "$cfg" "$seed"; done
done
$PY -m rlr analysis train
echo "ablations done"
