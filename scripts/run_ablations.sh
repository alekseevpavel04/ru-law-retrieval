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

train() {
  echo "=== $1 ${2:+seed $2} $(date +%H:%M:%S)"
  $PY -m rlr train "configs/train/$1.yaml" ${2:+--seed "$2"} 2>&1 | grep -E "dev base|dev best|Traceback|Error"
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
