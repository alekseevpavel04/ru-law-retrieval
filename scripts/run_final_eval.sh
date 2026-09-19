#!/usr/bin/env bash
# Final evaluation after all configurations are fixed on dev (E1-E3). Resumable per step.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
SETS="dev test golden external"
ev() { $PY -m rlr evaluate --sets $SETS --path "$2" --name "$1"; }
# fine-tuned models: chosen configs, 3 seeds each
ev ft-e5-small     models/e1_small_llm_hn/best
ev ft-e5-small-s43 models/e1_small_llm_hn_s43/best
ev ft-e5-small-s44 models/e1_small_llm_hn_s44/best
ev ft-e5-base      models/e2_base_llm_hn/best
ev ft-e5-base-s43  models/e2_base_llm_hn_s43/best
ev ft-e5-base-s44  models/e2_base_llm_hn_s44/best
# ablation checkpoints (E1, E2, E4) on dev+test for the ablation table
for r in e1_small_titles_inb e1_small_titles_hn e1_small_llm_inb e1_small_both_inb e1_small_both_hn e2_base_both_hn e4_small_llm_hn_f25 e4_small_llm_hn_f50; do
  $PY -m rlr evaluate --sets dev test --path models/$r/best --name "abl-$r"
done
# baselines on golden (corpus embeddings are cached)
$PY -m rlr baselines --bm25 --no-dense --sets dev test golden external
$PY -m rlr baselines --sets dev test golden external
# hybrids
$PY -m rlr baselines --sets dev test golden external --hybrid BM25 FRIDA
$PY -m rlr baselines --sets dev test golden external --hybrid BM25 ft-e5-small
$PY -m rlr bootstrap --pairs-file configs/bootstrap_pairs.yaml
echo "final eval done"
