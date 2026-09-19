#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
SETS="dev test golden external tk_hard"; PROTO="article chunk chunk_tkfmt"
ev() { $PY -m rlr evaluate --sets $SETS --protocols $PROTO --path "$2" --name "$1"; }
ev ft2-e5-small-s44 models/v2f_distill_from_v2e_ep3_s44/best
ev ft-e5-small models/e1_small_llm_hn/best
ev ft-e5-small-s43 models/e1_small_llm_hn_s43/best
ev ft-e5-small-s44 models/e1_small_llm_hn_s44/best
ev ft-e5-base models/e2_base_llm_hn/best
$PY -m rlr baselines --bm25 --no-dense --sets $SETS --protocols $PROTO
$PY -m rlr baselines --sets $SETS --protocols $PROTO
$PY -m rlr bootstrap --pairs-file configs/bootstrap_pairs_v2.yaml --out bootstrap_v2.csv
$PY -m rlr forgetting --models ft2-e5-small=models/v2f_distill_from_v2e_ep3/best
echo "v2 eval done"
