#!/usr/bin/env bash
# Final evaluation of v2 (after dev-only selection in run_v2_grid.py / run_v2_stage_e.py).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
SETS="dev test golden external tk_hard"
PROTO="article chunk chunk_tkfmt"
read -r FINAL S43 S44 < <($PY - <<'PYEOF'
import json
s = json.load(open("results/v2_selection.json", encoding="utf-8"))
f = s["final"]["model_dir"]
if s["final"].get("seeds"):
    seeds = s["final"]["seeds"]
elif s["stages"].get("E", {}).get("kept"):
    seeds = [r["model_dir"] for r in s["stages"]["E"]["seeds"]]
else:
    seeds = [r["model_dir"] for r in s["stages"]["D"]["runs"]]
print(f, *seeds)
PYEOF
)
echo "final=$FINAL seeds=$S43 $S44"
ev() { $PY -m rlr evaluate --sets $SETS --protocols $PROTO --path "$2" --name "$1"; }
ev ft2-e5-small "$FINAL"; ev ft2-e5-small-s43 "$S43"; ev ft2-e5-small-s44 "$S44"
ev ft-e5-small models/e1_small_llm_hn/best
ev ft-e5-small-s43 models/e1_small_llm_hn_s43/best
ev ft-e5-small-s44 models/e1_small_llm_hn_s44/best
ev ft-e5-base models/e2_base_llm_hn/best
$PY -m rlr baselines --bm25 --no-dense --sets $SETS --protocols $PROTO
$PY -m rlr baselines --sets $SETS --protocols $PROTO
$PY -m rlr bootstrap --pairs-file configs/bootstrap_pairs_v2.yaml --out bootstrap_v2.csv
$PY -m rlr forgetting --models ft2-e5-small="$FINAL"
echo "v2 eval done"
