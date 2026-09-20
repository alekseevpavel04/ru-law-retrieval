#!/usr/bin/env bash
# Final evaluation: every number in README and results/ is produced here.
#
# The recipe was chosen on dev alone (results/recipe_selection.json); this script is where the chosen
# checkpoint first meets test / tk_hard / golden. Each step is idempotent, so it can be re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
SETS="dev test golden external tk_hard"
PROTO="article chunk chunk_tkfmt"
ev() { $PY -m rlr evaluate --sets $SETS --protocols $PROTO --path "$2" --name "$1"; }

# --- published checkpoint (seed 43, best dev of three) and its seed repeats -------------------
ev e5-small-ru-law     models/v2f_distill_from_v2e_ep3_s43/best
ev e5-small-ru-law-s42 models/v2f_distill_from_v2e_ep3/best
ev e5-small-ru-law-s44 models/v2f_distill_from_v2e_ep3_s44/best

# --- ablations: data sources, hard negatives, model size, learning curve, seed repeats --------
for r in e1_small_titles_inb e1_small_titles_hn e1_small_llm_inb e1_small_llm_hn e1_small_both_inb \
         e1_small_both_hn e1_small_llm_hn_s43 e1_small_llm_hn_s44 \
         e2_base_llm_hn e2_base_both_hn e2_base_llm_hn_s43 e2_base_llm_hn_s44 \
         e4_small_llm_hn_f25 e4_small_llm_hn_f50; do
  ev "abl-$r" "models/$r/best"
done

# --- baselines: BM25 and 12 off-the-shelf embedders -------------------------------------------
$PY -m rlr baselines --bm25 --no-dense --sets $SETS --protocols $PROTO
$PY -m rlr baselines --sets $SETS --protocols $PROTO

# --- hybrid with BM25: equal-weight RRF, and RRF with the BM25 weight tuned on dev ------------
for m in FRIDA e5-small-ru-law; do
  $PY -m rlr baselines --hybrid BM25 "$m" --sets dev test --protocols chunk
  $PY -m rlr baselines --hybrid BM25 "$m" --tune-weight --sets dev test --protocols chunk
done

# --- significance, general-domain retrieval, speed and size -----------------------------------
$PY -m rlr bootstrap --pairs-file configs/bootstrap_pairs.yaml
$PY -m rlr forgetting --models e5-small=intfloat/multilingual-e5-small \
                               e5-small-ru-law=models/v2f_distill_from_v2e_ep3_s43/best
$PY -m rlr speed --models e5-small=intfloat/multilingual-e5-small \
                          e5-small-ru-law=models/v2f_distill_from_v2e_ep3_s43/best \
                          e5-base=intfloat/multilingual-e5-base USER2-base=deepvk/USER2-base \
                          bge-m3=BAAI/bge-m3 e5-large=intfloat/multilingual-e5-large \
                          FRIDA=ai-forever/FRIDA Qwen3-Emb-0.6B=Qwen/Qwen3-Embedding-0.6B

# --- tables and figures -----------------------------------------------------------------------
MODELS="FRIDA e5-large-instruct e5-large bge-m3 USER-bge-m3 Qwen3-Emb-0.6B ru-en-RoSBERTa \
        USER2-base e5-base e5-small USER-base BM25 rubert-tiny2"
FOCUS="BM25 e5-small e5-small-ru-law e5-large FRIDA"
$PY -m rlr analysis report --models $MODELS --finetuned e5-small-ru-law --focus $FOCUS
$PY -m rlr analysis hero
$PY -m rlr analysis train
$PY -m rlr analysis lexical
$PY -m rlr analysis learning --full-run e1_small_llm_hn \
    --fraction-runs e4_small_llm_hn_f25 e4_small_llm_hn_f50 --refs e5-base e5-large
$PY scripts/headline.py
$PY scripts/readme_tables.py
echo "evaluation done"
