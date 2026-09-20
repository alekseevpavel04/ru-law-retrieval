#!/usr/bin/env bash
# Build every dataset file from the raw generations, then annotate the training pairs with the
# teacher. Run after both generation passes (scripts/run_generation.sh) are complete.
# Every step is resumable: finished items are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
mkdir -p logs
server() {  # (re)start llama.cpp with a given model and wait for it to load
  taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true; sleep 2
  scripts/llm_server.sh "$1" "$2" > "logs/llm_$1_dataset.log" 2>&1 &
  for _ in $(seq 1 60); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  echo "llama-server did not come up"; exit 1
}

# 1. dev / test dataset from the first generation pass
$PY -m rlr build-dataset

# 2. harder TK test set: YandexGPT generates, Qwen3 judges
server yandex 3
$PY -m rlr generate --mode tkhard --out data/gen/tkhard_raw.jsonl --workers 3
server qwen 4
$PY -m rlr judge --inp data/gen/tkhard_raw.jsonl --out data/gen/tkhard_judged.jsonl --workers 4
taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true

# 3. tk_hard (deduped against every training question), then the training set of both passes
$PY -m rlr build-dataset --tk-hard
$PY -m rlr build-dataset --train-raw train_raw.jsonl train_raw_v2.jsonl --train-out train_llm_v12
$PY -m pytest -q   # leakage tests run on the real files

# 4. hard negatives mined by the base models (for the ablations) and by the teacher (for the recipe)
$PY -m rlr mine --base intfloat/multilingual-e5-small --name e5-small
$PY -m rlr mine --base intfloat/multilingual-e5-base --name e5-base
$PY -m rlr teacher --train train_llm_v12
$PY -m rlr export-mteb
echo "dataset done"
