#!/usr/bin/env bash
# v2 data + training pipeline (resumable steps). Run after data/gen/train_raw_v2.jsonl is complete.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
server() {  # (re)start llama.cpp with a given model
  taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true; sleep 2
  scripts/llm_server.sh "$1" "$2" > "logs/llm_$1_v2pipe.log" 2>&1 &
  for i in $(seq 1 60); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done; echo "server failed"; exit 1
}
# 1. harder TK test set: YandexGPT generates, Qwen3 judges
server yandex 3
$PY -m rlr generate --mode tkhard --out data/gen/tkhard_raw.jsonl --workers 3
server qwen 4
$PY -m rlr judge --inp data/gen/tkhard_raw.jsonl --out data/gen/tkhard_judged.jsonl --workers 4
taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true
# 2. datasets: tk_hard (deduped against all train questions), then train v1+v2 (deduped against dev/test/tk_hard)
$PY -m rlr build-dataset --tk-hard
$PY -m rlr build-dataset --train-raw train_raw.jsonl train_raw_v2.jsonl --train-out train_llm_v12
$PY -m pytest -q
# 3. teacher annotations (FRIDA)
$PY -m rlr teacher --train train_llm_v12
echo "v2 data done"
