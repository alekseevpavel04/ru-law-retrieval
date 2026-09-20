#!/usr/bin/env bash
# All question generation, on one local llama.cpp server at a time.
#   1. dev / test questions  - YandexGPT-5-Lite-8B
#   2. LLM judge for them    - Qwen3-8B (a different family, so it does not agree with itself)
#   3. training questions    - Qwen3-8B, two passes with different prompts
# Resumable: every step skips items that are already in its output file.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
mkdir -p logs data/gen
server() {  # (re)start llama.cpp with a given model and wait for it to load
  taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true; sleep 2
  scripts/llm_server.sh "$1" "$2" > "logs/llm_$1.log" 2>&1 &
  for _ in $(seq 1 60); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  echo "llama-server did not come up"; exit 1
}

server yandex 3
$PY -m rlr generate --mode eval --out data/gen/eval_raw.jsonl --workers 3

server qwen 4
$PY -m rlr judge --inp data/gen/eval_raw.jsonl --out data/gen/eval_judged.jsonl --workers 4
$PY -m rlr generate --mode train --out data/gen/train_raw.jsonl --workers 4
$PY -m rlr generate --mode train --prompt-version v2 --seed 1042 --out data/gen/train_raw_v2.jsonl --workers 4

taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true
echo "generation done"
