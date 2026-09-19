#!/usr/bin/env bash
# Stage 2 pipeline on one Qwen3 server session: judge dev/test questions, then generate train questions.
# Resumable: every step skips items that are already in its output file.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true
sleep 2
scripts/llm_server.sh qwen 4 > logs/llm_qwen.log 2>&1 &
for i in $(seq 1 60); do curl -s http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
$PY -m rlr judge --inp data/gen/eval_raw.jsonl --out data/gen/eval_judged.jsonl --workers 4
$PY -m rlr generate --mode train --out data/gen/train_raw.jsonl --workers 4
taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true
echo "generation pipeline done"
