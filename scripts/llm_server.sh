#!/usr/bin/env bash
# Start llama.cpp server with one of the generator models (Git Bash on Windows).
# Usage: scripts/llm_server.sh qwen|yandex [parallel_slots]
set -euo pipefail
LLAMA="${LLAMA_DIR:-/d/VScode_projects/llama.cpp/b11052}/llama-server.exe"
MODELS="${LLM_MODELS_DIR:-D:/VScode_projects/llm-models}"
SLOTS="${2:-3}"
CTX_PER_SLOT=3072
case "$1" in
  qwen)   MODEL="$MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"; ALIAS=Qwen3-8B-Q4_K_M ;;
  yandex) MODEL="$MODELS/yandexgpt5-lite-8b/YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"; ALIAS=YandexGPT-5-Lite-8B-instruct-Q4_K_M ;;
  *) echo "unknown model $1"; exit 1 ;;
esac
exec "$LLAMA" -m "$MODEL" --alias "$ALIAS" --host 127.0.0.1 --port 8080 \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np "$SLOTS" -c $((SLOTS * CTX_PER_SLOT)) --jinja
