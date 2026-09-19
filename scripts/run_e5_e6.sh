#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY -m rlr forgetting --models e5-small=intfloat/multilingual-e5-small ft-e5-small=models/e1_small_llm_hn/best e5-base=intfloat/multilingual-e5-base ft-e5-base=models/e2_base_llm_hn/best
$PY -m rlr speed --models e5-small=intfloat/multilingual-e5-small ft-e5-small=models/e1_small_llm_hn/best e5-base=intfloat/multilingual-e5-base e5-large=intfloat/multilingual-e5-large bge-m3=BAAI/bge-m3 FRIDA=ai-forever/FRIDA Qwen3-Emb-0.6B=Qwen/Qwen3-Embedding-0.6B USER2-base=deepvk/USER2-base
echo "e5e6 done"
