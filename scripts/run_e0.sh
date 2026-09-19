#!/usr/bin/env bash
# After generation: build dataset, leak tests, BM25 + all dense baselines (E0) on dev/test/external,
# hard negative mining for e5-small and e5-base. Every step is resumable / cached.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
until grep -q "generation pipeline done" logs/gen_pipeline.log 2>/dev/null; do sleep 30; done
taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true
$PY -m rlr build-dataset
$PY -m pytest -q
$PY -m rlr baselines --bm25 --no-dense --sets dev test external
$PY -m rlr baselines --sets dev test external
$PY -m rlr mine --base intfloat/multilingual-e5-small --name e5-small
$PY -m rlr mine --base intfloat/multilingual-e5-base --name e5-base
echo "E0 pipeline done"
