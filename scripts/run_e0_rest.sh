#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY -m rlr baselines --sets dev test external --models USER-base USER-bge-m3 USER2-base FRIDA ru-en-RoSBERTa Qwen3-Emb-0.6B rubert-tiny2
$PY -m rlr mine --base intfloat/multilingual-e5-small --name e5-small
$PY -m rlr mine --base intfloat/multilingual-e5-base --name e5-base
echo "E0 pipeline done"
