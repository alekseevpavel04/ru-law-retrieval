#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q '"E"' results/v2_selection.json 2>/dev/null; do sleep 30; done
(cd scripts && PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe run_v2_stage_f.py) || exit 1
scripts/run_v2_eval.sh
