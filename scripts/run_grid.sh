#!/usr/bin/env bash
# Run a list of training configs sequentially (finished runs are skipped), then summarize.
# Usage: scripts/run_grid.sh configs/train/e1_*.yaml
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for cfg in "$@"; do
  echo "=== $cfg $(date +%H:%M:%S)"
  $PY -m rlr train "$cfg" 2>&1 | grep -v -i "warn\|symlink\|Loading weights\|widget" | grep -E "^\{|dev base|dev best|Traceback|Error|skipping|\"name\"|seconds|peak" | grep -v "'loss'"
done
$PY -m rlr analysis train
echo "grid done"
