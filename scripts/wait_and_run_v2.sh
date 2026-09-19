#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "3636/3636" logs/gen_train_v2.log; do sleep 30; done
scripts/run_v2_pipeline.sh
