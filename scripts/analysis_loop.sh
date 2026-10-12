#!/usr/bin/env bash
# Validation sweep that follows training: re-run (cached) every 10 min while the training queue
# is running, then a final pass and the selection rule. Safe to restart at any time.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$STAGE1_ROOT"
while true; do
  finished=0; grep -q "queue finished" logs/train_queue.log 2>/dev/null && finished=1
  PYTHONUNBUFFERED=1 "$PY" analysis/pipeline.py analysis/configs/val_sweep.yaml 2>&1 \
    | grep --line-buffered -v -i "warn\|stddev" >> logs/analysis_sweep.log
  if [[ $finished == 1 ]]; then break; fi
  sleep 600
done
"$PY" analysis/select.py | tee -a logs/analysis_sweep.log
echo "analysis loop finished $(date -Is)" >> logs/analysis_sweep.log
