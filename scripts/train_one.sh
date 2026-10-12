#!/usr/bin/env bash
# Train (or resume, or skip if finished) one run: train_one.sh NAME CONFIG SOURCE SEED
set -euo pipefail
source "$(dirname "$0")/env.sh"
name=$1; config=$2; source_=$3; seed=$4
out="$STAGE1_ROOT/runs/$name"; log="$STAGE1_ROOT/logs/train_$name.log"
cd "$STAGE1_ROOT"
mkdir -p "$STAGE1_ROOT/logs"
if [[ -f "$out/training_summary.json" ]]; then echo "[skip] $name finished"; exit 0; fi
resume=()
if [[ -f "$out/last.pt" ]]; then resume=(--resume "$out/last.pt"); echo "[resume] $name"; else echo "[start] $name"; fi
"$PY" -m neomotion train --prompt-source "$source_" --config "$config" --seed "$seed" \
  --teacher-manifest "$MANIFEST" --patient-split-manifest "$SPLIT" --output-dir "$out" \
  --local-files-only "${resume[@]}" >> "$log" 2>&1
cp "$log" "$out/train.log"
echo "[done] $name"
