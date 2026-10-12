#!/usr/bin/env bash
# Run every job in jobs_training.tsv, N in parallel (default 3). Idempotent: finished runs are
# skipped, interrupted runs resume from last.pt. Optional filter: train_queue.sh [N] [PART]
set -uo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-3}; PART=${2:-}
grep -v '^#' "$STAGE1_ROOT/scripts/jobs_training.tsv" | awk -v p="$PART" -F'\t' 'p=="" || $5==p {print $1, $2, $3, $4}' \
  | xargs -P "$N" -L 1 bash "$STAGE1_ROOT/scripts/train_one.sh"
echo "queue finished $(date -Is)"
