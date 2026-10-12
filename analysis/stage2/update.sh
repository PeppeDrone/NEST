#!/usr/bin/env bash
# Refresh the Stage-2 follow-through analysis and the manuscript after new videos finish.
#   bash analysis/stage2/update.sh            # DWPose on new videos only (CPU), statistics, figures, PDF
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../scripts/env.sh"
cd "$STAGE1_ROOT"
export PYTHONPATH="$STAGE1_ROOT/src"
CUDA_VISIBLE_DEVICES="" "$PY" analysis/stage2/followthrough.py extract --provider cpu 2>&1 | grep -v -i warn | tee -a logs/stage2_followthrough.log
"$PY" analysis/stage2/followthrough.py analyze > /dev/null
bash manuscript/compile.sh
