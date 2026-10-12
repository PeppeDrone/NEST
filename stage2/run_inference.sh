#!/usr/bin/env bash
# Stage-2 (frozen UniAnimate/Wan2.1-14B) inference on the shortlisted manual test trajectories.
#   bash stage2/run_inference.sh          # build inputs (if needed) + run, resumable
#   tail -f stage2/run/logs/stage2.log
# Requires external/UniAnimate-DiT (code + Wan2.1-I2V-14B + adapter weights) under the repository root
# and a CUDA environment (CONDA_PREFIX set to the environment holding the NVIDIA pip wheels).
# Re-running skips finished videos (fingerprint-checked) and continues with the rest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/stage2/run"
PY="${PY:-python}"
cd "$ROOT"
mkdir -p "$HERE/logs"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_gpu.sh"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU already has a compute process; not starting (unified-memory OOM risk)." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi

[[ -f "$HERE/stage2_requests.jsonl" ]] || PYTHONPATH="$ROOT/src" "$PY" "$ROOT/stage2/build_inputs.py" 2>&1 | tee -a "$HERE/logs/build_inputs.log"

PYTHONPATH="$ROOT/src" "$PY" -m dmtbaby infer-stage2 --root "$ROOT" \
  --manifest "$HERE/stage2_requests.jsonl" \
  --persistent-dit-params 0 \
  2>&1 | tee -a "$HERE/logs/stage2.log"
