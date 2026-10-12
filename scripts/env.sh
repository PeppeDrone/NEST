# Shared paths for every script in this repository. Set PY to your Python interpreter (default: python);
# the pinned CLIP ViT-L/14 weights are read from the Hugging Face cache.
export STAGE1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="${PY:-python}"
export MANIFEST="$STAGE1_ROOT/data/dwpose_woface_62_v1/teacher_training_manifest.jsonl"
export SPLIT="$STAGE1_ROOT/data_refs/patient_split_v1.json"
export PYTHONPATH="$STAGE1_ROOT/src"
