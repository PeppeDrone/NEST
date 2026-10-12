#!/usr/bin/env bash
# Source after: conda activate dmtbaby-cascade
# Puts pip NVIDIA cuDNN/CUDA libs on LD_LIBRARY_PATH so ORT CUDA EP can dlopen
# libcudnn.so (the pip package ships libcudnn.so.9 without an unversioned soname).
#
# Do not leave set -euo pipefail on the caller's interactive shell — that makes
# later failing commands (or unset vars) kill the whole terminal.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate dmtbaby-cascade first" >&2
  return 1 2>/dev/null || exit 1
fi

CUDNN_LIB="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [[ ! -d "${CUDNN_LIB}" ]]; then
  echo "missing ${CUDNN_LIB}; install nvidia-cudnn-cu13 in this env" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -e "${CUDNN_LIB}/libcudnn.so" && -e "${CUDNN_LIB}/libcudnn.so.9" ]]; then
  ln -s libcudnn.so.9 "${CUDNN_LIB}/libcudnn.so" || {
    echo "could not create ${CUDNN_LIB}/libcudnn.so symlink" >&2
    return 1 2>/dev/null || exit 1
  }
fi

paths=("${CUDNN_LIB}")
for pkg in cublas cuda_runtime cufft curand cusolver cusparse nvjitlink cuda_nvrtc; do
  lib="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/${pkg}/lib"
  [[ -d "${lib}" ]] && paths+=("${lib}")
done
[[ -d /usr/local/cuda/lib64 ]] && paths+=(/usr/local/cuda/lib64)

export LD_LIBRARY_PATH="$(IFS=:; echo "${paths[*]}"):${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH primed for ORT CUDA (cudnn at ${CUDNN_LIB})"
