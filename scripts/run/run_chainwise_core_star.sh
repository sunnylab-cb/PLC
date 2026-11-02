#!/usr/bin/env bash
# Manual single-GPU launch only AFTER transfer verification. No model downloads.
set -euo pipefail
BUNDLE="${BUNDLE:-/data1/PLC/experiments/chainwise_esm_core_20261102}"
PYTHON="${PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}"
GPU="${1:?Specify exactly one free GPU index}"
[[ "${GPU}" =~ ^[0-9]+$ ]] || exit 2
exec 9>/data1/PLC/core_training.lock
flock -n 9 || { printf 'Another core training task holds the lock.\n'; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${BUNDLE}/code_v1:${BUNDLE}/deps:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${BUNDLE}/logs"
exec > >(tee -a "${BUNDLE}/logs/core_train.log") 2>&1
"${PYTHON}" -u "${BUNDLE}/code_v1/scripts/train/train_chainwise_core.py" --bundle "${BUNDLE}" --smoke --steps 100
"${PYTHON}" -u "${BUNDLE}/code_v1/scripts/train/train_chainwise_core.py" --bundle "${BUNDLE}" --amp --accumulation-steps 1
