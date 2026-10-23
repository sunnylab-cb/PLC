#!/usr/bin/env bash
# G0/S0/S1/T1: small-global/static-GVP/conditioned-GVP/local-triangle pilots.
# Exact expanded joint GLOBAL-SI30 data, fixed <5A labels, seed44 FROM SCRATCH.
# STAGE=smoke profiles real TRAIN loss/backward, cap-sized stress, and real VAL.
# STAGE=train requires ALL FOUR matching smoke passes. No TEST/HOLO inference.
# One GPU per array task, -u by default; no torch.compile or feature extraction.
# Best F1 + best finite loss only. Early stopping uses VAL F1, patience30.
#SBATCH --job-name=localscan
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-3%4

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_${SLURM_JOB_ID}"
cd "${CODE}"
variants=(G0_small_global S0_local_gvp S1_cond_gvp T1_cond_triangle)
variant="${variants[${SLURM_ARRAY_TASK_ID}]}"
case "${STAGE:?Set smoke or train}" in
    smoke)
        "${PYTHON}" -u scripts/diagnostics/smoke_local_spatial_scan.py \
            --experiment "${EXPERIMENT}" --variant "${variant}"
        ;;
    train)
        exec 9>"${EXPERIMENT}/${variant}.lock"
        flock -n 9 || { echo "Duplicate training run: ${variant}" >&2; exit 2; }
        "${PYTHON}" -u scripts/diagnostics/prepare_local_spatial_scan.py \
            --experiment "${EXPERIMENT}" --verify --require-smoke
        "${PYTHON}" -u scripts/train/train_local_spatial_scan.py \
            --config "${EXPERIMENT}/prepared/${variant}.yaml" --amp --accumulation-steps 1
        ;;
    *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
