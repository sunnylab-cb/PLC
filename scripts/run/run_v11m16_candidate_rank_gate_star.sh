#!/usr/bin/env bash
# Run exactly one V11m16 PLC task on one Star GPU.

set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_DIR="${PROJECT_DIR:-/data1/PLC/repo}"
STAR_ROOT="${STAR_ROOT:-/data1/PLC}"
CONDA_ROOT="${CONDA_ROOT:-/home/husrcf/anaconda3}"
CONDA_ENV="${CONDA_ENV:-gpsite_full}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${STAR_ROOT}/logs/v11m16_rankgate_${RUN_STAMP}_g${GPU_ID}.log"

mkdir -p "${STAR_ROOT}/logs" "${STAR_ROOT}/tmp"
echo "Starting one V11m16 task on Star GPU ${GPU_ID}; log=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PROJECT_DIR="${PROJECT_DIR}" \
ARCHIVE_ROOT="${STAR_ROOT}" \
OUTPUT_DIR="${STAR_ROOT}/outputs" \
CONDA_ROOT="${CONDA_ROOT}" \
CONDA_ENV="${CONDA_ENV}" \
SLURM_JOB_ID="star_v11m16_${RUN_STAMP}_g${GPU_ID}" \
SLURMD_NODENAME="star" \
  bash "${PROJECT_DIR}/scripts/slurm/run_v11m16_candidate_rank_gate.slurm" \
    >"${LOG_FILE}" 2>&1

echo "Completed V11m16 task on Star GPU ${GPU_ID}"
