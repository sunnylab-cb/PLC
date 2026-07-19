#!/usr/bin/env bash
# Run one or more V11m13 ESMC probes sequentially on a selected Star GPU.
#
# Usage:
#   run_v11m13_esmc_star.sh GPU_ID TASK_ID [TASK_ID ...]
# Example:
#   run_v11m13_esmc_star.sh 0 0 2 4

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 GPU_ID TASK_ID [TASK_ID ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
TASK_IDS=("$@")

PROJECT_DIR="${PROJECT_DIR:-/data1/PLC/repo}"
STAR_ROOT="${STAR_ROOT:-/data1/PLC}"
CONDA_ROOT="${CONDA_ROOT:-/home/husrcf/anaconda3}"
CONDA_ENV="${CONDA_ENV:-gpsite_full}"
BATCH_SIZE="${BATCH_SIZE:-24}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export PATH="${CONDA_ROOT}/bin:${PATH}"
mkdir -p "${STAR_ROOT}/logs" "${STAR_ROOT}/tmp"

for task_id in "${TASK_IDS[@]}"; do
  if [[ ! "${task_id}" =~ ^[0-5]$ ]]; then
    echo "ERROR: ESMC task id must be in [0,5], got ${task_id}" >&2
    exit 3
  fi

  run_id="star_esmc_${RUN_STAMP}_g${GPU_ID}_t${task_id}"
  log_file="${STAR_ROOT}/logs/${run_id}.log"
  echo "Starting task=${task_id} gpu=${GPU_ID} log=${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PROJECT_DIR="${PROJECT_DIR}" \
  ARCHIVE_ROOT="${STAR_ROOT}" \
  OUTPUT_DIR="${STAR_ROOT}/outputs" \
  CONDA_ENV="${CONDA_ENV}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SLURM_ARRAY_TASK_ID="${task_id}" \
  SLURM_JOB_ID="${run_id}" \
  SLURMD_NODENAME="star" \
    bash "${PROJECT_DIR}/scripts/slurm/run_v11m13_esmc_probes.slurm" \
      >"${log_file}" 2>&1

  echo "Completed task=${task_id} gpu=${GPU_ID}"
done
