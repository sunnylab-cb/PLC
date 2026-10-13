#!/usr/bin/env bash
# P1: no training, no checkpoint load, no GPU, no benchmark threshold fitting.
# Analyze frozen P0 predictions for candidate availability vs selection failure.
# Main radius 8 A; 6/10 A sensitivity controls are all reported, not selected.
# Oracle and same-chemistry-target metrics are explicitly diagnostic only.
#SBATCH --job-name=bench_candidate
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00

set -euo pipefail
ROOT=/hpc2hdd/home/shuang886/PLC/experiments/benchmark_integrity_20260919
CODE="${CODE:-${ROOT}/code_p1}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${CODE}"
"${PYTHON}" -u scripts/diagnostics/benchmark_candidate_audit.py \
  --eval-root "${ROOT}/eval_12793043" \
  --sdf-root /hpc2hdd/home/shuang886/jhspoolers/PLC/benchmark/ligands_sdf \
  --output "${ROOT}/candidate_${SLURM_JOB_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}" --radii 6,8,10 --min-size 3
