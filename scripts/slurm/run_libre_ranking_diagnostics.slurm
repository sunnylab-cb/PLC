#!/usr/bin/env bash
# CPU-only: compare frozen LiBRe and PLC GFGF predictions on the accepted cohort.
# Both datasets in ONE job; no GPU, inference, training, or benchmark gate fitting.
# Fixed Top-k/fractions and reciprocal native-count budgets isolate ranking.
# Multi-component and shared geometry candidates diagnose centroid limitations.
# GT affects metrics/strata/oracle ceilings only, never proposal generation/ranking.
#SBATCH --job-name=libre_rank_diag
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
ROOT="${ROOT:-/hpc2hdd/home/shuang886/PLC/external_baselines/libre}"
CODE="${CODE:-${ROOT}/runners/ranking_v2}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
"${PYTHON}" -u diagnose_libre_ranking.py \
  --libre-root "${ROOT}/results/benchmark_12960610" \
  --plc-root /hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006/benchmark_bestf1_12956268/GFGF \
  --accepted-report "${ROOT}/results/benchmark_12960610/accepted_report_v1/final_report.json" \
  --output "${ROOT}/results/ranking_${SLURM_JOB_ID}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
