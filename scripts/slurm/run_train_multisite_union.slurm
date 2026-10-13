#!/usr/bin/env bash
# TRAIN-only equivalent-geometry union coverage; does NOT approve new negatives.
# Complete-link residue/geometry groups avoid transitive misalignment.
# Reports heuristic site multiplicity, chemistry/QC and unknown label coverage.
# No model code, split or source features are changed. CPU only.
#SBATCH --job-name=train_site_union
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_site_parallel_20260926}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${EXPERIMENT}/code"
"${PYTHON}" -u scripts/diagnostics/audit_train_multisite_union.py \
  --root "${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final" \
  --chemistry-manifest "${ARCHIVE}/experiments/strict_assignment_20260919/train_chemistry.json" \
  --label-alignment "${ARCHIVE}/experiments/strict_assignment_20260919/label_alignment_verified.json" \
  --workers "${SLURM_CPUS_PER_TASK}" --output "${EXPERIMENT}/union_${SLURM_JOB_ID}"
