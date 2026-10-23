#!/usr/bin/env bash
# CPU-only nonshared-PDB analysis of the completed joint epoch25 HOLO4K run.
# Freeze ID-only nested cohorts, verify frozen predictions, report both models.
# Primary exclusion: PLC TRAIN+VAL PDB accessions, NOT an SI30 certificate.
# All cohorts and all fixed budgets remain visible; no TEST-driven selection.
# No inference/training, no GPU, no changes to the parent immutable snapshot.
#SBATCH --job-name=holo_nonshared
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${ANALYSIS_ROOT:?}"
PARENT=/hpc2hdd/home/shuang886/PLC/experiments/joint_holo_reference_20261023
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="${ANALYSIS_ROOT}/code_v1:${PYTHONPATH:-}"
cd "${ANALYSIS_ROOT}/code_v1"
exec 9>"${ANALYSIS_ROOT}/analysis.lock"
flock -n 9 || exit 2
"${PYTHON}" -u scripts/benchmark/analyze_holo_nonshared.py prepare \
    --parent "${PARENT}" --output "${ANALYSIS_ROOT}/results"
"${PYTHON}" -u scripts/benchmark/analyze_holo_nonshared.py analyze \
    --output "${ANALYSIS_ROOT}/results"
