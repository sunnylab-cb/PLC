#!/usr/bin/env bash
# Custom HOLO4K chain x exact-chemistry union label/QC preparation ONLY.
# All chains retained; no model inference, checkpoint choice or benchmark tuning.
# Failures are recorded and block subsequent evaluation pending explicit review.
#SBATCH --job-name=holo_chainunion_qc
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017}"
BENCHMARK="${BENCHMARK:-/hpc2hdd/home/shuang886/jhspoolers/PLC/benchmark}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${EXPERIMENT}/code_v1"
"${PYTHON}" -u scripts/benchmark/eval_chain_chem_union.py prepare \
    --work "${EXPERIMENT}/holo_union" --benchmark "${BENCHMARK}" \
    --stats "${SOURCE_ROOT}/dataset/train_only_stats.json"
