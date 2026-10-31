#!/usr/bin/env bash
# CPU strict-VAL audit: SASA loss from other chains, neighbor distance, saved G0 errors.
# Protein-heavy atoms only. No training or single-chain model inference.
#SBATCH --job-name=chain_background
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
BASE=/hpc2hdd/home/shuang886/PLC
SOURCE_ROOT="${SOURCE_ROOT:-${BASE}/experiments/joint_global_si30_gfgf_20261017}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${EXPERIMENT}/code_v1"
"${PYTHON}" -u scripts/diagnostics/audit_chain_background.py --experiment "${EXPERIMENT}" \
  --predictions "${BASE}/experiments/chain_chem_union_20261030/g0_gap/val" \
  --workers "${SLURM_CPUS_PER_TASK}" --controls-per-source 50
