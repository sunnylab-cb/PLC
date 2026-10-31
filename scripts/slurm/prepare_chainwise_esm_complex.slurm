#!/usr/bin/env bash
# Inventory chainwise ESM for unchanged full-complex G0; NO chain-only samples.
# Keep split/labels/normalization/caps; CPU checks and sequence-SHA dedup only.
#SBATCH --job-name=chainesm_inventory
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
BASE=/hpc2hdd/home/shuang886/PLC
G0_ROOT="${G0_ROOT:-${BASE}/experiments/local_spatial_scan_ampfix_u_20261024}"
SOURCE_ROOT="${SOURCE_ROOT:-${BASE}/experiments/joint_global_si30_gfgf_20261017}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${EXPERIMENT}/code_v1"
"${PYTHON}" -u preparation/protein/esm2_chainwise_complex.py prepare \
  --experiment "${EXPERIMENT}" --training "${G0_ROOT}" --upstream "${SOURCE_ROOT}" \
  --weights "${BASE}/external_baselines/libre/checkpoints/esm2/esm2_t33_650M_UR50D.pt" \
  --libre-cache "${BASE}/external_baselines/libre/features/chainwise_fp32_l33_v1" \
  --workers "${SLURM_CPUS_PER_TASK}"
