#!/usr/bin/env bash
# CPU: exact chemistry + receptor-chain union, unchanged strict SI30 membership.
# Keep all chains in audit; >=3-contact queries enter supervised sets.
# No feature duplication, model inference, normalization refit or TEST selection.
#SBATCH --job-name=chainunion_qc
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017}"
G0_ROOT="${G0_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024}"
POOL="${POOL:-/hpc2hdd/home/shuang886/PLC/experiments/biolip_feature_expansion_20261015/qc_pool/samples.jsonl}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${EXPERIMENT}/code_v1"
"${PYTHON}" -u scripts/diagnostics/prepare_chain_chem_union.py \
    --experiment "${EXPERIMENT}" --upstream "${SOURCE_ROOT}" \
    --base-config "${G0_ROOT}/prepared/G0_small_global.yaml" \
    --pool "${POOL}" --workers "${SLURM_CPUS_PER_TASK}"
