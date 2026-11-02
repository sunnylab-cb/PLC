#!/usr/bin/env bash
# Retry ESM2 extraction after the PyTorch Namespace safe-loading failure.
# Preserve frozen code_v1/manifest/G0/full-complex inputs; no new model variant.
# Existing CPU audit is complete and is NOT rerun. No TEST inference.
#SBATCH --job-name=chainesm_retry
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set original immutable experiment root}"
FIX_ROOT="${FIX_ROOT:?Set versioned compatibility runner root}"
BASE=/hpc2hdd/home/shuang886/PLC
SOURCE_ROOT="${SOURCE_ROOT:-${BASE}/experiments/joint_global_si30_gfgf_20261017}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${EXPERIMENT}/code_v1"
exec 9>"${EXPERIMENT}/extract.lock"
flock -n 9 || exit 2
"${PYTHON}" -u "${FIX_ROOT}/scripts/diagnostics/retry_chainwise_esm_extract.py" --experiment "${EXPERIMENT}"
"${PYTHON}" -u preparation/protein/esm2_chainwise_complex.py finalize --experiment "${EXPERIMENT}" --workers 8
