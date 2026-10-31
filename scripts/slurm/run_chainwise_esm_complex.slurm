#!/usr/bin/env bash
# Chainwise ESM reassembled into FULL complexes: extract, frozen VAL swap, scratch G0.
# STAGE=extract checks cache parity and all residue axes before materialization.
# STAGE=train requires real-data smoke, F1 early stopping, no resume/compile.
# Never use isolated-chain model inputs, chain-union labels, or TEST selection.
#SBATCH --job-name=chainesm_complex
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
BASE=/hpc2hdd/home/shuang886/PLC
SOURCE_ROOT="${SOURCE_ROOT:-${BASE}/experiments/joint_global_si30_gfgf_20261017}"
G0_ROOT="${G0_ROOT:-${BASE}/experiments/local_spatial_scan_ampfix_u_20261024}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${EXPERIMENT}/code_v1"
exec 9>"${EXPERIMENT}/${STAGE:?Set extract, eval or train}.lock"
flock -n 9 || exit 2
case "${STAGE}" in
  extract)
    "${PYTHON}" -u preparation/protein/esm2_chainwise_complex.py extract --experiment "${EXPERIMENT}"
    "${PYTHON}" -u preparation/protein/esm2_chainwise_complex.py finalize --experiment "${EXPERIMENT}" --workers 8
    ;;
  eval)
    "${PYTHON}" -u scripts/diagnostics/eval_chainwise_esm_complex.py --experiment "${EXPERIMENT}" \
      --checkpoint "${G0_ROOT}/outputs/G0_small_global/joint_si30_G0_small_global_scratch_seed44_localscan_20261025_044151/best_f1_c0_model.pt" --workers 8
    ;;
  train)
    "${PYTHON}" -u scripts/train/train_chainwise_esm_complex.py --experiment "${EXPERIMENT}" --smoke --steps 100
    "${PYTHON}" -u scripts/train/train_chainwise_esm_complex.py --experiment "${EXPERIMENT}" --amp --accumulation-steps 1
    ;;
  *) exit 2 ;;
esac
