#!/usr/bin/env bash
# Frozen best-F1 G0: 4096 deterministic TRAIN pairs + ALL strict VAL pairs.
# Same eval-mode BF16, 5A targets, caps, and saved VAL gate; full/chain control.
# F1/MCC/AP/AUC/Top-L and plain NLL, no training, no TEST/HOLO model selection.
#SBATCH --job-name=g0_trainval_gap
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017}"
G0_ROOT="${G0_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CHECKPOINT="${G0_ROOT}/outputs/G0_small_global/joint_si30_G0_small_global_scratch_seed44_localscan_20261025_044151/best_f1_c0_model.pt"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_${SLURM_JOB_ID}"
cd "${EXPERIMENT}/code_v1"
"${PYTHON}" -u scripts/diagnostics/eval_g0_train_val_gap.py \
    --work "${EXPERIMENT}/g0_gap" --training "${G0_ROOT}" --upstream "${SOURCE_ROOT}" \
    --checkpoint "${CHECKPOINT}" --workers 8 --train-samples 4096
