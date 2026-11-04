#!/usr/bin/env bash
# Frozen G0: strict-VAL-only baseline/GVP/position/both, plus chain reversal.
# One job loops over all treatments, same labels and global Top20 budget.
# Counts artificial backbone bridges and preserves predictions for paired audits.
# No TEST/HOLO inference, threshold sweep, or change to ongoing training.
#SBATCH --job-name=chainfix_val
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set experiment root}"
SOURCE=/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017
G0=/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024
export PYTHONPATH="${EXPERIMENT}/model_code:${EXPERIMENT}/code_v1:${SOURCE}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${EXPERIMENT}/code_v1"
exec 9>"${EXPERIMENT}/val_audit.lock"
flock -n 9 || exit 2
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u scripts/diagnostics/eval_chain_boundary_controls.py \
    --experiment "${EXPERIMENT}" --workers 8 \
    --checkpoint "${G0}/outputs/G0_small_global/joint_si30_G0_small_global_scratch_seed44_localscan_20261025_044151/best_f1_c0_model.pt"
