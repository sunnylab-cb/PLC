#!/usr/bin/env bash
# Three scratch controls of G0 + chainwise ESM, retaining FULL complexes:
# 0: GVP same-chain backbone/radius/sequence flags only.
# 1: propagate chain-gap positions into co-attention self-attention only.
# 2: both. Existing job13468092 is the untouched no-fix reference.
# Same seed44, split, labels, batch16, losses, LR and F1 early stopping.
# Real-data forward/backward + cap-sized smoke must pass before fresh training.
# No resume, TEST/HOLO selection, new weights, or file-transfer jobs.
#SBATCH --job-name=G0_chainfix
#SBATCH --partition=i64m1tga800ue
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set experiment root}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
SOURCE=/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017
export PYTHONPATH="${EXPERIMENT}/model_code:${EXPERIMENT}/code_v1:${SOURCE}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${EXPERIMENT}/code_v1"
variants=(gvp_boundary coattn_position both)
variant="${variants[${SLURM_ARRAY_TASK_ID}]}"
exec 9>"${EXPERIMENT}/${variant}.lock"
flock -n 9 || exit 2
"${PYTHON}" -u scripts/train/train_chain_boundary_controls.py --experiment "${EXPERIMENT}" --variant "${variant}" --smoke --steps 100
"${PYTHON}" -u scripts/train/train_chain_boundary_controls.py --experiment "${EXPERIMENT}" --variant "${variant}" --amp --accumulation-steps 1
