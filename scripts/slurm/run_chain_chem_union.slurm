#!/usr/bin/env bash
# Scratch G0 on NEW chain + exact-chemical union targets, seed44.
# Three matched controls: baseline; weight_decay .05->.10; dropout .15->.20.
# Each task runs its own real-data GPU smoke before starting full training.
# One GPU and 8 CPUs per task; VAL F1 stopping; best F1/best finite loss only.
# No resume, torch.compile, TEST/HOLO tuning or modification of old snapshots.
#SBATCH --job-name=chainunion_g0
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-2%3

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set immutable experiment root}"
SOURCE_ROOT="${SOURCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${EXPERIMENT}/code_v1:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_${SLURM_JOB_ID}"
cd "${EXPERIMENT}/code_v1"
variants=(union_base union_wd010 union_drop020)
variant="${variants[${SLURM_ARRAY_TASK_ID}]}"
exec 9>"${EXPERIMENT}/${variant}.lock"
flock -n 9 || { echo "Duplicate training run: ${variant}" >&2; exit 2; }
"${PYTHON}" -u scripts/diagnostics/smoke_chain_chem_union.py \
    --experiment "${EXPERIMENT}" --variant "${variant}" --steps 30
"${PYTHON}" -u scripts/diagnostics/prepare_chain_chem_union.py \
    --experiment "${EXPERIMENT}" --verify --variant "${variant}" --require-smoke
"${PYTHON}" -u scripts/train/train_chain_chem_union.py \
    --config "${EXPERIMENT}/prepared/${variant}.yaml" --amp --accumulation-steps 1
