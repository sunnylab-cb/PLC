#!/usr/bin/env bash
# Speed-only experiment: S0/S1/T1 x chunk128/512/1024 x old/batched statistics.
# Same TRAIN inputs, batch16, BF16, GradScaler, loss and checkpointing.
# Validate output/gradient/statistics parity before repeated same-GPU timing.
# No TEST/VAL selection, checkpoints, resume, or edits to existing training.
# One task runs the entire matrix; eight CPUs and ONE GPU on -u.
# Example: sbatch --export=ALL,EXPERIMENT=/new/root scripts/slurm/run_local_scan_speed.slurm
#SBATCH --job-name=localscan_speed
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set new speed-probe root}"
REFERENCE="${REFERENCE:-/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024}"
SOURCE_ROOT="/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${CODE}"
exec 9>"${EXPERIMENT}/profile.lock"
flock -n 9 || { echo "Duplicate speed probe" >&2; exit 2; }
"${PYTHON}" -u scripts/diagnostics/profile_local_scan_speed.py \
    --reference "${REFERENCE}" --output "${EXPERIMENT}/speed_profile.json" \
    --steps 40 --warmup 10 --repeats 3
