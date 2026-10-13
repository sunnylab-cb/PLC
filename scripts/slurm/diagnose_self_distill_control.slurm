#!/usr/bin/env bash
# Diagnose failed lambda-zero smoke using cold/warm repeated base steps and KD0.
# No optimization run, no pass artifact, no dependent training submission.
# Original frozen snapshot/cache remain read-only; all GPU work stays in Slurm.
#SBATCH --job-name=gfgf_sd_control_audit
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00

set -euo pipefail
AUDIT="${AUDIT:-/hpc2hdd/home/shuang886/PLC/experiments/self_distill_control_audit_20261012}"
EXPERIMENT="${EXPERIMENT:-/hpc2hdd/home/shuang886/PLC/experiments/self_distill_gfgf_20261012}"
export PLC_FROZEN_CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${PLC_FROZEN_CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${AUDIT}/triton_cache_${SLURM_JOB_ID}"
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u \
    "${AUDIT}/scripts/diagnostics/diagnose_self_distill_control.py" \
    --experiment "${EXPERIMENT}" --output "${AUDIT}/control_${SLURM_JOB_ID}.json"
