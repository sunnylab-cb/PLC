#!/usr/bin/env bash
# Submission-only CPU continuation: no model training/inference on CPU/login.
# Submit with afterok:<smoke_job>. Verify all four GPU checks and frozen hashes
# before submitting array 0-3%2 to UE. A failed smoke never launches training.
# This launcher lives outside the frozen code_v1 tree; it does not modify it.
#SBATCH --job-name=plc_libre_launch
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
SMOKE_JOB="${SMOKE_JOB:?Set successful prerequisite job ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u \
    "${EXPERIMENT}/launcher/submit_libre_transfer.py" \
    --experiment "${EXPERIMENT}" --smoke-job "${SMOKE_JOB}"
