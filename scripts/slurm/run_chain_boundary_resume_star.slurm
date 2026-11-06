#!/usr/bin/env bash
# Continue the three archived HPC controls from best-F1 optimizer states.
# Smoke: one critical GPU task, all arms sequential; discard its updates.
# Train: --array=0-2%1 --dependency=afterok:SMOKE; one allocated GPU at a time.
# No HPC execution, data transfers, downloads, compile, or automatic requeue.
#SBATCH --job-name=plc_chainresume
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=background
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT="${ROOT:-/data1/PLC/experiments/chain_boundary_resume_20261106}"
BUNDLE="${BUNDLE:-/data1/PLC/experiments/chainwise_esm_core_20261102}"
: "${PHASE:?Set PHASE=smoke or train}"
[[ "${PHASE}" == smoke || "${PHASE}" == train ]] || exit 2
[[ "${SLURMD_NODENAME:-}" == star ]] || { printf 'Star-only launcher.\n'; exit 2; }
exec 9>/data1/PLC/core_training.lock
flock -n 9 || { printf 'Another PLC task holds the single-GPU lock.\n'; exit 2; }
export PYTHONPATH="${ROOT}/code_v1:${BUNDLE}/deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${BUNDLE}/runtime/torch291/bin/python"
SCRIPT="${ROOT}/code_v1/scripts/train/train_chain_boundary_resume_star.py"
"${PYTHON}" -c 'import torch,triton; assert torch.__version__ == "2.9.1+cu126" and triton.__version__ == "3.5.1"; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("STAR_RESUME_GPU",torch.cuda.get_device_name(0),flush=True)'
variants=(gvp_boundary coattn_position both)
if [[ "${PHASE}" == smoke ]]; then
    for variant in "${variants[@]}"; do
        "${PYTHON}" -u "${SCRIPT}" smoke --root "${ROOT}" --variant "${variant}" --steps 20
    done
else
    : "${SLURM_ARRAY_TASK_ID:?Training requires a serial task array}"
    [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-2]$ ]] || exit 2
    "${PYTHON}" -u "${SCRIPT}" train --root "${ROOT}" --variant "${variants[$SLURM_ARRAY_TASK_ID]}"
fi
