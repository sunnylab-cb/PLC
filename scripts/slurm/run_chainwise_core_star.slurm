#!/usr/bin/env bash
# Star GPU jobs only: critical smoke, then an afterok scratch training job.
# No transfers, ESM extraction, model downloads, or automatic resume.
# Submit smoke with --account=labcritical --qos=critical --time=01:00:00.
# Submit train with --dependency=afterok:SMOKE_ID and the defaults below.
#SBATCH --job-name=plc_chainesm
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
BUNDLE="${BUNDLE:-/data1/PLC/experiments/chainwise_esm_core_20261102}"
PYTHON=/home/husrcf/anaconda3/envs/gpsite_full/bin/python
: "${PHASE:?Set PHASE=smoke or train}"
[[ "${PHASE}" == smoke || "${PHASE}" == train ]] || exit 2
exec 9>/data1/PLC/core_training.lock
flock -n 9 || { printf 'Another core training task holds the lock.\n'; exit 2; }
# Keep Slurm's device mapping; never override CUDA_VISIBLE_DEVICES here.
export PYTHONPATH="${BUNDLE}/code_v1:${BUNDLE}/deps:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf 'STAR_CORE phase=%s job=%s devices=%s\n' "${PHASE}" "${SLURM_JOB_ID}" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable inside allocation"; assert torch.cuda.device_count() == 1, "Exactly one visible GPU required"; print("CUDA", torch.__version__, torch.cuda.get_device_name(0))'
if [[ "${PHASE}" == smoke ]]; then
    "${PYTHON}" -u "${BUNDLE}/code_v1/scripts/train/train_chainwise_core.py" --bundle "${BUNDLE}" --smoke --steps 100
else
    "${PYTHON}" -u "${BUNDLE}/code_v1/scripts/train/train_chainwise_core.py" --bundle "${BUNDLE}" --amp --accumulation-steps 1
fi
