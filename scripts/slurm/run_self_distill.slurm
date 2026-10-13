#!/usr/bin/env bash
# Self-distillation: frozen GFGF seed44 best-F1 teacher -> SAME scratch student.
# Array 0: T=2 / lambda=.05. Array 1: T=2 / lambda=.10.
# Hard-label Focal/Tversky, CNC, original auxiliaries and best-F1 VAL unchanged.
# KL uses TRAIN real-pair cache ONLY; axis/hash checked; skip source-dropped rows.
# Epochs 1-5: no KL; epochs 6-10: ramp; afterwards full lambda.
# No resume, freeze, compile, new features, downloads or TEST-derived targets.
# Chain: cache eval afterok -> GPU smoke afterok -> this 0-1%2 array.
# One GPU per task. No GPU probes on login nodes. All outputs on HDD.
#SBATCH --job-name=gfgf_selfdistill
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=36:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
CODE="${EXPERIMENT}/code_v1"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_cache_${SLURM_JOB_ID:-manual}"
cd "${CODE}"
echo "Slurm job=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NODELIST:-none} stage=${STAGE:-smoke} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "${STAGE:-smoke}" == smoke ]]; then
    timeout 90m "${PYTHON}" -u scripts/diagnostics/smoke_self_distill.py --experiment "${EXPERIMENT}"
elif [[ "${STAGE}" == train ]]; then
    "${PYTHON}" scripts/diagnostics/prepare_self_distill.py --experiment "${EXPERIMENT}" --verify --check-smoke
    VARIANTS=(gfgf_sd_t2_lam005 gfgf_sd_t2_lam010)
    ID="${SLURM_ARRAY_TASK_ID:?Expected array 0-1}"
    [[ "${ID}" =~ ^[0-1]$ ]] || exit 2
    "${PYTHON}" -u scripts/train/train_self_distill.py \
        --config "${EXPERIMENT}/prepared/${VARIANTS[$ID]}.yaml" --amp --accumulation-steps 1
else
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
fi
