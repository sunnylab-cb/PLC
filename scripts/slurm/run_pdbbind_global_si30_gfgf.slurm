#!/usr/bin/env bash
# One GPU/task, exact V11m13 group_rescompat + GFGF seed44, FROM SCRATCH.
# STAGE=smoke: verify artifacts, FLA/reference check, production mixed-precision
# loss/backward and real VAL inference; discard the smoke model afterwards.
# STAGE=train: requires matching successful smoke; no resume/compile/warm-start.
# Dataset: independent PDBbind-only global SI<.30 TRAIN/VAL/TEST, fixed5 labels.
# Existing best-F1, AUPRC/AUROC and VAL gate utilities remain unchanged.
#SBATCH --job-name=pdb_gsi30_gfgf
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=36:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set new immutable experiment root}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_cache_${SLURM_JOB_ID:-manual}"
cd "${CODE}"
case "${STAGE:?Set smoke or train}" in
    smoke)
        "${PYTHON}" -u scripts/diagnostics/smoke_internal_gfgf.py --experiment "${EXPERIMENT}"
        ;;
    train)
        exec 9>"${EXPERIMENT}/training.lock"
        flock -n 9 || { echo "Training already running" >&2; exit 2; }
        "${PYTHON}" -u scripts/diagnostics/prepare_internal_gfgf.py \
            --experiment "${EXPERIMENT}" --verify --require-smoke
        "${PYTHON}" -u scripts/train/train_v11m13.py \
            --config "${EXPERIMENT}/prepared/GFGF.yaml" --amp --accumulation-steps 1
        ;;
    *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
