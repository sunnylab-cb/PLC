#!/usr/bin/env bash
# M0/C0/R0: replace ONLY G0's front protein context with MLP/TCN/BiGRU.
# Retain GVP/ESM fusion, co-attention, ligand graph and group_rescompat local54.
# Expanded joint GLOBAL-SI30, <5A contact labels, seed44, FROM SCRATCH.
# STAGE=smoke: production AMP/backward100 + cap1800 + VAL + strict CKPT load.
# STAGE=train: requires ALL THREE matching successful smoke reports.
# One GPU/task on -u; no compile, resume, TEST or external-benchmark selection.
# Unchanged loss/batch16; VAL-F1 patience30; best-F1 + best finite loss only.
# Example: sbatch --export=ALL,EXPERIMENT=/path,STAGE=smoke --time=02:00:00 ...
# Then submit STAGE=train with --dependency=afterok:<smoke-array-id>.
#SBATCH --job-name=seqcontext
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
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_${SLURM_JOB_ID}"
cd "${CODE}"
variants=(M0_pointwise C0_tcn R0_bigru)
variant="${variants[${SLURM_ARRAY_TASK_ID}]}"
case "${STAGE:?Set smoke or train}" in
    smoke)
        "${PYTHON}" -u scripts/diagnostics/smoke_sequence_context_controls.py \
            --experiment "${EXPERIMENT}" --variant "${variant}"
        ;;
    train)
        exec 9>"${EXPERIMENT}/${variant}.lock"
        flock -n 9 || { echo "Duplicate training: ${variant}" >&2; exit 2; }
        if compgen -G "${EXPERIMENT}/outputs/${variant}/*/config.yaml" > /dev/null; then
            echo "Existing scratch output found; refuse duplicate or implicit resume" >&2
            exit 2
        fi
        "${PYTHON}" -u scripts/diagnostics/prepare_sequence_context_controls.py \
            --experiment "${EXPERIMENT}" --verify --require-smoke
        "${PYTHON}" -u scripts/train/train_sequence_context_controls.py \
            --config "${EXPERIMENT}/prepared/${variant}.yaml" --amp --accumulation-steps 1
        ;;
    *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
