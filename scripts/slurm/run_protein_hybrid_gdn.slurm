#!/usr/bin/env bash
# Conservative protein-only Qwen-inspired probe, NOT an autoregressive LM.
# STAGE=smoke: FLA/reference forward+backward and real TRAIN-batch smoke.
# STAGE=train, array 0/1: FFFF control / GFGF bidirectional chain-reset GDN.
# Exact same strict train/val, seed44, random init, Co-attn/local54/loss/caps.
# One GPU per task. No torch.compile, no resume, no TEST/benchmark selection.
# Use afterok:<smoke_job> for training. Everything lives in HDD experiments.
#SBATCH --job-name=protein_hybrid
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
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
if [[ "${STAGE:-smoke}" == smoke ]]; then
    timeout 45m "${PYTHON}" -u scripts/diagnostics/smoke_protein_hybrid.py \
        --prepared "${EXPERIMENT}/prepared" --output "${EXPERIMENT}/smoke_pass.json"
elif [[ "${STAGE}" == train ]]; then
    "${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"] is True' "${EXPERIMENT}/smoke_pass.json"
    PATTERNS=(FFFF GFGF)
    ID="${SLURM_ARRAY_TASK_ID:?Expected array 0-1}"
    [[ "${ID}" == 0 || "${ID}" == 1 ]] || exit 2
    "${PYTHON}" -u scripts/train/train_v11m13.py \
        --config "${EXPERIMENT}/prepared/${PATTERNS[$ID]}.yaml" --amp --accumulation-steps 1
else
    echo "Unknown STAGE: ${STAGE}" >&2
    exit 2
fi
