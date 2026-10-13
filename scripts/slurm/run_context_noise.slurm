#!/usr/bin/env bash
# ORIGINAL GFGF/V11m13 scratch seed44: protein-context noise, NOT distillation.
# Six arms: random/background x sigma .05/.10/.20; foreground floor .5.
# Protein-only V4 geometry candidates, 25% TRAIN batches, private noise RNG.
# Labels and ligand do not generate masks; clean inference and original losses.
# Smoke: TRAIN mask QC + real B16 BF16 backward + original-class checkpoint load.
# Train only after successful smoke; one A800 per task, at most two concurrent.
# No teacher/cache/KL, resume/freeze/compile, downloads or TEST selection.
# All artifacts on HDD. Query GPU allocation through Slurm, not login nvidia-smi.
#SBATCH --job-name=gfgf_context_noise
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
echo "job=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NODELIST:-none} stage=${STAGE:-smoke} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "${STAGE:-smoke}" == smoke ]]; then
    timeout 90m "${PYTHON}" -u scripts/diagnostics/smoke_context_noise.py --experiment "${EXPERIMENT}"
elif [[ "${STAGE}" == train ]]; then
    "${PYTHON}" scripts/diagnostics/prepare_context_noise.py --experiment "${EXPERIMENT}" --verify --check-smoke
    VARIANTS=(gfgf_random_noise05 gfgf_background_noise05 gfgf_random_noise10 gfgf_background_noise10 gfgf_random_noise20 gfgf_background_noise20)
    ID="${SLURM_ARRAY_TASK_ID:?Expected array 0-5}"
    [[ "${ID}" =~ ^[0-5]$ ]] || exit 2
    "${PYTHON}" -u scripts/train/train_context_noise.py \
        --config "${EXPERIMENT}/prepared/${VARIANTS[$ID]}.yaml" --amp --accumulation-steps 1
else
    echo "Unknown STAGE: ${STAGE}" >&2
    exit 2
fi
