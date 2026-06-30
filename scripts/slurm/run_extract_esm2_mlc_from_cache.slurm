#!/bin/bash
# Extract ESM2 multi-layer residue embeddings from existing data/esm2_latents.
#
# This reuses the sequence/residue metadata in the existing final-layer ESM2
# cache, so it does not need original PDB files and keeps residue alignment
# identical to current training.

#SBATCH --job-name=esm2_mlc
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/esm2_mlc_%A_%a.out
#SBATCH --error=logs/esm2_mlc_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
INPUT_ROOT="${INPUT_ROOT:-data/esm2_latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/esm2_multilayer_latents}"
REPR_LAYERS="${REPR_LAYERS:-2,6,12,18,24,30,33}"
MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

SUBDIRS=(
  "1981-2000"
  "2001-2010"
  "2011-2019"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBDIR="${SUBDIRS[$TASK_ID]}"
if [ -z "${SUBDIR}" ]; then
  echo "ERROR: invalid ESM MLC extract task id: ${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Subdir: ${SUBDIR}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Layers: ${REPR_LAYERS}"
echo "============================================"

if command -v module &> /dev/null; then
  module purge || true
fi

if command -v conda &> /dev/null; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda command not found in PATH"
  exit 1
fi
conda activate af3

python preparation/protein/esm2_extract_multilayer_from_cache.py \
  --input "${INPUT_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --subdir "${SUBDIR}" \
  --model "${MODEL_NAME}" \
  --repr-layers "${REPR_LAYERS}" \
  --device cuda \
  ${OVERWRITE_FLAG}

echo "Done ESM2 MLC extraction ${SUBDIR} at $(date)"
