#!/bin/bash
# Test ligand-dependency eval for the two strongest ESM2-MLC warm probes.
#
# Runs `ligand_mode=all` on the internal test split:
#   real, shuffle, zero
#
# Purpose:
#   Verify that the small val-F1 gain from MLC did not reduce ligand awareness,
#   especially real-shuffle and real-zero gaps.

#SBATCH --job-name=mlc_ligdep
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mlc_ligdep_%A_%a.out
#SBATCH --error=logs/mlc_ligdep_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAVE_ROWS="${NUM_SAVE_ROWS:-1}"

RUN_NAMES=(
  "esmmlc_wsum_only"
  "esmmlc_concat_only"
)

RUN_DIRS=(
  "outputs/pocket_gvp_fusion_seed44_esmmlc_wsum_only_v11m13_20260630_183257"
  "outputs/pocket_gvp_fusion_seed44_esmmlc_concat_only_v11m13_20260630_184500"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
RUN_DIR="${RUN_DIRS[$TASK_ID]}"
if [ -z "${RUN_NAME}" ] || [ -z "${RUN_DIR}" ]; then
  echo "ERROR: invalid MLC ligdep task id=${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
CONFIG="${RUN_DIR}/config.yaml"
OUT_DIR="results/ligdep_v11m13_esmmlc_${SLURM_ARRAY_JOB_ID:-manual}/${RUN_NAME}_test_all"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
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

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 3
fi
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config not found: ${CONFIG}"
  exit 4
fi

mkdir -p "${OUT_DIR}"

python scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${CKPT}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --ligand_mode all \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_save_rows "${NUM_SAVE_ROWS}" \
  --output "${OUT_DIR}" \
  --amp

echo "Done: ${RUN_NAME} test all at $(date)"
