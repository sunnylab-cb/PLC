#!/bin/bash
# Ligand-dependency eval for the current key ESM/MLC checkpoints.
#
# Scope:
#   3 checkpoints x 3 eval groups on the internal test split.
#
# Checkpoints:
#   0 esm3_global_prior_fullfinetune
#       Best current val signal: full-parameter finetune from V11m13 with
#       normalized ESM3 global prior.
#   1 esm3_global_prior_scratch
#       Scratch control with normalized ESM3 global prior.
#   2 esmmlc_scratch_wsum_uniform
#       Completed ESM2-MLC scratch weighted-sum run; late small positive val
#       signal, mostly precision-driven.
#
# Eval groups:
#   all          -> real, shuffle, zero
#   source_split -> real, shuffle, zero, graph/UniMol source controls
#   feature_split -> real, zero_global55, zero_local54
#
# Purpose:
#   Validate whether the val-F1 gains are true ligand-aware improvements:
#   real F1 must improve while real-shuffle / real-zero gaps remain healthy.

#SBATCH --job-name=e3mlc_lgd
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/e3mlc_ligdep_%A_%a.out
#SBATCH --error=logs/e3mlc_ligdep_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_SAVE_ROWS="${NUM_SAVE_ROWS:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

RUN_NAMES=(
  "esm3_global_prior_fullfinetune"
  "esm3_global_prior_scratch"
  "esmmlc_scratch_wsum_uniform"
)

RUN_DIRS=(
  "outputs/pocket_gvp_fusion_seed44_esm3_global_prior_fullfinetune_v11m13e3ft_20260702_023232"
  "outputs/pocket_gvp_fusion_seed44_esm3_global_prior_scratch_v11m13e3ft_20260702_034855"
  "outputs/pocket_gvp_fusion_seed44_esmmlc_scratch_wsum_uniform_v11m13_20260701_094833"
)

MODE_GROUPS=(
  "all"
  "source_split"
  "feature_split"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_IDX=$((TASK_ID / 3))
MODE_IDX=$((TASK_ID % 3))
RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
RUN_DIR="${RUN_DIRS[$RUN_IDX]}"
MODE_GROUP="${MODE_GROUPS[$MODE_IDX]}"

if [ -z "${RUN_NAME}" ] || [ -z "${RUN_DIR}" ] || [ -z "${MODE_GROUP}" ]; then
  echo "ERROR: invalid task id=${TASK_ID} run_idx=${RUN_IDX} mode_idx=${MODE_IDX}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
CONFIG="${RUN_DIR}/config.yaml"
OUT_ROOT="results/ligdep_v11m13_esm3_mlc_key_${SLURM_ARRAY_JOB_ID:-manual}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}/${MODE_GROUP}"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Mode group: ${MODE_GROUP}"
echo "Split: ${SPLIT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Checkpoint: ${CKPT}"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
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

echo "Python: $(which python)"
python --version
echo "Eval command:"
echo "python scripts/infer/infer_v11_internal_ligdep.py --checkpoint ${CKPT} --config ${CONFIG} --split ${SPLIT} --ligand_mode ${MODE_GROUP} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --max_samples ${MAX_SAMPLES} --shuffle_seed ${SHUFFLE_SEED} --num_save_rows ${NUM_SAVE_ROWS} --output ${OUT_DIR} --amp"

python scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${CKPT}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --ligand_mode "${MODE_GROUP}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_samples "${MAX_SAMPLES}" \
  --shuffle_seed "${SHUFFLE_SEED}" \
  --num_save_rows "${NUM_SAVE_ROWS}" \
  --output "${OUT_DIR}" \
  --amp

echo "Done: ${RUN_NAME} ${MODE_GROUP} at $(date)"
