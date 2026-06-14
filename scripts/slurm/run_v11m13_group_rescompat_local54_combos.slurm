#!/bin/bash
# Local54 pair/triple/core-combo ablation for V11m13 group_rescompat.
#
# Purpose:
#   Single subgroup ablation showed base33 is the largest individual group, but
#   full local54 drop is much larger than any single group.  This job scans
#   pair/triple/core combinations to test whether the useful signal is a
#   combinatorial base33-anchor + electronic/sasa/flex effect.
#
# Execution:
#   array=0 seed44, array=1 seed45.  Each task runs the combinations
#   sequentially to avoid launching many tiny GPU jobs.

#SBATCH --job-name=v11m13_l54cmb
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m13_l54cmb_%A_%a.out
#SBATCH --error=logs/v11m13_l54cmb_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m13_group_rescompat_local54_combos_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

RUN_NAMES=(
  "seed44_group_rescompat_lr2e4"
  "seed45_group_rescompat_lr2e4"
)

CKPTS=(
  "outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed45_group_rescompat_lr2e4_v11m13_20260614_022638/best_f1_c0_model.pt"
)

COMBO_NAMES=(
  "base33_electronic3"
  "base33_sasa_hbond4"
  "base33_flex_rot5"
  "electronic3_sasa_hbond4"
  "electronic3_flex_rot5"
  "sasa_hbond4_flex_rot5"
  "base33_electronic3_sasa_hbond4"
  "base33_electronic3_flex_rot5"
  "base33_sasa_hbond4_flex_rot5"
  "electronic3_sasa_hbond4_flex_rot5"
  "base33_electronic3_sasa_hbond4_flex_rot5"
)

COMBO_RANGES=(
  "0:33,33:36"
  "0:33,38:42"
  "0:33,36:38,44:47"
  "33:36,38:42"
  "33:36,36:38,44:47"
  "38:42,36:38,44:47"
  "0:33,33:36,38:42"
  "0:33,33:36,36:38,44:47"
  "0:33,38:42,36:38,44:47"
  "33:36,38:42,36:38,44:47"
  "0:33,33:36,38:42,36:38,44:47"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
CKPT="${CKPTS[$TASK_ID]}"

if [ -z "${RUN_NAME}" ] || [ -z "${CKPT}" ]; then
  echo "ERROR: invalid task id ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"
echo "Output root: ${OUT_ROOT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================"

mkdir -p logs

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

cd "${PROJECT_DIR}" || exit 1

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 3
fi

export PYTHONUNBUFFERED=1
echo "Python: $(which python)"
python --version

REAL_OUT="${OUT_ROOT}/${RUN_NAME}_test_real"
mkdir -p "${REAL_OUT}"
echo "Running real baseline: ${REAL_OUT}"
python scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${CKPT}" \
  --split test \
  --ligand_mode real \
  --output "${REAL_OUT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_save_rows 1 \
  --amp
EXIT_CODE=$?
if [ "${EXIT_CODE}" -ne 0 ]; then
  echo "ERROR: real baseline failed with ${EXIT_CODE}"
  exit "${EXIT_CODE}"
fi

for IDX in "${!COMBO_NAMES[@]}"; do
  NAME="${COMBO_NAMES[$IDX]}"
  RANGES="${COMBO_RANGES[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${RUN_NAME}_test_custom_${NAME}_zero"
  mkdir -p "${OUT_DIR}"
  echo "============================================"
  echo "Custom local54 combo: ${NAME}"
  echo "Ranges: ${RANGES}"
  echo "Output: ${OUT_DIR}"
  echo "============================================"
  python scripts/infer/infer_v11_internal_ligdep.py \
    --checkpoint "${CKPT}" \
    --split test \
    --ligand_mode custom_local54 \
    --custom_local54_name "${NAME}" \
    --custom_local54_ranges "${RANGES}" \
    --custom_local54_op zero \
    --output "${OUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_save_rows 1 \
    --amp
  EXIT_CODE=$?
  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: custom combo ${NAME} failed with ${EXIT_CODE}"
    exit "${EXIT_CODE}"
  fi
done

echo "Finished at: $(date)"
