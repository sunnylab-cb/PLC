#!/bin/bash
# Broad local54 dependency scan for the current strongest V11m8x co-attn probe.
#
# Purpose:
#   Identify which local54 atom-chemistry dimensions/groups/combinations are
#   carrying the ligand-aware residue signal in the co-attn model.  This is an
#   eval-only job: no training, no checkpoint mutation.
#
# Default checkpoint:
#   outputs/pocket_gvp_fusion_seed44_coattn_g60_v11m8x_*/best_f1_c0_model.pt
#
# Scan design:
#   1) Baseline dependency controls: real/shuffle/zero/local54/global55/edge/topology
#   2) Single local54 dims 0..53 with zero + atom-shuffle perturbations
#   3) Whole groups with zero + atom-shuffle + mean + matched-noise perturbations
#   4) High-priority chemistry combinations with the same four perturbations
#
# Perturbation semantics for --ligand_mode custom_local54:
#   zero    : set selected local54 dims to zero
#   shuffle : permute selected dim values across atoms within each ligand
#   mean    : replace selected dim values by the per-ligand per-dim mean
#   noise   : replace selected dim values by per-ligand Gaussian noise matching
#             that ligand's selected-dim mean/std

#SBATCH --job-name=v11m8x_l54wide
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-178
#SBATCH --output=logs/v11m8x_l54wide_%A_%a.out
#SBATCH --error=logs/v11m8x_l54wide_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m8x_coattn_local54_wide_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="${SPLIT:-test}"
VARIANT="${VARIANT_NAME:-seed44_coattn_g60}"

BASE_MODES=(
  "real"
  "shuffle"
  "zero"
  "zero_ligand_local54_only"
  "zero_ligand_global55_only"
  "zero_ligand_edge_attr_only"
  "shuffle_ligand_topology_only"
)

SINGLE_OPS=("zero" "shuffle")

GROUP_NAMES=(
  "base33"
  "electronic3"
  "sasa_hbond4"
  "flex_rot5"
  "metal4"
  "stereo5"
)
GROUP_RANGES=(
  "0:33"
  "33:36"
  "38:42"
  "36:38,44:47"
  "47:51"
  "42:44,51:54"
)
GROUP_OPS=("zero" "shuffle" "mean" "noise")

COMBO_NAMES=(
  "electronic3_sasa_hbond4"
  "electronic3_flex_rot5"
  "sasa_hbond4_flex_rot5"
  "base33_electronic3"
  "base33_sasa_hbond4"
  "base33_flex_rot5"
  "electronic3_sasa_hbond4_flex_rot5"
  "base33_electronic3_sasa_hbond4_flex_rot5"
  "polar_surface_electronic"
  "nonlocal_tail_metal_stereo"
)
COMBO_RANGES=(
  "33:36,38:42"
  "33:36,36:38,44:47"
  "38:42,36:38,44:47"
  "0:33,33:36"
  "0:33,38:42"
  "0:33,36:38,44:47"
  "33:36,38:42,36:38,44:47"
  "0:33,33:36,38:42,36:38,44:47"
  "33:36,38:42,47:51"
  "42:44,47:51,51:54"
)
COMBO_OPS=("zero" "shuffle" "mean" "noise")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
IS_CUSTOM=0
LIGAND_MODE=""
FEATURE_NAME=""
FEATURE_RANGES=""
FEATURE_OP=""

N_BASE=${#BASE_MODES[@]}
N_SINGLE_OPS=${#SINGLE_OPS[@]}
N_SINGLE_TASKS=$((54 * N_SINGLE_OPS))
N_GROUPS=${#GROUP_NAMES[@]}
N_GROUP_OPS=${#GROUP_OPS[@]}
N_GROUP_TASKS=$((N_GROUPS * N_GROUP_OPS))
N_COMBOS=${#COMBO_NAMES[@]}
N_COMBO_OPS=${#COMBO_OPS[@]}
N_COMBO_TASKS=$((N_COMBOS * N_COMBO_OPS))
N_TOTAL=$((N_BASE + N_SINGLE_TASKS + N_GROUP_TASKS + N_COMBO_TASKS))

idx="${TASK_ID}"
if [ "${idx}" -lt "${N_BASE}" ]; then
  LIGAND_MODE="${BASE_MODES[$idx]}"
elif [ "$((idx - N_BASE))" -lt "${N_SINGLE_TASKS}" ]; then
  idx=$((idx - N_BASE))
  dim=$((idx / N_SINGLE_OPS))
  op_idx=$((idx % N_SINGLE_OPS))
  IS_CUSTOM=1
  FEATURE_NAME=$(printf "dim%02d" "${dim}")
  FEATURE_RANGES="${dim}:$((dim + 1))"
  FEATURE_OP="${SINGLE_OPS[$op_idx]}"
  LIGAND_MODE="custom_local54"
elif [ "$((idx - N_BASE - N_SINGLE_TASKS))" -lt "${N_GROUP_TASKS}" ]; then
  idx=$((idx - N_BASE - N_SINGLE_TASKS))
  group_idx=$((idx / N_GROUP_OPS))
  op_idx=$((idx % N_GROUP_OPS))
  IS_CUSTOM=1
  FEATURE_NAME="${GROUP_NAMES[$group_idx]}"
  FEATURE_RANGES="${GROUP_RANGES[$group_idx]}"
  FEATURE_OP="${GROUP_OPS[$op_idx]}"
  LIGAND_MODE="custom_local54"
elif [ "$((idx - N_BASE - N_SINGLE_TASKS - N_GROUP_TASKS))" -lt "${N_COMBO_TASKS}" ]; then
  idx=$((idx - N_BASE - N_SINGLE_TASKS - N_GROUP_TASKS))
  combo_idx=$((idx / N_COMBO_OPS))
  op_idx=$((idx % N_COMBO_OPS))
  IS_CUSTOM=1
  FEATURE_NAME="${COMBO_NAMES[$combo_idx]}"
  FEATURE_RANGES="${COMBO_RANGES[$combo_idx]}"
  FEATURE_OP="${COMBO_OPS[$op_idx]}"
  LIGAND_MODE="custom_local54"
else
  echo "ERROR: invalid task id ${TASK_ID}; total tasks=${N_TOTAL}"
  exit 2
fi

if [ "${IS_CUSTOM}" -eq 1 ]; then
  MODE_LABEL="custom_local54_${FEATURE_NAME}_${FEATURE_OP}"
else
  MODE_LABEL="${LIGAND_MODE}"
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}/${N_TOTAL}"
echo "Variant: ${VARIANT}"
echo "Split: ${SPLIT}"
echo "Ligand mode: ${LIGAND_MODE}"
echo "Mode label: ${MODE_LABEL}"
echo "Feature name: ${FEATURE_NAME}"
echo "Feature ranges: ${FEATURE_RANGES}"
echo "Feature op: ${FEATURE_OP}"
echo "Output root: ${OUT_ROOT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1; exiting after task mapping check."
  exit 0
fi

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

RUN_DIR="$(ls -dt "outputs/pocket_gvp_fusion_${VARIANT}_v11m8x_"* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: no output directory found for variant: ${VARIANT}"
  exit 3
fi

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 4
fi

OUT_DIR="${OUT_ROOT}/${VARIANT}_${SPLIT}_${MODE_LABEL}"
mkdir -p "${OUT_DIR}"

echo "Run dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Python: $(which python)"
python --version

CMD=(
  python scripts/infer/infer_v11_internal_ligdep.py
  --checkpoint "${CKPT}"
  --split "${SPLIT}"
  --ligand_mode "${LIGAND_MODE}"
  --output "${OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_samples "${MAX_SAMPLES}"
  --num_save_rows 1
  --amp
)

if [ "${IS_CUSTOM}" -eq 1 ]; then
  CMD+=(
    --custom_local54_name "${FEATURE_NAME}"
    --custom_local54_ranges "${FEATURE_RANGES}"
    --custom_local54_op "${FEATURE_OP}"
  )
fi

echo "Command: ${CMD[*]}"
echo "============================================"
"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
