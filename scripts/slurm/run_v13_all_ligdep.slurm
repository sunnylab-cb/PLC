#!/bin/bash
# Ligand-dependency eval for all completed V13 candidates.
#
# Scope:
#   12 checkpoints x 3 eval groups on the internal test split.
#
# Eval groups:
#   all          -> real, shuffle, zero
#   source_split -> real, shuffle, zero, graph/UniMol source controls
#   feature_split -> real, zero_global55, zero_local54
#
# Purpose:
#   Determine whether each V13 branch keeps/improves ligand-aware behavior:
#   real F1, real-shuffle gap, real-zero gap, and local54/graph dependency.

#SBATCH --job-name=v13_ligdep
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/v13_ligdep_%A_%a.out
#SBATCH --error=logs/v13_ligdep_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_SAVE_ROWS="${NUM_SAVE_ROWS:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

RUN_NAMES=(
  "v13m0_scratch_coattn_resatom"
  "v13m0_scratch_coattn_resatom_tri"
  "v13m0_scratch_noncoattn_resatom"
  "v13m0_scratch_noncoattn_resatom_tri"
  "v13m1_coattn_slots16"
  "v13m1_noncoattn_slots16"
  "v13m2_coattn_qkreparam"
  "v13m2_noncoattn_qkreparam"
  "v13m0_warm_resatom_adapter_only"
  "v13m0_warm_resatom_adapter_head"
  "v13m0_warm_value_enrich_only"
  "v13m0_warm_value_enrich_head"
)

RUN_DIRS=(
  "outputs/pocket_gvp_fusion_coattn_scratch_resatom_b16_acc2_h96_v13m0_20260628_232952"
  "outputs/pocket_gvp_fusion_coattn_scratch_resatom_tri_b16_acc2_h96_v13m0_20260628_233355"
  "outputs/pocket_gvp_fusion_noncoattn_scratch_resatom_b16_acc2_h96_v13m0_20260629_001411"
  "outputs/pocket_gvp_fusion_noncoattn_scratch_resatom_tri_b16_acc2_h96_v13m0_20260629_002412"
  "outputs/pocket_gvp_fusion_coattn_slots16_b16_acc2_h128_v13m1_20260629_003321"
  "outputs/pocket_gvp_fusion_noncoattn_slots16_b16_acc2_h128_v13m1_20260629_004239"
  "outputs/pocket_gvp_fusion_coattn_qkreparam_b16_acc2_h96_v13m2_20260629_094439"
  "outputs/pocket_gvp_fusion_noncoattn_qkreparam_b16_acc2_h96_v13m2_20260629_094438"
  "outputs/pocket_gvp_fusion_seed44_resatom_adapter_only_v13m0_20260629_200559"
  "outputs/pocket_gvp_fusion_seed44_resatom_adapter_head_v13m0_20260629_201528"
  "outputs/pocket_gvp_fusion_seed44_value_enrich_only_v13m0_20260629_202547"
  "outputs/pocket_gvp_fusion_seed44_value_enrich_head_v13m0_20260629_202620"
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
  echo "ERROR: invalid array task id=${TASK_ID} run_idx=${RUN_IDX} mode_idx=${MODE_IDX}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
CONFIG="${RUN_DIR}/config.yaml"
OUT_ROOT="results/ligdep_v13_all_${SLURM_ARRAY_JOB_ID:-manual}"
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
echo "Project: ${PROJECT_DIR}"
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
