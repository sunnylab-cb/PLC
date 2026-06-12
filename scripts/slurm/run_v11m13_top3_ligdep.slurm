#!/bin/bash
# Test ligand-dependency eval for the three most informative V11m13 probes.
#
# Purpose:
#   V11m13 local-evidence probes did not clearly raise val F1, but three runs
#   are still worth checking on test dependency:
#     - group_rescompat_lr2e4: best peak among low-LR interaction variants.
#     - all3_lr5e4: aggressive full-combination control.
#     - chem_adapter_lr2e4: most stable late/tail behavior.
#
# Evaluation:
#   split=test only; each run is evaluated with:
#     all           -> real, shuffle, zero
#     source_split  -> graph/UniMol split controls
#     feature_split -> local54/global55 and local54 subgroup controls
#
# Primary readout:
#   test real F1, real-shuffle gap, real-zero gap, local54/graph drops.

#SBATCH --job-name=v11m13_ligdep
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/v11m13_ligdep_%A_%a.out
#SBATCH --error=logs/v11m13_ligdep_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m13_top3_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"

RUN_NAMES=(
  "group_rescompat_lr2e4"
  "all3_lr5e4"
  "chem_adapter_lr2e4"
)

RUN_DIRS=(
  "outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114"
  "outputs/pocket_gvp_fusion_seed44_all3_lr5e4_v11m13_20260611_205603"
  "outputs/pocket_gvp_fusion_seed44_chem_adapter_lr2e4_v11m13_20260611_205604"
)

MODES=(
  "all"
  "source_split"
  "feature_split"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_IDX=$(( TASK_ID / 3 ))
MODE_IDX=$(( TASK_ID % 3 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
RUN_DIR="${RUN_DIRS[$RUN_IDX]}"
MODE="${MODES[$MODE_IDX]}"
CKPT="${RUN_DIR}/best_f1_c0_model.pt"

if [ -z "${RUN_NAME}" ] || [ -z "${RUN_DIR}" ] || [ -z "${MODE}" ]; then
  echo "ERROR: invalid V11m13 top3 ligand-dependency task id: ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Split: ${SPLIT}"
echo "Ligand mode bundle: ${MODE}"
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

OUT_DIR="${OUT_ROOT}/${RUN_NAME}_${SPLIT}_${MODE}"
mkdir -p "${OUT_DIR}"

echo "Python: $(which python)"
python --version

CMD=(
  python scripts/infer/infer_v11_internal_ligdep.py
  --checkpoint "${CKPT}"
  --split "${SPLIT}"
  --ligand_mode "${MODE}"
  --output "${OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_samples "${MAX_SAMPLES}"
  --num_save_rows 1
  --amp
)

echo "Command: ${CMD[*]}"
echo "============================================"
"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
