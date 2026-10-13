#!/bin/bash
#SBATCH --job-name=v11m6_ligdep
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/v11m6_ligdep_%A_%a.out
#SBATCH --error=logs/v11m6_ligdep_%A_%a.err

# V11m6 ligand-dependency matrix:
#   runs:   main, no_gate, local54_only, no_detach_protein
#   splits: val, test
#   modes:  all(real/shuffle/zero), source_split, feature_split

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m6_controls_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

RUN_NAMES=(
  "main"
  "no_gate"
  "local54_only"
  "no_detach_protein"
)

CKPTS=(
  "outputs/pocket_gvp_fusion_v11m6_20260527_185025/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_no_gate_v11m6_20260527_225519/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_local54_only_v11m6_20260527_230450/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_no_detach_protein_v11m6_20260528_010519/best_f1_c0_model.pt"
)

SPLITS=("val" "test")
MODES=("all" "source_split" "feature_split")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_IDX=$(( TASK_ID / 6 ))
SPLIT_IDX=$(( (TASK_ID / 3) % 2 ))
MODE_IDX=$(( TASK_ID % 3 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
CKPT="${CKPTS[$RUN_IDX]}"
SPLIT="${SPLITS[$SPLIT_IDX]}"
MODE="${MODES[$MODE_IDX]}"

if [ -z "${RUN_NAME}" ] || [ -z "${CKPT}" ] || [ -z "${SPLIT}" ] || [ -z "${MODE}" ]; then
  echo "ERROR: invalid task mapping for task id ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"
echo "Split: ${SPLIT}"
echo "Ligand mode: ${MODE}"
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
