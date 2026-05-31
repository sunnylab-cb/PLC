#!/bin/bash
# Runs paper-evidence ligand-dependency evaluation for:
#   1) moment-init winner qknorm_a14_tiny3e4_g60
#   2) grad060_no_gate seeds 43/44/45/46
#
# For each checkpoint, this array runs test split bundles:
#   all           -> real, shuffle, zero
#   source_split  -> real, shuffle, zero, graph/UniMol decoupling
#   feature_split -> real, local54/global55 feature split controls
#
# Purpose:
#   - Decide whether moment-init should be kept.
#   - Collect grad060 multi-seed ligand-aware evidence for mean±std reporting.

#SBATCH --job-name=v11m6n_ligdep_mg60
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/v11m6n_ligdep_mg60_%A_%a.out
#SBATCH --error=logs/v11m6n_ligdep_mg60_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m6n_moment_g60seeds_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"

RUN_NAMES=(
  "qknorm_a14_tiny3e4_g60"
  "seed43_grad060_no_gate"
  "seed44_grad060_no_gate"
  "seed45_grad060_no_gate"
  "seed46_grad060_no_gate"
)

CKPTS=(
  "outputs/pocket_gvp_fusion_qknorm_a14_tiny3e4_g60_v11m6n_20260531_091911/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed43_grad060_no_gate_v11m6n_20260530_142644/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_grad060_no_gate_v11m6n_20260530_142644/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed45_grad060_no_gate_v11m6n_20260530_142644/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed46_grad060_no_gate_v11m6n_20260530_142644/best_f1_c0_model.pt"
)

MODES=("all" "source_split" "feature_split")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_IDX=$(( TASK_ID / 3 ))
MODE_IDX=$(( TASK_ID % 3 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
CKPT="${CKPTS[$RUN_IDX]}"
MODE="${MODES[$MODE_IDX]}"

if [ -z "${RUN_NAME}" ] || [ -z "${CKPT}" ] || [ -z "${MODE}" ]; then
  echo "ERROR: invalid task mapping for task id ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run: ${RUN_NAME}"
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
