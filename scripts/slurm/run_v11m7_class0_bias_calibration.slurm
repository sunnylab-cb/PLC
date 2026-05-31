#!/bin/bash
# Class0/contact logit-bias calibration for existing V11m6n checkpoints.
#
# What this runs:
#   - Sweep additive class-0 logit bias on VAL real mode.
#   - Select the bias with best VAL global_f1_c0.
#   - Re-run TEST real/shuffle/zero with the selected fixed bias.
#
# Why:
#   qknorm showed higher precision but lower recall; grad060 seeds showed better
#   recall but lower precision.  This checks whether inference-only calibration
#   can improve F1 while preserving real-shuffle ligand dependency.

#SBATCH --job-name=v11m7_c0cal
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m7_c0cal_%A_%a.out
#SBATCH --error=logs/v11m7_c0cal_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/v11m7_class0_bias_calibration_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BIAS_GRID="${BIAS_GRID:--0.80,-0.75,-0.70,-0.65,-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80}"

RUN_NAMES=(
  "grad060_no_gate_seed42"
  "qknorm_a14_tiny3e4_g60"
)

CKPTS=(
  "outputs/pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_qknorm_a14_tiny3e4_g60_v11m6n_20260531_091911/best_f1_c0_model.pt"
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

OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1

CMD=(
  python scripts/infer/sweep_v11_class0_bias.py
  --checkpoint "${CKPT}"
  --output "${OUT_DIR}"
  --biases "${BIAS_GRID}"
  --select_metric global_f1_c0
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_samples "${MAX_SAMPLES}"
  --num_save_rows 1
  --amp
)

echo "Python: $(which python)"
python --version
echo "Command: ${CMD[*]}"
echo "============================================"
"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
