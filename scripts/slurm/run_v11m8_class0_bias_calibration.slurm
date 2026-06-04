#!/bin/bash
# Inference-only class0/contact logit-bias calibration for completed V11m8 runs.
#
# Purpose:
#   V11m8 all_light improves ligand-aware recall, but precision/recall balance
#   varies by seed.  This runs a val-selected additive class-0 logit bias and
#   then evaluates test real/shuffle/zero with the selected fixed bias.
#
# This is post-hoc calibration only.  It does not change training or checkpoint
# weights, and should be judged by test real F1 plus real-shuffle gap.

#SBATCH --job-name=v11m8_c0cal
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m8_c0cal_%A_%a.out
#SBATCH --error=logs/v11m8_c0cal_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/v11m8_class0_bias_calibration_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BIAS_GRID="${BIAS_GRID:--0.80,-0.75,-0.70,-0.65,-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80}"

VARIANTS=(
  "all_light_g60"
  "seed43_all_light_g60"
  "seed44_all_light_g60"
  "seed46_all_light_g60"
  "seed42_all_light_center010_g60"
  "seed42_all_light_nocenter_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m8 calibration task id: ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
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

RUN_DIR="$(ls -dt "outputs/pocket_gvp_fusion_${VARIANT}_v11m8_"* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: no output directory found for variant: ${VARIANT}"
  exit 3
fi

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 4
fi

OUT_DIR="${OUT_ROOT}/${VARIANT}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1

CMD=(
  python scripts/infer/sweep_v11_class0_bias.py
  --checkpoint "${CKPT}"
  --output "${OUT_DIR}"
  "--biases=${BIAS_GRID}"
  --select_metric global_f1_c0
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_samples "${MAX_SAMPLES}"
  --num_save_rows 1
  --amp
)

echo "Run dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
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
