#!/bin/bash
# Ligand-dependency eval for V11m6n grad060/no_gate + light ligand-source dropout.
#
# This script is intended to run after run_v11m6n_ligdrop_probe.slurm completes.
# It resolves each variant's newest output directory and evaluates:
#   all          -> real / shuffle / zero
#   source_split -> graph-vs-UniMol decoupling
#   feature_split -> local54-vs-global55 ablation

#SBATCH --job-name=v11m6n_ldeval
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/v11m6n_ldeval_%A_%a.out
#SBATCH --error=logs/v11m6n_ldeval_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m6n_ligdrop_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"

VARIANTS=(
  "seed42_ligdrop02_g60_no_gate"
  "seed42_ligdrop03_g60_no_gate"
  "seed45_ligdrop02_g60_no_gate"
  "seed45_ligdrop03_g60_no_gate"
)

MODE_GROUPS=(
  "all"
  "source_split"
  "feature_split"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT_INDEX=$((TASK_ID / 3))
MODE_INDEX=$((TASK_ID % 3))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
LIGAND_MODE="${MODE_GROUPS[$MODE_INDEX]}"

if [ -z "${VARIANT}" ] || [ -z "${LIGAND_MODE}" ]; then
  echo "ERROR: invalid ligdrop eval task id: ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Split: ${SPLIT}"
echo "Ligand mode group: ${LIGAND_MODE}"
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

RUN_DIR="$(ls -dt "outputs/pocket_gvp_fusion_${VARIANT}_v11m6n_"* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: no output directory found for variant: ${VARIANT}"
  exit 3
fi

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 4
fi

OUT_DIR="${OUT_ROOT}/${VARIANT}_${SPLIT}_${LIGAND_MODE}"
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

echo "Command: ${CMD[*]}"
echo "============================================"
"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
