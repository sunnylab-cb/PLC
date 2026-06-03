#!/bin/bash
# Ligand-dependency eval for V11m8 all_light follow-up probes.
#
# Purpose:
#   Convert the V11m8 follow-up training runs into directly comparable
#   test-split evidence: real/shuffle/zero plus graph/UniMol and local54/global55
#   ablations.  This is intentionally scheduled as a dependency after training.

#SBATCH --job-name=v11m8_foldev
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/v11m8_foldev_%A_%a.out
#SBATCH --error=logs/v11m8_foldev_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m8_all_light_followup_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"

VARIANTS=(
  "seed43_all_light_g60"
  "seed44_all_light_g60"
  "seed46_all_light_g60"
  "seed42_all_light_nocenter_g60"
  "seed42_all_light_center010_g60"
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
  echo "ERROR: invalid V11m8 follow-up eval task id: ${TASK_ID}"
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
