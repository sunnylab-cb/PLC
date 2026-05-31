#!/bin/bash
# Runs focused mechanism ablation for the current main candidate:
#   V11m6n grad060_no_gate / best_f1_c0_model.pt
#
# This array evaluates test split modes that split local54 atom chemistry and
# ligand graph contributions:
#   real
#   zero_ligand_local54_only
#   zero_ligand_local54_base33
#   zero_ligand_local54_electronic3
#   zero_ligand_local54_sasa_hbond4
#   zero_ligand_local54_flex_rot5
#   zero_ligand_local54_metal4
#   zero_ligand_local54_stereo5
#   zero_ligand_edge_attr_only
#   shuffle_ligand_topology_only
#
# Purpose:
#   identify which part of local54/ligand graph is carrying the residue-level
#   ligand-aware signal before designing V11m7.

#SBATCH --job-name=v11m6n_l54sub
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/v11m6n_l54sub_%A_%a.out
#SBATCH --error=logs/v11m6n_l54sub_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m6n_grad060_local54_subgroups_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"
RUN_NAME="grad060_no_gate"
CKPT="outputs/pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523/best_f1_c0_model.pt"

MODES=(
  "real"
  "zero_ligand_local54_only"
  "zero_ligand_local54_base33"
  "zero_ligand_local54_electronic3"
  "zero_ligand_local54_sasa_hbond4"
  "zero_ligand_local54_flex_rot5"
  "zero_ligand_local54_metal4"
  "zero_ligand_local54_stereo5"
  "zero_ligand_edge_attr_only"
  "shuffle_ligand_topology_only"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODE="${MODES[$TASK_ID]}"

if [ -z "${MODE}" ]; then
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
