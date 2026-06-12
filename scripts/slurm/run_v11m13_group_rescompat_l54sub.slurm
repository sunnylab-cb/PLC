#!/bin/bash
# Local54 subgroup and ligand-edge topology scan for V11m13 group_rescompat.
#
# Purpose:
#   group_rescompat_lr2e4 is the current best V11m13 single-seed test result
#   (test real F1_c0 ~= 0.6158, real-shuffle gap ~= 0.080).  This job runs the
#   full local54_subgroups dependency bundle to identify which local chemistry
#   groups and graph controls carry the gain:
#     real
#     zero_ligand_local54_only
#     zero_ligand_local54_base33
#     zero_ligand_local54_electronic3
#     zero_ligand_local54_sasa_hbond4
#     zero_ligand_local54_flex_rot5
#     zero_ligand_local54_metal4
#     zero_ligand_local54_stereo5
#     zero_ligand_edge_attr_only
#     shuffle_ligand_topology_only

#SBATCH --job-name=v11m13_l54sub
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/v11m13_l54sub_%j.out
#SBATCH --error=logs/v11m13_l54sub_%j.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
RUN_NAME="group_rescompat_lr2e4"
RUN_DIR="outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114"
CKPT="${RUN_DIR}/best_f1_c0_model.pt"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m13_group_rescompat_l54sub_${SLURM_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="test"
LIGAND_MODE="local54_subgroups"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Split: ${SPLIT}"
echo "Ligand mode: ${LIGAND_MODE}"
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

OUT_DIR="${OUT_ROOT}/${RUN_NAME}_${SPLIT}_${LIGAND_MODE}"
mkdir -p "${OUT_DIR}"

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
