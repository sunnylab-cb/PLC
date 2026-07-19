#!/bin/bash
# Test real/shuffle/zero for the 9977123 fixed-vs-trainable scale selectors.
#
# This closes the selector experiment: val ranking favors fixed scale, but the
# promotion decision requires test ligand dependency.  Evaluate both best-F1
# and best-AUPRC checkpoints because they occur at different epochs.

#SBATCH --job-name=e3scale_lgd
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/e3scale_ligdep_%A_%a.out
#SBATCH --error=logs/e3scale_ligdep_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/hpc2hdd/home/shuang886/PLC}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DECISION_MODE="${DECISION_MODE:-argmax}"
CONTACT_THRESHOLD="${CONTACT_THRESHOLD:-}"

RUN_NAMES=("scale_fixed" "scale_trainable")
RUN_DIRS=(
  "${ARCHIVE_ROOT}/outputs/pocket_gvp_fusion_seed44_esm3_scratch_scale_fixed_selectors_v11m13e3scale_20260715_015933"
  "${ARCHIVE_ROOT}/outputs/pocket_gvp_fusion_seed44_esm3_scratch_scale_trainable_selectors_v11m13e3scale_20260715_020902"
)
CHECKPOINT_NAMES=("best_f1_c0_model.pt" "best_auprc_c0_model.pt")
SELECTOR_NAMES=("best_f1" "best_auprc")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_IDX=$((TASK_ID / 2))
SELECTOR_IDX=$((TASK_ID % 2))
RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
RUN_DIR="${RUN_DIRS[$RUN_IDX]}"
CHECKPOINT_NAME="${CHECKPOINT_NAMES[$SELECTOR_IDX]}"
SELECTOR_NAME="${SELECTOR_NAMES[$SELECTOR_IDX]}"

if [ -z "${RUN_NAME}" ] || [ -z "${CHECKPOINT_NAME}" ]; then
  echo "ERROR: invalid task=${TASK_ID}"
  exit 2
fi

CKPT="${RUN_DIR}/${CHECKPOINT_NAME}"
CONFIG="${RUN_DIR}/config.yaml"
OUT_ROOT="${PROJECT_DIR}/results/ligdep_v11m13_esm3_scale_selector_${SLURM_ARRAY_JOB_ID:-manual}_${DECISION_MODE}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}_${SELECTOR_NAME}"

DECISION_ARGS=()
case "${DECISION_MODE}" in
  argmax)
    ;;
  checkpoint_gate)
    DECISION_ARGS+=(--use_checkpoint_gate)
    ;;
  threshold)
    if [ -z "${CONTACT_THRESHOLD}" ]; then
      echo "ERROR: CONTACT_THRESHOLD is required when DECISION_MODE=threshold"
      exit 5
    fi
    DECISION_ARGS+=(--contact_threshold "${CONTACT_THRESHOLD}")
    ;;
  *)
    echo "ERROR: DECISION_MODE must be argmax, checkpoint_gate, or threshold"
    exit 6
    ;;
esac

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

echo "run=${RUN_NAME} selector=${SELECTOR_NAME} checkpoint=${CKPT}"
echo "purpose=test real/shuffle/zero for fixed-vs-trainable local delta scale"
echo "decision_mode=${DECISION_MODE} contact_threshold=${CONTACT_THRESHOLD:-checkpoint/default}"
echo "start=$(date) node=${SLURMD_NODENAME} job=${SLURM_JOB_ID}_${TASK_ID}"

if command -v module &> /dev/null; then module purge || true; fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate af3

test -f "${CKPT}" || { echo "missing checkpoint: ${CKPT}"; exit 3; }
test -f "${CONFIG}" || { echo "missing config: ${CONFIG}"; exit 4; }

python scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${CKPT}" \
  --config "${CONFIG}" \
  --split test \
  --ligand_mode all \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_samples 0 \
  --shuffle_seed 42 \
  --num_save_rows 1 \
  --output "${OUT_DIR}" \
  "${DECISION_ARGS[@]}" \
  --amp

echo "end=$(date)"
