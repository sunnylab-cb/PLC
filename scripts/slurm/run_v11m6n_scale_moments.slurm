#!/bin/bash
#SBATCH --job-name=v11m6n_moments
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v11m6n_scale_moments_%A_%a.out
#SBATCH --error=logs/v11m6n_scale_moments_%A_%a.err

# Extract numeric-scale moments from already trained V11m6/V11m6n checkpoints.
#
# Purpose:
#   We want initialization priors based on numeric scale distributions, not
#   hard-coded attention patterns.  This job measures trained-model moments:
#     - base/final/delta logits mean/std/quantiles
#     - effective residual delta magnitude relative to base logits
#     - local_delta QK score distribution and attention entropy/top1
#     - learned delta residual scale and QK alpha
#     - sampled train label priors
#
# How to use the output:
#   Use train-split moments to set future initialization ranges for:
#     local_delta_final_init_std,
#     local_delta_logit_scale_init,
#     local_delta_qk_norm_scale_init,
#     optional output/delta bias priors.
#   Do not use val/test moments to choose priors; those are only sanity checks.
#
# Array variants:
#   0 grad05_no_gate: stable protected no-detach baseline at grad_scale=0.50.
#   1 grad055_no_gate: left-mid grad-scale probe.
#   2 grad060_no_gate: current strongest seed42 ligand-aware candidate.
#   3 grad065_no_gate: right-mid grad-scale probe.
#   4 grad070_no_gate: over-strong protein-gradient control.
#   5 seed44_grad05_no_gate: high-val seed at grad_scale=0.50.
#   6 v11m6_no_detach_protein: full no-detach brittle/high-zero-gap control.

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
RESULT_DIR="${RESULT_DIR:-results/v11m6n_scale_moments_${SLURM_ARRAY_JOB_ID:-manual}}"
SPLIT="${SPLIT:-train}"
MAX_BATCHES="${MAX_BATCHES:-48}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="grad05_no_gate"
    CKPT="outputs/pocket_gvp_fusion_grad05_no_gate_v11m6n_20260528_195453/best_f1_c0_model.pt"
    ;;
  1)
    RUN_NAME="grad055_no_gate"
    CKPT="outputs/pocket_gvp_fusion_grad055_no_gate_v11m6n_20260529_132503/best_f1_c0_model.pt"
    ;;
  2)
    RUN_NAME="grad060_no_gate"
    CKPT="outputs/pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523/best_f1_c0_model.pt"
    ;;
  3)
    RUN_NAME="grad065_no_gate"
    CKPT="outputs/pocket_gvp_fusion_grad065_no_gate_v11m6n_20260529_132512/best_f1_c0_model.pt"
    ;;
  4)
    RUN_NAME="grad070_no_gate"
    CKPT="outputs/pocket_gvp_fusion_grad070_no_gate_v11m6n_20260529_202523/best_f1_c0_model.pt"
    ;;
  5)
    RUN_NAME="seed44_grad05_no_gate"
    CKPT="outputs/pocket_gvp_fusion_seed44_grad05_no_gate_v11m6n_20260529_132503/best_f1_c0_model.pt"
    ;;
  6)
    RUN_NAME="v11m6_no_detach_protein"
    CKPT="outputs/pocket_gvp_fusion_no_detach_protein_v11m6_20260528_010519/best_f1_c0_model.pt"
    ;;
  *)
    echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"
echo "Split: ${SPLIT}"
echo "Max batches: ${MAX_BATCHES}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Partition: i64m1tga800u"
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
mkdir -p "${RESULT_DIR}"

echo "Python: $(which python)"
python --version

echo "GPU devices:"
if command -v rocm-smi &> /dev/null; then
  rocm-smi
elif command -v nvidia-smi &> /dev/null; then
  nvidia-smi
else
  echo "No rocm-smi/nvidia-smi found"
fi

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 3
fi

OUT_JSON="${RESULT_DIR}/${RUN_NAME}_${SPLIT}_moments.json"
OUT_CSV="${RESULT_DIR}/summary_${SPLIT}.csv"

CMD=(
  python scripts/analysis/summarize_v11m6n_scale_moments.py
  --checkpoint "${CKPT}"
  --split "${SPLIT}"
  --max-batches "${MAX_BATCHES}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --amp
  --output-json "${OUT_JSON}"
  --output-csv "${OUT_CSV}"
)

echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "Output JSON: ${OUT_JSON}"
echo "Output CSV: ${OUT_CSV}"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
