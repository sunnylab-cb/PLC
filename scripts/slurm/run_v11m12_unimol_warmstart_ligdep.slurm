#!/bin/bash
# V11m12 pure-late UniMol warm-start ligand-dependency evaluation.
#
# Purpose:
#   Test whether the small Val bonus from pure-late UniMol memory is genuinely
#   ligand-aware, not just validation calibration/noise.
#
# Candidates:
#   - groupgate_lr2e4: stable control; little Val gain.
#   - mem1b2_lr2e4/lr5e4: completed positive memory probes.
#   - mem4b2_lr2e4/lr5e4: strongest running memory probes.
#
# Mode groups:
#   - source_split includes real/shuffle/zero plus UniMol-vs-graph decoupling:
#       shuffle_graph_keep_unimol, keep_graph_shuffle_unimol,
#       zero_ligand_graph_only, zero_unimol_only
#   - feature_split checks local54 vs global55 ligand feature dependence.
#
# Recommended submit:
#   sbatch --dependency=afterok:9827213 scripts/slurm/run_v11m12_unimol_warmstart_ligdep.slurm

#SBATCH --job-name=v11m12_udep
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/v11m12_udep_%A_%a.out
#SBATCH --error=logs/v11m12_udep_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m12_unimol_warmstart_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="${SPLIT:-test}"

VARIANTS=(
  "seed44_u_groupgate_lr2e4"
  "seed44_u_mem1b2_lr2e4"
  "seed44_u_mem1b2_lr5e4"
  "seed44_u_mem4b2_lr2e4"
  "seed44_u_mem4b2_lr5e4"
)

MODE_GROUPS=(
  "source_split"
  "feature_split"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_MODES="${#MODE_GROUPS[@]}"
VARIANT_INDEX=$((TASK_ID / N_MODES))
MODE_INDEX=$((TASK_ID % N_MODES))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
LIGAND_MODE="${MODE_GROUPS[$MODE_INDEX]}"

if [ -z "${VARIANT}" ] || [ -z "${LIGAND_MODE}" ]; then
  echo "ERROR: invalid V11m12 UniMol warm-start ligdep task id: ${TASK_ID}"
  echo "VARIANT_INDEX=${VARIANT_INDEX}, MODE_INDEX=${MODE_INDEX}"
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

RUN_DIR="$(ls -dt "outputs/pocket_gvp_fusion_${VARIANT}_v11m12_"* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: no output directory found for variant: ${VARIANT}"
  exit 3
fi

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  echo "If training is still running, submit this script with --dependency=afterok:<training_job_id>."
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
