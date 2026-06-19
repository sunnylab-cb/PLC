#!/bin/bash
# Benchmark an existing V11m15 checkpoint without retraining.
#
# Use this after assignment probe training finishes but post-train benchmark
# fails or needs to be repeated.  It runs COACH420/HOLO4K in the current
# per-ligand, no atom cap setting and writes PDR/LAA/BP metrics.

#SBATCH --job-name=v11m15_eval
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v11m15_eval_%j.out
#SBATCH --error=logs/v11m15_eval_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
CKPT="${CKPT:-outputs/pocket_gvp_fusion_seed44_assign_inbatch_embed_lam005_v11m15_20260618_231733/best_f1_c0_model.pt}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT_DIR}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_DIR}/data/stats_output.pt}"
RUN_NAME="${RUN_NAME:-v11m15_assign_inbatch_embed_lam005}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/v11m15_existing_benchmark_${RUN_NAME}_${SLURM_JOB_ID}}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT_DIR}"

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

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 2
fi
if [ ! -f "${FEATURE_STATS}" ]; then
  echo "ERROR: feature stats not found: ${FEATURE_STATS}" >&2
  exit 3
fi

python -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${CKPT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --feature-stats "${FEATURE_STATS}" \
  --datasets coach420,holo4k \
  --tag-mode per_ligand \
  --ligand-mode real \
  --output "${OUT_DIR}" \
  --batch-size 2 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --max-seq-len 1800 \
  --max-atoms -1 \
  --amp

echo "End: $(date)"
cat "${OUT_DIR}/coach420_summary.json"
cat "${OUT_DIR}/holo4k_summary.json"
