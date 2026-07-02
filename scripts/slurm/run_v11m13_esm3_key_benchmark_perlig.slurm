#!/bin/bash
# Run the two promising ESM3 global-prior checkpoints on COACH420/HOLO4K as
# single protein-ligand pairs, without ligand atom truncation.
#
# Requires:
#   benchmark/esm3_latents_norm/{coach420,holo4k}/{id}.pt
# Generate it with scripts/slurm/run_benchmark_extract_esm3_norm.slurm.
#
# Array:
#   0 esm3_global_prior_fullfinetune_seed44
#   1 esm3_global_prior_scratch_seed44

#SBATCH --job-name=e3_bench
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/esm3_key_benchmark_%A_%a.out
#SBATCH --error=logs/esm3_key_benchmark_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BENCH_ROOT="${PROJECT_DIR}/benchmark"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_DIR}/data/stats_output.pt}"
ESM3_ROOT="${ESM3_ROOT:-esm3_latents_norm}"

RUN_NAMES=(
  "esm3_global_prior_fullfinetune_seed44"
  "esm3_global_prior_scratch_seed44"
)

CHECKPOINTS=(
  "outputs/pocket_gvp_fusion_seed44_esm3_global_prior_fullfinetune_v11m13e3ft_20260702_023232/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_esm3_global_prior_scratch_v11m13e3ft_20260702_034855/best_f1_c0_model.pt"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
CKPT="${PROJECT_DIR}/${CHECKPOINTS[$TASK_ID]}"
OUT_DIR="${PROJECT_DIR}/results/esm3_key_benchmark_perlig_noatom_${RUN_NAME}_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run name: ${RUN_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Checkpoint: ${CKPT}"
echo "Benchmark root: ${BENCH_ROOT}"
echo "Feature stats: ${FEATURE_STATS}"
echo "ESM3 root: benchmark/${ESM3_ROOT}"
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

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${FEATURE_STATS}" ]]; then
  echo "ERROR: feature stats not found: ${FEATURE_STATS}" >&2
  exit 1
fi
if [[ ! -d "${BENCH_ROOT}/${ESM3_ROOT}/coach420" || ! -d "${BENCH_ROOT}/${ESM3_ROOT}/holo4k" ]]; then
  echo "ERROR: normalized benchmark ESM3 features not found under ${BENCH_ROOT}/${ESM3_ROOT}" >&2
  exit 2
fi

echo "Preflight counts:"
printf "protein coach/holo: "
find "${BENCH_ROOT}/features_fixed_conservative/protein_v4/coach420" -type f -name "*_protein.pt" | wc -l
find "${BENCH_ROOT}/features_fixed_conservative/protein_v4/holo4k" -type f -name "*_protein.pt" | wc -l
printf "esm2 coach/holo: "
find "${BENCH_ROOT}/esm2/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${BENCH_ROOT}/esm2/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
printf "esm3 coach/holo: "
find "${BENCH_ROOT}/${ESM3_ROOT}/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${BENCH_ROOT}/${ESM3_ROOT}/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
printf "ligand coach/holo: "
find "${BENCH_ROOT}/ligand_features_v5/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${BENCH_ROOT}/ligand_features_v5/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
ls -lh "${BENCH_ROOT}/residue_ligand_tags/atom_h/"*_per_ligand_dmin.pt

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
  --esm3-root "${ESM3_ROOT}" \
  --amp

echo "End: $(date)"
echo "Summaries:"
cat "${OUT_DIR}/coach420_summary.json"
cat "${OUT_DIR}/holo4k_summary.json"
