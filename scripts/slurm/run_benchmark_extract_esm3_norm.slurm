#!/bin/bash
# Extract and normalize ESM3 features for COACH420/HOLO4K benchmark.
#
# Output:
#   benchmark/esm3_latents/{coach420,holo4k}/{protein}.pt
#   benchmark/esm3_latents_norm/{coach420,holo4k}/{protein}.pt
#
# The benchmark inference script aligns ESM3 to ESM2 by residue keys, so these
# files must preserve chain_id/residue_index/insertion_code metadata.

#SBATCH --job-name=bench_e3
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_esm3_extract_%j.out
#SBATCH --error=logs/benchmark_esm3_extract_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PDB_ROOT="${PDB_ROOT:-benchmark/p2rank-datasets-fixed_hres}"
RAW_OUT="${RAW_OUT:-benchmark/esm3_latents}"
NORM_OUT_DIR="${NORM_OUT_DIR:-esm3_latents_norm}"
STATS="${STATS:-data/stats_output.pt}"
DEVICES="${DEVICES:-0}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "PDB root: ${PDB_ROOT}"
echo "Raw ESM3 out: ${RAW_OUT}"
echo "Normalized out: benchmark/${NORM_OUT_DIR}"
echo "Stats: ${STATS}"
echo "Devices: ${DEVICES}"

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

if [ ! -d "${PDB_ROOT}" ]; then
  echo "ERROR: PDB root not found: ${PDB_ROOT}" >&2
  exit 2
fi
if [ ! -f "${STATS}" ]; then
  echo "ERROR: stats file not found: ${STATS}" >&2
  exit 3
fi

echo "Input PDB counts:"
find "${PDB_ROOT}/coach420" -maxdepth 1 -type f -name "*.pdb" | wc -l
find "${PDB_ROOT}/holo4k" -maxdepth 1 -type f -name "*.pdb" | wc -l

python preparation/protein/esm3_extract_latents.py \
  --input "${PDB_ROOT}" \
  --pattern "*.pdb" \
  --output "${RAW_OUT}" \
  --devices "${DEVICES}"

python scripts/apply_feature_normalization.py \
  --data-root benchmark \
  --stats "${STATS}" \
  --output-root benchmark \
  --features esm3 \
  --all \
  --esm3-in-dir "$(basename "${RAW_OUT}")" \
  --esm3-out-dir "${NORM_OUT_DIR}" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Output ESM3 counts:"
find "${RAW_OUT}/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${RAW_OUT}/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
find "benchmark/${NORM_OUT_DIR}/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "benchmark/${NORM_OUT_DIR}/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
echo "End: $(date)"
