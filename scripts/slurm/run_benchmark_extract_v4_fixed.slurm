#!/bin/bash
# Extract protein V4 features from conservative PDBFixer outputs.
#
# Input:
#   benchmark/p2rank-datasets-fixed/{coach420,holo4k}/*.pdb
#
# Output:
#   benchmark/features_fixed_conservative/protein_v4/{coach420,holo4k}/{id}/{id}_protein.pt
#
# Uses the same V4 normalization stats as training.

#SBATCH --job-name=bench_v4fix
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_extract_v4_fixed_%j.out
#SBATCH --error=logs/benchmark_extract_v4_fixed_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
INPUT_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets-fixed"
OUTPUT_ROOT="${PROJECT_DIR}/benchmark/features_fixed_conservative/protein_v4"
STATS_JSON="${PROJECT_DIR}/configs/protein_v4_stats_new_test.json"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_ROOT}/coach420" "${OUTPUT_ROOT}/holo4k"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Stats: ${STATS_JSON}"

cd "${PROJECT_DIR}"

echo "Extracting coach420 fixed protein V4 features..."
conda run -n af3 python -u preparation/protein/extract_complete_features_v4.py extract \
  --input "${INPUT_ROOT}/coach420" \
  --output "${OUTPUT_ROOT}/coach420" \
  --stats-json "${STATS_JSON}" \
  --mode complex \
  --pattern "*.pdb" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Extracting holo4k fixed protein V4 features..."
conda run -n af3 python -u preparation/protein/extract_complete_features_v4.py extract \
  --input "${INPUT_ROOT}/holo4k" \
  --output "${OUTPUT_ROOT}/holo4k" \
  --stats-json "${STATS_JSON}" \
  --mode complex \
  --pattern "*.pdb" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "End: $(date)"
echo "Feature counts:"
find "${OUTPUT_ROOT}/coach420" -type f -name "*.pt" | wc -l
find "${OUTPUT_ROOT}/holo4k" -type f -name "*.pt" | wc -l
