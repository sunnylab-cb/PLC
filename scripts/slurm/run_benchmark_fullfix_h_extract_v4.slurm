#!/bin/bash
# Build a feature-only fixed benchmark with missing residues + hydrogens,
# then extract protein V4 features from it. If PDBFixer fails while
# inserting missing residues for an edge-case structure, retry that
# structure with hydrogens only so benchmark feature extraction can finish.
#
# Input:
#   benchmark/p2rank-datasets/{coach420,holo4k}/*.pdb
#   benchmark/p2rank-datasets/{coach420,holo4k}.ds
#
# Fixed PDB output:
#   benchmark/p2rank-datasets-fixed_hres/{coach420,holo4k}/*.pdb
#
# Feature output:
#   benchmark/features_fixed_hres/protein_v4/{coach420,holo4k}/{id}/{id}_protein.pt

#SBATCH --job-name=bench_hresv4
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=logs/benchmark_fullfix_h_extract_v4_%j.out
#SBATCH --error=logs/benchmark_fullfix_h_extract_v4_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
RAW_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets"
FIXED_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets-fixed_hres"
FEATURE_ROOT="${PROJECT_DIR}/benchmark/features_fixed_hres/protein_v4"
STATS_JSON="${PROJECT_DIR}/configs/protein_v4_stats_new_test.json"
RESULT_DIR="${PROJECT_DIR}/results/benchmark_fullfix_hres_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${FIXED_ROOT}" "${FEATURE_ROOT}/coach420" "${FEATURE_ROOT}/holo4k" "${RESULT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Raw input: ${RAW_ROOT}"
echo "H/full fixed output: ${FIXED_ROOT}"
echo "Feature output: ${FEATURE_ROOT}"
echo "Stats: ${STATS_JSON}"

cd "${PROJECT_DIR}"

echo "Running full PDBFixer policy: add missing residues + add hydrogens..."
conda run -n af3 python -u scripts/benchmark/fix_p2rank_benchmark_pdbs.py \
  --input-root "${RAW_ROOT}" \
  --output-root "${FIXED_ROOT}" \
  --manifest "${RAW_ROOT}/coach420.ds" \
  --manifest "${RAW_ROOT}/holo4k.ds" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --add-missing-residues \
  --add-hydrogens \
  --fallback-no-missing-residues-on-fail \
  --status-jsonl "${RESULT_DIR}/status.jsonl"

echo "Extracting coach420 H/full-fixed protein V4 features..."
conda run -n af3 python -u preparation/protein/extract_complete_features_v4.py extract \
  --input "${FIXED_ROOT}/coach420" \
  --output "${FEATURE_ROOT}/coach420" \
  --stats-json "${STATS_JSON}" \
  --mode complex \
  --pattern "*.pdb" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Extracting holo4k H/full-fixed protein V4 features..."
conda run -n af3 python -u preparation/protein/extract_complete_features_v4.py extract \
  --input "${FIXED_ROOT}/holo4k" \
  --output "${FEATURE_ROOT}/holo4k" \
  --stats-json "${STATS_JSON}" \
  --mode complex \
  --pattern "*.pdb" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "End: $(date)"
echo "Fixed PDB counts:"
find "${FIXED_ROOT}/coach420" -maxdepth 1 -type f -name "*.pdb" | wc -l
find "${FIXED_ROOT}/holo4k" -maxdepth 1 -type f -name "*.pdb" | wc -l
echo "Feature counts:"
find "${FEATURE_ROOT}/coach420" -type f -name "*.pt" | wc -l
find "${FEATURE_ROOT}/holo4k" -type f -name "*.pt" | wc -l
