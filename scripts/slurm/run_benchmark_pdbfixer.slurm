#!/bin/bash
# Run conservative PDBFixer over P2Rank benchmark proteins.
#
# Inputs:
#   benchmark/p2rank-datasets/{coach420,holo4k}/*.pdb
#   benchmark/p2rank-datasets/{coach420,holo4k}.ds
#
# Outputs:
#   benchmark/p2rank-datasets-fixed/{coach420,holo4k}/*.pdb
#   results/benchmark_pdbfixer_${SLURM_JOB_ID}/status.jsonl
#
# Policy:
#   protein-only output; remove heterogens/water; replace non-standard residues;
#   add missing heavy atoms; do NOT add missing residues or hydrogens by default.

#SBATCH --job-name=bench_pdbfix
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_pdbfixer_%j.out
#SBATCH --error=logs/benchmark_pdbfixer_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
INPUT_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets"
OUTPUT_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets-fixed"
RESULT_DIR="${PROJECT_DIR}/results/benchmark_pdbfixer_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${RESULT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_ROOT}"

cd "${PROJECT_DIR}"

conda run -n af3 python -u scripts/benchmark/fix_p2rank_benchmark_pdbs.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest "${INPUT_ROOT}/coach420.ds" \
  --manifest "${INPUT_ROOT}/holo4k.ds" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --status-jsonl "${RESULT_DIR}/status.jsonl"

echo "End: $(date)"
echo "Fixed PDB counts:"
find "${OUTPUT_ROOT}/coach420" -maxdepth 1 -type f -name '*.pdb' | wc -l
find "${OUTPUT_ROOT}/holo4k" -maxdepth 1 -type f -name '*.pdb' | wc -l
