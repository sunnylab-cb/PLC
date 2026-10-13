#!/bin/bash
# Extract V5 ligand features for benchmark ligands after SDF QC.
#
# Inputs:
#   benchmark/ligands_sdf/{coach420,holo4k}/*.sdf
#
# Outputs:
#   benchmark/ligand_features_v5/{coach420,holo4k}/*.pt
#
# Policy:
#   - flat SDF directory mode, one PT per ligand SDF key
#   - 109D V5 atom features: 54D local + 45D global + 10D shape
#   - 50 conformers to match the current V5 training feature policy
#   - loose mode for robustness on benchmark ligand edge cases

#SBATCH --job-name=bench_ligv5
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_extract_ligand_v5_%j.out
#SBATCH --error=logs/benchmark_extract_ligand_v5_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
SDF_ROOT="${PROJECT_DIR}/benchmark/ligands_sdf"
FEATURE_ROOT="${PROJECT_DIR}/benchmark/ligand_features_v5"
RESULT_DIR="${PROJECT_DIR}/results/benchmark_ligand_v5_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${FEATURE_ROOT}/coach420" "${FEATURE_ROOT}/holo4k" "${RESULT_DIR}"

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "SDF root: ${SDF_ROOT}"
echo "Feature root: ${FEATURE_ROOT}"

echo "Input SDF counts:"
find "${SDF_ROOT}/coach420" -maxdepth 1 -type f -name "*.sdf" | wc -l
find "${SDF_ROOT}/holo4k" -maxdepth 1 -type f -name "*.sdf" | wc -l

echo "Extracting COACH420 V5 ligand features..."
conda run -n af3 python -u scripts/benchmark/ligand/extract_ligand_features_v5.py \
  --input "${SDF_ROOT}/coach420" \
  --output "${FEATURE_ROOT}/coach420" \
  --flat \
  --conformers 50 \
  --loose \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --rdkit-threads 1

echo "Extracting HOLO4K V5 ligand features..."
conda run -n af3 python -u scripts/benchmark/ligand/extract_ligand_features_v5.py \
  --input "${SDF_ROOT}/holo4k" \
  --output "${FEATURE_ROOT}/holo4k" \
  --flat \
  --conformers 50 \
  --loose \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --rdkit-threads 1

echo "End: $(date)"
echo "Feature counts:"
find "${FEATURE_ROOT}/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${FEATURE_ROOT}/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
