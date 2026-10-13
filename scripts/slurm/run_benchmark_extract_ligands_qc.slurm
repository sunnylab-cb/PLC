#!/bin/bash
# Extract ligand SDF files for COACH420/HOLO4K and run QC only.
# This job intentionally does not extract V5 ligand features yet. The next
# feature job should be launched only after QC confirms coverage/readability.

#SBATCH --job-name=bench_ligsdf
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=logs/benchmark_extract_ligands_qc_%j.out
#SBATCH --error=logs/benchmark_extract_ligands_qc_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
RAW_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets"
SDF_ROOT="${PROJECT_DIR}/benchmark/ligands_sdf"
QC_ROOT="${PROJECT_DIR}/results/benchmark_ligand_sdf_qc_${SLURM_JOB_ID}"
CCD_CACHE="${SDF_ROOT}/_ccd_cache.json"

mkdir -p "${PROJECT_DIR}/logs" "${SDF_ROOT}/coach420" "${SDF_ROOT}/holo4k" "${QC_ROOT}"

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Raw root: ${RAW_ROOT}"
echo "SDF root: ${SDF_ROOT}"
echo "QC root: ${QC_ROOT}"

echo "Extracting COACH420 ligand SDF..."
conda run -n af3 python -u scripts/benchmark/ligand/extract_ligands_from_pdb.py \
  --input "${RAW_ROOT}/coach420" \
  --output "${SDF_ROOT}/coach420" \
  --mlig-ds "${RAW_ROOT}/coach420(mlig).ds" \
  --pdb-ds "${RAW_ROOT}/coach420.ds" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --bond-timeout 8 \
  --ccd-cache "${CCD_CACHE}"

echo "Extracting HOLO4K ligand SDF..."
conda run -n af3 python -u scripts/benchmark/ligand/extract_ligands_from_pdb.py \
  --input "${RAW_ROOT}/holo4k" \
  --output "${SDF_ROOT}/holo4k" \
  --mlig-ds "${RAW_ROOT}/holo4k(mlig).ds" \
  --pdb-ds "${RAW_ROOT}/holo4k.ds" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --bond-timeout 8 \
  --ccd-cache "${CCD_CACHE}"

echo "QC COACH420 ligand SDF..."
conda run -n af3 python -u scripts/benchmark/qc_ligand_sdf.py \
  --dataset coach420 \
  --sdf-dir "${SDF_ROOT}/coach420" \
  --pdb-ds "${RAW_ROOT}/coach420.ds" \
  --mlig-ds "${RAW_ROOT}/coach420(mlig).ds" \
  --output-dir "${QC_ROOT}"

echo "QC HOLO4K ligand SDF..."
conda run -n af3 python -u scripts/benchmark/qc_ligand_sdf.py \
  --dataset holo4k \
  --sdf-dir "${SDF_ROOT}/holo4k" \
  --pdb-ds "${RAW_ROOT}/holo4k.ds" \
  --mlig-ds "${RAW_ROOT}/holo4k(mlig).ds" \
  --output-dir "${QC_ROOT}"

echo "End: $(date)"
echo "SDF counts:"
find "${SDF_ROOT}/coach420" -maxdepth 1 -type f -name "*.sdf" | wc -l
find "${SDF_ROOT}/holo4k" -maxdepth 1 -type f -name "*.sdf" | wc -l
echo "QC summaries:"
ls -lh "${QC_ROOT}"/*_summary.json
