#!/bin/bash
# Build atom+H fixed protein PDBs without missing-residue insertion, then
# generate residue-ligand distance tags for benchmark ligands.
#
# Protein policy:
#   - remove heterogens/water
#   - replace non-standard residues
#   - add missing heavy atoms
#   - add hydrogens
#   - do NOT insert missing residues
#
# Tag policy:
#   - use benchmark/ligands_sdf/{coach420,holo4k}/*.sdf as ligand coordinates
#   - write both per-ligand tags and aggregate-by-PDB tags

#SBATCH --job-name=bench_atomhtag
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_atomh_tags_%j.out
#SBATCH --error=logs/benchmark_atomh_tags_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
RAW_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets"
ATOMH_ROOT="${PROJECT_DIR}/benchmark/p2rank-datasets-fixed_atom_h"
SDF_ROOT="${PROJECT_DIR}/benchmark/ligands_sdf"
TAG_ROOT="${PROJECT_DIR}/benchmark/residue_ligand_tags/atom_h"
RESULT_DIR="${PROJECT_DIR}/results/benchmark_atomh_tags_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${ATOMH_ROOT}" "${TAG_ROOT}" "${RESULT_DIR}"

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Atom+H PDB root: ${ATOMH_ROOT}"
echo "SDF root: ${SDF_ROOT}"
echo "Tag root: ${TAG_ROOT}"

echo "Building atom+H protein-only PDBs without missing-residue insertion..."
conda run -n af3 python -u scripts/benchmark/fix_p2rank_benchmark_pdbs.py \
  --input-root "${RAW_ROOT}" \
  --output-root "${ATOMH_ROOT}" \
  --manifest "${RAW_ROOT}/coach420.ds" \
  --manifest "${RAW_ROOT}/holo4k.ds" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --add-hydrogens \
  --status-jsonl "${RESULT_DIR}/fix_status.jsonl"

echo "Generating per-ligand COACH420 tags..."
conda run -n af3 python -u scripts/benchmark/tag/calculate_distances_dcenter.py \
  --flat \
  --input "${ATOMH_ROOT}/coach420" \
  --ligand-dir "${SDF_ROOT}/coach420" \
  --pdb-ds "${RAW_ROOT}/coach420.ds" \
  --mlig-ds "${RAW_ROOT}/coach420(mlig).ds" \
  --output "${TAG_ROOT}/coach420_per_ligand_dmin.pt" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Generating per-ligand HOLO4K tags..."
conda run -n af3 python -u scripts/benchmark/tag/calculate_distances_dcenter.py \
  --flat \
  --input "${ATOMH_ROOT}/holo4k" \
  --ligand-dir "${SDF_ROOT}/holo4k" \
  --pdb-ds "${RAW_ROOT}/holo4k.ds" \
  --mlig-ds "${RAW_ROOT}/holo4k(mlig).ds" \
  --output "${TAG_ROOT}/holo4k_per_ligand_dmin.pt" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Generating aggregate-by-PDB COACH420 tags..."
conda run -n af3 python -u scripts/benchmark/tag/calculate_distances_dcenter.py \
  --flat \
  --input "${ATOMH_ROOT}/coach420" \
  --ligand-dir "${SDF_ROOT}/coach420" \
  --pdb-ds "${RAW_ROOT}/coach420.ds" \
  --mlig-ds "${RAW_ROOT}/coach420(mlig).ds" \
  --aggregate-by-pdb \
  --output "${TAG_ROOT}/coach420_aggregate_pdb_dmin.pt" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Generating aggregate-by-PDB HOLO4K tags..."
conda run -n af3 python -u scripts/benchmark/tag/calculate_distances_dcenter.py \
  --flat \
  --input "${ATOMH_ROOT}/holo4k" \
  --ligand-dir "${SDF_ROOT}/holo4k" \
  --pdb-ds "${RAW_ROOT}/holo4k.ds" \
  --mlig-ds "${RAW_ROOT}/holo4k(mlig).ds" \
  --aggregate-by-pdb \
  --output "${TAG_ROOT}/holo4k_aggregate_pdb_dmin.pt" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "End: $(date)"
echo "Atom+H PDB counts:"
find "${ATOMH_ROOT}/coach420" -maxdepth 1 -type f -name "*.pdb" | wc -l
find "${ATOMH_ROOT}/holo4k" -maxdepth 1 -type f -name "*.pdb" | wc -l
echo "Tag files:"
ls -lh "${TAG_ROOT}"/*.pt
