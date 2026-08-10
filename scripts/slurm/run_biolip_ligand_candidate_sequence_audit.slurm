#!/usr/bin/env bash
# Pre-filter BioLip pairs that have protein V4 + chain ESM2 but lack ligand V6.
#
# The input currently contains about 38k regular-organic pairs. Before spending
# CPU time on conformer/chemistry extraction, this job removes:
#   1. direct PDBbind val/test PDB IDs recorded in the manifest;
#   2. direct COACH420/HOLO4K PDB IDs;
#   3. sequence homologs to PDBbind val/test and COACH420/HOLO4K at
#      >=30% identity over >=80% of the candidate BioLip chain.
#
# It only writes FASTA, hit tables, and clean TSV manifests. Ligand extraction
# is deliberately a separate downstream job after retained-size review.

#SBATCH --job-name=biolip_ligcand_qc
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_ligcand_qc_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_ligcand_qc_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip}"
QC_DIR="${QC_DIR:-${BIOLIP}/qc_training_manifest_20260811}"
OUTPUT="${OUTPUT:-${BIOLIP}/qc_ligand_candidates_sequence_clean_20260811}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

module load biological/mmseqs

echo "BioLip ligand-candidate sequence prefilter"
echo "  manifest=${QC_DIR}/biolip_needs_ligand_extraction.tsv"
echo "  esm=${BIOLIP}/esm2_features"
echo "  output=${OUTPUT}"

"${PYTHON}" -u scripts/diagnostics/audit_biolip_sequence_leakage.py \
  --candidate-manifest "${QC_DIR}/biolip_needs_ligand_extraction.tsv" \
  --biolip-esm-dir "${BIOLIP}/esm2_features" \
  --exclude-direct-pdbbind \
  --exclude-pdb-ids "${QC_DIR}/exclude_coach420_holo4k_pdb_ids.txt" \
  --pdbbind-data-root "${PROJECT}/data" \
  --pdbbind-split-file "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "$(command -v mmseqs)" \
  --min-seq-id 0.30 \
  --query-coverage 0.80 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

echo "BioLip ligand-candidate sequence prefilter completed."
