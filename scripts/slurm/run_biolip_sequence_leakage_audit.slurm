#!/usr/bin/env bash
# Sequence-homology leakage audit for the first BioLip V11 pilot.
#
# Query: every exact BioLip PDB-chain used by the 632-pair pilot.
# Targets: PDBbind val/test and all COACH420/HOLO4K ESM2 sequences.
# Rule: MMseqs identity >=30% over >=80% of the BioLip query chain.
# Output: best-hit tables plus a group-safe BioLip PDB exclusion list.
#
# This job is diagnostic only. It does not mutate the pilot or training data.

#SBATCH --job-name=biolip_seq_audit
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_seq_audit_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_seq_audit_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PILOT="${PILOT:-/hpc2hdd/home/shuang886/biolip/v11_pilot_strict_nobench_20260811}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/qc_sequence_leakage_20260811}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

module load biological/mmseqs

echo "BioLip sequence-homology leakage audit"
echo "  pilot=${PILOT}"
echo "  output=${OUTPUT}"
echo "  mmseqs=$(command -v mmseqs)"

"${PYTHON}" -u scripts/diagnostics/audit_biolip_sequence_leakage.py \
  --pilot-root "${PILOT}" \
  --pdbbind-data-root "${PROJECT}/data" \
  --pdbbind-split-file "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "$(command -v mmseqs)" \
  --min-seq-id 0.30 \
  --query-coverage 0.80 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

echo "BioLip sequence-homology leakage audit completed."
