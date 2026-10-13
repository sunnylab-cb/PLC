#!/usr/bin/env bash
# Read-only PDBbind exclusion breakdown + BioLiP candidate tensor/mask audit.
# Writes isolated candidate manifests, NOT approved training IDs. No GPU,
# feature extraction, MMseqs search, model training or production split changes.
#SBATCH --job-name=training_coverage_qc
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated output root}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
BIO=/hpc2hdd/home/shuang886/biolip
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "job=${SLURM_JOB_ID} read-only coverage/QC, no training"
"${PYTHON}" -u "${EXPERIMENT}/scripts/diagnostics/audit_pdbbind_coverage.py" \
  --full-split "${EXPERIMENT}/preparation/dataset_split/mmseqs_splits_strict.txt" \
  --legacy-split "${EXPERIMENT}/configs/splits/splits.txt" \
  --legacy-exclude "${EXPERIMENT}/configs/splits/excluded_samples.txt" \
  --data-root /hpc2hdd/home/shuang886/jhspoolers/PLC/data \
  --chain-audit-root "${BIO}/pdbbind_chain_si30_split_11283646" \
  --final-root "${BIO}/pdbbind_chain_si30_split_11283646_final" \
  --output "${EXPERIMENT}/pdbbind_coverage.json"
"${PYTHON}" -u "${EXPERIMENT}/scripts/diagnostics/audit_biolip_expansion_tensors.py" \
  --biolip-root "${BIO}" \
  --split-root "${BIO}/pdbbind_chain_si30_split_11283646" \
  --final-root "${BIO}/pdbbind_chain_si30_split_11283646_final" \
  --output-dir "${EXPERIMENT}/biolip_tensor_qc"
echo COVERAGE_AUDIT_COMPLETE
