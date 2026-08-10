#!/bin/bash
# BioLip side-path preflight phase 2:
# load every strict, feature-ready protein-ligand pair and verify exact
# receptor-chain/ESM2 residue-key alignment plus finite V4/ESM2/ligand tensors.
# This is read-only QC; it does not rewrite any source feature file.
#SBATCH --job-name=biolip_tensor_qc
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_tensor_qc_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_tensor_qc_%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/shuang886/jhspoolers/PLC
BIOLIP=/hpc2hdd/home/shuang886/biolip
QC_DIR=${BIOLIP}/qc_training_manifest_20260811
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "BioLip tensor alignment QC"
echo "  host=$(hostname)"
echo "  manifest=${QC_DIR}/biolip_current_feature_ready.tsv"
echo "  output=${QC_DIR}/tensor_alignment_qc_strict.json"

${PYTHON} ${PROJECT}/scripts/diagnostics/inspect_biolip_feature_alignment.py \
  --biolip-root ${BIOLIP} \
  --manifest ${QC_DIR}/biolip_current_feature_ready.tsv \
  --output-json ${QC_DIR}/tensor_alignment_qc_strict.json \
  --max-samples 0 \
  --strict-no-pdbbind
