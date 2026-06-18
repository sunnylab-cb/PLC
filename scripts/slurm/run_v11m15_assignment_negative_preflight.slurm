#!/bin/bash
# Preflight for V11m15 assignment-loss experiments.
#
# Goal:
#   Before adding same-protein hard negatives, verify that the PDBbind training
#   split has enough exact residue-order groups where another ligand pocket can
#   safely be used as a negative mask on the current residue logits.
#
# Output:
#   results/v11m15_assignment_preflight_${SLURM_JOB_ID}/assignment_negative_preflight.json

#SBATCH --job-name=assign_pf
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/assign_preflight_%j.out
#SBATCH --error=logs/assign_preflight_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
CONFIG="${CONFIG:-${PROJECT_DIR}/outputs/pocket_gvp_fusion_seed44_u_mem4b2_lr2e4_v11m12_20260611_111611/config.yaml}"
SPLITS="${SPLITS:-${PROJECT_DIR}/outputs/pocket_gvp_fusion_seed44_u_mem4b2_lr2e4_v11m12_20260611_111611/splits.txt}"
OUT_DIR="${PROJECT_DIR}/results/v11m15_assignment_preflight_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Config: ${CONFIG}"
echo "Splits: ${SPLITS}"
echo "Output: ${OUT_DIR}"

test -f "${CONFIG}"
test -f "${SPLITS}"

conda run -n af3 python -u scripts/diagnostics/analyze_assignment_negatives.py \
  --config "${CONFIG}" \
  --splits "${SPLITS}" \
  --split train \
  --output "${OUT_DIR}/assignment_negative_preflight.json" \
  --jaccard-thresh 0.25 \
  --centroid-min-dist 7.0 \
  --overlap-count-thresh 4

echo "End: $(date)"
