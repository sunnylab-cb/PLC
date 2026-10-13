#!/usr/bin/env bash
# CPU-only, no model training/inference: reuse all 647 strict VAL predictions.
# Union the previous 19 proposal protocols, retain the original prediction,
# and compare raw vs boundary-preserving pools under two GT-free selectors.
# Do not touch test/COACH/HOLO, fit thresholds, or count oracle F1 as performance.
# Both frozen model arms are processed sequentially in this one CPU job.
#SBATCH --job-name=strict_val_hybrid
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_assignment_20260919}"
CODE="${CODE:-${EXPERIMENT}/code_hybrid}"
SOURCE="${SOURCE:-${EXPERIMENT}/val_12804287}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
OUT="${EXPERIMENT}/hybrid_${SLURM_JOB_ID:-manual}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
for name in base44 esm3gp44; do
  "${PYTHON}" -u scripts/diagnostics/strict_val_hybrid_candidates.py \
    --export "${SOURCE}/${name}" \
    --splits "${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final/splits.txt" \
    --output "${OUT}/${name}"
done
echo "STRICT_VAL_HYBRID_JOB_COMPLETE=${OUT}"
