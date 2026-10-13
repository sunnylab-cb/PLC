#!/usr/bin/env bash
# CPU-only audit of CURRENT strict merged TRAIN (10317 rows), never old PDBbind.
# Verify residue axes, chemical identity and rigid-coordinate compatibility.
# Distinguish same-axis upper bounds from conservative valid negatives; include
# zero-count rows and report PDBbind/BioLiP separately. Does not train or create
# a negative-mask artifact. Local PDBbind SDF identities are supplied by manifest.
#SBATCH --job-name=strict_train_neg
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_assignment_20260919}"
CODE="${CODE:-${EXPERIMENT}/code}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
LABEL_ARGS=()
if [[ -n "${LABEL_ALIGNMENT:-}" ]]; then LABEL_ARGS=(--label-alignment "${LABEL_ALIGNMENT}"); fi
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
"${PYTHON}" -u scripts/diagnostics/audit_strict_train_negatives.py \
  --root "${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final" \
  --chemistry-manifest "${EXPERIMENT}/train_chemistry.json" \
  --workers "${SLURM_CPUS_PER_TASK}" --output "${EXPERIMENT}/negative_${SLURM_JOB_ID:-manual}" "${LABEL_ARGS[@]}"
