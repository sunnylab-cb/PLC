#!/usr/bin/env bash
# CPU-only: scan all 647 VAL protein/ESM/label axes and four failed TRAIN rows.
# Export metadata for source PDB/SDF verification locally. Does NOT alter labels.
#SBATCH --job-name=strict_label_axis
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
EXPERIMENT="${EXPERIMENT:-/hpc2hdd/home/shuang886/PLC/experiments/strict_assignment_20260919}"
CODE="${CODE:-${EXPERIMENT}/code_label_fix}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
"${PYTHON}" scripts/diagnostics/prepare_internal_label_alignment.py request \
  --root /hpc2hdd/home/shuang886/PLC/datasets/pdbbind_biolip_chain_si30_final \
  --audit-csv "${EXPERIMENT}/negative_12804060/samples.csv" \
  --output "${EXPERIMENT}/label_axis_request.json"
