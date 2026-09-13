#!/usr/bin/env bash
# CPU audit of every strict merged ESM3 tensor and ESM2 residue-key alignment.

#SBATCH --job-name=si30_e3_audit
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_audit_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_audit_%j.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/hpc2hdd/home/shuang886/PLC/datasets/pdbbind_biolip_chain_si30_esm3}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/PLC/results/strict_esm3_feature_audit_${SLURM_JOB_ID}.json}"
mkdir -p "${PROJECT}/logs" "$(dirname "${OUTPUT}")"
cd "${PROJECT}"
"${PYTHON}" scripts/diagnostics/audit_strict_esm3_features.py \
  --data-root "${DATA_ROOT}" --output "${OUTPUT}" --workers "${SLURM_CPUS_PER_TASK}"
