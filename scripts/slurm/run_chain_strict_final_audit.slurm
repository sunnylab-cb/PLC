#!/usr/bin/env bash
# Independent final audit of the exact PDBbind+BioLiP train allowlist.

#SBATCH --job-name=chain_si30_audit
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/chain_si30_audit_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/chain_si30_audit_%j.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
MMSEQS="${MMSEQS:-/hpc2ssd/softwares/mmseqs/bin/mmseqs}"
SPLIT_ROOT="${SPLIT_ROOT:?set SPLIT_ROOT}"
OUTPUT="${OUTPUT:-${SPLIT_ROOT}/final_allowlist_audit}"

cd "${PROJECT}"
mkdir -p "${PROJECT}/logs" "${OUTPUT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts/diagnostics:${PYTHONPATH:-}"
"${PYTHON}" scripts/diagnostics/audit_chain_strict_final_allowlist.py \
  --split-root "${SPLIT_ROOT}" --output-dir "${OUTPUT}" \
  --mmseqs "${MMSEQS}" --threads 64 \
  --min-seq-id 0.30 --min-coverage 0.80 --search-floor 0.28
