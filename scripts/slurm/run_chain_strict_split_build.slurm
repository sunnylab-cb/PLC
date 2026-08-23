#!/usr/bin/env bash
# Build a fail-closed, polymer-chain SI30 PDBbind split and BioLiP train allowlist.
# Homology predicate: identity>=0.30 AND (qcov>=0.80 OR tcov>=0.80).

#SBATCH --job-name=chain_si30_split
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/chain_si30_split_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/chain_si30_split_%j.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
MMSEQS="${MMSEQS:-/hpc2ssd/softwares/mmseqs/bin/mmseqs}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/pdbbind_chain_si30_split_${SLURM_JOB_ID:-manual}}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts/diagnostics:${PYTHONPATH:-}"

"${PYTHON}" scripts/diagnostics/build_chain_strict_pdbbind_biolip_split.py \
  --pdbbind-root "${PROJECT}/data" \
  --pdbbind-ids "${PROJECT}/configs/splits/splits.txt" \
  --biolip-root "${BIOLIP}" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" --mmseqs "${MMSEQS}" \
  --min-seq-id 0.30 --min-coverage 0.80 --search-floor 0.28 \
  --threads 64 --seed 42

echo "Chain-strict split completed: ${OUTPUT}"
