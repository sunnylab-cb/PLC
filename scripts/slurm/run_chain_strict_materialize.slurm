#!/usr/bin/env bash
# Materialize the audited strict PDBbind+BioLiP root using directory symlinks.

#SBATCH --job-name=si30_materialize
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_materialize_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_materialize_%j.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
SPLIT_ROOT="${SPLIT_ROOT:?set SPLIT_ROOT}"
DATA_ROOT="${DATA_ROOT:-/hpc2hdd/home/shuang886/PLC/datasets/pdbbind_biolip_chain_si30_final}"

cd "${PROJECT}"
mkdir -p "${PROJECT}/logs" "$(dirname "${DATA_ROOT}")"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts/diagnostics:${PYTHONPATH:-}"
"${PYTHON}" scripts/diagnostics/materialize_chain_strict_merged_root.py \
  --split-root "${SPLIT_ROOT}" --pdbbind-root "${PROJECT}/data" \
  --biolip-root "${BIOLIP}" --output-root "${DATA_ROOT}"
