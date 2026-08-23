#!/usr/bin/env bash
# Build and pairwise-screen a unified BioLiP + PDBbind-train manifest.
# This job creates metadata only; it does not duplicate feature tensors.

#SBATCH --job-name=biopdb_merge_qc
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biopdb_merge_qc_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biopdb_merge_qc_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
MMSEQS="${MMSEQS:-/hpc2ssd/softwares/mmseqs/bin/mmseqs}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/merged_pdbbind_pairwise_qc_${SLURM_JOB_ID:-manual}}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

"${PYTHON}" scripts/diagnostics/prepare_biolip_pdbbind_merged_manifest.py \
  --biolip-root "${BIOLIP}" \
  --pdbbind-root "${PROJECT}/data" \
  --pdbbind-splits "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "${MMSEQS}" \
  --min-seq-id 0.30 --coverage 0.80 --cov-mode 0 --threads 32

echo "Merged manifest preflight completed: ${OUTPUT}"
