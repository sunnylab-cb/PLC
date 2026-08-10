#!/usr/bin/env bash
# BioLip side-path phase 0: metadata/asset readiness and multi-site supervision QC.
#
# This CPU-only job loads the existing pairs_cache_v6.pkl, indexes current V4,
# ESM2, fixed receptor, and ligand V6 assets, then writes atomic TSV/JSON ledgers.
# It never modifies source BioLip data and does not start model training.

#SBATCH --job-name=biolip_manifest_qc
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/biolip_manifest_qc_%j.out
#SBATCH --error=logs/biolip_manifest_qc_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BIOLIP_ROOT="${BIOLIP_ROOT:-/hpc2hdd/home/shuang886/biolip}"
OUT_DIR="${OUT_DIR:-${BIOLIP_ROOT}/qc_training_manifest_20260811}"
PDBBIND_SPLITS="${PDBBIND_SPLITS:-${PROJECT_DIR}/configs/splits/splits.txt}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate af3
fi

echo "BioLip manifest QC"
echo "  host=$(hostname)"
echo "  root=${BIOLIP_ROOT}"
echo "  output=${OUT_DIR}"
echo "  splits=${PDBBIND_SPLITS}"
echo "  python=$(command -v python)"

python -u scripts/diagnostics/build_biolip_training_manifest.py \
  --biolip-root "${BIOLIP_ROOT}" \
  --pairs-cache "${BIOLIP_ROOT}/pairs_cache_v6.pkl" \
  --output-dir "${OUT_DIR}" \
  --pdbbind-splits "${PDBBIND_SPLITS}" \
  --min-heavy-atoms 6 \
  --max-heavy-atoms 128 \
  --min-binding-residues 3
