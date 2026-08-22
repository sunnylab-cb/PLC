#!/usr/bin/env bash
# Formal connected-component leakage audit for the exact BioLip training set.
# MMseqs cluster-mode 1 detects transitive train/evaluation sequence clusters.

#SBATCH --job-name=biolip_cc_audit
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_connected_cluster_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_connected_cluster_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PILOT="${PILOT:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/qc_connected_cluster_${SLURM_JOB_ID}}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
module load biological/mmseqs

"${PYTHON}" -u scripts/diagnostics/audit_biolip_connected_clusters.py \
  --pilot-root "${PILOT}" \
  --pdbbind-data-root "${PROJECT}/data" \
  --pdbbind-split-file "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "$(command -v mmseqs)" \
  --min-seq-id 0.30 \
  --coverage 0.80 \
  --cov-mode 0 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

cat "${OUTPUT}/connected_cluster_summary.json"
