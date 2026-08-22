#!/usr/bin/env bash
# Pairwise sequence-identity sensitivity analysis on the exact BioLip training
# manifest. This supplements the formal SI<0.30/qcov>=0.80 leakage audit.
#
# Variants:
#   0 qcov80_lowfloor: search from SI=0.10 with BioLip query coverage >=80%;
#      reports the observed maximum coverage-qualified pairwise SI.
#   1 bidir80_lowfloor: SI=0.10 with both query and target coverage >=80%;
#      stricter full-length-like pairwise comparison.
#   2 local_si30_control: SI>=0.30 with no coverage constraint; diagnoses
#      short local motifs and must not be interpreted as cluster leakage.

#SBATCH --job-name=biolip_si_sens
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_si_sensitivity_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_si_sensitivity_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PILOT="${PILOT:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NAMES=(qcov80_lowfloor bidir80_lowfloor local_si30_control)
MIN_IDS=(0.10 0.10 0.30)
COVERAGES=(0.80 0.80 0.0)
COV_MODES=(2 0 0)
NAME="${NAMES[$TASK_ID]:-}"
MIN_ID="${MIN_IDS[$TASK_ID]:-}"
COVERAGE="${COVERAGES[$TASK_ID]:-}"
COV_MODE="${COV_MODES[$TASK_ID]:-}"
OUTPUT="/hpc2hdd/home/shuang886/biolip/qc_pairwise_si_${NAME}_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"

if [ -z "${NAME}" ]; then
  echo "ERROR: invalid task ${TASK_ID}" >&2
  exit 2
fi

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
module load biological/mmseqs

echo "Pairwise SI sensitivity: ${NAME}"
echo "  min_id=${MIN_ID} coverage=${COVERAGE} cov_mode=${COV_MODE}"

"${PYTHON}" -u scripts/diagnostics/audit_biolip_sequence_leakage.py \
  --pilot-root "${PILOT}" \
  --pdbbind-data-root "${PROJECT}/data" \
  --pdbbind-split-file "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "$(command -v mmseqs)" \
  --min-seq-id "${MIN_ID}" \
  --query-coverage "${COVERAGE}" \
  --cov-mode "${COV_MODE}" \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

"${PYTHON}" - "${OUTPUT}/best_hit_per_query_category.tsv" "${OUTPUT}/si_distribution.json" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

source, output = map(Path, sys.argv[1:])
groups = defaultdict(list)
with source.open() as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        groups[row["category"]].append(
            {
                "identity": float(row["identity"]),
                "alignment_length": int(row["alignment_length"]),
                "query_coverage": float(row["query_coverage"]),
                "target_coverage": float(row["target_coverage"]),
                "pdb_id": row["pdb_id"],
                "chain": row["chain"],
                "source_id": row["source_id"],
            }
        )

def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)

summary = {}
for category, rows in groups.items():
    rows.sort(key=lambda row: (row["identity"], row["alignment_length"]), reverse=True)
    identities = [row["identity"] for row in rows]
    summary[category] = {
        "queries_with_reported_hit": len(rows),
        "identity_max": max(identities, default=None),
        "identity_p50": quantile(identities, 0.50),
        "identity_p90": quantile(identities, 0.90),
        "identity_p99": quantile(identities, 0.99),
        "n_ge_0_30": sum(value >= 0.30 for value in identities),
        "top10": rows[:10],
    }
output.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "Saved ${OUTPUT}/si_distribution.json"
