#!/usr/bin/env bash
# Extract ligand V6 features only for the strict all-evaluation-clean BioLip set.
#
# Scientific purpose:
#   Expand the BioLip candidate-ranking pretraining set without leaking protein
#   homologs into PDBbind val/test, COACH420, or HOLO4K. The upstream MMseqs
#   audit retained 6,999 regular-organic pairs (30% identity, 80% query
#   coverage). This single CPU job traverses that allowlist, rather than
#   launching thousands of concurrent jobs or processing the rejected pairs.
#
# Engineering behavior:
#   - existing .pt files are skipped, so the job is resumable;
#   - one RDKit thread per worker prevents nested CPU oversubscription;
#   - per-pair timeout falls back to the no-conformer V6 route;
#   - an exact-manifest tensor QC runs even if some extraction tasks fail.

#SBATCH --job-name=biolip_ligv6_clean
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_ligv6_clean_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_ligv6_clean_%j.err

set -uo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip}"
AUDIT="${AUDIT:-${BIOLIP}/qc_ligand_candidates_sequence_clean_20260811}"
MANIFEST="${MANIFEST:-${AUDIT}/retained_pilot_manifest_all_evaluation.tsv}"
OUTPUT="${OUTPUT:-${BIOLIP}/ligand_features}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
WORKERS="${WORKERS:-96}"
QC_JSON="${AUDIT}/ligand_v6_extraction_qc.json"
QC_FAILURES="${AUDIT}/ligand_v6_extraction_failures.tsv"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}" || exit 1
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: strict BioLip manifest missing: ${MANIFEST}" >&2
  exit 2
fi

echo "BioLip strict-clean ligand V6 extraction"
echo "  manifest=${MANIFEST}"
echo "  output=${OUTPUT}"
echo "  workers=${WORKERS}"
echo "  python=${PYTHON}"

"${PYTHON}" -u preparation/ligand/biolip_ligand_extract.py \
  --biolip_root "${BIOLIP}" \
  --output "${OUTPUT}" \
  --pairs-cache "${BIOLIP}/pairs_cache_v6.pkl" \
  --manifest "${MANIFEST}" \
  --workers "${WORKERS}" \
  --num_conformers 50 \
  --rdkit-threads 1 \
  --per-pair-timeout 600
extract_status=$?

"${PYTHON}" - "${MANIFEST}" "${OUTPUT}" "${QC_JSON}" "${QC_FAILURES}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

manifest, feature_root, summary_path, failure_path = map(Path, sys.argv[1:])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
failures = []
atom_counts = []
completeness = []
routes = Counter()
schemas = Counter()

for row in rows:
    stem = row["feature_stem"]
    path = feature_root / row["pdb_id"].lower() / f"{stem}.pt"
    if not path.is_file():
        failures.append((stem, "missing", str(path)))
        continue
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        x = data.get("atom_features", data.get("x"))
        coords = data.get("coords", data.get("pos"))
        edge_index = data.get("edge_index")
        edge_attr = data.get("edge_attr")
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 109:
            raise ValueError(f"atom features must be [N,109], got {getattr(x, 'shape', None)}")
        if not isinstance(coords, torch.Tensor) or coords.shape != (x.shape[0], 3):
            raise ValueError(f"coords must be [N,3], got {getattr(coords, 'shape', None)}")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2,E], got {getattr(edge_index, 'shape', None)}")
        if not isinstance(edge_attr, torch.Tensor) or edge_attr.shape != (edge_index.shape[1], 6):
            raise ValueError(f"edge_attr must be [E,6], got {getattr(edge_attr, 'shape', None)}")
        for name, value in (("atom_features", x), ("coords", coords), ("edge_attr", edge_attr)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite {name}")
        atom_counts.append(int(x.shape[0]))
        completeness.append(float(data.get("feature_completeness_ratio", 0.0)))
        routes[str(data.get("ligand_route", "missing"))] += 1
        schemas[str(data.get("feature_schema_version", "missing"))] += 1
    except Exception as exc:
        failures.append((stem, type(exc).__name__, str(exc)))


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


summary = {
    "manifest": str(manifest),
    "expected_pairs": len(rows),
    "valid_pairs": len(atom_counts),
    "failed_pairs": len(failures),
    "atom_count": {
        "min": min(atom_counts, default=None),
        "p50": quantile(atom_counts, 0.50),
        "p90": quantile(atom_counts, 0.90),
        "p99": quantile(atom_counts, 0.99),
        "max": max(atom_counts, default=None),
    },
    "feature_completeness": {
        "p10": quantile(completeness, 0.10),
        "p50": quantile(completeness, 0.50),
        "min": min(completeness, default=None),
    },
    "routes": dict(sorted(routes.items())),
    "schemas": dict(sorted(schemas.items())),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
tmp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
os.replace(tmp, summary_path)
with failure_path.open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(("feature_stem", "error_type", "error"))
    writer.writerows(failures)
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if not failures else 3)
PY
qc_status=$?

if [ "${extract_status}" -ne 0 ] || [ "${qc_status}" -ne 0 ]; then
  echo "BioLip ligand V6 extraction needs retry: extraction=${extract_status}, qc=${qc_status}" >&2
  exit 1
fi

echo "BioLip strict-clean ligand V6 extraction and QC completed."
