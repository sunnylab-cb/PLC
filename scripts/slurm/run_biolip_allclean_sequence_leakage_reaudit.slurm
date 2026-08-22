#!/usr/bin/env bash
# Re-audit the exact 5,786-pair BioLip training manifest against every
# evaluation protein. This is diagnostic only and does not mutate data.
#
# MMseqs leakage rule follows the dataset construction protocol:
#   sequence identity >=30% over >=80% of the BioLip query chain.
# Direct PDB-ID intersections are also written explicitly.

#SBATCH --job-name=biolip_reaudit
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_allclean_reaudit_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_allclean_reaudit_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PILOT="${PILOT:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/qc_allclean_sequence_reaudit_${SLURM_JOB_ID}}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

module load biological/mmseqs

"${PYTHON}" - "${PILOT}/pilot_manifest.tsv" "${PROJECT}" "${OUTPUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest, project, output = map(Path, sys.argv[1:])
with manifest.open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
train_ids = {row["pdb_id"].lower() for row in rows}

external = {}
for dataset in ("coach420", "holo4k"):
    root = project / "benchmark" / "esm2" / dataset
    ids = {path.stem.split("_", 1)[0].lower() for path in root.glob("*.pt")}
    external[dataset] = {
        "training_pdbs": len(train_ids),
        "benchmark_pdbs": len(ids),
        "direct_pdb_overlap_count": len(train_ids & ids),
        "direct_pdb_overlap_ids": sorted(train_ids & ids),
    }

out = output / "direct_pdb_overlap.json"
out.write_text(json.dumps(external, indent=2, sort_keys=True))
print(json.dumps(external, indent=2, sort_keys=True))
PY

"${PYTHON}" -u scripts/diagnostics/audit_biolip_sequence_leakage.py \
  --pilot-root "${PILOT}" \
  --pdbbind-data-root "${PROJECT}/data" \
  --pdbbind-split-file "${PROJECT}/configs/splits/splits.txt" \
  --coach-esm-dir "${PROJECT}/benchmark/esm2/coach420" \
  --holo-esm-dir "${PROJECT}/benchmark/esm2/holo4k" \
  --output-dir "${OUTPUT}" \
  --mmseqs "$(command -v mmseqs)" \
  --min-seq-id 0.30 \
  --query-coverage 0.80 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

echo "Actual-manifest BioLip leakage re-audit completed: ${OUTPUT}"
cat "${OUTPUT}/direct_pdb_overlap.json"
cat "${OUTPUT}/summary.json"
