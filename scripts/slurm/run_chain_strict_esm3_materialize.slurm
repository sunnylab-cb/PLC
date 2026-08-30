#!/usr/bin/env bash
# Build the strict merged feature root only after all BioLiP ESM3 shards pass.

#SBATCH --job-name=si30_e3_mat
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_mat_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_mat_%j.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
SPLIT_ROOT="${SPLIT_ROOT:-/hpc2hdd/home/shuang886/biolip/pdbbind_chain_si30_split_11283646_final}"
DATA_ROOT="${DATA_ROOT:-${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_esm3}"

mkdir -p "${PROJECT}/logs"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/scripts/diagnostics:${PROJECT}:${PYTHONPATH:-}"

"${PYTHON}" scripts/diagnostics/materialize_chain_strict_merged_root.py \
  --split-root "${SPLIT_ROOT}" --pdbbind-root "${PROJECT}/data" \
  --biolip-root "${BIOLIP}" --output-root "${DATA_ROOT}" --include-esm3

"${PYTHON}" - "${DATA_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
assert summary["merged_train"] == 10317
assert summary["include_esm3"] is True
assert summary["feature_links"] == summary["labels"] * 4
print(json.dumps(summary, indent=2))
PY
