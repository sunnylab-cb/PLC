#!/usr/bin/env bash
# Materialize the first leakage-filtered BioLip side-path dataset for V11.
#
# What this job does:
#   1. Removes direct PDB-ID overlap with PDBbind and COACH420/HOLO4K.
#   2. Keeps whole PDB structures in one split to avoid chain/ligand leakage.
#   3. Builds exact chain-level V4/GVP sidecars without changing source files.
#   4. Links ESM2 and ligand V6 features into the PocketDatasetV8 layout.
#   5. Recomputes residue-heavy-atom/ligand-heavy-atom d_min labels at 5/8 A.
#   6. Reloads every pair with the production dataset class as a final QC.
#
# This is preprocessing only: no GPU is requested and no model is trained.

#SBATCH --job-name=biolip_v11prep
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_v11prep_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_v11prep_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip}"
QC_DIR="${QC_DIR:-${BIOLIP}/qc_training_manifest_20260811}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BIOLIP}/v11_pilot_strict_nobench_20260811}"
EXTRA_EXCLUDE="${EXTRA_EXCLUDE:-}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"

mkdir -p "${PROJECT}/logs"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "BioLip V11 pilot materialization"
echo "  source=${BIOLIP}"
echo "  manifest=${QC_DIR}/biolip_current_feature_ready.tsv"
echo "  exclusions=${QC_DIR}/exclude_coach420_holo4k_pdb_ids.txt"
echo "  extra_exclusions=${EXTRA_EXCLUDE:-none}"
echo "  output=${OUTPUT_ROOT}"

EXCLUDE_ARGS=(
  --exclude-pdb-ids "${QC_DIR}/exclude_coach420_holo4k_pdb_ids.txt"
)
if [ -n "${EXTRA_EXCLUDE}" ]; then
  if [ ! -f "${EXTRA_EXCLUDE}" ]; then
    echo "ERROR: EXTRA_EXCLUDE does not exist: ${EXTRA_EXCLUDE}" >&2
    exit 2
  fi
  EXCLUDE_ARGS+=(--exclude-pdb-ids "${EXTRA_EXCLUDE}")
fi

"${PYTHON}" -u scripts/diagnostics/prepare_biolip_v11_pilot.py \
  --biolip-root "${BIOLIP}" \
  --manifest "${QC_DIR}/biolip_current_feature_ready.tsv" \
  "${EXCLUDE_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT}"

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from collections import Counter

import torch

from src.data.collate import pocket_collate_fn_v10
from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8

root = sys.argv[1]
train_ids, val_ids, test_ids = load_splits(f"{root}/splits.txt")
common = dict(
    data_root=root,
    max_seq_len=1800,
    max_atoms=128,
    distances_file=f"{root}/residue_ligand_distances_dcenter.pt",
    esm_features_dir="esm2_latents",
    ligand_features_dir="ligand_features_v5",
    protein_features_dir="complete_features_v4",
    nma_features_dir="complete_features_v4_nma",
    label_contact_thresh=5.0,
    label_near_thresh=8.0,
    use_nma=False,
    verbose=False,
)
datasets = {
    "train": PocketDatasetV8(pdb_ids=train_ids, split="train", **common),
    "val": PocketDatasetV8(pdb_ids=val_ids, split="val", **common),
}
counts = Counter()
max_residues = 0
max_atoms = 0
for split, dataset in datasets.items():
    if not len(dataset):
        raise RuntimeError(f"empty {split} dataset")
    for sample in dataset:
        if sample["labels"].numel() != sample["n_residues"]:
            raise RuntimeError(f"{sample['pdb_id']}: label/residue length mismatch")
        if not bool((sample["labels"] == 0).any()):
            raise RuntimeError(f"{sample['pdb_id']}: no contact label")
        for key in (
            "esm",
            "protein_physchem",
            "protein_spatial_scalar",
            "protein_spatial_vector",
            "gvp_coords",
            "gvp_vectors",
            "ligand_features",
            "ligand_coords",
        ):
            value = sample[key]
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{sample['pdb_id']}: non-finite {key}")
        counts.update(int(value) for value in sample["labels"].tolist())
        max_residues = max(max_residues, int(sample["n_residues"]))
        max_atoms = max(max_atoms, int(sample["n_atoms"]))
    pocket_collate_fn_v10([datasets[split][0]])

qc = {
    "dataset_counts": {name: len(dataset) for name, dataset in datasets.items()},
    "label_counts": {str(key): value for key, value in sorted(counts.items())},
    "max_residues": max_residues,
    "max_atoms": max_atoms,
    "production_loader_qc": "pass",
}
with open(f"{root}/dataset_qc.json", "w") as handle:
    json.dump(qc, handle, indent=2, sort_keys=True)
print(json.dumps(qc, indent=2, sort_keys=True))
PY

echo "BioLip V11 pilot preprocessing and production-loader QC completed."
