#!/usr/bin/env bash
# Full normalization pipeline:
#   1. Generate normalized staging (tmp/norm_stage_ligunimol/)
#   2. Validate staging
#   3. Directory swap (original → _ori, staging → canonical names)
set -euo pipefail

REPO=/hpc2hdd/home/shuang886/jhspoolers/PLC
DATA=$REPO/data
STAGE=$REPO/tmp/norm_stage_ligunimol
STATS=$DATA/stats_output.pt
WORKERS=128
LOG=$REPO/tmp/norm_swap.log

mkdir -p "$REPO/tmp"

echo "============================================================"
echo " Normalization pipeline  $(date)"
echo " REPO:    $REPO"
echo " STAGING: $STAGE"
echo " STATS:   $STATS"
echo " WORKERS: $WORKERS"
echo "============================================================"

# ── Step 1: generate staging ──────────────────────────────────────────────────
echo ""
echo "[1/3] Running apply_feature_normalization.py → staging ..."
conda run -n af3 python "$REPO/scripts/apply_feature_normalization.py" \
    --data-root "$DATA" \
    --stats     "$STATS" \
    --output-root "$STAGE" \
    --features ligand unimol \
    --all \
    --overwrite \
    --workers "$WORKERS" \
    --log-every 1000 \
    --ligand-in-dir  ligand_features_v5 \
    --ligand-out-dir ligand_features_v5 \
    --unimol-in-dir  unimol_embeddings \
    --unimol-out-dir unimol_embeddings
echo "[1/3] Done."

# ── Step 2: validate staging ──────────────────────────────────────────────────
echo ""
echo "[2/3] Validating staging ..."
conda run -n af3 python - << 'PYEOF'
import sys, torch
from pathlib import Path

STAGE = Path("/hpc2hdd/home/shuang886/jhspoolers/PLC/tmp/norm_stage_ligunimol")
DATA  = Path("/hpc2hdd/home/shuang886/jhspoolers/PLC/data")

def check(label, root, pattern, key, expected_norm):
    files = sorted(root.rglob(pattern))
    n = len(files)
    if n == 0:
        print(f"  FAIL {label}: 0 files found in {root}")
        sys.exit(1)
    # spot-check first, middle, last
    samples = [files[0], files[n//2], files[-1]]
    for f in samples:
        d = torch.load(f, map_location="cpu", weights_only=False)
        t = d if isinstance(d, torch.Tensor) else d.get(key)
        if t is None:
            print(f"  FAIL {label}: missing key '{key}' in {f}")
            sys.exit(1)
        if not torch.isfinite(t.float()).all():
            print(f"  FAIL {label}: non-finite values in {f}")
            sys.exit(1)
        has_meta = isinstance(d, dict) and "feature_normalization" in d
        if has_meta != expected_norm:
            print(f"  FAIL {label}: expected feature_normalization={expected_norm}, got {has_meta} in {f}")
            sys.exit(1)
    print(f"  OK   {label}: {n} files, normalized={expected_norm}, finite=True, meta={expected_norm}")
    return n

n_lig_stage  = check("staging ligand",  STAGE / "ligand_features_v5",  "**/*_ligand.pt",  "atom_features", True)
n_uni_stage  = check("staging unimol",  STAGE / "unimol_embeddings",   "**/*_unimol.pt",  "embedding",     True)
n_lig_ori    = check("original ligand", DATA  / "ligand_features_v5",  "**/*_ligand.pt",  "atom_features", False)
n_uni_ori    = check("original unimol", DATA  / "unimol_embeddings",   "**/*_unimol.pt",  "embedding",     False)

if n_lig_stage != n_lig_ori:
    print(f"  FAIL count mismatch: staging ligand={n_lig_stage} vs original={n_lig_ori}")
    sys.exit(1)
if n_uni_stage != n_uni_ori:
    print(f"  FAIL count mismatch: staging unimol={n_uni_stage} vs original={n_uni_ori}")
    sys.exit(1)

# scale sanity: spot-check 10gs absmax
import re
def find_sample(root, name):
    hits = list(root.rglob(f"**/{name}"))
    return hits[0] if hits else None

f_lig_s = find_sample(STAGE / "ligand_features_v5", "10gs_ligand.pt")
f_lig_o = find_sample(DATA  / "ligand_features_v5",  "10gs_ligand.pt")
f_uni_s = find_sample(STAGE / "unimol_embeddings",  "10gs_unimol.pt")
f_uni_o = find_sample(DATA  / "unimol_embeddings",   "10gs_unimol.pt")

if f_lig_s and f_lig_o:
    am_o = torch.load(f_lig_o, map_location="cpu", weights_only=False)["atom_features"].float().abs().max().item()
    am_s = torch.load(f_lig_s, map_location="cpu", weights_only=False)["atom_features"].float().abs().max().item()
    print(f"  ligand 10gs absmax: {am_o:.2f} -> {am_s:.2f}")
if f_uni_s and f_uni_o:
    am_o = torch.load(f_uni_o, map_location="cpu", weights_only=False)["embedding"].float().abs().max().item()
    am_s = torch.load(f_uni_s, map_location="cpu", weights_only=False)["embedding"].float().abs().max().item()
    print(f"  UniMol 10gs absmax: {am_o:.2f} -> {am_s:.2f}")

print("Validation passed.")
PYEOF
echo "[2/3] Validation done."

# ── Step 3: directory swap ────────────────────────────────────────────────────
echo ""
echo "[3/3] Directory swap ..."

mv "$DATA/ligand_features_v5"  "$DATA/ligand_features_v5_ori"
mv "$STAGE/ligand_features_v5" "$DATA/ligand_features_v5"
echo "  ligand_features_v5_ori  ← original"
echo "  ligand_features_v5      ← normalized"

mv "$DATA/unimol_embeddings"  "$DATA/unimol_embeddings_ori"
mv "$STAGE/unimol_embeddings" "$DATA/unimol_embeddings"
echo "  unimol_embeddings_ori   ← original"
echo "  unimol_embeddings       ← normalized"

echo "[3/3] Swap done."

# ── Final check ───────────────────────────────────────────────────────────────
echo ""
echo "Final spot-check on canonical paths ..."
conda run -n af3 python - << 'PYEOF'
import torch
from pathlib import Path

DATA = Path("/hpc2hdd/home/shuang886/jhspoolers/PLC/data")

def spot(path, key):
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = d[key].float()
    norm = "feature_normalization" in d
    am = t.abs().max().item()
    return norm, am

p_lig = next((DATA / "ligand_features_v5").rglob("10gs_ligand.pt"))
p_uni = next((DATA / "unimol_embeddings").rglob("10gs_unimol.pt"))
p_lig_o = next((DATA / "ligand_features_v5_ori").rglob("10gs_ligand.pt"))
p_uni_o = next((DATA / "unimol_embeddings_ori").rglob("10gs_unimol.pt"))

n_lig,  am_lig  = spot(p_lig,   "atom_features")
n_uni,  am_uni  = spot(p_uni,   "embedding")
n_lig_o,am_lig_o= spot(p_lig_o, "atom_features")
n_uni_o,am_uni_o= spot(p_uni_o, "embedding")

print(f"data/ligand_features_v5:      normalized={n_lig}  absmax={am_lig:.2f}")
print(f"data/ligand_features_v5_ori:  normalized={n_lig_o} absmax={am_lig_o:.2f}")
print(f"data/unimol_embeddings:       normalized={n_uni}  absmax={am_uni:.2f}")
print(f"data/unimol_embeddings_ori:   normalized={n_uni_o} absmax={am_uni_o:.2f}")
PYEOF

echo ""
echo "============================================================"
echo " Pipeline complete  $(date)"
echo "============================================================"
