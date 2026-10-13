#!/usr/bin/env python3
"""Extract audited missing ligands into a NEW isolated directory, then QC.

This is feature preparation only. No labels, normalization stats, model inputs
or production split are changed; passing tensors are NOT approved TRAIN rows.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

try:
    from .audit_biolip_expansion_tensors import tensor_qc
except ImportError:
    from audit_biolip_expansion_tensors import tensor_qc


def verify_manifest(rows):
    if not rows or len({r["feature_stem"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate candidate manifest")
    for row in rows:
        if row["known_train_chain_key"] != "1" or row["known_sequence_matches"] != "True":
            raise ValueError(f"Unverified TRAIN chain: {row['feature_stem']}")
        if not (6 <= int(row["heavy_atoms_meta"]) <= 512 and int(row["binding_residue_count"]) >= 3):
            raise ValueError("Candidate policy mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--biolip-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.audit_dir/"summary.json").read_text())
    manifest = args.audit_dir/"missing_ligand_known_chain_candidates.tsv"
    with manifest.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    verify_manifest(rows)
    if (len(rows) != summary["missing_known_chain_candidates"] or
            summary["missing_rows_sequence_verified"] != len(rows) or
            summary["known_chain_sequence_failures"] != 0):
        raise ValueError("Upstream sequence audit/count mismatch")
    cache = args.biolip_root/"pairs_cache_v6.pkl"
    if not cache.is_file():
        raise FileNotFoundError("Refuse full discovery or cache modification")
    output = args.output_dir.resolve()
    if output == args.biolip_root.resolve() or args.biolip_root.resolve() in output.parents:
        raise ValueError("Output must be outside production BioLiP tree")
    output.mkdir(parents=True, exist_ok=False)
    code_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code_root/"preparation/ligand"))
    import biolip_ligand_extract as extractor
    import extract_ligand_features_v5 as v5
    if not (extractor.V5_AVAILABLE and extractor.V6_TYPES_AVAILABLE and
            v5.PYPROPEL_AVAILABLE and v5.V4_AVAILABLE):
        raise RuntimeError("V5/V6/pypropel dependencies unavailable; no silent V5-only fallback")
    command = [sys.executable, "-u", str(code_root/"preparation/ligand/biolip_ligand_extract.py"),
               "--biolip_root", str(args.biolip_root), "--output", str(output/"ligand_features"),
               "--pairs-cache", str(cache), "--manifest", str(manifest), "--workers", str(args.workers),
               "--num_conformers", "50", "--rdkit-threads", "1", "--per-pair-timeout", "600"]
    provenance = dict(command=command, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      expected_pairs=len(rows), approved_for_training=False)
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("START selective ligand extraction", json.dumps(provenance), flush=True)
    proc = subprocess.run(command)
    routes, failures, counts = Counter(), Counter(), Counter()
    with (output/"tensor_qc.jsonl").open("w") as f:
        for row in rows:
            path = output/"ligand_features"/row["pdb_id"]/f"{row['feature_stem']}.pt"
            try:
                qc = tensor_qc(torch.load(path,map_location="cpu",weights_only=False), row)
            except Exception as exc:
                qc = dict(tensor_route=-999, qc_fail_reasons=[f"load_or_schema:{type(exc).__name__}"],
                          error=str(exc), old_route_gate=False, expanded_route_gate=False)
            failures.update(qc["qc_fail_reasons"])
            routes[qc["tensor_route"]] += 1
            for key in ("old_route_gate", "expanded_route_gate", "over_model_atom_cap_256"):
                counts[key] += bool(qc.get(key))
            counts["tensor_qc_pass"] += not qc["qc_fail_reasons"]
            f.write(json.dumps(dict(feature_stem=row["feature_stem"], **qc, approved_for_training=False))+"\n")
    result = dict(expected=len(rows), extraction_returncode=proc.returncode, counts=dict(counts),
                  routes=dict(routes), failures=dict(failures),
                  remaining=["Protein/ESM residue-axis check", "Receptor-ligand geometry and labels",
                             "Canonical chemical/site deduplication", "Mask-aware normalization and model compatibility"],
                  approved_training_rows=0)
    (output/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    raise SystemExit(proc.returncode or (3 if counts["tensor_qc_pass"] != len(rows) else 0))


if __name__ == "__main__":
    main()
