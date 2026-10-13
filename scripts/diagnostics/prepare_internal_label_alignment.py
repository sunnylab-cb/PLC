#!/usr/bin/env python3
"""Recover internal-label axes from source PDB/SDF, without changing datasets.

request: run on HPC CPU, scan ALL strict val axes plus failed TRAIN rows.
verify: run locally where original New/ PDB/SDF files exist. Reproduce the full
historical distance vector before mapping keys; bind the overlay to source
distances, protein metadata/coordinates and the strict split hash.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.strict_assignment_common import digest_object, find_feature, read_strict_splits, write_json
from scripts.infer.benchmark_integrity import (
    legacy_tag_axis, require_finite, residue_keys, safe_tensor_load, sha256_file, verify_tag_distances,
)


def reference_digest(protein):
    return digest_object([residue_keys(protein), list(map(str, protein["residue_names"])),
                          np.asarray(protein["ca_coords"], dtype=np.float32).tolist()])


def load_overlay(path, split_path, allowed_ids):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if data["split_sha256"] != sha256_file(split_path):
        raise ValueError("Label overlay split hash mismatch")
    return {sid: row for sid, row in data["rows"].items() if sid in set(allowed_ids)}


def aligned_values(sid, protein, distance, overlay):
    if isinstance(distance, dict) and "distances_vdw_adjusted" in distance:
        raise ValueError("Label alignment only supports dcenter labels")
    raw = torch.as_tensor(distance["distances"] if isinstance(distance, dict) else distance).float()
    require_finite(raw, sid + " distances")
    if sid not in overlay:
        return raw
    row = overlay[sid]
    if reference_digest(protein) != row["reference_sha256"] or digest_object(raw.tolist()) != row["raw_distances_sha256"]:
        raise ValueError(f"Label overlay provenance mismatch: {sid}")
    if not row["verification"]["distance_verified"]:
        raise ValueError("Unverified label overlay")
    idx = torch.as_tensor(row["source_indices"], dtype=torch.long)
    if idx.shape != (len(protein["ca_coords"]),) or len(set(idx.tolist())) != len(idx) or (idx < 0).any() or (idx >= len(raw)).any():
        raise ValueError("Invalid overlay source indices")
    return raw[idx]


def collect_request(args):
    split_path = args.root / "splits.txt"
    groups = read_strict_splits(split_path)
    failed = [row["sample_id"] for row in csv.DictReader(args.audit_csv.open()) if row["error"]]
    if set(failed) - set(groups["train"]):
        raise ValueError("Audit failures are not TRAIN-only")
    ids = groups["val"] + failed
    distances = safe_tensor_load(args.root / "residue_ligand_distances_dcenter.pt")
    requests, counts = {}, {"val": 0, "train": 0}
    for i, sid in enumerate(ids):
        p = safe_tensor_load(find_feature(args.root, "complete_features_v4_nma", sid, True))
        esm = safe_tensor_load(find_feature(args.root, "esm2_latents", sid, True))
        n = len(p["ca_coords"])
        keys, ekeys = residue_keys(p, n), residue_keys(esm, len(esm["embeddings"]))
        if keys[:min(n, len(ekeys))] != ekeys[:min(n, len(ekeys))]:
            raise ValueError(f"Non-label ESM axis shift requires separate repair: {sid}")
        d = distances[sid]
        raw = torch.as_tensor(d["distances"] if isinstance(d, dict) else d).float()
        if raw.shape != (n,):
            subset = "val" if sid in groups["val"] else "train"
            counts[subset] += 1
            requests[sid] = {"split": subset, "keys": keys, "names": list(map(str, p["residue_names"])),
                             "ca": np.asarray(p["ca_coords"], dtype=np.float32).tolist(),
                             "reference_sha256": reference_digest(p), "distances": raw.tolist()}
        if (i + 1) % 100 == 0:
            print(f"LABEL_AXIS_SCAN {i+1}/{len(ids)}", flush=True)
    write_json(args.output, {"split_sha256": sha256_file(split_path), "scanned": len(ids), "mismatch_counts": counts, "rows": requests})
    print("LABEL_REQUEST_COMPLETE", counts, flush=True)


def verify_request(args):
    from Bio.PDB import PDBParser

    data = json.loads(args.request.read_text())
    rows = {}
    for sid, request in data["rows"].items():
        pdbs = list(args.source.glob(f"*/{sid}/{sid}_protein.pdb"))
        if len(pdbs) != 1:
            raise ValueError(f"Ambiguous/missing source PDB: {sid}")
        pdb = pdbs[0]
        sdf = pdb.with_name(f"{sid}_ligand.sdf")
        tag = legacy_tag_axis(pdb)
        raw = torch.tensor(request["distances"], dtype=torch.float32)
        verification = verify_tag_distances(raw, tag, [sdf])
        keys = residue_keys(tag, len(raw))
        index = {key: i for i, key in enumerate(keys)}
        structure = PDBParser(QUIET=True).get_structure(sid, str(pdb))[0]
        source_indices, max_ca = [], 0.
        for key, name, ca in zip(request["keys"], request["names"], request["ca"]):
            key = (str(key[0]), int(key[1]), str(key[2]))
            residue = structure[key[0]][(" ", key[1], key[2] or " ")]
            if residue.resname != name or "CA" not in residue:
                raise ValueError(f"Source/cached protein identity mismatch: {sid}/{key}")
            error = float(np.linalg.norm(residue["CA"].coord - ca))
            if error > .01:
                raise ValueError(f"Source/cached protein CA differs: {sid}/{key}: {error}")
            max_ca = max(max_ca, error)
            source_indices.append(index[key])
        mapped = raw[source_indices]
        n = len(source_indices)
        classes = lambda d: torch.where(d < 5., 0, torch.where(d < 8., 1, 2))
        rows[sid] = {"split": request["split"], "source_indices": source_indices,
                     "reference_sha256": request["reference_sha256"], "raw_distances_sha256": digest_object(raw.tolist()),
                     "verification": verification, "source_pdb": str(pdb), "source_pdb_sha256": sha256_file(pdb),
                     "max_ca_error": max_ca, "n_raw": len(raw), "n_protein": n,
                     "changed_classes_vs_legacy_prefix": int((classes(mapped) != classes(raw[:n])).sum()),
                     "changed_contacts_vs_legacy_prefix": int(((mapped < 5.) != (raw[:n] < 5.)).sum()),
                     "positional_shifts": sum(i != j for i, j in enumerate(source_indices))}
        print(sid, {k: rows[sid][k] for k in ("split", "n_raw", "n_protein", "positional_shifts", "changed_contacts_vs_legacy_prefix")}, flush=True)
    write_json(args.output, {"split_sha256": data["split_sha256"], "rows": rows,
                            "scope": "diagnostic overlay only; no source data changed; no test sample used"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["request", "verify"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect_request(args) if args.mode == "request" else verify_request(args)
