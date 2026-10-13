#!/usr/bin/env python3
"""Read-only first-stage tensor QC for BioLiP expansion, not a train allowlist.

Checks existing ligand tensors and writes missing-ligand candidate manifests.
Known TRAIN chains get an exact cached-sequence check; new chains still need
homology, geometry, normalization and residue-axis audits before any training.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import torch

try:
    from .audit_biolip_expansion import asset_index, policies, sha256
except ImportError:
    from audit_biolip_expansion import asset_index, policies, sha256


def tensor_qc(data, row):
    reasons = []
    x = torch.as_tensor(data["atom_features"])
    xyz = torch.as_tensor(data["coords"])
    edge = torch.as_tensor(data["edge_index"])
    attr = torch.as_tensor(data["edge_attr"])
    n = len(x)
    if x.shape != (n, 109) or n == 0:
        reasons.append("atom_feature_shape")
    if xyz.shape != (n, 3):
        reasons.append("coordinate_shape")
    for key, value in (("features", x), ("coords", xyz), ("edge_attr", attr)):
        if not torch.isfinite(value).all():
            reasons.append(f"nonfinite_{key}")
    if edge.ndim != 2 or edge.shape[0] != 2:
        reasons.append("edge_shape")
    elif attr.shape != (edge.shape[1], 6):
        reasons.append("edge_attr_shape")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n or edge.is_floating_point()):
        reasons.append("edge_indices")
    identity = {"pdb_id": "pdb_id", "protein_chain": "protein_chain",
                "binding_site_id": "binding_site_id", "ligand_id": "ligand_id"}
    for key, column in identity.items():
        if str(data.get(key, "")) != row[column]:
            reasons.append(f"identity_{key}")
    mask = data.get("x_group_mask")
    means, completeness = [], None
    if mask is None:
        reasons.append("missing_group_mask")
    else:
        mask = torch.as_tensor(mask).float()
        if mask.shape != (n, 14):
            reasons.append("group_mask_shape")
        elif not ((mask == 0) | (mask == 1)).all():
            reasons.append("group_mask_nonbinary")
        else:
            means = mask.mean(0).tolist()
            completeness = float(mask.mean())
            stored = float(data.get("feature_completeness_ratio", -1))
            if not abs(stored-completeness) < 1e-5:
                reasons.append("completeness_mismatch")
    heavy = int(data.get("num_heavy_atoms", -1))
    if heavy != int(row["heavy_atoms_meta"]):
        reasons.append("heavy_atom_count_mismatch")
    route = int(data.get("ligand_route", -1))
    good = not reasons and completeness is not None and completeness >= .8
    return dict(tensor_route=route, total_atoms=n, tensor_heavy_atoms=heavy,
                over_model_atom_cap_256=n > 256, completeness=completeness,
                group_valid_fraction=means, qc_fail_reasons=reasons,
                old_route_gate=bool(good and route == 5),
                expanded_route_gate=bool(good and route in (3, 4, 5)))


def sequence_from_cache(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    value = data.get("sequence", data.get("residue_name_1"))
    if value is None:
        raise ValueError("missing explicit sequence/residue_name_1")
    sequence = value if isinstance(value, str) else "".join(value)
    sequence = "".join(sequence.split()).replace(":", "").replace("-", "").upper()
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWYBXZUOJ" for c in sequence):
        raise ValueError("invalid sequence")
    return sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--biolip-root", type=Path, required=True)
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--final-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    root, final = args.biolip_root, args.final_root
    ids = set((final/"biolip_train_ids.txt").read_text().split())
    known = defaultdict(set)
    with (root/"v11_allclean_route5_20260811/pilot_manifest.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["sample_id"] in ids:
                known[(row["pdb_id"], row["protein_chain"])].add(row["sample_id"])
    old_sequences = {}
    with (args.split_root/"biolip_chains.metadata.jsonl").open() as f:
        for row in map(json.loads, f):
            for m in row["members"]:
                if m["sample_id"] in ids:
                    old_sequences[m["sample_id"]] = row["sequence"]
    if set(old_sequences) != ids:
        raise ValueError("Incomplete current TRAIN sequence provenance")
    excluded = set()
    for line in (final/"pdbbind_chain_si30_splits.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            split, key = line.split()[:2]
            if split != "train":
                excluded.add(key)
    with (final/"external_chains.metadata.jsonl").open() as f:
        for row in map(json.loads, f):
            excluded.update(m["sample_id"].split(":", 1)[-1][:4].lower() for m in row["members"])
    assets = asset_index(root)
    candidates, missing = [], []
    metadata = root/"qc_training_manifest_allclean_20260811/biolip_pairs_metadata.tsv.gz"
    with gzip.open(metadata, "rt", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if not policies(row)["chemistry_6_512_contact3"]:
                continue
            p, c, key = row["pdb_id"], row["protein_chain"], row["feature_stem"]
            if key in ids or p in excluded:
                continue
            if not (p in assets["protein"] and p in assets["fixed"] and f"{p}_{c}" in assets["esm"]):
                continue
            row["known_train_chain_key"] = int((p,c) in known)
            if (p,key) in assets["ligand"]:
                candidates.append(row)
            elif (p,c) in known:
                missing.append(row)
    print(f"CANDIDATES existing={len(candidates)} missing_known_chain={len(missing)}", flush=True)

    seq_status = {}
    for p,c in sorted({(r["pdb_id"],r["protein_chain"]) for r in candidates+missing if r["known_train_chain_key"]}):
        try:
            seq = sequence_from_cache(root/"esm2_features"/f"{p}_{c}_esm2.pt")
            expected = {old_sequences[key] for key in known[(p,c)]}
            seq_status[(p,c)] = dict(known_sequence_matches=expected == {seq},
                                      sequence_sha256=hashlib.sha256(seq.encode()).hexdigest())
        except Exception as exc:
            seq_status[(p,c)] = dict(known_sequence_matches=False, sequence_error=f"{type(exc).__name__}: {exc}")
    routes, failures, counts = Counter(), Counter(), Counter()
    by_known = defaultdict(Counter)
    with (args.output_dir/"tensor_qc.jsonl").open("w") as f:
        for i,row in enumerate(candidates):
            p, key = row["pdb_id"], row["feature_stem"]
            try:
                data = torch.load(root/"ligand_features"/p/f"{key}.pt", map_location="cpu", weights_only=False)
                qc = tensor_qc(data, row)
            except Exception as exc:
                qc = dict(tensor_route=-999, qc_fail_reasons=[f"load_or_schema:{type(exc).__name__}"],
                          error=str(exc), old_route_gate=False, expanded_route_gate=False)
            routes[qc["tensor_route"]] += 1
            failures.update(qc["qc_fail_reasons"])
            group = "known_chain" if row["known_train_chain_key"] else "new_chain"
            for flag in ("old_route_gate", "expanded_route_gate", "over_model_atom_cap_256"):
                counts[flag] += bool(qc.get(flag))
                by_known[group][flag] += bool(qc.get(flag))
            by_known[group]["rows"] += 1
            record = dict(row, **qc, **seq_status.get((p,row["protein_chain"]), {}), approved_for_training=False)
            f.write(json.dumps(record, sort_keys=True)+"\n")
            if (i+1) % 500 == 0:
                print("QC", i+1, "seconds", round(time.monotonic()-started,1), flush=True)
    for name, rows in (("existing_file_candidates.tsv", candidates), ("missing_ligand_known_chain_candidates.tsv", missing)):
        if not rows:
            continue
        fields = list(rows[0])+["known_sequence_matches", "sequence_sha256", "sequence_error", "approved_for_training"]
        with (args.output_dir/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, **seq_status.get((row["pdb_id"],row["protein_chain"]), {}), approved_for_training=0))
    summary = dict(existing_candidates=len(candidates), missing_known_chain_candidates=len(missing),
                   tensor_routes=dict(routes), qc_failures=dict(failures), gate_counts=dict(counts),
                   by_known_chain={k:dict(v) for k,v in by_known.items()},
                   known_chain_sequence_checks=len(seq_status),
                   known_chain_sequence_failures=sum(not v["known_sequence_matches"] for v in seq_status.values()),
                   missing_rows_sequence_verified=sum(seq_status[(r["pdb_id"],r["protein_chain"])]["known_sequence_matches"] for r in missing),
                   elapsed_seconds=time.monotonic()-started,
                   provenance={str(p):sha256(p) for p in (metadata,final/"biolip_train_ids.txt",Path(__file__))},
                   limitations=["Tensor gates are not training approval", "No new homology search, receptor geometry or residue-axis QC",
                                "Feature validity and normalization handling must be fixed before broadening routes",
                                "No original data, model, split or features modified"])
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
