#!/usr/bin/env python3
"""TRAIN-only negative coverage audit; no training artifact or loss is produced.

Report separate upper bounds (mask-only) and conservative chemistry-distinct,
rigid-geometry-verified negatives. Never compare centroids in unrelated frames.
Counts include every requested train row, including zeros and QC failures.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.benchmark_candidate_audit import chemical_identity
from scripts.diagnostics.strict_assignment_common import (
    YEARS, digest_object, find_feature, quantiles, read_strict_splits, write_csv, write_json,
)
from scripts.infer.benchmark_integrity import residue_keys, require_finite, safe_tensor_load, sha256_file
from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, load_overlay


def prepare_chemistry(args, train):
    """Local original PDBbind SDFs, keyed only by TRAIN sample ID."""
    rows = {}
    for sid in train:
        paths = [args.sdf_root / year / sid / f"{sid}_ligand.sdf" for year in YEARS]
        existing = [p for p in paths if p.exists()]
        if not existing:
            continue
        if len(existing) != 1:
            raise ValueError(f"Ambiguous original SDF for {sid}")
        path = existing[0]
        identity, error = chemical_identity(str(path))
        rows[sid] = {"smiles": identity, "error": error, "path": str(path), "sha256": sha256_file(path)}
    write_json(args.output, {"split_sha256": sha256_file(args.root / "splits.txt"), "split": "train", "rows": rows})
    print(f"TRAIN chemistry manifest: {len(rows)} files; {sum(bool(r['smiles']) for r in rows.values())} sanitized identities", flush=True)


def rigid_errors(a, b):
    """Align b to a using ALL corresponding CAs, without pocket masks."""
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(bc.T @ ac)
    fix = np.eye(3)
    fix[-1, -1] = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    fitted = bc @ (u @ fix @ vt)
    err = np.linalg.norm(fitted - ac, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(err.max())


def random_batch_hit_probability(population, negatives, batch_size):
    """Uniform sampling without replacement, conditional on this anchor."""
    draws = min(batch_size - 1, population - 1)
    if not negatives or draws <= 0:
        return 0.
    miss = 1.
    for j in range(draws):
        miss *= max(0., (population - 1 - negatives - j) / (population - 1 - j))
    return 1. - miss


def pair_status(a, b, jaccard_max=.2, center_min=8., overlap_max=3):
    am, bm = a["mask"], b["mask"]
    if not am.any() or not bm.any():
        return {"status": "zero_contact"}
    overlap = int((am & bm).sum())
    jac = overlap / int((am | bm).sum())
    result = {"jaccard": jac, "overlap": overlap}
    if jac > jaccard_max or overlap >= overlap_max:
        return {**result, "status": "high_overlap"}
    # Map BOTH masks into each common residue axis before measuring centers.
    centers = [float(np.linalg.norm(c[am].mean(0) - c[bm].mean(0))) for c in (a["ca"], b["ca"])]
    result.update(center_min=min(centers), center_max=max(centers))
    if min(centers) < center_min:
        return {**result, "status": "nearby_or_ambiguous"}
    if a["geometry_key"] == b["geometry_key"]:
        rmsd = maximum = 0.
    else:
        rmsd, maximum = rigid_errors(a["ca"], b["ca"])
    result.update(rmsd=rmsd, max_ca_error=maximum)
    # <=0.05 A admits rigid frame changes/rounding, not conformational changes.
    result["geometry_verified"] = maximum <= .05
    result["geometry_compatible_exploratory"] = rmsd <= 2. and maximum <= 5.
    relation = "unknown" if not a["smiles"] or not b["smiles"] else (
        "same" if a["smiles"] == b["smiles"] else "different")
    result["chemistry_relation"] = relation
    result["status"] = "distinct_mask_candidate"
    result["valid_conservative"] = bool(result["geometry_verified"] and relation == "different")
    return result


_DISTANCES = None
_ROOT = None
_CHEMISTRY = None
_SOURCES = None
_ALIGNMENT = None


def initialize_worker(root, distances, chemistry, sources, alignment):
    global _ROOT, _DISTANCES, _CHEMISTRY, _SOURCES, _ALIGNMENT
    torch.set_num_threads(1)
    _ROOT, _DISTANCES, _CHEMISTRY, _SOURCES = Path(root), distances, chemistry, sources
    _ALIGNMENT = alignment


def load_sample(sid):
    try:
        protein_file = find_feature(_ROOT, "complete_features_v4_nma", sid, True)
        ligand_file = find_feature(_ROOT, "ligand_features_v5", sid)
        prot, lig = safe_tensor_load(protein_file), safe_tensor_load(ligand_file)
        ca = np.asarray(prot["ca_coords"], dtype=np.float64)
        keys = residue_keys(prot, len(ca))
        names = list(map(str, prot["residue_names"]))
        if len(names) != len(keys) or ca.shape != (len(keys), 3):
            raise ValueError("Protein identity/coordinate length mismatch")
        distance = _DISTANCES[sid]
        if isinstance(distance, dict) and "distances_vdw_adjusted" in distance:
            raise ValueError("Expected dcenter labels, found vdw")
        raw_values = np.asarray(distance["distances"] if isinstance(distance, dict) else distance)
        values = aligned_values(sid, prot, distance, _ALIGNMENT).numpy()
        if values.shape != (len(ca),):
            raise ValueError(f"Label/CA length mismatch: {values.shape} vs {len(ca)}")
        require_finite(ca, sid + " CA")
        require_finite(values, sid + " labels")
        # Auxiliary feature length controls the real training-visible axis.
        esm_file = find_feature(_ROOT, "esm2_latents", sid, True)
        esm = safe_tensor_load(esm_file)
        ekeys = residue_keys(esm, len(esm["embeddings"]))
        aligned = keys[:min(len(keys), len(ekeys))] == ekeys[:min(len(keys), len(ekeys))]
        visible = min(len(ca), len(ekeys), 1800)
        training_contacts_match = np.array_equal(values[:visible] < 5., raw_values[:visible] < 5.)
        path = Path(lig.get("ligand_path", "__missing__"))
        smiles, error, chemical_source = None, "SDF not available", "unknown"
        if path.is_file() and path.suffix.lower() == ".sdf":
            smiles, error = chemical_identity(str(path))
            chemical_source = str(path)
        elif sid in _CHEMISTRY:
            entry = _CHEMISTRY[sid]
            if Path(entry["path"]).name != path.name:
                raise ValueError("Local chemistry manifest/feature ligand filename mismatch")
            smiles, error, chemical_source = entry["smiles"], entry["error"], entry["path"]
        atoms = int(lig["atom_features"].shape[0])
        return {"sample_id": sid, "source": _SOURCES[sid], "axis_key": digest_object([keys, names]),
                "geometry_key": digest_object([keys, names, ca.tolist()]), "ca": ca,
                "mask": values < 5., "smiles": smiles, "chemistry_error": error,
                "chemical_source": chemical_source, "visible": visible, "esm_axis_aligned": aligned,
                "label_overlay_used": sid in _ALIGNMENT, "training_contacts_match": bool(training_contacts_match),
                "n_atoms": atoms, "protein_file": str(protein_file.resolve()), "error": None}
    except Exception as exc:
        return {"sample_id": sid, "source": _SOURCES[sid], "error": f"{type(exc).__name__}: {exc}"}


def main():
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chemistry-manifest", type=Path)
    ap.add_argument("--label-alignment", type=Path)
    ap.add_argument("--prepare-chemistry-only", action="store_true")
    ap.add_argument("--sdf-root", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    split_path = args.root / "splits.txt"
    splits = read_strict_splits(split_path)
    train = splits["train"]
    alignment = load_overlay(args.label_alignment, split_path, train)
    if args.prepare_chemistry_only:
        if args.sdf_root is None:
            raise ValueError("--sdf-root required")
        prepare_chemistry(args, train)
        return
    if len(train) != 10317 or len(splits["val"]) != 647 or len(splits["test"]) != 647:
        raise ValueError("This audit expects the verified strict merged 10317/647/647 split")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for line in (args.root / "provenance.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["sample_id"] in sources and sources[row["sample_id"]] != row["source"]:
            raise ValueError("Conflicting dataset provenance")
        sources[row["sample_id"]] = row["source"]
    chemistry = {}
    if args.chemistry_manifest:
        manifest = json.loads(args.chemistry_manifest.read_text())
        if manifest["split_sha256"] != sha256_file(split_path) or manifest["split"] != "train":
            raise ValueError("Chemistry manifest split mismatch")
        chemistry = manifest["rows"]
        if set(chemistry) - set(train):
            raise ValueError("Non-train chemistry records")
    distances = safe_tensor_load(args.root / "residue_ligand_distances_dcenter.pt")
    # No held-out labels are passed into workers or candidate construction.
    distances = {sid: distances[sid] for sid in train}
    samples = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.root), distances, chemistry, sources, alignment)) as pool:
        for i, sample in enumerate(pool.map(load_sample, train, chunksize=8)):
            samples.append(sample)
            if (i + 1) % 250 == 0:
                print(f"TRAIN_QC {i+1}/{len(train)}", flush=True)
    groups = defaultdict(list)
    counters = {sid: Counter() for sid in train}
    for sample in samples:
        if not sample["error"]:
            groups[sample["axis_key"]].append(sample)
    pair_counts, relations, jaccards, eligible_pairs = Counter(), Counter(), [], []
    for group in groups.values():
        for sample in group:
            counters[sample["sample_id"]]["same_axis_peers"] = len(group) - 1
        for a, b in combinations(group, 2):
            result = pair_status(a, b)
            pair_counts[result["status"]] += 1
            if "jaccard" in result:
                jaccards.append(result["jaccard"])
            if result["status"] != "distinct_mask_candidate":
                continue
            relations[result["chemistry_relation"]] += 1
            visible = min(a["visible"], b["visible"])
            av, bv = dict(a), dict(b)
            for sample in (av, bv):
                sample["mask"], sample["ca"] = sample["mask"][:visible], sample["ca"][:visible]
            cropped = pair_status(av, bv) if visible else {"status": "zero_contact"}
            train_visible = bool(result["valid_conservative"] and a["esm_axis_aligned"] and b["esm_axis_aligned"]
                                 and a["training_contacts_match"] and b["training_contacts_match"]
                                 and cropped.get("valid_conservative", False))
            for sample in (a, b):
                c = counters[sample["sample_id"]]
                c["mask_distinct_upper_bound"] += 1
                c["geometry_verified"] += int(result["geometry_verified"])
                c["geometry_compatible_exploratory"] += int(result["geometry_compatible_exploratory"] and result["chemistry_relation"] == "different")
                c["valid_conservative"] += int(result["valid_conservative"])
                c["valid_training_visible"] += int(train_visible)
            eligible_pairs.append({"sample_a": a["sample_id"], "sample_b": b["sample_id"],
                                   "source_a": a["source"], "source_b": b["source"],
                                   "train_visible": train_visible, **result})
    columns = ("same_axis_peers", "mask_distinct_upper_bound", "geometry_verified",
               "geometry_compatible_exploratory", "valid_conservative", "valid_training_visible")
    rows = []
    for s in samples:
        row = {"sample_id": s["sample_id"], "source": s["source"], "error": s["error"],
               "axis_key": s.get("axis_key"), "geometry_key": s.get("geometry_key"),
               "n_full": len(s["mask"]) if not s["error"] else None, "n_visible": s.get("visible"),
               "n_atoms": s.get("n_atoms"), "esm_axis_aligned": s.get("esm_axis_aligned"),
               "label_overlay_used": s.get("label_overlay_used"), "training_contacts_match": s.get("training_contacts_match"),
               "chemistry_known": bool(s.get("smiles")), "smiles": s.get("smiles"),
               "chemical_source": s.get("chemical_source"), "chemistry_error": s.get("chemistry_error"),
               **{k: counters[s["sample_id"]][k] for k in columns}}
        rows.append(row)
    summaries = {}
    for source in ("all", "pdbbind", "biolip"):
        subset = rows if source == "all" else [r for r in rows if r["source"] == source]
        summaries[source] = {"n_requested": len(subset), "qc_failures": sum(bool(r["error"]) for r in subset),
                             "chemistry_known": sum(r["chemistry_known"] for r in subset),
                             "esm_axis_mismatches": sum(r["esm_axis_aligned"] is False for r in subset),
                             "n_atoms_gt256": sum((r["n_atoms"] or 0) > 256 for r in subset)}
        for col in columns:
            summaries[source][col] = {"covered_samples": sum(r[col] > 0 for r in subset),
                                      "coverage": sum(r[col] > 0 for r in subset) / max(1, len(subset)),
                                      "negatives_per_sample_including_zeros": quantiles(r[col] for r in subset)}
    valid_groups = defaultdict(list)
    for row in rows:
        if row["axis_key"]:
            valid_groups[row["axis_key"]].append(row)
    group_coverage = {key: sum(any(r[key] > 0 for r in group) for group in valid_groups.values())
                      / max(1, len(valid_groups)) for key in columns}
    batch_coverage = {str(b): float(np.mean([random_batch_hit_probability(len(train), r["valid_training_visible"], b)
                                           for r in rows])) for b in (8, 16, 32, 64)}
    write_csv(args.output / "samples.csv", rows)
    write_csv(args.output / "distinct_pairs.csv", eligible_pairs)
    write_json(args.output / "summary.json", {"split": "train", "split_sha256": sha256_file(split_path),
        "label_overlay_sha256": sha256_file(args.label_alignment) if args.label_alignment else None,
        "label_overlay_rows": sum(bool(r["label_overlay_used"]) for r in rows),
        "population": {k: len(v) for k, v in splits.items()}, "summaries": summaries,
        "n_axis_groups": len(groups), "n_multi_axis_groups": sum(len(g) > 1 for g in groups.values()),
        "axis_group_coverage": group_coverage,
        "expected_anchor_coverage_under_uniform_random_batches": batch_coverage,
        "pair_status_counts": dict(pair_counts), "distinct_pair_chemistry_counts": dict(relations),
        "jaccard_all_same_axis_pairs": quantiles(jaccards),
        "criteria": {"jaccard_max": .2, "overlap_must_be_less_than": 3, "both_frame_center_min_A": 8.,
                     "rigid_fit_max_ca_error_A": .05, "exploratory_rmsd_A": 2., "exploratory_max_CA_A": 5.},
        "limitations": ["Conservative coverage is a lower bound, not all homolog pocket negatives.",
                        "Canonical exact isomeric SMILES only; distinct SMILES need not imply nonbinding or chemical dissimilarity.",
                        "Labels checked against prepared residue axes, not regenerated from original structures.",
                        "Same-axis grouping omits renumbered/partially aligned homologs.",
                        "Exploratory compatible geometry is not authorized for direct region losses.",
                        "Atom-cropped rows flagged; full-molecule chemistry may differ from visible atom subsets."],
        "training_artifact_created": False})
    print("STRICT_TRAIN_NEGATIVE_AUDIT_COMPLETE", args.output, flush=True)
    if any(r["error"] for r in rows):
        raise RuntimeError("QC failures recorded in samples.csv; audit is incomplete, not a clean pass")


if __name__ == "__main__":
    main()
