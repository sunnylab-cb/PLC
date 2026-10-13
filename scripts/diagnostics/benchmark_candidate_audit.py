#!/usr/bin/env python3
"""CPU-only diagnostics on frozen benchmark predictions; never fits on test labels.

Candidates are spatial components of the saved predicted contact mask. Primary
radius=8 A, minimum size=3, score=top-20%-mean contact probability. Radii 6/10 A
are prespecified sensitivity controls, not a search for the best test setting.
GT is used only to evaluate candidates, compute oracle ceilings and stratify.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def binary_f1(pred, target):
    denom = int(pred.sum()) + int(target.sum())
    return 2.0 * int((pred & target).sum()) / denom if denom else 0.0


def contact_components(ca, active, radius, min_size):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    parent = np.arange(len(indices))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(ca[indices]).query_pairs(radius):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for i, index in enumerate(indices):
        groups[root(i)].append(index)
    return [np.asarray(g, dtype=int) for g in sorted(groups.values(), key=lambda x: x[0])
            if len(g) >= min_size]


def score_component(probability, indices, fraction=0.2):
    values = probability[indices]
    k = max(1, math.ceil(len(values) * fraction))
    return float(np.sort(values)[-k:].mean())


def candidate_metrics(ca, probabilities, active, target, radius=8., min_size=3):
    components = contact_components(ca, active, radius, min_size)
    masks = []
    for indices in components:
        mask = np.zeros(len(active), dtype=bool)
        mask[indices] = True
        masks.append(mask)
    scores = [score_component(probabilities, indices) for indices in components]
    selected_index = int(np.argmax(scores)) if scores else None
    selected = masks[selected_index] if masks else np.zeros_like(active)
    f1s = [binary_f1(mask, target) for mask in masks]
    ious = [int((mask & target).sum()) / max(1, int((mask | target).sum())) for mask in masks]
    centers = [ca[idx].mean(0) for idx in components]
    has_target = bool(target.any())
    distances = [float(np.linalg.norm(center - ca[target].mean(0))) for center in centers] if has_target else []
    row = {
        "baseline_f1": binary_f1(active, target),
        "selected_f1": binary_f1(selected, target),
        "oracle_f1": max(f1s, default=0.),
        "oracle_minus_selected_f1": max(f1s, default=0.) - binary_f1(selected, target),
        "n_candidates": len(components), "no_candidates": int(not components),
        "n_true_contact": int(target.sum()), "n_pred_contact": int(active.sum()),
        "n_selected_contact": int(selected.sum()),
        "n_discarded_small_component_residues": int(active.sum()) - sum(map(len, components)),
        "candidate_hit_dcc4": float(any(d < 4. for d in distances)) if has_target else None,
        "selected_hit_dcc4": float(distances[selected_index] < 4.) if has_target and components else (0. if has_target else None),
        "candidate_hit_iou025": float(any(i >= 0.25 for i in ious)) if has_target else None,
        "candidate_hit_iou050": float(any(i >= 0.50 for i in ious)) if has_target else None,
        "selected_is_f1_oracle": float(bool(f1s) and np.isclose(f1s[selected_index], max(f1s))) if has_target else None,
    }
    return row, selected, masks


@lru_cache(maxsize=20000)
def chemical_identity(path):
    """Exact sanitized heavy-atom isomeric SMILES, not a ligand-code proxy."""
    from rdkit import Chem, rdBase

    try:
        with rdBase.BlockLogs():
            supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False, strictParsing=False)
            mol = next((m for m in supplier if m is not None), None)
            if mol is None:
                raise ValueError("no readable molecule")
            mol = Chem.RemoveHs(mol, sanitize=False)
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    n = len(arrays["labels"])
    if arrays["probs"].shape != (n, 3) or arrays["ca_coords"].shape != (n, 3) or arrays["preds"].shape != (n,):
        raise ValueError(f"Malformed prediction arrays: {path}")
    if not np.isfinite(arrays["probs"]).all() or not np.isfinite(arrays["ca_coords"]).all():
        raise ValueError(f"Nonfinite prediction arrays: {path}")
    if not np.isin(arrays["labels"], [0, 1, 2]).all() or not np.isin(arrays["preds"], [0, 1, 2]).all():
        raise ValueError(f"Invalid class labels: {path}")
    if arrays["residue_keys"].shape != (n, 3):
        raise ValueError(f"Missing canonical residue keys: {path}")
    if not np.allclose(arrays["probs"].sum(-1), 1., atol=2e-3):
        raise ValueError(f"Invalid probabilities: {path}")
    return arrays


def process_protein(task):
    run_root, dataset, protein, csv_rows, sdf_root, radii, min_size = task
    arrays, identities, errors = [], [], []
    for row in csv_rows:
        key = row["pdb_id"]
        if Path(key).name != key:
            raise ValueError(f"Unsafe key {key}")
        a = read_prediction(Path(run_root) / "predictions" / dataset / f"{key}.npz")
        if str(a["protein_key"]) != protein or str(a["ligand_key"]) != row["ligand_key"]:
            raise ValueError(f"Prediction/CSV key mismatch {key}")
        if arrays and (not np.array_equal(a["residue_keys"], arrays[0]["residue_keys"])
                       or not np.allclose(a["ca_coords"], arrays[0]["ca_coords"], atol=1e-5)):
            raise ValueError(f"Cannot compare different protein axes: {protein}")
        value = binary_f1(a["preds"] == 0, a["labels"] == 0)
        if not math.isclose(value, float(row["f1_c0"]), abs_tol=1e-7):
            raise ValueError(f"Saved prediction/CSV F1 mismatch {key}")
        identity, error = chemical_identity(str(Path(sdf_root) / dataset / f"{row['ligand_key']}.sdf"))
        arrays.append(a)
        identities.append(identity)
        errors.append(error)
    targets = [a["labels"] == 0 for a in arrays]
    active = [a["preds"] == 0 for a in arrays]
    centers = [a["ca_coords"][target].mean(0) if target.any() else None for a, target in zip(arrays, targets)]
    distinct = np.zeros((len(arrays), len(arrays)), dtype=bool)
    for i in range(len(arrays)):
        for j in range(i):
            if centers[i] is None or centers[j] is None:
                continue
            overlap = int((targets[i] & targets[j]).sum()) / max(1, int((targets[i] | targets[j]).sum()))
            distinct[i, j] = distinct[j, i] = overlap <= 0.2 and np.linalg.norm(centers[i] - centers[j]) >= 8.
    rows, pairs = [], []
    for i, (a, original) in enumerate(zip(arrays, csv_rows)):
        equivalents = [j for j in range(len(arrays)) if identities[i] is not None and identities[i] == identities[j]]
        for radius in radii:
            metrics, selected, masks = candidate_metrics(a["ca_coords"], a["probs"][:, 0], active[i], targets[i], radius, min_size)
            row = {
                "dataset": dataset, "protein_key": protein, "sample_key": original["pdb_id"],
                "radius": radius, "n_ligand_rows": len(arrays),
                "has_distinct_other_pocket": int(distinct[i].any()),
                "distinct_chem_distinct_pocket": int(any(distinct[i,j] and identities[i] is not None
                    and identities[j] is not None and identities[i] != identities[j] for j in range(len(arrays)))),
                "same_chemical_copies": len(equivalents), "chemical_identity_known": int(identities[i] is not None),
                "chemical_identity_error": errors[i], **metrics,
                "equiv_baseline_f1": max((binary_f1(active[i], targets[j]) for j in equivalents), default=None),
                "equiv_selected_f1": max((binary_f1(selected, targets[j]) for j in equivalents), default=None),
                "equiv_oracle_f1": max((binary_f1(m, targets[j]) for m in masks for j in equivalents), default=0.) if equivalents else None,
            }
            rows.append(row)
    for i in range(len(arrays)):
        for j in range(i):
            if not distinct[i, j]:
                continue
            relation = "unknown_chemistry" if identities[i] is None or identities[j] is None else (
                "same_chemistry" if identities[i] == identities[j] else "different_chemistry")
            pairs.append({
                "dataset": dataset, "protein_key": protein, "sample_a": csv_rows[i]["pdb_id"],
                "sample_b": csv_rows[j]["pdb_id"], "relation": relation,
                "prediction_jaccard": int((active[i] & active[j]).sum()) / max(1, int((active[i] | active[j]).sum())),
                "contact_probability_mae": float(np.abs(arrays[i]["probs"][:, 0] - arrays[j]["probs"][:, 0]).mean()),
                "symmetric_swap_f1_gap": 0.5 * (binary_f1(active[i], targets[i]) - binary_f1(active[j], targets[i])
                    + binary_f1(active[j], targets[j]) - binary_f1(active[i], targets[j])),
                "target_center_distance": float(np.linalg.norm(centers[i] - centers[j])),
            })
    return rows, pairs


METRICS = ("baseline_f1", "selected_f1", "oracle_f1", "oracle_minus_selected_f1", "n_candidates",
           "no_candidates", "candidate_hit_dcc4", "selected_hit_dcc4", "candidate_hit_iou025",
           "candidate_hit_iou050", "n_pred_contact", "n_selected_contact", "equiv_baseline_f1",
           "equiv_selected_f1", "equiv_oracle_f1", "chemical_identity_known")


def means(rows, metrics):
    out = {}
    for name in metrics:
        valid = [float(r[name]) for r in rows if r.get(name) is not None and math.isfinite(float(r[name]))]
        out[name] = {"mean": float(np.mean(valid)) if valid else None, "n": len(valid)}
    return out


def summarize(rows, metrics=METRICS):
    groups = defaultdict(list)
    for row in rows:
        groups[row["protein_key"]].append(row)
    protein_means = [{k: v["mean"] for k, v in means(group, metrics).items()} for group in groups.values()]
    return {"n_rows": len(rows), "n_proteins": len(groups), "sample_mean": means(rows, metrics),
            "protein_mean": means(protein_means, metrics)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--sdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--radii", default="6,8,10")
    parser.add_argument("--min-size", type=int, default=3)
    args = parser.parse_args()
    radii = [float(r) for r in args.radii.split(",")]
    if any(r <= 0 for r in radii) or args.min_size < 1 or 8. not in radii:
        parser.error("Use positive radii including prespecified primary 8 A and min-size >= 1")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"protocol": {
        "primary_radius": 8., "sensitivity_radii": radii, "minimum_component_residues": args.min_size,
        "candidate_source": "saved predicted contact masks, not GT",
        "selector": "top-20%-mean contact probability; deterministic residue-order ties",
        "oracle": "GT-selected F1 upper bound within generated component family; NOT inference performance",
        "threshold": "unchanged saved validation gate", "test_tuning": False,
        "chemical_equivalence": "exact sanitized canonical heavy-atom isomeric SMILES; unknowns retained separately",
        "equiv_metrics": "max over identical-chemistry targets, diagnostic only, not strict per-instance performance",
        "source_eval_root": str(args.eval_root),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, "runs": {}}
    for name in ("base44", "esm3gp44"):
        run = args.eval_root / name / "keys_protein"
        meta_path = run / "benchmark_summary_all.json"
        meta = json.loads(meta_path.read_text())
        if meta["residue_alignment"] != "keys" or meta["residue_coverage"] != "protein":
            raise ValueError(f"P1 requires completed corrected predictions: {run}")
        for dataset in ("coach420", "holo4k"):
            with (run / f"{dataset}_all_results.csv").open() as handle:
                source = list(csv.DictReader(handle))
            if len({r["pdb_id"] for r in source}) != len(source):
                raise ValueError("Duplicate source sample keys")
            expected = {f"{r['pdb_id']}.npz" for r in source}
            if {p.name for p in (run / "predictions" / dataset).glob("*.npz")} != expected:
                raise ValueError(f"Incomplete/stale prediction cache: {run}/{dataset}")
            groups = defaultdict(list)
            for row in source:
                groups[row["protein_key"]].append(row)
            tasks = [(str(run), dataset, protein, group, str(args.sdf_root), radii, args.min_size)
                     for protein, group in sorted(groups.items())]
            rows, pairs = [], []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for i, (r, p) in enumerate(pool.map(process_protein, tasks, chunksize=4), 1):
                    rows.extend(r)
                    pairs.extend(p)
                    if i % 200 == 0:
                        print(f"{name}/{dataset}: {i}/{len(tasks)} proteins", flush=True)
            if len(rows) != len(source) * len(radii):
                raise ValueError("Candidate audit lost samples")
            result = {"source_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(), "radii": {}}
            for radius in radii:
                subset = [r for r in rows if r["radius"] == radius]
                buckets = {
                    "all": subset,
                    "single_ligand_row": [r for r in subset if r["n_ligand_rows"] == 1],
                    "multi_ligand_row": [r for r in subset if r["n_ligand_rows"] > 1],
                    "distinct_other_pocket": [r for r in subset if r["has_distinct_other_pocket"]],
                    "distinct_chem_distinct_pocket": [r for r in subset if r["distinct_chem_distinct_pocket"]],
                    "repeated_chemistry": [r for r in subset if r["same_chemical_copies"] > 1],
                    "multiple_predicted_candidates": [r for r in subset if r["n_candidates"] > 1],
                }
                result["radii"][str(radius)] = {k: summarize(v) for k, v in buckets.items()}
            result["swap_pairs"] = {relation: summarize([p for p in pairs if p["relation"] == relation],
                ("prediction_jaccard", "contact_probability_mae", "symmetric_swap_f1_gap", "target_center_distance"))
                for relation in ("same_chemistry", "different_chemistry", "unknown_chemistry")}
            write_csv(args.output / f"{name}_{dataset}_candidates.csv", rows)
            write_csv(args.output / f"{name}_{dataset}_swap_pairs.csv", pairs)
            summary["runs"][f"{name}/{dataset}"] = result
            (args.output / "candidate_summary.partial.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            print(f"COMPLETED {name}/{dataset}: {len(source)} ligand rows, {len(pairs)} distinct-pocket pairs", flush=True)
    (args.output / "candidate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
