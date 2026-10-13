#!/usr/bin/env python3
"""CPU-only matched LiBRe/PLC ranking and candidate diagnostics.

No training, test-threshold selection, chemistry repair, or input mutation.
Fixed cardinalities isolate ranking from prediction extent. Shared geometry-only
proposals isolate ranking conditional on that proposal pool, not true LAA.
Ground truth enters metrics/strata only; oracle results are diagnostic ceilings.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from benchmark_candidate_audit import binary_f1, contact_components, score_component


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    result = {r["pdb_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample keys: {path}")
    return result


def top_mask(probability, k):
    if probability.ndim != 1 or not np.isfinite(probability).all() or k < 0:
        raise ValueError("Invalid ranking input")
    mask = np.zeros(len(probability), bool)
    mask[np.argsort(-probability, kind="stable")[:min(k, len(mask))]] = True
    return mask


def budgets(n, plc_count, libre_count):
    out = {f"top{k}": min(n, k) for k in [10, 20, 40, 80]}
    out.update({f"top{int(f * 100)}pct": max(1, math.ceil(n * f)) for f in [.05, .10, .20]})
    out.update(match_plc_count=plc_count, match_libre_count=libre_count)
    return out


def tie_statistics(probability, target, k):
    """Exact F1 expectation/bounds for uniform selection at a tied Top-k cutoff.

    Labels evaluate the random-tie policy; they never choose which tied residues
    to predict. Bounds are sensitivity diagnostics, not attainable model scores.
    """
    k = min(k, len(probability))
    if k <= 0:
        return {"f1_tie_expected": 0., "f1_tie_min": 0., "f1_tie_max": 0.,
                "cutoff_tie": 0, "cutoff_tie_at_one": 0}
    cutoff = np.partition(probability, len(probability) - k)[len(probability) - k]
    above, tied = probability > cutoff, probability == cutoff
    slots = k - int(above.sum())
    count, positives, true_above = int(tied.sum()), int((tied & target).sum()), int((above & target).sum())
    denom = k + int(target.sum())
    ambiguous = slots < count
    return {
        "f1_tie_expected": 2 * (true_above + slots * positives / count) / denom,
        "f1_tie_min": 2 * (true_above + max(0, slots - (count - positives))) / denom,
        "f1_tie_max": 2 * (true_above + min(slots, positives)) / denom,
        "cutoff_tie": int(ambiguous), "cutoff_tie_at_one": int(ambiguous and cutoff == 1.),
    }


def fixed_geometry_candidates(ca, radius=8., budget=32):
    """Farthest-point CA seeds and spheres; neither probabilities nor labels enter."""
    first = int(np.argmin(((ca - ca.mean(0)) ** 2).sum(-1)))
    seeds, closest = [], np.full(len(ca), np.inf)
    index = first
    for _ in range(min(budget, len(ca))):
        seeds.append(index)
        closest = np.minimum(closest, ((ca - ca[index]) ** 2).sum(-1))
        closest[seeds] = -1
        index = int(np.argmax(closest))
    tree = cKDTree(ca)
    unique = {tuple(sorted(tree.query_ball_point(ca[i], radius))) for i in seeds}
    return [np.asarray(c, int) for c in sorted(unique) if len(c) >= 3]


def evaluate_candidates(ca, p, target, candidates):
    # All proposal scoring and selection precede label-dependent evaluation.
    ranked = sorted(candidates, key=lambda c: (-score_component(p, c), tuple(c)))
    f1, ious, distances = [], [], []
    for indices in ranked:
        mask = np.zeros(len(target), bool)
        mask[indices] = True
        f1.append(binary_f1(mask, target))
        ious.append(int((mask & target).sum()) / max(1, int((mask | target).sum())))
        if target.any():
            distances.append(float(np.linalg.norm(ca[indices].mean(0) - ca[target].mean(0))))
    has_target = bool(target.any())
    top_ties = [i for i, c in enumerate(ranked)
                if score_component(p, c) == score_component(p, ranked[0])] if ranked else []
    return {
        "n_candidates": len(ranked), "selected_f1": f1[0] if f1 else 0.,
        "oracle_f1": max(f1, default=0.),
        "oracle_minus_selected_f1": max(f1, default=0.) - (f1[0] if f1 else 0.),
        "n_selected": len(ranked[0]) if ranked else 0,
        "selected_hit_dcc4": float(bool(distances) and distances[0] < 4.) if has_target else None,
        "top3_hit_dcc4": float(any(d < 4. for d in distances[:3])) if has_target else None,
        "any_hit_dcc4": float(any(d < 4. for d in distances)) if has_target else None,
        "selected_hit_iou025": float(bool(ious) and ious[0] >= .25) if has_target else None,
        "top3_hit_iou025": float(any(v >= .25 for v in ious[:3])) if has_target else None,
        "any_hit_iou025": float(any(v >= .25 for v in ious)) if has_target else None,
        "n_top_score_ties": len(top_ties),
        "selected_f1_tie_mean": float(np.mean([f1[i] for i in top_ties])) if top_ties else 0.,
        "selected_hit_dcc4_tie_mean": float(np.mean([distances[i] < 4. for i in top_ties]))
            if has_target and top_ties else (0. if has_target else None),
        "selected_hit_iou025_tie_mean": float(np.mean([ious[i] >= .25 for i in top_ties]))
            if has_target and top_ties else (0. if has_target else None),
    }


def load_pair(libre_path, plc_path, libre_row, plc_row):
    with np.load(libre_path, allow_pickle=False) as a:
        lib = {k: a[k] for k in a.files}
    with np.load(plc_path, allow_pickle=False) as a:
        plc = {k: a[k] for k in a.files}
    n = int(plc_row["n_residues"])
    if plc["probs"].shape != (n, 3) or lib["probabilities"].shape != (n,):
        raise ValueError("Probability axis mismatch")
    if not np.isin(plc["labels"], [0, 1, 2]).all() or lib["labels"].dtype != np.bool_:
        raise ValueError("Label encoding mismatch")
    if lib["labels"].shape != (n,) or plc["labels"].shape != (n,) or plc["preds"].shape != (n,):
        raise ValueError("Label/prediction axis mismatch")
    if not np.isin(plc["preds"], [0, 1, 2]).all():
        raise ValueError("Invalid prediction classes")
    if lib["residue_keys"].shape != (n, 3) or not np.array_equal(lib["residue_keys"], plc["residue_keys"]):
        raise ValueError("Residue key mismatch")
    if lib["ca_coords"].shape != (n, 3) or plc["ca_coords"].shape != (n, 3):
        raise ValueError("Coordinate axis mismatch")
    if not np.allclose(lib["ca_coords"], plc["ca_coords"], rtol=0, atol=1e-5):
        raise ValueError("Coordinate mismatch")
    target = plc["labels"] == 0
    if not np.array_equal(target, lib["labels"]):
        raise ValueError("Per-residue contact labels differ")
    if str(plc["protein_key"]) != plc_row["protein_key"] or str(plc["ligand_key"]) != plc_row["ligand_key"]:
        raise ValueError("Sample identity mismatch")
    probabilities = {"libre": lib["probabilities"], "plc": plc["probs"][:, 0]}
    if not np.isfinite(plc["probs"]).all() or not np.allclose(plc["probs"].sum(-1), 1., atol=2e-3):
        raise ValueError("Invalid PLC probability normalization")
    if not np.isfinite(lib["ca_coords"]).all():
        raise ValueError("Non-finite coordinates")
    native = {"libre": probabilities["libre"] > .6, "plc": plc["preds"] == 0}
    for name, row in [("libre", libre_row), ("plc", plc_row)]:
        p = probabilities[name]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probabilities")
        if int(native[name].sum()) != int(row["n_pred_contact"]) or int(target.sum()) != int(row["n_true_contact"]):
            raise ValueError("Saved prediction/CSV count mismatch")
        if not math.isclose(binary_f1(native[name], target), float(row["f1_c0"]), abs_tol=1e-7):
            raise ValueError("Saved prediction/CSV F1 mismatch")
    return probabilities, native, target, lib["ca_coords"], lib["residue_keys"], str(lib["smiles"])


def process_protein(task):
    dataset, protein, pairs, libre_dir, plc_dir, original_multiplicity = task
    loaded = []
    for key, lr, pr in pairs:
        if Path(key).name != key:
            raise ValueError("Unsafe sample key")
        lp, pp = Path(libre_dir) / f"{key}.npz", Path(plc_dir) / f"{key}.npz"
        p, native, target, ca, keys, smiles = load_pair(lp, pp, lr, pr)
        if loaded and (not np.array_equal(keys, loaded[0][6]) or not np.allclose(ca, loaded[0][5], rtol=0, atol=1e-5)):
            raise ValueError(f"Within-protein residue axes differ: {protein}")
        loaded.append((key, p, native, target, smiles, ca, keys, sha(lp), sha(pp)))
    shared = fixed_geometry_candidates(loaded[0][5])
    targets = [r[3] for r in loaded]
    centroids = [r[5][r[3]].mean(0) if r[3].any() else None for r in loaded]
    out, candidates, audit = [], [], []
    for i, (key, p, native, y, smiles, ca, keys, lib_sha, plc_sha) in enumerate(loaded):
        distinct = []
        for j in range(len(loaded)):
            if j == i or centroids[i] is None or centroids[j] is None:
                continue
            overlap = int((y & targets[j]).sum()) / max(1, int((y | targets[j]).sum()))
            if overlap <= .2 and np.linalg.norm(centroids[i] - centroids[j]) >= 8.:
                distinct.append(j)
        base = {"dataset": dataset, "protein_key": protein, "sample_key": key,
                "n_ligand_rows": original_multiplicity, "has_distinct_other_pocket": int(bool(distinct)),
                "distinct_chem_distinct_pocket": int(any(smiles != loaded[j][4] for j in distinct)),
                "n_residues": len(y), "n_true_contact": int(y.sum())}
        counts = budgets(len(y), int(native["plc"].sum()), int(native["libre"].sum()))
        masks = {name: {**{rule: top_mask(p[name], k) for rule, k in counts.items()}, "native": native[name]}
                 for name in p}
        for name in p:
            ap = float(average_precision_score(y, p[name])) if y.any() else None
            for rule, active in masks[name].items():
                tp, count, positive = int((active & y).sum()), int(active.sum()), int(y.sum())
                out.append({**base, "model": name, "rule": rule, "f1": binary_f1(active, y),
                            "precision": tp / count if count else 0., "recall": tp / positive if positive else 0.,
                            "n_predicted": count, "ap": ap,
                            "n_probability_one": int((p[name] == 1.).sum()),
                            **tie_statistics(p[name], y, count)})
            for rule in ["native", "top20", "top40", "top10pct"]:
                proposed = contact_components(ca, masks[name][rule], radius=8., min_size=3)
                candidates.append({**base, "model": name, "rule": f"components8_{rule}",
                                   **evaluate_candidates(ca, p[name], y, proposed)})
            candidates.append({**base, "model": name, "rule": "shared_geometry_fps32_r8",
                               **evaluate_candidates(ca, p[name], y, shared)})
        audit.append({"dataset": dataset, "protein_key": protein, "sample_key": key,
                      "libre_sha256": lib_sha, "plc_sha256": plc_sha,
                      "aligned_labels_keys_coords": True})
    return out, candidates, audit


def aggregate(rows, fields):
    proteins = defaultdict(list)
    for row in rows:
        proteins[row["protein_key"]].append(row)
    def mean(values):
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None
    return {"n_samples": len(rows), "n_proteins": len(proteins),
            "sample_mean": {k: mean([r[k] for r in rows]) for k in fields},
            "metric_n_samples": {k: sum(r[k] is not None for r in rows) for k in fields},
            "protein_mean": {k: mean([mean([r[k] for r in group]) for group in proteins.values()]) for k in fields}}


def summaries(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        strata = ["all", "one_ligand" if row["n_ligand_rows"] == 1 else
                  "two_three_ligands" if row["n_ligand_rows"] < 4 else "four_plus_ligands"]
        if row["has_distinct_other_pocket"]:
            strata.append("distinct_pockets")
        if row["distinct_chem_distinct_pocket"]:
            strata.append("distinct_chem_distinct_pockets")
        for stratum in strata:
            groups[(row["dataset"], row["model"], row["rule"], stratum)].append(row)
    return [{"dataset": ds, "model": model, "rule": rule, "stratum": st, **aggregate(group, fields)}
            for (ds, model, rule, st), group in sorted(groups.items())]


def paired_difference(rows, field, replicates=2000):
    matched = defaultdict(dict)
    for r in rows:
        key = (r["protein_key"], r["sample_key"])
        if r["model"] in matched[key]:
            raise ValueError("Duplicate paired model row")
        matched[key][r["model"]] = r[field]
    groups = defaultdict(list)
    for (protein, key), values in matched.items():
        if set(values) != {"libre", "plc"}:
            raise ValueError("Incomplete paired model rows")
        groups[protein].append(values["libre"] - values["plc"])
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    means = sums / counts
    rng = np.random.default_rng(20261010)
    sample_boot, protein_boot = [], []
    for _ in range(replicates):
        draw = rng.integers(0, len(groups), len(groups))
        sample_boot.append(sums[draw].sum() / counts[draw].sum())
        protein_boot.append(means[draw].mean())
    return {"delta": "LiBRe minus PLC", "field": field, "bootstrap_unit": "protein",
            "replicates": replicates, "seed": 20261010, "n_pairs": len(matched),
            "sample_difference": float(sums.sum() / counts.sum()),
            "sample_ci95": np.quantile(sample_boot, [.025, .975]).tolist(),
            "protein_difference": float(means.mean()),
            "protein_ci95": np.quantile(protein_boot, [.025, .975]).tolist()}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libre-root", type=Path, required=True)
    ap.add_argument("--plc-root", type=Path, required=True)
    ap.add_argument("--accepted-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-proteins", type=int, default=0, help="Smoke only; zero for the full matched population")
    args = ap.parse_args()
    if args.workers < 1 or args.limit_proteins < 0:
        ap.error("Invalid workers or protein limit")
    args.output.mkdir(parents=True, exist_ok=False)
    accepted = json.loads(args.accepted_report.read_text())
    protocol = {"accepted_report_sha256": sha(args.accepted_report), "runner_sha256": sha(__file__),
                "helper_sha256": sha(sys.modules["benchmark_candidate_audit"].__file__),
                "libre_root": str(args.libre_root), "plc_root": str(args.plc_root),
                "fixed_k": [10, 20, 40, 80], "fixed_fractions": [.05, .10, .20],
                "fraction_rounding": "ceil, minimum one", "tie_break": "stable shared residue order",
                "primary_equal_budget": "top20", "cross_model_budgets": "each frozen model's native count",
                "component_radius": 8., "component_min_size": 3,
                "shared_proposals": "32 farthest-point CA seeds, radius8 spheres, minimum3, geometry only",
                "candidate_rank": "top20percent mean probability; stable residue-index tie break",
                "tie_audit": "v2 follow-up: analytic uniform cutoff-tie F1 expectation/bounds and mean over tied top-scoring candidates; not test selection",
                "label_usage": "metrics and strata only; oracle is diagnostic, not a deployed selector",
                "warning": "Descriptive benchmark diagnostics, no test tuning or selected best protocol; not standard SOTA pocket metrics.",
                "smoke_limit_proteins": args.limit_proteins}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks, expected_keys = [], set()
    for ds, acceptance in accepted.items():
        directory = args.libre_root / ds
        plcdir = args.plc_root / ds / "val_gate"
        reference_path = plcdir / f"{ds}_all_results.csv"
        for name in ["samples.csv", "failures.json", "summary.json", "protocol.json"]:
            if sha(directory / name) != acceptance["input_sha256"][name]:
                raise ValueError(f"Frozen LiBRe input changed: {ds}/{name}")
        if sha(reference_path) != acceptance["input_sha256"]["reference.csv"]:
            raise ValueError("Frozen PLC input changed")
        reference, libre = csv_rows(reference_path), csv_rows(directory / "samples.csv")
        excluded = {r["sample_key"] for r in acceptance["report_acceptance"]["excluded_from_both_models"]}
        if set(libre) != set(reference) - excluded or len(libre) != acceptance["n_successful"]:
            raise ValueError("Approved population changed")
        key_hash = hashlib.sha256(("\n".join(sorted(libre)) + "\n").encode()).hexdigest()
        if key_hash != acceptance["report_acceptance"]["matched_sample_keys_sha256"]:
            raise ValueError("Approved key hash changed")
        counts, groups = defaultdict(int), defaultdict(list)
        for row in reference.values():
            counts[row["protein_key"]] += 1
        for key, row in libre.items():
            groups[row["protein_key"]].append((key, row, reference[key]))
        selected = sorted(groups)
        if args.limit_proteins:
            selected = selected[:args.limit_proteins]
        expected_keys.update((ds, key) for protein in selected for key, _, _ in groups[protein])
        tasks.extend((ds, protein, groups[protein], str(directory / "predictions"),
                      str(plcdir / "predictions" / ds), counts[protein]) for protein in selected)
    start = time.time()
    ranking, candidates, audits = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (r, c, a) in enumerate(executor.map(process_protein, tasks, chunksize=1), 1):
            ranking.extend(r)
            candidates.extend(c)
            audits.extend(a)
            if i % 50 == 0 or i == len(tasks):
                progress = {"proteins": i, "total_proteins": len(tasks), "pairs": len(audits), "seconds": time.time() - start}
                (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
    rank_fields = ["f1", "precision", "recall", "n_predicted", "ap", "cutoff_tie",
                   "f1_tie_expected", "f1_tie_min", "f1_tie_max", "cutoff_tie_at_one", "n_probability_one"]
    candidate_fields = list(evaluate_candidates(np.zeros((1, 3)), np.ones(1), np.zeros(1, bool), []).keys())
    if len(audits) != len(expected_keys) or {(r["dataset"], r["sample_key"]) for r in audits} != expected_keys:
        raise ValueError("Final prediction audit population mismatch")
    paired = []
    for ds in accepted:
        for rule in ["native", "top20", "top10pct", "match_plc_count", "match_libre_count", "shared_geometry_fps32_r8"]:
            source, metric = (candidates, "selected_f1") if rule.startswith("shared") else (ranking, "f1")
            chosen = [r for r in source if r["dataset"] == ds and r["rule"] == rule]
            paired.append({"dataset": ds, "rule": rule, **paired_difference(chosen, metric)})
            tie_metric = "selected_f1_tie_mean" if rule.startswith("shared") else "f1_tie_expected"
            paired.append({"dataset": ds, "rule": rule, **paired_difference(chosen, tie_metric)})
    result = {"complete": True, "is_smoke": bool(args.limit_proteins), "protocol": protocol,
              "n_pairs": len(audits), "elapsed_seconds": time.time() - start,
              "ranking": summaries(ranking, rank_fields), "candidates": summaries(candidates, candidate_fields),
              "paired_differences": paired}
    write_csv(args.output / "ranking_rows.csv", ranking)
    write_csv(args.output / "candidate_rows.csv", candidates)
    write_csv(args.output / "prediction_audit.csv", audits)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("MATCHED_RANKING_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
