#!/usr/bin/env python3
"""CPU-only follow-up on existing strict VAL exports, with no fitted parameters.

Compare the union of the previous 19 proposal protocols with shape-preserving
intersections with the original contact prediction. Always retain the original
prediction, including an empty one. Rank by top-20%-mean or a soft-count F1
proxy, never by labels. The proxy is a ratio of expected counts, NOT expected
F1 or a guarantee of improvement. This is exploratory validation, not test.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.benchmark_candidate_audit import binary_f1, score_component
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_csv, write_json
from scripts.diagnostics.strict_val_candidates import generate_candidates, protocols


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_candidates(candidates):
    result, seen = [], set()
    for candidate in candidates:
        indices = np.unique(np.asarray(candidate, dtype=np.int64))
        key = tuple(indices.tolist())
        if key not in seen:
            seen.add(key)
            result.append(indices)
    return result


def build_pools(ca, probability, active):
    """No GT argument. Each pool starts with the original prediction as fallback."""
    raw = unique_candidates([
        candidate
        for family, radius, budget in protocols()
        for candidate in generate_candidates(ca, probability, active, family, radius, budget)
    ])
    clipped = [indices[active[indices]] for indices in raw]
    original = [np.flatnonzero(active)]
    return {
        "raw_union": unique_candidates(original + raw),
        "boundary_preserving": unique_candidates(original + clipped),
        "hybrid_union": unique_candidates(original + raw + clipped),
    }


def rank_candidates(probability, candidates, method):
    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid contact probabilities")
    if not candidates:
        raise ValueError("Original prediction must be retained")
    if method == "top20_mean":
        scores = np.array([score_component(probability, indices) if len(indices) else 0.
                           for indices in candidates])
    elif method == "soft_f1_proxy":
        total = probability.sum()
        scores = np.array([2 * probability[indices].sum() / (len(indices) + total)
                           if len(indices) + total > 0 else 0. for indices in candidates])
    else:
        raise ValueError(method)
    # The original is first: deterministic ties never force a replacement.
    return int(np.argmax(scores)), scores


def evaluate_pool(ca, target, active, candidates, selected):
    sizes = np.array([len(indices) for indices in candidates])
    tp = np.array([int(target[indices].sum()) for indices in candidates])
    denominator = sizes + target.sum()
    f1 = np.divide(2. * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    unions = sizes + target.sum() - tp
    iou = np.divide(tp, unions, out=np.zeros(len(tp), dtype=float), where=unions > 0)
    coverage = np.zeros(len(target), dtype=bool)
    for indices in candidates:
        coverage[indices] = True
    return {
        "baseline_f1": binary_f1(active, target), "selected_f1": float(f1[selected]),
        "oracle_f1": float(f1.max()), "n_candidates": len(candidates),
        "selected_original": selected == 0, "n_selected": int(sizes[selected]),
        "n_original": int(active.sum()), "n_true_contacts": int(target.sum()),
        "selected_iou": float(iou[selected]), "oracle_iou": float(iou.max()),
        "hit_iou050": float(iou.max() >= .5) if target.any() else None,
        "target_residue_coverage": float(target[coverage].sum() / target.sum()) if target.any() else None,
        "protein_fraction_covered": float(coverage.mean()),
        "n_full_residues": len(ca),
    }


def summarize(rows):
    summaries = []
    for pool in ("raw_union", "boundary_preserving", "hybrid_union"):
        for method in ("top20_mean", "soft_f1_proxy"):
            chosen = [r for r in rows if r["pool"] == pool and r["method"] == method]
            groups = defaultdict(list)
            for row in chosen:
                groups[row["protein_key"]].append(row)
            metrics = ("baseline_f1", "selected_f1", "oracle_f1", "n_candidates", "selected_original",
                       "n_selected", "n_original", "n_true_contacts", "selected_iou", "oracle_iou",
                       "hit_iou050", "target_residue_coverage", "protein_fraction_covered")
            entry = {"pool": pool, "method": method, "n_samples": len(chosen), "n_proteins": len(groups)}
            for metric in metrics:
                values = [r[metric] for r in chosen if r[metric] is not None]
                balanced = [np.mean([r[metric] for r in g if r[metric] is not None])
                            for g in groups.values() if any(r[metric] is not None for r in g)]
                entry[metric] = float(np.mean(values)) if values else None
                entry[f"protein_mean_{metric}"] = float(np.mean(balanced)) if balanced else None
            delta = np.array([np.mean([r["selected_f1"] - r["baseline_f1"] for r in groups[key]])
                              for key in sorted(groups)])
            rng = np.random.default_rng(20260920)
            draws = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)])
            entry.update(protein_delta=float(delta.mean()),
                         protein_delta_ci025=float(np.quantile(draws, .025)),
                         protein_delta_ci975=float(np.quantile(draws, .975)),
                         improved_rows=sum(r["selected_f1"] > r["baseline_f1"] for r in chosen),
                         worsened_rows=sum(r["selected_f1"] < r["baseline_f1"] for r in chosen))
            summaries.append(entry)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ids = read_strict_splits(args.splits)["val"]
    export = json.loads((args.export / "export_summary.json").read_text())
    if export["split"] != "val" or export["split_sha256"] != sha256(args.splits) or export["n"] != len(ids):
        raise ValueError("Export/split provenance mismatch")
    pred_dir = args.export / "predictions"
    if {p.stem for p in pred_dir.glob("*.npz")} != set(ids):
        raise ValueError("VAL prediction population mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, manifest = [], []
    for i, sid in enumerate(ids):
        path = pred_dir / f"{sid}.npz"
        with np.load(path, allow_pickle=False) as a:
            ca, probability = a["ca_coords"], a["contact_probability"]
            target, active, observed = a["labels"] == 0, a["contact_prediction"].astype(bool), a["observed"].astype(bool)
            protein_key = str(a["protein_key"])
        if ca.shape != (len(target), 3) or probability.shape != target.shape or not np.isfinite(ca).all():
            raise ValueError(f"Malformed axis: {sid}")
        if not np.array_equal(active, observed & (probability >= export["gate"])) or np.any(probability[~observed] != 0):
            raise ValueError(f"Prediction/gate provenance mismatch: {sid}")
        pools = build_pools(ca, probability, active)
        for name, candidates in pools.items():
            for method in ("top20_mean", "soft_f1_proxy"):
                selected, scores = rank_candidates(probability, candidates, method)
                metrics = evaluate_pool(ca, target, active, candidates, selected)
                rows.append({"sample_id": sid, "protein_key": protein_key, "pool": name, "method": method,
                             "n_observed": int(observed.sum()), "selected_score": float(scores[selected]),
                             "original_score": float(scores[0]), **metrics})
        manifest.append({"sample_id": sid, "prediction_sha256": sha256(path)})
        if (i + 1) % 50 == 0:
            print(f"HYBRID_VAL {i+1}/{len(ids)}", flush=True)
    summary = summarize(rows)
    write_csv(args.output / "rows.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_json(args.output / "summary.json", {
        "split": "val", "split_sha256": sha256(args.splits), "n_samples": len(ids),
        "source_export": export, "protocols": summary,
        "script_sha256": sha256(__file__),
        "candidate_generator_sha256": sha256(Path(__file__).with_name("strict_val_candidates.py")),
        "warning": "Exploratory VAL follow-up; no fitted weights/gates; oracle is not performance; bootstrap is not across seeds or sequence clusters.",
        "soft_f1_proxy": "2 * sum(p in candidate) / (candidate size + sum(p over protein)); ratio of expected counts, not expected F1",
        "original_fallback": "Original retained, with ties preferring it; proxy improvement does not guarantee true F1 improvement.",
    })
    write_json(args.output / "prediction_manifest.json", manifest)
    print("STRICT_VAL_HYBRID_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
