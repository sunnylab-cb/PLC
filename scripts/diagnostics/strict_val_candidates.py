#!/usr/bin/env python3
"""Evaluate whole-protein proposals on strict VAL, never external test labels.

Proposal generation and ranking receive only CA coordinates and frozen model
probabilities. Labels enter evaluate_candidates only. Spheres are NOT cropped
to the final predicted contact mask. Geometry-FPS is a coverage control, not a
trained pocket detector. Oracle F1 is a proposal diagnostic, not model accuracy.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.benchmark_candidate_audit import binary_f1, contact_components, score_component
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_csv, write_json


def fps_seeds(ca, budget):
    if not len(ca) or budget < 1:
        return []
    first = int(np.argmin(((ca - ca.mean(0)) ** 2).sum(-1)))
    seeds, closest = [], np.full(len(ca), np.inf)
    index = first
    for _ in range(min(budget, len(ca))):
        seeds.append(index)
        closest = np.minimum(closest, ((ca - ca[index]) ** 2).sum(-1))
        closest[seeds] = -1
        index = int(np.argmax(closest))
    return seeds


def generate_candidates(ca, probability, active, family, radius, budget):
    """No target/ligand pose argument: no GT-guided candidate construction."""
    if family.startswith("components"):
        selected = active if family == "components_gate" else probability >= float(family.split("_")[1])
        candidates = contact_components(ca, selected, radius, 3)
    else:
        tree = cKDTree(ca)
        if family == "fps":
            seeds = fps_seeds(ca, budget)
        elif family == "prob_nms":
            # Rank every protein residue by mean local probability, not GT centers.
            neighborhoods = tree.query_ball_point(ca, 8.)
            density = np.array([probability[idx].mean() for idx in neighborhoods])
            available, seeds = np.ones(len(ca), bool), []
            for i in np.argsort(-density, kind="stable"):
                if available[i]:
                    seeds.append(int(i))
                    available[tree.query_ball_point(ca[i], 8.)] = False
                    if len(seeds) == budget:
                        break
        else:
            raise ValueError(family)
        candidates = [np.asarray(sorted(tree.query_ball_point(ca[i], radius)), int) for i in seeds]
    unique = {tuple(c.tolist()): c for c in candidates if len(c) >= 3}
    # This selector is GT-independent. Top-20%-mean matches the earlier audit.
    ranked = sorted(unique.values(), key=lambda c: (-score_component(probability, c), tuple(c)))
    return ranked[:budget]


def evaluate_candidates(ca, target, active, candidates):
    masks = []
    for indices in candidates:
        mask = np.zeros(len(ca), bool)
        mask[indices] = True
        masks.append(mask)
    selected = masks[0] if masks else np.zeros(len(ca), bool)
    f1 = [binary_f1(m, target) for m in masks]
    iou = [int((m & target).sum()) / max(1, int((m | target).sum())) for m in masks]
    dcc = [float(np.linalg.norm(ca[m].mean(0) - ca[target].mean(0))) for m in masks] if target.any() else []
    union = np.logical_or.reduce(masks) if masks else np.zeros(len(ca), bool)
    return {
        "baseline_f1": binary_f1(active, target), "selected_f1": binary_f1(selected, target),
        "oracle_f1": max(f1, default=0.), "n_candidates": len(masks),
        "n_selected": int(selected.sum()), "mean_candidate_size": float(np.mean([len(c) for c in candidates])) if candidates else 0.,
        "protein_fraction_covered": float(union.mean()),
        "target_residue_coverage": float((union & target).sum() / target.sum()) if target.any() else None,
        "candidate_hit_iou025": float(any(x >= .25 for x in iou)) if target.any() else None,
        "candidate_hit_iou050": float(any(x >= .5 for x in iou)) if target.any() else None,
        "candidate_hit_dcc4": float(any(x < 4. for x in dcc)) if target.any() else None,
        "selected_hit_dcc4": float(bool(dcc) and dcc[0] < 4.) if target.any() else None,
    }


def protocols():
    result = [("components_gate", 8., 16), ("components_0.1", 6., 16), ("components_0.2", 6., 16)]
    result += [(family, radius, budget) for family in ("prob_nms", "fps")
               for radius in (8., 12.) for budget in (4, 8, 16, 32)]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ids = read_strict_splits(args.splits)["val"]
    actual = {p.stem for p in args.predictions.glob("*.npz")}
    if actual != set(ids):
        raise ValueError(f"VAL prediction population mismatch: expected {len(ids)}, actual {len(actual)}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        with np.load(args.predictions / f"{sid}.npz", allow_pickle=False) as a:
            ca, probability, target, active = a["ca_coords"], a["contact_probability"], a["labels"] == 0, a["contact_prediction"].astype(bool)
            observed = a["observed"].astype(bool)
            protein_key = str(a["protein_key"])
        if ca.shape != (len(target), 3) or probability.shape != target.shape or not np.isfinite(ca).all() or not np.isfinite(probability).all():
            raise ValueError(f"Malformed full-protein prediction: {sid}")
        for family, radius, budget in protocols():
            candidates = generate_candidates(ca, probability, active, family, radius, budget)
            metrics = evaluate_candidates(ca, target, active, candidates)
            rows.append({"sample_id": sid, "protein_key": protein_key, "family": family, "radius": radius,
                         "budget": budget, "n_full_residues": len(target), "n_model_residues": int(observed.sum()),
                         "unobserved_true_contacts": int((target & ~observed).sum()), **metrics})
    write_csv(args.output / "candidate_rows.csv", rows)
    summary = []
    for family, radius, budget in protocols():
        chosen = [r for r in rows if (r["family"], r["radius"], r["budget"]) == (family, radius, budget)]
        groups = defaultdict(list)
        for row in chosen:
            groups[row["protein_key"]].append(row)
        metrics = list(evaluate_candidates(np.zeros((1, 3)), np.ones(1, bool), np.zeros(1, bool), []).keys())
        entry = {"family": family, "radius": radius, "budget": budget, "n_samples": len(chosen), "n_proteins": len(groups)}
        for metric in metrics:
            values = [r[metric] for r in chosen if r[metric] is not None]
            balanced = [np.mean([r[metric] for r in g if r[metric] is not None]) for g in groups.values() if any(r[metric] is not None for r in g)]
            entry[metric] = float(np.mean(values)) if values else None
            entry[f"protein_mean_{metric}"] = float(np.mean(balanced)) if balanced else None
        summary.append(entry)
    write_json(args.output / "summary.json", {"split": "val", "n_samples": len(ids), "protocols": summary,
        "unobserved_probability_policy": "zero; geometry proposals still cover full CA axis",
        "warning": "Validation diagnostic, not held-out performance. Oracle is not an inference selector; no HOLO/COACH labels used."})
    write_csv(args.output / "summary.csv", summary)
    print("STRICT_VAL_CANDIDATES_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
