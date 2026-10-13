#!/usr/bin/env python3
"""Read-only audit of conditional metrics and composition in benchmark CSVs.

No model or benchmark outputs are changed. Ligand-count bins refer to evaluated
rows, not distinct biological sites. Ligand-code repetitions are only a proxy
for chemical equivalence, not proof of identical model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    "f1_c0", "precision_c0", "recall_c0", "ap_c0", "f1_binary",
    "dcc_contact_success", "pdr_union_success", "laa_success", "bp_f1_c0",
    "n_residues", "n_true_contact", "n_pred_contact",
)


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.mean(values) if values else None


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key not in ("dataset", "pdb_id", "protein_key", "ligand_key"):
                row[key] = float(value) if value else float("nan")
    return rows


def stats(rows):
    result = {"n": len(rows), "n_proteins": len({r["protein_key"] for r in rows})}
    result.update({k: mean([r[k] for r in rows]) for k in METRICS})
    result["contact_fraction"] = mean([r["n_true_contact"] / r["n_residues"] for r in rows])
    return result


def audit(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["protein_key"]].append(row)
    for group in groups.values():
        codes = Counter(r["ligand_key"].split("_")[1] for r in group)
        for row in group:
            row["ligands_per_protein"] = len(group)
            row["same_code_copies"] = codes[row["ligand_key"].split("_")[1]]

    multi = [r for r in rows if math.isfinite(r["laa_success"])]
    detected = [r for r in rows if r["dcc_contact_success"] == 1]
    failed = [r for r in rows if r["dcc_contact_success"] != 1]
    any_hit_multi = [r for r in multi if r["pdr_union_success"] == 1]
    matched = [r for r in rows if r["len_protein"] == r["len_esm"] == r["len_labels"]]
    mismatch = [r for r in rows if not r["len_protein"] == r["len_esm"] == r["len_labels"]]
    assert all(math.isfinite(r["f1_c0"]) for r in rows)
    reconstructed = sum(len(part) * mean([r["f1_c0"] for r in part]) for part in (detected, failed) if part) / len(rows)
    assert math.isclose(reconstructed, mean([r["f1_c0"] for r in rows]), abs_tol=1e-12)
    result = {
        "all": stats(rows),
        "protein_balanced_f1": mean([mean([r["f1_c0"] for r in g]) for g in groups.values()]),
        "protein_best_row_f1": mean([max(r["f1_c0"] for r in g) for g in groups.values()]),
        "detected_current": stats(detected),
        "failed_current": stats(failed),
        "multi_evaluable": stats(multi),
        "multi_any_hit": stats(any_hit_multi),
        "multi_rank1_ignoring_distance": mean([float(r["laa_rank"] == 1) for r in multi]),
        "conditional_nearest_correct_given_any_hit": mean([r["laa_success"] for r in any_hit_multi]),
        "lengths_match": stats(matched),
        "lengths_mismatch": stats(mismatch),
        "protein_over1800": stats([r for r in rows if r["len_protein"] > 1800]),
        "zero_true_contact": stats([r for r in rows if r["n_true_contact"] == 0]),
        "ap_nonfinite_positive_label_rows": sum(not math.isfinite(r["ap_c0"]) and r["n_true_contact"] > 0 for r in rows),
        "zero_pred_contact": sum(r["n_pred_contact"] == 0 for r in rows),
        "ligand_cropped": sum(r["ligand_atoms_raw"] != r["n_atoms"] for r in rows),
        "ligand_over_training256": stats([r for r in rows if r["ligand_atoms_raw"] > 256]),
        "same_ligand_code_repeated": stats([r for r in rows if r["same_code_copies"] > 1]),
        "distinct_codes_multi": stats([r for r in rows if r["ligands_per_protein"] > 1 and r["same_code_copies"] == 1]),
        "by_ligand_count": {},
        "by_length": {},
    }
    for name, lo, hi in (("1", 1, 2), ("2", 2, 3), ("3-4", 3, 5), ("5-9", 5, 10), ("10+", 10, math.inf)):
        result["by_ligand_count"][name] = stats([r for r in rows if lo <= r["ligands_per_protein"] < hi])
    for name, lo, hi in (("<200", 0, 200), ("200-399", 200, 400), ("400-799", 400, 800), ("800-1799", 800, 1800), ("1800", 1800, math.inf)):
        result["by_length"][name] = stats([r for r in rows if lo <= r["n_residues"] < hi])
    result["zero_contact_examples"] = [r["pdb_id"] for r in rows if r["n_true_contact"] == 0][:20]
    repeated = defaultdict(list)
    for row in rows:
        repeated[(row["protein_key"], row["ligand_key"].split("_")[1])].append(row)

    def center(row, prefix):
        return [row[f"{prefix}_{axis}"] for axis in "xyz"]

    separated, stationary = 0, 0
    for group in repeated.values():
        if len(group) < 2 or any(r["n_true_contact"] == 0 for r in group):
            continue
        target_spread = max(math.dist(center(a, "true_contact_center"), center(b, "true_contact_center")) for a in group for b in group)
        if target_spread <= 15:
            continue
        separated += 1
        predicted_spread = max(math.dist(center(a, "pred_contact_center"), center(b, "pred_contact_center")) for a in group for b in group)
        stationary += predicted_spread <= 1
    result["repeated_code_target_spread_gt15A_groups"] = separated
    result["of_those_predicted_spread_le1A_groups"] = stationary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = {}
    for path in sorted(args.root.glob("seed*/val_gate/*_all_results.csv")):
        result[str(path.relative_to(args.root))] = audit(read_rows(path))
    if not result:
        raise SystemExit("No seed*/val_gate/*_all_results.csv files found")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
