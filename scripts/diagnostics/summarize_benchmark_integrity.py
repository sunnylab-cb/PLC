#!/usr/bin/env python3
"""Paired P0 result analysis; protein-resampled CIs are not multi-seed uncertainty."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.audit_per_ligand_csv import audit, read_rows


def paired_delta(old, new, repetitions=2000):
    old = {r["pdb_id"]: r for r in old}
    new = {r["pdb_id"]: r for r in new}
    if old.keys() != new.keys():
        raise ValueError("Sample sets differ between control arms")
    groups = defaultdict(list)
    for key in sorted(old):
        if old[key]["protein_key"] != new[key]["protein_key"]:
            raise ValueError("Protein key mismatch")
        groups[old[key]["protein_key"]].append(new[key]["f1_c0"] - old[key]["f1_c0"])
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    means = sums / sizes
    rng = np.random.default_rng(44)
    sample, protein = [], []
    for _ in range(repetitions):
        idx = rng.integers(0, len(groups), size=len(groups))
        sample.append(float(sums[idx].sum() / sizes[idx].sum()))
        protein.append(float(means[idx].mean()))
    return {"n_rows": len(old), "n_proteins": len(groups),
            "sample_delta": float(sums.sum() / sizes.sum()),
            "sample_cluster_bootstrap_95ci": np.quantile(sample, [.025, .975]).tolist() if len(groups) > 1 else None,
            "protein_balanced_delta": float(means.mean()),
            "protein_bootstrap_95ci": np.quantile(protein, [.025, .975]).tolist() if len(groups) > 1 else None,
            "replicates": repetitions, "rng_seed": 44}


def qc_summary(path):
    rows = [json.loads(line) for line in path.open()]
    if any(r.get("status", "passed") != "passed" for r in rows):
        raise ValueError("QC source is not an all-passed preflight")
    result = {"n_rows": len(rows), "n_proteins": len({r["protein_key"] for r in rows})}
    for source, fields in {
        "esm2": ["missing_evaluated", "positional_mismatches"],
        "esm3": ["missing_evaluated", "positional_mismatches", "coordinate_rekeyed",
                 "source_duplicate_rows", "source_name3_conflicts"],
        "tag_axis": ["positional_mismatches", "extra_tag_residues"],
    }.items():
        for field in fields:
            flagged = [r for r in rows if r["qc"].get(source, {}).get(field, 0) > 0]
            result[f"{source}/{field}"] = {"n_rows": len(flagged), "n_proteins": len({r["protein_key"] for r in flagged})}
    result["rows_with_sasa_fallback"] = sum(bool(r["qc"]["ligand_repairs"]) for r in rows)
    result["max_label_distance_error"] = max(r["qc"]["tag_axis"]["max_distance_error"] for r in rows)
    return result, {r["sample_key"]: r["qc"] for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root / "eval_12793043"
    arms = ("legacy_bf16", "keys_legacy_budget", "keys_protein")
    result = {"protocol": "same checkpoint, same val gate, same stats, same ligand rows; seed44 only",
              "bootstrap_caveat": "protein-resampled evaluation uncertainty, not training-seed uncertainty",
              "rows": [], "comparisons": {}, "qc": {}, "strata": {}, "metadata": {}}
    all_rows = {}
    for name in ("base44", "esm3gp44"):
        metadata = [json.loads((root / name / arm / "benchmark_summary_all.json").read_text()) for arm in arms]
        for key in ("checkpoint_sha256", "contact_threshold", "feature_stats_sha256", "max_seq_len", "max_atoms"):
            if len({m[key] for m in metadata}) != 1:
                raise ValueError(f"Control mismatch: {name}/{key}")
        result["metadata"][name] = {k: metadata[0][k] for k in
            ("checkpoint_sha256", "contact_threshold", "feature_stats_sha256", "max_seq_len", "max_atoms")}
        for dataset in ("coach420", "holo4k"):
            controls = {}
            for arm in arms:
                run = root / name / arm
                rows = read_rows(run / f"{dataset}_all_results.csv")
                if len({r["pdb_id"] for r in rows}) != len(rows):
                    raise ValueError("Duplicate sample IDs")
                summary = json.loads((run / f"{dataset}_summary.json").read_text())
                if not summary["input_integrity"]["all_evaluated_logits_finite"] or summary["input_integrity"]["n_skipped"]:
                    raise ValueError("Incomplete/nonfinite inference results")
                metrics = summary["sample_mean"]
                result["rows"].append({"model": name, "dataset": dataset, "arm": arm,
                    "f1": metrics["Mean_F1_c0"], "protein_f1": summary["protein_mean_of_ligand_mean"]["Mean_F1_c0"],
                    "protein_best_f1": summary["protein_mean_of_ligand_max"]["Mean_F1_c0"],
                    "precision": metrics["Mean_Prec_c0"], "recall": metrics["Mean_Rec_c0"],
                    "ap": metrics["Mean_AP_c0"], "laa": metrics["LAA"], "pdr": metrics["PDR"],
                    "conditional_bp_f1": metrics["BP_F1_c0"], "coverage": summary["metric_coverage"],
                    "contact_counts": summary["contact_count_distribution"]})
                controls[arm] = rows
            for old, new in ((arms[0], arms[1]), (arms[1], arms[2]), (arms[0], arms[2])):
                result["comparisons"][f"{name}/{dataset}/{old}->{new}"] = paired_delta(controls[old], controls[new])
            result["strata"][f"{name}/{dataset}"] = audit(controls["keys_protein"])
            all_rows[name, dataset] = controls
    for dataset in ("coach420", "holo4k"):
        result["comparisons"][f"{dataset}/base44->esm3gp44/keys_protein"] = paired_delta(
            all_rows["base44", dataset]["keys_protein"], all_rows["esm3gp44", dataset]["keys_protein"])
        result["qc"][dataset], qc = qc_summary(args.root / "preflight_12793034" / f"{dataset}_input_qc.jsonl")
        flags = {
            "esm2_missing_or_shift": lambda q: q["esm2"].get("missing_evaluated", 0) > 0 or q["esm2"].get("positional_mismatches", 0) > 0,
            "tag_axis_shift": lambda q: q["tag_axis"].get("positional_mismatches", 0) > 0,
            "sasa_fallback": lambda q: bool(q["ligand_repairs"]),
        }
        for name in ("base44", "esm3gp44"):
            for flag, predicate in flags.items():
                keys = {key for key, q in qc.items() if predicate(q)}
                if not keys:
                    continue
                control = all_rows[name, dataset]
                old = [r for r in control["legacy_bf16"] if r["pdb_id"] in keys]
                new = [r for r in control["keys_protein"] if r["pdb_id"] in keys]
                result["comparisons"][f"{name}/{dataset}/audit_flag/{flag}"] = paired_delta(old, new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("model dataset arm sample_f1 protein_f1 LAA conditional_BP")
    for r in result["rows"]:
        print(r["model"], r["dataset"], r["arm"], *(f"{r[k]:.6f}" for k in ("f1", "protein_f1", "laa", "conditional_bp_f1")))
    for name, value in result["comparisons"].items():
        print(name, f"delta={value['sample_delta']:.6f}", value["sample_cluster_bootstrap_95ci"])


if __name__ == "__main__":
    main()
