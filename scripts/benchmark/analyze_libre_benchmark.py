#!/usr/bin/env python3
"""Validate full accounting and compare LiBRe/PLC on identical successful rows.

Missing predictions stay missing. Bounds illustrate their possible contribution
to a full-population sample mean, not imputed or measured scores. Bootstrap units
are proteins, not correlated ligand rows. No thresholds are tuned here.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    by_key = {r["pdb_id"]: r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"Duplicate sample keys: {path}")
    return by_key


def analyze_extraction_methods(path, predictions, reference, failures):
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    metadata = {row["stem"]: row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("Duplicate historical QC sample keys")
    if set(metadata) != set(reference):
        raise ValueError("Historical QC keys differ from the reference population")
    methods = {key: row["extraction_method"] for key, row in metadata.items()}
    if not all(methods.values()):
        raise ValueError("Missing historical extraction method")
    counts = Counter(methods.values())
    rejected = Counter(methods[row["sample_key"]] for row in failures)
    grouped = defaultdict(list)
    for key in predictions:
        grouped[methods[key]].append(key)
    results = {}
    for method, count in sorted(counts.items()):
        keys = grouped[method]
        libre = float(np.mean([float(predictions[k]["f1_c0"]) for k in keys])) if keys else None
        plc = float(np.mean([float(reference[k]["f1_c0"]) for k in keys])) if keys else None
        results[method] = {
            "n_requested": count, "n_successful": len(keys), "n_rejected": rejected[method],
            "libre_sample_f1": libre, "plc_sample_f1": plc,
            "difference": libre - plc if keys else None,
        }
    return {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_keyset_match": True,
        "caveat": "Historical method metadata; matching keys do not verify unchanged current SDF chemistry. "
                  "Groups are descriptive, not a causal chemistry ablation.",
        "methods": results,
    }


def accept_exclusions(policy, dataset, predictions, reference, failures):
    rule = policy["datasets"][dataset]
    approved = rule["excluded_sample_keys"]
    rejected = [r["sample_key"] for r in failures]
    if len(approved) != len(set(approved)) or len(rejected) != len(set(rejected)):
        raise ValueError("Duplicate approved or rejected sample keys")
    if len(reference) != rule["requested_count"]:
        raise ValueError("Approved reference count mismatch")
    if set(approved) != set(rejected):
        raise ValueError("Failures differ from explicitly approved exclusions")
    if not set(approved) <= set(reference) or set(predictions) != set(reference) - set(approved):
        raise ValueError("Matched cohort differs from approved reference minus exclusions")
    for failure in failures:
        if any(failure.get(k) != v for k, v in policy["failure_signature"].items()):
            raise ValueError("Failure reason differs from approved chemistry rejection")
    return {
        "complete_under_approved_exclusion_policy": True,
        "all_original_rows_have_predictions": not failures,
        "policy_id": policy["policy_id"], "approval": policy["approval"],
        "n_matched_pairs_per_model": len(predictions),
        "excluded_from_both_models": sorted(failures, key=lambda r: r["sample_key"]),
        "matched_sample_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(predictions)) + "\n").encode()).hexdigest(),
    }


def analyze(root, dataset, replicates, historical_qc_root=None, accepted_exclusions=None):
    directory = root / dataset
    summary = json.loads((directory / "summary.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    failures = json.loads((directory / "failures.json").read_text())
    predictions = read_csv(directory / "samples.csv")
    reference = read_csv(directory / "reference.csv")
    failed_keys = [r["sample_key"] for r in failures]
    if len(set(failed_keys)) != len(failed_keys):
        raise ValueError("Duplicate rejected rows")
    if set(predictions) & set(failed_keys) or set(predictions) | set(failed_keys) != set(reference):
        raise ValueError("Successful/rejected rows do not partition the reference population")
    if progress["proteins_processed"] != progress["proteins_total"]:
        raise ValueError("Dataset traversal is incomplete")
    if len(reference) != summary["n_expected"] or len(failures) != summary["n_failed"]:
        raise ValueError("Summary coverage mismatch")
    groups = defaultdict(list)
    for key, row in predictions.items():
        ref = reference[key]
        for field in ["protein_key", "ligand_key", "n_residues", "n_true_contact"]:
            if row[field] != ref[field]:
                raise ValueError(f"Reference alignment mismatch: {key} {field}")
        score, baseline = float(row["f1_c0"]), float(ref["f1_c0"])
        if not 0 <= score <= 1 or not 0 <= baseline <= 1:
            raise ValueError("Invalid F1")
        groups[row["protein_key"]].append(score - baseline)
    sizes = np.array([len(v) for v in groups.values()])
    sums = np.array([sum(v) for v in groups.values()])
    means = sums / sizes
    rng = np.random.default_rng(20261010)
    sample_bootstrap, protein_bootstrap = [], []
    for _ in range(replicates):
        draw = rng.integers(0, len(sizes), size=len(sizes))
        sample_bootstrap.append(sums[draw].sum() / sizes[draw].sum())
        protein_bootstrap.append(means[draw].mean())
    n = len(predictions)
    total = len(reference)
    measured_sum = sum(float(r["f1_c0"]) for r in predictions.values())
    if not np.isclose(measured_sum / n, summary["libre"]["sample_mean"]["f1_c0"], atol=1e-12):
        raise ValueError("Summary sample F1 mismatch")
    matched_plc_f1 = np.mean([float(reference[k]["f1_c0"]) for k in predictions])
    if not np.isclose(matched_plc_f1, summary["plc_frozen_reference_same_subset"]["sample_mean"]["f1_c0"],
                      rtol=0, atol=1e-12):
        raise ValueError("Matched PLC sample F1 mismatch")
    if any(summary[key]["n_samples"] != n for key in ["libre", "plc_frozen_reference_same_subset"]):
        raise ValueError("Matched summary row count mismatch")
    full_multiplicity = Counter(r["protein_key"] for r in reference.values())
    buckets = defaultdict(list)
    for key, row in predictions.items():
        count = full_multiplicity[row["protein_key"]]
        name = "1" if count == 1 else "2-3" if count < 4 else "4+"
        buckets[name].append(key)
    report = {
        "all_reference_rows_accounted": True,
        "all_rows_have_predictions": not failures,
        "n_expected": total, "n_successful": n, "n_rejected": len(failures),
        "coverage_fraction": n / total, "n_successful_proteins": len(groups),
        "elapsed_seconds": summary["elapsed_seconds"],
        "libre": summary["libre"],
        "plc_matched": summary["plc_frozen_reference_same_subset"],
        "pooled_contact_metrics": summary["pooled_contact_metrics"],
        "paired_difference_LiBRe_minus_PLC": {
            "bootstrap_unit": "protein", "replicates": replicates, "seed": 20261010,
            "sample_mean_f1": float(sums.sum() / sizes.sum()),
            "sample_mean_f1_ci95": np.quantile(sample_bootstrap, [.025, .975]).tolist(),
            "protein_mean_f1": float(means.mean()),
            "protein_mean_f1_ci95": np.quantile(protein_bootstrap, [.025, .975]).tolist(),
            "caveat": "sampling uncertainty within this matched dataset, not seed variance or leakage control",
        },
        "unmeasured_full_population_sample_f1_bounds": [measured_sum / total, (measured_sum + total - n) / total],
        "plc_full_population_sample_f1": float(np.mean([float(r["f1_c0"]) for r in reference.values()])),
        "ligand_row_multiplicity_buckets": {
            name: {"n_samples": len(keys), "n_proteins": len({predictions[k]["protein_key"] for k in keys}),
                   "libre_sample_f1": float(np.mean([float(predictions[k]["f1_c0"]) for k in keys])),
                   "plc_sample_f1": float(np.mean([float(reference[k]["f1_c0"]) for k in keys]))}
            for name, keys in buckets.items()
        },
    }
    if historical_qc_root is not None:
        report["historical_extraction_methods"] = analyze_extraction_methods(
            historical_qc_root / f"{dataset}_sdf_qc.tsv", predictions, reference, failures)
    if accepted_exclusions is not None:
        policy = json.loads(accepted_exclusions.read_text())
        report["report_acceptance"] = accept_exclusions(policy, dataset, predictions, reference, failures)
        report["report_acceptance"]["policy_sha256"] = hashlib.sha256(accepted_exclusions.read_bytes()).hexdigest()
        report["input_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ["summary.json", "samples.csv", "reference.csv", "failures.json",
                         "protocol.json", "progress.json", "qc.json"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--historical-qc-root", type=Path)
    parser.add_argument("--accepted-exclusions", type=Path,
                        help="Explicit user-approved exclusion manifest; does not modify original coverage")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    report = {ds: analyze(args.root, ds, args.bootstrap, args.historical_qc_root, args.accepted_exclusions)
              for ds in ["coach420", "holo4k"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for ds, data in report.items():
        print(ds, json.dumps({k: data[k] for k in ["n_successful", "n_rejected",
              "paired_difference_LiBRe_minus_PLC", "ligand_row_multiplicity_buckets"]}), flush=True)


if __name__ == "__main__":
    main()
