#!/usr/bin/env python3
"""Metadata-only accession cohorts, then CPU scoring of frozen HOLO predictions.

Post-hoc reference analysis, not a prospectively isolated benchmark. No new
inference, threshold/K selection, homology certificate, or training changes.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark.diagnose_libre_ranking import csv_rows, load_pair
from scripts.benchmark.run_libre_joint_test import compare
from scripts.diagnostics.recompute_joint_topk_reference import METRICS, metrics, score_row, sha, write_json

PARENT_SHA = "05fd2dd155624124c58f6a3ff1876e2c38726a78772001d4ac1eb441ad41d904"
COHORTS = {
    "all_matched": (),
    "no_train_pdb": ("train",),
    "no_train_val_pdb": ("train", "val"),
    "no_internal_pdb": ("train", "val", "test"),
}
POLICY = dict(
    primary_cohort="no_train_val_pdb", primary_rule="top20",
    reference_only=True, test_driven_iteration_allowed=False,
    grouping="exact PDB accession, all ligand rows of a protein stay together",
    selection_inputs="split manifest and successful LiBRe sample IDs only; no scores or labels",
    post_hoc_metadata_analysis=True, sequence_disjoint_certified=False,
    si30_certified=False, libre_train_isolation_certified=False,
    threshold="original PLC VAL gate, released LiBRe FP32 p>0.6",
    report_all_cohorts=True, select_best_cohort_or_k=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protein_key(sample_id):
    return sample_id.split("_", 1)[0]


def accession(protein):
    require(bool(re.fullmatch(r"[0-9A-Za-z]{4}[0-9A-Za-z]*", protein)),
            f"Invalid benchmark protein key: {protein}")
    return protein[:4].lower()


def make_cohorts(split_path, sample_ids):
    """Membership uses identifiers only, never predictions or contact counts."""
    groups = {s: set() for s in ("train", "val", "test")}
    with Path(split_path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split, pdb = row["split"], row["pdb_id"].lower()
            require(split in groups and bool(re.fullmatch(r"[0-9a-z]{4}", pdb)),
                    "Invalid internal split/PDB accession")
            groups[split].add(pdb)
    require(all(groups.values()), "Missing internal split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not groups[a] & groups[b], f"Internal PDB accession crosses {a}/{b}")
    ids = sorted(sample_ids)
    require(len(ids) == len(set(ids)) and bool(ids), "Empty or duplicate matched sample IDs")
    pdb_by_id = {sid: accession(protein_key(sid)) for sid in ids}
    cohorts = {}
    for name, splits in COHORTS.items():
        blocked = set().union(*(groups[s] for s in splits))
        kept = [sid for sid in ids if pdb_by_id[sid] not in blocked]
        cohorts[name] = dict(
            excluded_splits=list(splits), sample_ids=kept, n_samples=len(kept),
            n_protein_keys=len({protein_key(s) for s in kept}),
            n_pdb_accessions=len({pdb_by_id[s] for s in kept}),
            excluded_rows=len(ids) - len(kept),
        )
    exposure = {pdb: [s for s in groups if pdb in groups[s]] for pdb in sorted(set(pdb_by_id.values()))}
    return dict(cohorts=cohorts, pdb_exposure=exposure,
                internal_unique_pdb_counts={s: len(v) for s, v in groups.items()})


def source_hashes():
    files = [Path(__file__), ROOT / "scripts/benchmark/diagnose_libre_ranking.py",
             ROOT / "scripts/benchmark/run_libre_joint_test.py",
             ROOT / "scripts/benchmark/run_libre_benchmark.py",
             ROOT / "scripts/infer/benchmark_integrity.py",
             ROOT / "scripts/diagnostics/recompute_joint_topk_reference.py",
             ROOT / "scripts/diagnostics/benchmark_candidate_audit.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare(parent, output):
    protocol_path = parent / "protocol.json"
    require(sha(protocol_path) == PARENT_SHA, "Not the frozen epoch25 HOLO reference")
    p = json.loads(protocol_path.read_text())
    summary_path = parent / "matched_summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary["complete"] and summary["protocol_sha256"] == PARENT_SHA,
            "Parent evaluation incomplete or changed")
    split_paths = [Path(s) for s in p["asset_sha256"] if s.endswith("/split/candidate_split.tsv")]
    require(len(split_paths) == 1, "Missing/ambiguous frozen split manifest")
    split_path = split_paths[0]
    libre = Path(p["libre"])
    require(sha(split_path) == p["asset_sha256"][str(split_path)], "Changed training split")
    require(sha(libre / "samples.csv") == p["asset_sha256"][str(libre / "samples.csv")], "Changed LiBRe population")
    lib_rows = csv_rows(libre / "samples.csv")
    require(len(lib_rows) == p["expected_matched"] == summary["matched_rows"], "Wrong matched population")
    manifest = make_cohorts(split_path, list(lib_rows))
    # Freeze membership before any NPZ is read. Earlier full-cohort results are
    # already known, so this is explicitly post-hoc, not preregistered testing.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "cohorts.json", manifest)
    paths = [protocol_path, summary_path, split_path, libre / "samples.csv",
             parent / "matched_prediction_audit.json", parent / "inference/holo4k_all_results.csv",
             output / "cohorts.json"]
    contract = dict(policy=POLICY, parent=str(parent), libre=str(libre), val_gate=p["val_gate"],
                    parent_protocol_sha256=PARENT_SHA, source_sha256=source_hashes(),
                    input_sha256={str(path): sha(path) for path in paths},
                    job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(output / "protocol.json", contract)
    print("COHORTS_FROZEN", json.dumps({k: {a: b for a, b in v.items() if a != "sample_ids"}
          for k, v in manifest["cohorts"].items()}), flush=True)


def add_protein_means(result, rows, libre_scores, gate):
    values = defaultdict(list)
    for row in rows:
        y = row["labels"].numpy() == 0
        for model in ("plc", "libre"):
            p = row["scores"].numpy() if model == "plc" else libre_scores[row["sample_id"]]
            rules = score_row(p, y, gate)
            rules.pop("at_val_gate")
            selected = (np.asarray(p, dtype=np.float32) > np.float32(.6) if model == "libre"
                        else np.asarray(p, dtype=np.float64) >= gate)
            rules["native_threshold"] = dict(n=len(y), positive=int(y.sum()),
                                              predicted=int(selected.sum()), tp=int(y[selected].sum()))
            for rule, c in rules.items():
                values[(model, rule, protein_key(row["sample_id"]))].append(
                    metrics(c["n"], c["positive"], c["predicted"], c["tp"]))
    for model in ("plc", "libre"):
        for rule, record in result["models"][model]["all"]["rules"].items():
            groups = [v for (m, r, _), v in values.items() if m == model and r == rule]
            record["protein_mean_of_ligand_mean"] = {
                k: float(np.mean([np.mean([x[k] for x in v]) for v in groups])) for k in METRICS}
            record["protein_mean_of_ligand_max_f1"] = float(np.mean([max(x["f1"] for x in v) for v in groups]))
            record["n_proteins"] = len(groups)


def same_numbers(current, expected, path="models"):
    if isinstance(expected, dict):
        require(current.keys() == expected.keys(), f"Full-cohort fields changed: {path}")
        for key, value in expected.items():
            same_numbers(current[key], value, f"{path}/{key}")
    elif isinstance(expected, (float, int)):
        require(math.isclose(current, expected, rel_tol=0, abs_tol=1e-12), f"Full-cohort metric changed: {path}")
    else:
        require(current == expected, f"Full-cohort value changed: {path}")


def analyze(output):
    start = time.monotonic()
    contract = json.loads((output / "protocol.json").read_text())
    require(contract["policy"] == POLICY and contract["source_sha256"] == source_hashes(), "Changed code/policy")
    for path, digest in contract["input_sha256"].items():
        require(sha(path) == digest, f"Changed input: {path}")
    parent, libre, gate = Path(contract["parent"]), Path(contract["libre"]), contract["val_gate"]
    manifest = json.loads((output / "cohorts.json").read_text())
    lib_rows = csv_rows(libre / "samples.csv")
    plc_rows = csv_rows(parent / "inference/holo4k_all_results.csv")
    audits = json.loads((parent / "matched_prediction_audit.json").read_text())
    audit = {a["sample_id"]: a for a in audits}
    require(len(audit) == len(audits) and set(audit) == set(lib_rows) <= set(plc_rows), "Prediction audit coverage changed")
    rows, lib_scores = {}, {}
    for index, sid in enumerate(manifest["cohorts"]["all_matched"]["sample_ids"]):
        pp = parent / "inference/predictions/holo4k" / (sid + ".npz")
        lp = libre / "predictions" / (sid + ".npz")
        require(sha(pp) == audit[sid]["plc_sha256"] and sha(lp) == audit[sid]["libre_sha256"],
                f"Changed frozen predictions: {sid}")
        p, native, y, _, _, _ = load_pair(lp, pp, lib_rows[sid], plc_rows[sid])
        require(protein_key(sid) == plc_rows[sid]["protein_key"], "Unexpected protein-key convention")
        require(np.array_equal(native["plc"], p["plc"].astype(np.float64) >= gate), "Native PLC gate changed")
        rows[sid] = dict(sample_id=sid, source="holo4k", scores=torch.from_numpy(p["plc"]),
                         labels=torch.from_numpy(np.where(y, 0, 2).astype(np.uint8)))
        lib_scores[sid] = p["libre"]
        if (index + 1) % 1000 == 0:
            print("PREDICTIONS_VERIFIED", index + 1, flush=True)
    results = {}
    for name, cohort in manifest["cohorts"].items():
        selected = [rows[sid] for sid in cohort["sample_ids"]]
        if not selected:
            results[name] = dict(status="empty", n_samples=0)
            continue
        result = compare(selected, lib_scores, gate)
        add_protein_means(result, selected, lib_scores, gate)
        result.update({k: v for k, v in cohort.items() if k != "sample_ids"})
        if name == "all_matched":
            frozen = json.loads((parent / "matched_summary.json").read_text())
            same_numbers(result["models"], frozen["models"])
        results[name] = result
        print("COHORT_COMPLETE", name, json.dumps({m: result["models"][m]["all"]["rules"]["top20"]["sample_macro"]
              for m in ("plc", "libre")}), flush=True)
    for path, digest in contract["input_sha256"].items():
        require(sha(path) == digest, f"Input changed during analysis: {path}")
    write_json(output / "summary.json", dict(
        complete=True, policy=POLICY, results=results, protocol_sha256=sha(output / "protocol.json"),
        parent_protocol_sha256=PARENT_SHA, prediction_files_verified=len(rows) * 2,
        all_cohort_reproduces_parent=True, seconds=time.monotonic() - start,
        job_id=os.environ.get("SLURM_JOB_ID"),
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("prepare", "analyze"))
    ap.add_argument("--parent", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.phase == "prepare":
        require(args.parent is not None, "prepare needs --parent")
        prepare(args.parent.resolve(), args.output.resolve())
    else:
        analyze(args.output.resolve())
