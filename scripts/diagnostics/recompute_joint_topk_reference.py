#!/usr/bin/env python3
"""CPU-only, fixed-budget diagnostics from frozen reference TEST predictions.

Every budget is reported. There is no K/threshold/checkpoint selection, model
inference or training. Labels enter metrics only, never prediction budgets.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch


MODES = ("real", "shuffle", "zero")
METRICS = ("f1", "mcc", "precision", "recall")
REFERENCE_SHA = "86ca9c922daef5f4d121c064b85b28c56ff5ca6f6f08bbc7568f41115f0313ec"
POLICY = dict(reference_only=True, test_driven_iteration_allowed=False,
              select_best_k=False, primary="top20", fixed_k=[10, 20, 40, 80],
              length_divisors=[20, 10, 5], rounding="ceil, minimum one",
              ties="stable saved residue order; also analytic uniform-tie expectation",
              scope="each saved protein-ligand row, same original capped residue axis",
              controls=["at_0_5", "at_val_gate"], precision="saved FP32 scores, CPU float64 metrics")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def budgets(n):
    if n <= 0:
        raise ValueError("Empty residue axis")
    return {**{f"top{k}": min(n, k) for k in POLICY["fixed_k"]},
            **{f"top_l{d}": max(1, math.ceil(n / d)) for d in POLICY["length_divisors"]}}


def metrics(n, positive, predicted, tp):
    # Python scalars avoid integer overflow in the pooled MCC denominator.
    n, positive, predicted, tp = map(float, (n, positive, predicted, tp))
    fp, fn, tn = predicted - tp, positive - tp, n - positive - predicted + tp
    denominator = math.sqrt(predicted * positive * (n - positive) * (n - predicted))
    return dict(f1=2 * tp / (predicted + positive) if predicted + positive else 0.,
                mcc=(tp * tn - fp * fn) / denominator if denominator else 0.,
                precision=tp / predicted if predicted else 0.,
                recall=tp / positive if positive else 0.)


def score_row(scores, target, gate):
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    if scores.ndim != 1 or scores.shape != target.shape or not len(scores):
        raise ValueError("Invalid residue axis")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid/nonfinite probability")
    if not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("Invalid frozen VAL gate")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.concatenate(([0], np.cumsum(target[order], dtype=np.int64)))
    n, positive = len(scores), int(target.sum())
    result = {}
    for rule, k in budgets(n).items():
        cutoff = sorted_scores[k - 1]
        left = int(np.searchsorted(-sorted_scores, -cutoff, side="left"))
        right = int(np.searchsorted(-sorted_scores, -cutoff, side="right"))
        expected_tp = float(cumulative[left] + (k - left) *
                            (cumulative[right] - cumulative[left]) / (right - left))
        result[rule] = dict(n=n, positive=positive, predicted=k, tp=int(cumulative[k]),
                            cutoff_tie=int(right > k), expected_tp=expected_tp)
    for rule, threshold in [("at_0_5", .5), ("at_val_gate", gate)]:
        selected = scores >= threshold
        tp = int(target[selected].sum())
        result[rule] = dict(n=n, positive=positive, predicted=int(selected.sum()),
                            tp=tp, cutoff_tie=0, expected_tp=float(tp))
    return result


class Aggregate:
    def __init__(self):
        self.counts = Counter()
        self.macro = Counter()
        self.expected_macro = Counter()

    def add(self, row):
        self.counts.update({k: row[k] for k in ("n", "positive", "predicted", "tp", "cutoff_tie", "expected_tp")})
        self.counts["rows"] += 1
        self.macro.update(metrics(row["n"], row["positive"], row["predicted"], row["tp"]))
        self.expected_macro.update(metrics(row["n"], row["positive"], row["predicted"], row["expected_tp"]))

    def finish(self):
        c = self.counts
        if not c["rows"]:
            raise ValueError("Empty aggregate")
        return dict(n_samples=c["rows"], n_residues=c["n"], n_positive=c["positive"],
                    n_predicted=c["predicted"], mean_predicted=c["predicted"] / c["rows"],
                    cutoff_tie_rows=c["cutoff_tie"], cutoff_tie_fraction=c["cutoff_tie"] / c["rows"],
                    pooled=metrics(c["n"], c["positive"], c["predicted"], c["tp"]),
                    sample_macro={k: self.macro[k] / c["rows"] for k in METRICS},
                    uniform_tie_expected_pooled=metrics(c["n"], c["positive"], c["predicted"], c["expected_tp"]),
                    uniform_tie_expected_sample_macro={k: self.expected_macro[k] / c["rows"] for k in METRICS})


def validate_rows(rows, protocol, mode):
    expected = protocol["samples"]
    if len(rows) != len(expected):
        raise ValueError("Prediction population differs from locked TEST")
    axis = hashlib.sha256()
    for row, spec in zip(rows, expected):
        if row["sample_id"] != spec["sample_id"] or row["source"] != spec["source"]:
            raise ValueError("Sample/source axis differs from locked TEST")
        labels = row["labels"].numpy()
        if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("Invalid saved labels")
        if row["scores"].shape != row["labels"].shape:
            raise ValueError("Saved score/label length differs")
        if mode == "shuffle" and row["donor_id"] != spec["donor_id"]:
            raise ValueError("Changed shuffle donor")
        axis.update(row["sample_id"].encode() + b"\0" + labels.astype(np.int64).tobytes())
    return axis.hexdigest()


def evaluate(input_root, output):
    torch.set_num_threads(1)
    start = time.monotonic()
    paths = [input_root / name for name in ["protocol.json", "reference_test_summary.json",
             *[f"predictions_{mode}.pt" for mode in MODES]]]
    hashes = {str(p): sha(p) for p in paths}
    protocol = json.loads(paths[0].read_text())
    reference = json.loads(paths[1].read_text())
    if hashes[str(paths[0])] != REFERENCE_SHA or reference["protocol_sha256"] != REFERENCE_SHA:
        raise ValueError("Not the locked epoch25 reference TEST")
    if not reference["complete"] or not reference["reference_only"] or reference["test_driven_iteration_allowed"]:
        raise ValueError("Invalid reference evaluation contract")
    if reference["epoch"] != 25 or reference["val_gate"] != protocol["val_gate"]:
        raise ValueError("Changed checkpoint/gate")
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(policy=POLICY, input_sha256=hashes, script_sha256=sha(__file__),
                    parent_protocol_sha256=REFERENCE_SHA, epoch=reference["epoch"],
                    val_gate=reference["val_gate"], job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(output / "protocol.json", contract)
    summaries = {}
    for mode in MODES:
        rows = torch.load(input_root / f"predictions_{mode}.pt", map_location="cpu", weights_only=True)
        axis = validate_rows(rows, protocol, mode)
        if axis != reference["summaries"][mode]["runtime"]["residue_axis_sha256"]:
            raise ValueError("Changed saved residue/label axis")
        aggregate = defaultdict(Aggregate)
        with (output / f"rows_{mode}.csv").open("w", newline="") as handle:
            fields = ["sample_id", "source", "rule", "n", "positive", "predicted", "tp", "cutoff_tie", "expected_tp", *METRICS]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                results = score_row(row["scores"].numpy(), row["labels"].numpy() == 0, protocol["val_gate"])
                for rule, counts in results.items():
                    for source in ("all", row["source"]):
                        aggregate[(source, rule)].add(counts)
                    writer.writerow(dict(sample_id=row["sample_id"], source=row["source"], rule=rule,
                        **counts, **metrics(counts["n"], counts["positive"], counts["predicted"], counts["tp"])))
        summary = {source: {rule: a.finish() for (s, rule), a in aggregate.items() if s == source}
                   for source in ("all", "pdbbind", "biolip")}
        for source in summary:
            original = reference["summaries"][mode][source]
            for rule in ("at_0_5", "at_val_gate"):
                current = summary[source][rule]
                if (current["n_samples"], current["n_residues"], current["n_positive"]) != (
                    original["n_samples"], original["n_residues"], original["n_positive"]):
                    raise ValueError("Changed coverage")
                for metric in METRICS:
                    if not math.isclose(current["pooled"][metric], original[rule + "_global"][metric], abs_tol=1e-12, rel_tol=0):
                        raise ValueError(f"Frozen pooled control mismatch: {mode}/{source}/{rule}/{metric}")
                    if not math.isclose(current["sample_macro"][metric], original["sample_macro"][rule + "_" + metric], abs_tol=1e-12, rel_tol=0):
                        raise ValueError("Frozen sample-macro control mismatch")
        write_json(output / f"summary_{mode}.json", summary)
        summaries[mode] = summary
        del rows
        print("MODE_COMPLETE", mode, flush=True)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError("Input changed during analysis")
    gaps = {source: {rule: {f"real_minus_{mode}": {
        aggregation: {metric: summaries["real"][source][rule][aggregation][metric] -
                     summaries[mode][source][rule][aggregation][metric] for metric in METRICS}
        for aggregation in ("pooled", "sample_macro")} for mode in ("shuffle", "zero")}
        for rule in summaries["real"][source]} for source in ("all", "pdbbind", "biolip")}
    result = dict(complete=True, protocol=contract, summaries=summaries, gaps=gaps,
                  ranking_metrics_unchanged={m: {s: reference["summaries"][m][s]["ranking_global"]
                    for s in ("all", "pdbbind", "biolip")} for m in MODES},
                  seconds=time.monotonic() - start)
    write_json(output / "topk_reference_summary.json", result)
    print("TOPK_REFERENCE_COMPLETE", json.dumps(dict(seconds=result["seconds"], rows=len(protocol["samples"]))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input.resolve(), args.output.resolve())
