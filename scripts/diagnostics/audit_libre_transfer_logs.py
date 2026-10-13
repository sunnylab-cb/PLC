"""Read-only TB audit for LiBRe-inspired PLC hypotheses, without model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize_events(events):
    """Keep the last wall-time value per step, but expose conflicting repeats."""
    by_step = {}
    for event in events:
        value = float(event.value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite scalar at step {event.step}")
        by_step.setdefault(int(event.step), []).append(event)
    if not by_step:
        return None
    unique = [max(group, key=lambda item: item.wall_time)
              for _, group in sorted(by_step.items())]
    values = [float(event.value) for event in unique]
    best = max(unique, key=lambda item: item.value)
    return {
        "n_events": len(events),
        "n_unique_steps": len(unique),
        "duplicate_steps": sum(len(group) > 1 for group in by_step.values()),
        "max_duplicate_value_spread": max(
            max(float(e.value) for e in group) - min(float(e.value) for e in group)
            for group in by_step.values()
        ),
        "first": [int(unique[0].step), values[0]],
        "last": [int(unique[-1].step), values[-1]],
        "max": [int(best.step), float(best.value)],
        "min": min(values),
        "mean_over_unique_steps": statistics.mean(values),
        "tail5_mean": statistics.mean(values[-5:]),
        "series": [[int(event.step), float(event.value)] for event in unique],
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(tag):
    return tag.startswith(("CNC/", "Health/", "Gradients/")) or tag in {
        "Train/grad_norm", "Val/f1_c0", "Val/auprc_c0_global",
        "Val/auroc_c0_global", "Val/f1_c0_at_best_gate_global",
        "Val/f1_c0_at_0_5_global", "Val/precision_c0", "Val/recall_c0",
    }


def audit_run(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((run / "runs").glob("events.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {run / 'runs'}")
    reader = EventAccumulator(str(run / "runs"), size_guidance={"scalars": 0})
    reader.Reload()
    scalars = {
        tag: summarize_events(reader.Scalars(tag))
        for tag in sorted(reader.Tags().get("scalars", [])) if wanted(tag)
    }
    for tag, summary in scalars.items():
        if summary and not (tag.startswith("Val/") or tag == "CNC/epoch_clip_fraction"):
            summary.pop("series")
    for required in ("CNC/epoch_clip_fraction", "Val/f1_c0", "Val/auprc_c0_global"):
        if not scalars.get(required):
            raise ValueError(f"Missing {required} in {run}")
    return {
        "run": str(run),
        "config_sha256": sha256(run / "config.yaml"),
        "events": [{"path": str(path), "size": path.stat().st_size,
                    "sha256": sha256(path)} for path in event_files],
        "scalars": scalars,
        "same_best_f1_auprc_step": (
            scalars["Val/f1_c0"]["max"][0]
            == scalars["Val/auprc_c0_global"]["max"][0]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    report = {
        "scope": "Existing training TB only; no inference, intervention or retraining.",
        "limitations": [
            "Health/ and CNC/ are TRAIN statistics, not strict-VAL intervention results.",
            "Co-attention norms average padded positions and precede residual gates.",
            "Means over scalar steps are not pooled residue-weighted estimates.",
            "Gradient norms are not causal feature contributions.",
        ],
        "runs": {run.name: audit_run(run) for run in args.run},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = {}
    for name, run in report["runs"].items():
        scalars = run["scalars"]
        compact[name] = {
            "best_val_f1": scalars["Val/f1_c0"]["max"],
            "best_val_auprc": scalars["Val/auprc_c0_global"]["max"],
            "cnc_max": scalars["CNC/epoch_clip_fraction"]["max"],
            "cnc_tail5": scalars["CNC/epoch_clip_fraction"]["tail5_mean"],
            "max_duplicate_spread": max(s["max_duplicate_value_spread"]
                                        for s in scalars.values() if s),
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
