#!/usr/bin/env python3
"""Summarize V11 class0-bias calibration sweep outputs into a flat table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _get(dct: Dict, key: str, default=None):
    value = dct.get(key, default)
    return default if value is None else value


def _row(path: Path) -> Dict:
    with path.open() as f:
        obj = json.load(f)

    final = dict(obj.get("final_metrics") or {})
    test = obj.get("test_summaries") or {}
    selected = obj.get("selected") or {}

    if not final and test:
        real = test.get("real", {})
        shuffle = test.get("shuffle", {})
        zero = test.get("zero", {})
        real_f1 = real.get("global_f1_c0")
        shuffle_f1 = shuffle.get("global_f1_c0")
        zero_f1 = zero.get("global_f1_c0")
        final = {
            "selected_bias": obj.get("test_selected_bias"),
            "val_select_metric": obj.get("select_metric"),
            "val_select_value": selected.get("select_value"),
            "val_global_f1_c0": selected.get("global_f1_c0"),
            "val_global_precision_c0": selected.get("global_precision_c0"),
            "val_global_recall_c0": selected.get("global_recall_c0"),
            "test_real_global_f1_c0": real_f1,
            "test_real_global_precision_c0": real.get("global_precision_c0"),
            "test_real_global_recall_c0": real.get("global_recall_c0"),
            "test_shuffle_global_f1_c0": shuffle_f1,
            "test_zero_global_f1_c0": zero_f1,
            "test_real_shuffle_gap": None if real_f1 is None or shuffle_f1 is None else real_f1 - shuffle_f1,
            "test_real_zero_gap": None if real_f1 is None or zero_f1 is None else real_f1 - zero_f1,
        }

    return {
        "run": path.parent.name,
        "summary": str(path),
        "model_tag": obj.get("model_tag"),
        "checkpoint": obj.get("checkpoint"),
        "selected_bias": _get(final, "selected_bias"),
        "val_select_value": _get(final, "val_select_value"),
        "val_f1": _get(final, "val_global_f1_c0"),
        "val_precision": _get(final, "val_global_precision_c0"),
        "val_recall": _get(final, "val_global_recall_c0"),
        "test_real_f1": _get(final, "test_real_global_f1_c0"),
        "test_shuffle_f1": _get(final, "test_shuffle_global_f1_c0"),
        "test_zero_f1": _get(final, "test_zero_global_f1_c0"),
        "test_real_shuffle_gap": _get(final, "test_real_shuffle_gap"),
        "test_real_zero_gap": _get(final, "test_real_zero_gap"),
        "test_precision": _get(final, "test_real_global_precision_c0"),
        "test_recall": _get(final, "test_real_global_recall_c0"),
    }


def _find_summaries(roots: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("class0_bias_sweep_summary.json")))
    return sorted(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize class0-bias calibration sweeps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("roots", nargs="+", help="summary JSON files or result directories")
    parser.add_argument("--output_tsv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rows = [_row(path) for path in _find_summaries(args.roots)]
    if not rows:
        raise SystemExit("No class0_bias_sweep_summary.json files found")

    fieldnames = list(rows[0].keys())
    if args.output_tsv:
        out = Path(args.output_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(rows, f, indent=2)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
