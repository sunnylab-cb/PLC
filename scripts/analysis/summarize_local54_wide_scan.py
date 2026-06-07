#!/usr/bin/env python3
"""Summarize broad local54 dependency scan outputs.

Input:
  results/ligdep_v11m8x_coattn_local54_wide_<jobid>/

Outputs:
  local54_wide_scan_summary.csv
  local54_wide_scan_top_drops.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_combined_summaries(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("ligand_dependency_summary.json"))


def _first_summary(obj: Dict) -> Dict:
    summaries = obj.get("summaries", {})
    if not summaries:
        return {}
    return next(iter(summaries.values()))


def _row_from_file(path: Path, root: Path) -> Dict:
    with open(path, "r") as f:
        obj = json.load(f)
    summary = _first_summary(obj)
    mode = summary.get("mode") or (obj.get("modes") or [""])[0]
    return {
        "run_dir": str(path.parent.relative_to(root)),
        "mode": mode,
        "base_mode": summary.get("base_mode", mode),
        "custom_name": summary.get("custom_local54_name", ""),
        "custom_ranges": summary.get("custom_local54_ranges", ""),
        "custom_op": summary.get("custom_local54_op", ""),
        "custom_dim_labels": ";".join(summary.get("custom_local54_dim_labels", [])),
        "global_f1_c0": summary.get("global_f1_c0", ""),
        "global_precision_c0": summary.get("global_precision_c0", ""),
        "global_recall_c0": summary.get("global_recall_c0", ""),
        "mean_sample_f1_c0": summary.get("mean_sample_f1_c0", ""),
        "n_samples": summary.get("n_samples", obj.get("n_samples", "")),
    }


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict], top_k: int) -> None:
    top = sorted(rows, key=lambda r: _as_float(r.get("delta_global_f1_c0")), reverse=True)[:top_k]
    with open(path, "w") as f:
        f.write("# Local54 Wide Scan Top Drops\n\n")
        f.write("| rank | mode | op | ranges | delta F1 | F1 | P/R | dim labels |\n")
        f.write("|---:|---|---|---|---:|---:|---|---|\n")
        for i, row in enumerate(top, 1):
            pr = f"{_as_float(row.get('global_precision_c0')):.4f}/{_as_float(row.get('global_recall_c0')):.4f}"
            labels = str(row.get("custom_dim_labels", "")).replace("|", "/")
            if len(labels) > 180:
                labels = labels[:177] + "..."
            f.write(
                "| {rank} | {mode} | {op} | {ranges} | {delta:.4f} | {f1:.4f} | {pr} | {labels} |\n".format(
                    rank=i,
                    mode=row.get("custom_name") or row.get("mode"),
                    op=row.get("custom_op") or row.get("base_mode"),
                    ranges=row.get("custom_ranges"),
                    delta=_as_float(row.get("delta_global_f1_c0")),
                    f1=_as_float(row.get("global_f1_c0")),
                    pr=pr,
                    labels=labels,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize local54 wide dependency scan results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="scan output root")
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    root = args.root
    rows = [_row_from_file(path, root) for path in _iter_combined_summaries(root)]
    if not rows:
        raise SystemExit(f"no ligand_dependency_summary.json files found under {root}")

    real_f1_values = [
        _as_float(row["global_f1_c0"])
        for row in rows
        if row.get("mode") == "real" and row.get("global_f1_c0") != ""
    ]
    if not real_f1_values:
        raise SystemExit("no real baseline found in scan root")
    real_f1 = real_f1_values[0]

    for row in rows:
        f1 = _as_float(row.get("global_f1_c0"))
        row["real_global_f1_c0"] = f"{real_f1:.8f}"
        row["delta_global_f1_c0"] = f"{real_f1 - f1:.8f}"

    rows = sorted(rows, key=lambda r: _as_float(r.get("delta_global_f1_c0")), reverse=True)
    csv_path = root / "local54_wide_scan_summary.csv"
    md_path = root / "local54_wide_scan_top_drops.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, top_k=args.top_k)

    print(f"real_global_f1_c0={real_f1:.6f}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print("top drops:")
    for row in rows[: min(args.top_k, 20)]:
        print(
            "{delta:.4f} {name} {op} {ranges} f1={f1:.4f}".format(
                delta=_as_float(row.get("delta_global_f1_c0")),
                name=row.get("custom_name") or row.get("mode"),
                op=row.get("custom_op") or row.get("base_mode"),
                ranges=row.get("custom_ranges"),
                f1=_as_float(row.get("global_f1_c0")),
            )
        )


if __name__ == "__main__":
    main()
