#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(event_path: Path):
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    rows = {}
    for tag in ea.Tags().get("scalars", []):
        rows[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return rows


def print_tag(rows, tag, limit=12):
    vals = rows.get(tag)
    if not vals:
        return
    print(f"\n[{tag}] n={len(vals)}")
    for step, value in vals[:limit]:
        print(f"  step={step:<6d} value={value:.6g}")
    if len(vals) > limit:
        print("  ...")
        for step, value in vals[-5:]:
            print(f"  step={step:<6d} value={value:.6g}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_tb_v12.py EVENT_FILE [EVENT_FILE...]")
    for arg in sys.argv[1:]:
        path = Path(arg)
        print("=" * 100)
        print(path)
        rows = load_scalars(path)
        print(f"scalar_tags={len(rows)}")
        for tag in sorted(rows):
            if (
                tag.startswith("Val/")
                or tag.startswith("Train/")
                or tag.startswith("Gradients/")
                or tag.startswith("Health/")
                or tag.startswith("CNC/")
            ):
                print_tag(rows, tag)


if __name__ == "__main__":
    main()
