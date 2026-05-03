#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely delete fixed_receptors/{pdb_id} directories listed in a text file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--pdbid-list-file", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixed_root = Path(args.fixed_root).resolve()
    ids = [
        line.strip().lower()
        for line in Path(args.pdbid_list_file).read_text().splitlines()
        if line.strip()
    ]

    deleted = 0
    missing = 0
    bad = []
    for pdb_id in ids:
        target = (fixed_root / pdb_id).resolve()
        if target == fixed_root or not str(target).startswith(str(fixed_root) + "/"):
            bad.append(str(target))
            continue
        if not target.exists():
            missing += 1
            continue
        if args.dry_run:
            deleted += 1
            continue
        shutil.rmtree(target)
        deleted += 1

    print(
        f"listed={len(ids)} "
        f"{'would_delete' if args.dry_run else 'deleted'}={deleted} "
        f"missing={missing} bad={len(bad)}"
    )
    if bad:
        print("bad_paths:")
        for path in bad[:20]:
            print(path)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
