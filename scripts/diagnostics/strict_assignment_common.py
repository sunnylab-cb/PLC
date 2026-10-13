"""Small, fail-closed helpers shared by strict-val/train diagnostics."""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np

YEARS = ("1981-2000", "2001-2010", "2011-2019")


def read_strict_splits(path):
    groups = {k: [] for k in ("train", "val", "test")}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2 or parts[0] not in groups:
            raise ValueError(f"Unexpected split row: {line}")
        if Path(parts[1]).name != parts[1]:
            raise ValueError("Unsafe sample ID")
        groups[parts[0]].append(parts[1])
    flattened = sum(groups.values(), [])
    if not all(groups.values()) or len(flattened) != len(set(flattened)):
        raise ValueError("Empty split, duplicate ID, or train/val/test overlap")
    return groups


def find_feature(root, directory, sample, protein=False):
    suffixes = ("protein_conf0", "protein") if protein else ("ligand",)
    for year in YEARS:
        for suffix in suffixes:
            path = Path(root) / directory / year / sample / f"{sample}_{suffix}.pt"
            if path.exists():
                return path
    raise FileNotFoundError(f"{directory}/{sample}")


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def quantiles(values):
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {k: None for k in ("mean", "p0", "p25", "p50", "p75", "p90", "p100")}
    return {"mean": float(a.mean()), **{f"p{q}": float(np.percentile(a, q)) for q in (0, 25, 50, 75, 90, 100)}}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
