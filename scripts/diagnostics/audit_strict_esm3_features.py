#!/usr/bin/env python3
"""Audit ESM3 finiteness, scale, and residue-key coverage on a materialized root."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


YEARS = ("1981-2000", "2001-2010", "2011-2019")


def find(root: Path, feature: str, sample: str) -> Path:
    for year in YEARS:
        path = root / feature / year / sample / f"{sample}_protein.pt"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{feature}: {sample}")


def keys(obj: dict) -> list[tuple[str, int, str]] | None:
    chains = obj.get("chain_id", obj.get("chain_ids"))
    residues = obj.get("residue_index", obj.get("residue_indices"))
    insertions = obj.get("insertion_code", obj.get("insertion_codes"))
    if chains is None or residues is None:
        return None
    if insertions is None:
        insertions = [""] * len(residues)
    return [(str(c), int(r), str(i).strip()) for c, r, i in zip(chains, residues, insertions)]


def inspect(item: tuple[str, str, str]) -> dict:
    root_s, split, sample = item
    root = Path(root_s)
    try:
        e2 = torch.load(find(root, "esm2_latents", sample), map_location="cpu", weights_only=False)
        e3 = torch.load(find(root, "esm3_latents_norm", sample), map_location="cpu", weights_only=False)
        x = e3["embeddings"].float()
        k2, k3 = keys(e2), keys(e3)
        if k2 is None or k3 is None:
            matched = min(len(e2["embeddings"]), len(x))
            reference = len(e2["embeddings"])
        else:
            k3_set = set(k3)
            matched = sum(k in k3_set for k in k2)
            reference = len(k2)
        return {
            "split": split, "sample": sample, "rows": int(x.shape[0]),
            "nonfinite": int((~torch.isfinite(x)).sum()),
            "abs_max": float(torch.nan_to_num(x).abs().max()),
            "matched": matched, "reference": reference,
        }
    except Exception as exc:
        return {"split": split, "sample": sample, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    items = []
    for line in (args.data_root / "splits.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            split, sample = line.split()[:2]
            items.append((str(args.data_root), split, sample))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, items, chunksize=8))
    summary = {}
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        valid = [r for r in part if "error" not in r]
        summary[split] = {
            "samples": len(part), "errors": sum("error" in r for r in part),
            "samples_nonfinite": sum(r.get("nonfinite", 0) > 0 for r in part),
            "values_nonfinite": sum(r.get("nonfinite", 0) for r in part),
            "samples_partial_alignment": sum(r.get("matched") != r.get("reference") for r in valid),
            "matched_rows": sum(r["matched"] for r in valid),
            "reference_rows": sum(r["reference"] for r in valid),
            "max_abs": max((r["abs_max"] for r in valid), default=float("nan")),
        }
    payload = {"summary": summary, "problems": [r for r in rows if "error" in r or r.get("nonfinite", 0) or r.get("matched") != r.get("reference")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if any(v["errors"] or v["samples_nonfinite"] for v in summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
