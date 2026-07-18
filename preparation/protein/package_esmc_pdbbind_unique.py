#!/usr/bin/env python3
"""Validate and convert PDBbind ESM-C unique features for PLC training.

The star extractor stores selected ESM-C layers concatenated as ``[L, 3456]``
FP32.  PLC's multi-layer fusion consumes ``[L, K, D]``.  This utility selects
only sequences referenced by the PDBbind manifest, reshapes to ``[L,3,1152]``,
and stores FP16 (lossless relative to the BF16 model output for normal values)
to avoid duplicating an unnecessarily large cache on HPC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict payload")
    return payload


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _feature_path(root: Path, row: dict[str, str]) -> Path:
    return root / row["seq_sha256"][:2] / f"{row['unique_id']}.pt"


def _valid_output(path: Path, row: dict[str, str], dtype: torch.dtype) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load(path)
        embeddings = torch.as_tensor(payload["embeddings"])
        return (
            payload.get("sequence_sha256") == row["seq_sha256"]
            and payload.get("sequence") == row["sequence"]
            and payload.get("embedding_layout") == "residue_layer_dim"
            and list(payload.get("repr_layers", [])) == [12, 24, 36]
            and tuple(embeddings.shape) == (int(row["n_residues"]), 3, 1152)
            and embeddings.dtype == dtype
            and bool(torch.isfinite(embeddings).all())
        )
    except Exception:
        return False


def _convert_one(
    row: dict[str, str], source_root: Path, output_root: Path, dtype: torch.dtype, overwrite: bool
) -> tuple[str, int, int, bool]:
    source = _feature_path(source_root, row)
    target = _feature_path(output_root, row)
    if not overwrite and _valid_output(target, row, dtype):
        return row["unique_id"], int(row["n_residues"]), target.stat().st_size, True
    if not source.exists():
        raise FileNotFoundError(source)

    payload = _load(source)
    embeddings = torch.as_tensor(payload.get("embeddings"))
    length = int(row["n_residues"])
    expected = (length, 3456)
    if payload.get("sequence_sha256") != row["seq_sha256"]:
        raise ValueError(f"{source}: SHA-256 metadata mismatch")
    if payload.get("sequence") != row["sequence"]:
        raise ValueError(f"{source}: sequence metadata mismatch")
    if list(payload.get("repr_layers", [])) != [12, 24, 36]:
        raise ValueError(f"{source}: unexpected repr_layers={payload.get('repr_layers')}")
    if payload.get("hidden_normalization") != "final_layernorm":
        raise ValueError(
            f"{source}: unexpected hidden_normalization={payload.get('hidden_normalization')}"
        )
    if tuple(embeddings.shape) != expected:
        raise ValueError(f"{source}: shape={tuple(embeddings.shape)}, expected={expected}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{source}: non-finite embeddings")

    layered = embeddings.reshape(length, 3, 1152).to(dtype=dtype).contiguous()
    converted = dict(payload)
    converted.update(
        {
            "embeddings": layered,
            "embedding_layout": "residue_layer_dim",
            "embedding_combine": "stack",
            "embedding_dim": 1152,
            "embedding_num_layers": 3,
            "embedding_dtype": str(layered.dtype),
            "source_feature_path": str(source),
        }
    )
    _atomic_save(converted, target)
    return row["unique_id"], length, target.stat().st_size, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unique_rows = _read_tsv(args.manifest_dir / "unique_sequences.tsv")
    pdb_rows = _read_tsv(args.manifest_dir / "pdb_manifest.tsv")
    dtype = getattr(torch, args.dtype)
    unique_output = args.output_root / "unique"
    unique_output.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    residues = 0
    bytes_written = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _convert_one,
                row,
                args.source_root,
                unique_output,
                dtype,
                args.overwrite,
            ): row
            for row in unique_rows
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                _, length, size, was_skipped = future.result()
                skipped += int(was_skipped)
                completed += int(not was_skipped)
                residues += length
                bytes_written += size
            except Exception as exc:
                failures.append((row["unique_id"], str(exc)))
            if index % 250 == 0 or index == len(futures):
                print(
                    f"processed={index}/{len(futures)} completed={completed} "
                    f"skipped={skipped} failed={len(failures)}",
                    flush=True,
                )

    if failures:
        failure_path = args.output_root / "failures.tsv"
        with failure_path.open("w") as handle:
            handle.write("unique_id\terror\n")
            for unique_id, error in failures:
                handle.write(f"{unique_id}\t{error.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
        raise RuntimeError(f"{len(failures)} conversions failed; see {failure_path}")

    # Put manifests at the cache root so PocketDatasetV8 can auto-discover it.
    for name in ("unique_sequences.tsv", "pdb_manifest.tsv", "summary.json"):
        source = args.manifest_dir / name
        if source.exists():
            shutil.copy2(source, args.output_root / name)

    summary = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "unique_sequence_count": len(unique_rows),
        "pdb_count": len(pdb_rows),
        "completed": completed,
        "valid_existing": skipped,
        "residues": residues,
        "output_bytes": bytes_written,
        "dtype": str(dtype),
        "embedding_shape": ["L", 3, 1152],
        "repr_layers": [12, 24, 36],
        "hidden_normalization": "final_layernorm",
    }
    (args.output_root / "package_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
