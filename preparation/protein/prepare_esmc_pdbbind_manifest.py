#!/usr/bin/env python3
"""Build an exact-sequence ESM-C extraction manifest for PLC/PDBbind.

The PLC dataset is indexed by PDB ID, while ESM-C extraction is deduplicated by
the SHA-256 of the complete residue sequence.  This utility validates the
dataset sequence table against the existing residue-aligned ESM2 cache, then
writes both a unique-sequence extraction manifest and a PDB-to-sequence map.

The output ``unique_sequences.tsv`` is directly accepted by
``extract_esmc_unique.py`` from the PPC ESM-C pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def _sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _normalize_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def _load_mapping(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no rows")
    required = {"pdb_id", "sequence_length", "sequence"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    normalized = []
    seen = set()
    for row in rows:
        pdb_id = row["pdb_id"].strip().lower()
        sequence = _normalize_sequence(row["sequence"])
        expected_length = int(row["sequence_length"])
        if not pdb_id or pdb_id in seen:
            raise ValueError(f"duplicate or empty pdb_id: {pdb_id!r}")
        if not sequence or len(sequence) != expected_length:
            raise ValueError(
                f"{pdb_id}: sequence length {len(sequence)} != {expected_length}"
            )
        try:
            sequence.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{pdb_id}: non-ASCII sequence") from exc
        seen.add(pdb_id)
        normalized.append({"pdb_id": pdb_id, "sequence": sequence})
    return normalized


def _index_esm2(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: defaultdict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*_protein.pt")):
        pdb_id = path.name.removesuffix("_protein.pt").lower()
        if pdb_id in index:
            duplicates[pdb_id].extend([index[pdb_id], path])
        else:
            index[pdb_id] = path
    if duplicates:
        details = "; ".join(
            f"{pdb_id}: {[str(p) for p in paths]}"
            for pdb_id, paths in list(duplicates.items())[:10]
        )
        raise ValueError(f"duplicate ESM2 cache IDs under {root}: {details}")
    return index


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_esm2(path: Path, pdb_id: str) -> str:
    payload = _load_torch(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict payload")
    cached_sequence = _normalize_sequence(payload.get("sequence", ""))
    if not cached_sequence:
        raise ValueError(f"{pdb_id}: ESM2 cache has no sequence")
    embeddings = torch.as_tensor(payload.get("embeddings"))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cached_sequence):
        raise ValueError(
            f"{pdb_id}: invalid ESM2 shape {tuple(embeddings.shape)}, expected "
            f"({len(cached_sequence)}, D)"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{pdb_id}: ESM2 embeddings contain non-finite values")
    return cached_sequence


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return int(ordered[index])


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("preparation/dataset_split/dataset_proteins.mapping.txt"),
    )
    parser.add_argument("--esm2-root", type=Path, default=Path("data/esm2_latents"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequence-source",
        choices=["esm2", "mapping"],
        default="esm2",
        help=(
            "Use the residue-aligned ESM2 cache sequence by default. The split mapping "
            "can differ from the tensors actually consumed by training."
        ),
    )
    parser.add_argument(
        "--skip-esm2-payload-qc",
        action="store_true",
        help="Only require each PDB ID to have an ESM2 file; do not torch.load payloads.",
    )
    args = parser.parse_args()

    rows = _load_mapping(args.mapping)
    esm2_index = _index_esm2(args.esm2_root)
    mapping_ids = {row["pdb_id"] for row in rows}
    missing_esm2 = sorted(mapping_ids - set(esm2_index))
    extra_esm2 = sorted(set(esm2_index) - mapping_ids)
    if missing_esm2:
        raise FileNotFoundError(
            f"missing ESM2 caches for {len(missing_esm2)} IDs: {missing_esm2[:20]}"
        )

    sequence_to_pdbs: defaultdict[str, list[str]] = defaultdict(list)
    mapping_mismatches: list[dict[str, Any]] = []
    pdb_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        pdb_id = row["pdb_id"]
        mapping_sequence = row["sequence"]
        esm2_path = esm2_index[pdb_id]
        if args.skip_esm2_payload_qc:
            if args.sequence_source == "esm2":
                raise ValueError(
                    "--sequence-source esm2 requires ESM2 payload QC; remove "
                    "--skip-esm2-payload-qc"
                )
            sequence = mapping_sequence
        else:
            esm2_sequence = _validate_esm2(esm2_path, pdb_id)
            if esm2_sequence != mapping_sequence:
                mapping_mismatches.append(
                    {
                        "pdb_id": pdb_id,
                        "mapping_length": len(mapping_sequence),
                        "esm2_length": len(esm2_sequence),
                        "length_delta": len(esm2_sequence) - len(mapping_sequence),
                        "mapping_sha256": _sha256(mapping_sequence),
                        "esm2_sha256": _sha256(esm2_sequence),
                    }
                )
            sequence = esm2_sequence if args.sequence_source == "esm2" else mapping_sequence
        digest = _sha256(sequence)
        unique_id = f"esmcuseq_L{len(sequence):05d}_{digest[:16]}"
        sequence_to_pdbs[sequence].append(pdb_id)
        pdb_rows.append(
            {
                "pdb_id": pdb_id,
                "unique_id": unique_id,
                "seq_sha256": digest,
                "n_residues": len(sequence),
                "esm2_relpath": esm2_path.relative_to(args.esm2_root).as_posix(),
                "sequence": sequence,
            }
        )
        if index % 1000 == 0 or index == len(rows):
            print(f"validated {index}/{len(rows)} PDB caches", flush=True)

    unique_rows = []
    for sequence, pdb_ids in sequence_to_pdbs.items():
        digest = _sha256(sequence)
        unique_rows.append(
            {
                "unique_id": f"esmcuseq_L{len(sequence):05d}_{digest[:16]}",
                "seq_sha256": digest,
                "n_residues": len(sequence),
                "n_pdb_occurrences": len(pdb_ids),
                "sequence": sequence,
            }
        )
    unique_rows.sort(key=lambda row: (int(row["n_residues"]), row["seq_sha256"]))
    pdb_rows.sort(key=lambda row: row["pdb_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        args.output_dir / "unique_sequences.tsv",
        ["unique_id", "seq_sha256", "n_residues", "n_pdb_occurrences", "sequence"],
        unique_rows,
    )
    _write_tsv(
        args.output_dir / "pdb_manifest.tsv",
        [
            "pdb_id",
            "unique_id",
            "seq_sha256",
            "n_residues",
            "esm2_relpath",
            "sequence",
        ],
        pdb_rows,
    )
    _write_tsv(
        args.output_dir / "mapping_sequence_mismatches.tsv",
        [
            "pdb_id",
            "mapping_length",
            "esm2_length",
            "length_delta",
            "mapping_sha256",
            "esm2_sha256",
        ],
        mapping_mismatches,
    )

    lengths = [int(row["n_residues"]) for row in unique_rows]
    occurrence_histogram = Counter(len(pdb_ids) for pdb_ids in sequence_to_pdbs.values())
    summary = {
        "mapping_path": str(args.mapping.resolve()),
        "esm2_root": str(args.esm2_root.resolve()),
        "esm2_payload_qc": not args.skip_esm2_payload_qc,
        "sequence_source": args.sequence_source,
        "pdb_count": len(pdb_rows),
        "unique_sequence_count": len(unique_rows),
        "duplicate_sequence_pdb_count": len(pdb_rows) - len(unique_rows),
        "extra_esm2_cache_count": len(extra_esm2),
        "mapping_sequence_mismatch_count": len(mapping_mismatches),
        "mapping_sequence_mismatch_examples": mapping_mismatches[:20],
        "sequence_length": {
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "gt_1024": sum(length > 1024 for length in lengths),
            "gt_1600": sum(length > 1600 for length in lengths),
        },
        "pdb_occurrences_per_unique_sequence": {
            str(count): frequency for count, frequency in sorted(occurrence_histogram.items())
        },
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
