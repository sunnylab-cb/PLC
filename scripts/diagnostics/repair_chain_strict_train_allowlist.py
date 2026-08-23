#!/usr/bin/env python3
"""Remove train samples implicated by an independent pairwise audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_biolip_sequence_leakage import read_split_file


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    current = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current is not None:
                result[current] = "".join(chunks)
            current, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if current is not None:
        result[current] = "".join(chunks)
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    audit = args.audit_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_fasta = read_fasta(audit / "train.fasta")
    offending_ids = set()
    for comparison in ("train_vs_val", "train_vs_test", "train_vs_external"):
        path = audit / f"{comparison}_violations.jsonl"
        if path.is_file():
            offending_ids.update(row["query"] for row in read_jsonl(path))
    missing = offending_ids - set(train_fasta)
    if missing:
        raise RuntimeError(f"audit query IDs absent from train FASTA: {sorted(missing)}")
    offending_sequences = {train_fasta[record_id] for record_id in offending_ids}

    pdb_rows = read_jsonl(source / "pdbbind_chains.metadata.jsonl")
    bio_rows = read_jsonl(source / "biolip_chains.metadata.jsonl")
    splits = read_split_file(source / "pdbbind_chain_si30_splits.txt")
    bio_ids = {
        line.strip() for line in (source / "biolip_train_ids.txt").read_text().splitlines()
        if line.strip()
    }
    pdb_train = set(splits["train"])
    pdb_drop = {
        member["sample_id"]
        for row in pdb_rows if row["sequence"] in offending_sequences
        for member in row["members"] if member["sample_id"] in pdb_train
    }
    bio_drop = {
        member["sample_id"]
        for row in bio_rows if row["sequence"] in offending_sequences
        for member in row["members"] if member["sample_id"] in bio_ids
    }
    retained_pdb_train = sorted(pdb_train - pdb_drop)
    retained_bio = sorted(bio_ids - bio_drop)

    with (output / "pdbbind_chain_si30_splits.txt").open("w") as handle:
        handle.write("# Repaired after independent final pairwise audit\n")
        for sample_id in retained_pdb_train:
            handle.write(f"train\t{sample_id}\n")
        for split in ("val", "test"):
            for sample_id in splits[split]:
                handle.write(f"{split}\t{sample_id}\n")
    (output / "biolip_train_ids.txt").write_text("\n".join(retained_bio) + "\n")

    for name in (
        "pdbbind_chains.metadata.jsonl", "biolip_chains.metadata.jsonl",
        "external_chains.metadata.jsonl",
    ):
        destination = output / name
        if not destination.exists():
            os.symlink(source / name, destination)
    summary = {
        "source_root": str(source),
        "source_audit": str(audit),
        "offending_train_sequences": len(offending_sequences),
        "pdbbind_train_before": len(pdb_train),
        "pdbbind_train_removed": len(pdb_drop),
        "pdbbind_train_after": len(retained_pdb_train),
        "biolip_train_before": len(bio_ids),
        "biolip_train_removed": len(bio_drop),
        "biolip_train_after": len(retained_bio),
        "removed_pdbbind_ids": sorted(pdb_drop),
        "removed_biolip_ids": sorted(bio_drop),
    }
    (output / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
