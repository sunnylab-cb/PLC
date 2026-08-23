#!/usr/bin/env python3
"""Independently audit the final strict split and merged train allowlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_biolip_sequence_leakage import read_split_file, write_fasta
from build_chain_strict_pdbbind_biolip_split import parse_violations, run_search


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sequences_for_samples(rows: list[dict], sample_ids: set[str]) -> set[str]:
    result = set()
    for row in rows:
        if any(member["sample_id"] in sample_ids for member in row["members"]):
            result.add(row["sequence"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--search-floor", type=float, default=0.28)
    args = parser.parse_args()

    root = args.split_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = read_split_file(root / "pdbbind_chain_si30_splits.txt")
    bio_ids = {
        line.strip() for line in (root / "biolip_train_ids.txt").read_text().splitlines()
        if line.strip()
    }
    pdb_rows = read_jsonl(root / "pdbbind_chains.metadata.jsonl")
    bio_rows = read_jsonl(root / "biolip_chains.metadata.jsonl")
    ext_rows = read_jsonl(root / "external_chains.metadata.jsonl")

    split_sequences = {
        name: sequences_for_samples(pdb_rows, set(ids)) for name, ids in splits.items()
    }
    bio_sequences = sequences_for_samples(bio_rows, bio_ids)
    train_sequences = split_sequences["train"] | bio_sequences
    external_sequences = {row["sequence"] for row in ext_rows}

    fasta_sets = {
        "train": train_sequences,
        "val": split_sequences["val"],
        "test": split_sequences["test"],
        "external": external_sequences,
    }
    for name, sequences in fasta_sets.items():
        write_fasta(
            [(f"{name}_{index:06d}", sequence) for index, sequence in enumerate(sorted(sequences))],
            out / f"{name}.fasta",
        )

    comparisons = (
        ("train_vs_val", "train", "val"),
        ("train_vs_test", "train", "test"),
        ("val_vs_test", "val", "test"),
        ("train_vs_external", "train", "external"),
    )
    commands = []
    counts = {}
    for label, query, target in comparisons:
        raw = out / f"{label}.tsv"
        commands.append(run_search(
            args.mmseqs, out / f"{query}.fasta", out / f"{target}.fasta",
            raw, args.threads, args.search_floor,
        ))
        violations = parse_violations(raw, args.min_seq_id, args.min_coverage)
        counts[label] = len(violations)
        with (out / f"{label}_violations.jsonl").open("w") as handle:
            for row in violations:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "predicate": {
            "minimum_alignment_identity": args.min_seq_id,
            "minimum_coverage": args.min_coverage,
            "coverage_rule": "qcov_or_tcov",
            "search_identity_floor": args.search_floor,
        },
        "samples": {
            "pdbbind_train": len(splits["train"]),
            "biolip_train": len(bio_ids),
            "pdbbind_val": len(splits["val"]),
            "pdbbind_test": len(splits["test"]),
        },
        "unique_chain_sequences": {name: len(values) for name, values in fasta_sets.items()},
        "violating_alignments": counts,
        "passed": all(value == 0 for value in counts.values()),
        "commands": commands,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("final strict split audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
