#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['record_id']}\n")
            seq = row["sequence"]
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx:idx + 80] + "\n")


def build_group_index(group_rows: List[dict]) -> Dict[str, dict]:
    return {row["group_id"]: row for row in group_rows}


def load_representative_sequences(seq_rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in seq_rows:
        grouped.setdefault(row["group_id"], []).append(row)
    for group_id in grouped:
        grouped[group_id].sort(key=lambda x: int(x["sequence_index"]))
    return grouped


def make_record_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:07d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MMseqs2-ready L2 input FASTA and mapping from L1 manifest outputs."
    )
    parser.add_argument(
        "--l1-dir",
        required=True,
        help="Directory produced by build_l1_manifest.py",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for L2 MMseqs2 input files",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Suggested MMseqs2 coverage threshold written into command template",
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=0.95,
        help="Suggested MMseqs2 minimum sequence identity written into command template",
    )
    args = parser.parse_args()

    l1_dir = Path(args.l1_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    groups_path = l1_dir / "l1_groups.tsv"
    reps_path = l1_dir / "l1_representative_sequences.tsv"
    summary_path = l1_dir / "summary.json"
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path}")
    if not reps_path.exists():
        raise FileNotFoundError(f"Missing {reps_path}")

    group_rows = read_tsv(groups_path)
    seq_rows = read_tsv(reps_path)
    group_index = build_group_index(group_rows)
    sequences_by_group = load_representative_sequences(seq_rows)
    l1_summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            l1_summary = json.load(handle)

    all_unique_rows: List[dict] = []
    longest_rows: List[dict] = []
    group_catalog_rows: List[dict] = []

    record_counter = 1
    longest_counter = 1
    for group_id in sorted(group_index):
        group_meta = group_index[group_id]
        seqs = sequences_by_group.get(group_id, [])
        if not seqs:
            continue

        rep_entry_key = group_meta["representative_entry_key"]
        rep_pdb_id = group_meta["representative_pdb_id"]
        group_size = int(group_meta["group_size"])

        group_catalog_rows.append(
            {
                "group_id": group_id,
                "group_size": group_size,
                "representative_entry_key": rep_entry_key,
                "representative_pdb_id": rep_pdb_id,
                "n_representative_unique_contact_sequences": len(seqs),
                "sequence_lengths": json.dumps([int(row["sequence_length"]) for row in seqs]),
                "member_entry_keys": group_meta["member_entry_keys"],
                "member_pdb_ids": group_meta["member_pdb_ids"],
            }
        )

        for row in seqs:
            record_id = make_record_id("l2all", record_counter)
            record_counter += 1
            out_row = {
                "record_id": record_id,
                "group_id": group_id,
                "group_size": group_size,
                "representative_entry_key": rep_entry_key,
                "representative_pdb_id": rep_pdb_id,
                "sequence_index": int(row["sequence_index"]),
                "chain_hash": row["chain_hash"],
                "sequence_length": int(row["sequence_length"]),
                "sequence": row["sequence"],
                "mode": "all_unique_contact_chains",
            }
            all_unique_rows.append(out_row)

        longest_seq = seqs[0]
        longest_rows.append(
            {
                "record_id": make_record_id("l2long", longest_counter),
                "group_id": group_id,
                "group_size": group_size,
                "representative_entry_key": rep_entry_key,
                "representative_pdb_id": rep_pdb_id,
                "sequence_index": int(longest_seq["sequence_index"]),
                "chain_hash": longest_seq["chain_hash"],
                "sequence_length": int(longest_seq["sequence_length"]),
                "sequence": longest_seq["sequence"],
                "mode": "longest_contact_chain",
            }
        )
        longest_counter += 1

    write_fasta(outdir / "l2_all_unique_contact_chains.fasta", all_unique_rows)
    write_tsv(
        outdir / "l2_all_unique_contact_chains.tsv",
        all_unique_rows,
        fieldnames=[
            "record_id",
            "group_id",
            "group_size",
            "representative_entry_key",
            "representative_pdb_id",
            "sequence_index",
            "chain_hash",
            "sequence_length",
            "sequence",
            "mode",
        ],
    )
    write_fasta(outdir / "l2_longest_contact_chain.fasta", longest_rows)
    write_tsv(
        outdir / "l2_longest_contact_chain.tsv",
        longest_rows,
        fieldnames=[
            "record_id",
            "group_id",
            "group_size",
            "representative_entry_key",
            "representative_pdb_id",
            "sequence_index",
            "chain_hash",
            "sequence_length",
            "sequence",
            "mode",
        ],
    )
    write_tsv(
        outdir / "l2_group_catalog.tsv",
        group_catalog_rows,
        fieldnames=[
            "group_id",
            "group_size",
            "representative_entry_key",
            "representative_pdb_id",
            "n_representative_unique_contact_sequences",
            "sequence_lengths",
            "member_entry_keys",
            "member_pdb_ids",
        ],
    )

    command_template = f"""#!/usr/bin/env bash
set -euo pipefail

# Suggested MMseqs2 95% clustering commands
# Mode 1: all unique contacting chains from each L1 representative
mmseqs easy-cluster \\
  {outdir / 'l2_all_unique_contact_chains.fasta'} \\
  {outdir / 'mmseqs95_all_unique'} \\
  {outdir / 'tmp_all_unique'} \\
  --min-seq-id {args.min_seq_id} -c {args.coverage}

# Mode 2: only the longest contacting chain from each L1 representative
mmseqs easy-cluster \\
  {outdir / 'l2_longest_contact_chain.fasta'} \\
  {outdir / 'mmseqs95_longest'} \\
  {outdir / 'tmp_longest'} \\
  --min-seq-id {args.min_seq_id} -c {args.coverage}
"""
    with (outdir / "run_mmseqs2_95.sh").open("w", encoding="utf-8") as handle:
        handle.write(command_template)

    summary = {
        "l1_dir": str(l1_dir),
        "source_total_groups": l1_summary.get("total_groups"),
        "source_total_entries": l1_summary.get("total_entries"),
        "all_unique_contact_chains_records": len(all_unique_rows),
        "longest_contact_chain_records": len(longest_rows),
        "groups_with_multiple_unique_contact_chains": sum(
            1 for row in group_catalog_rows if int(row["n_representative_unique_contact_sequences"]) > 1
        ),
        "suggested_mmseqs2_min_seq_id": args.min_seq_id,
        "suggested_mmseqs2_coverage": args.coverage,
        "outputs": {
            "all_unique_fasta": str(outdir / "l2_all_unique_contact_chains.fasta"),
            "all_unique_table": str(outdir / "l2_all_unique_contact_chains.tsv"),
            "longest_fasta": str(outdir / "l2_longest_contact_chain.fasta"),
            "longest_table": str(outdir / "l2_longest_contact_chain.tsv"),
            "group_catalog": str(outdir / "l2_group_catalog.tsv"),
            "command_template": str(outdir / "run_mmseqs2_95.sh"),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
