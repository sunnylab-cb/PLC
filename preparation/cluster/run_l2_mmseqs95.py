#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
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


def run_mmseqs_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
) -> None:
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
    ]
    subprocess.run(cmd, check=True)


def hash_items(items: List[str]) -> str:
    if not items:
        return ""
    return hashlib.sha1("|".join(sorted(items)).encode()).hexdigest()


def parse_cluster_tsv(cluster_tsv: Path) -> Dict[str, str]:
    member_to_rep: Dict[str, str] = {}
    with cluster_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            representative, member = row[0].strip(), row[1].strip()
            member_to_rep[member] = representative
    return member_to_rep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MMseqs2 95% clustering on L2 inputs and backfill group/entry manifests."
    )
    parser.add_argument("--l1-dir", required=True, help="Directory from build_l1_manifest.py")
    parser.add_argument("--l2-input-dir", required=True, help="Directory from build_l2_mmseqs_input.py")
    parser.add_argument("--outdir", required=True, help="Output directory for MMseqs2 results and manifests")
    parser.add_argument("--min-seq-id", type=float, default=0.95)
    parser.add_argument("--coverage", type=float, default=0.8)
    args = parser.parse_args()

    l1_dir = Path(args.l1_dir).resolve()
    l2_input_dir = Path(args.l2_input_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fasta_path = l2_input_dir / "l2_all_unique_contact_chains.fasta"
    seq_table_path = l2_input_dir / "l2_all_unique_contact_chains.tsv"
    l1_manifest_path = l1_dir / "l1_manifest.tsv"
    if not fasta_path.exists():
        raise FileNotFoundError(f"Missing {fasta_path}")
    if not seq_table_path.exists():
        raise FileNotFoundError(f"Missing {seq_table_path}")
    if not l1_manifest_path.exists():
        raise FileNotFoundError(f"Missing {l1_manifest_path}")

    mmseqs_prefix = outdir / "mmseqs95_all_unique"
    mmseqs_tmp = outdir / "mmseqs_tmp"
    cluster_tsv = outdir / "mmseqs95_all_unique_cluster.tsv"

    run_mmseqs_easy_cluster(
        fasta_path=fasta_path,
        out_prefix=mmseqs_prefix,
        tmp_dir=mmseqs_tmp,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
    )

    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected MMseqs2 cluster TSV at {cluster_tsv}")

    seq_rows = read_tsv(seq_table_path)
    l1_manifest_rows = read_tsv(l1_manifest_path)
    member_to_rep = parse_cluster_tsv(cluster_tsv)
    seq_by_record_id = {row["record_id"]: row for row in seq_rows}
    group_to_cluster_reps: Dict[str, List[str]] = defaultdict(list)
    representative_record_to_cluster_size: Counter[str] = Counter(member_to_rep.values())

    chain_cluster_rows: List[dict] = []
    for record_id, representative_id in sorted(member_to_rep.items()):
        seq_row = seq_by_record_id[record_id]
        group_id = seq_row["group_id"]
        group_to_cluster_reps[group_id].append(representative_id)
        chain_cluster_rows.append(
            {
                "record_id": record_id,
                "group_id": group_id,
                "representative_entry_key": seq_row["representative_entry_key"],
                "representative_pdb_id": seq_row["representative_pdb_id"],
                "sequence_index": seq_row["sequence_index"],
                "chain_hash": seq_row["chain_hash"],
                "sequence_length": seq_row["sequence_length"],
                "mmseqs95_chain_cluster_representative": representative_id,
                "mmseqs95_chain_cluster_size": representative_record_to_cluster_size[representative_id],
            }
        )

    l2_group_rows: List[dict] = []
    group_to_l2_signature: Dict[str, str] = {}
    signature_to_groups: Dict[str, List[str]] = defaultdict(list)
    for group_id, rep_ids in sorted(group_to_cluster_reps.items()):
        unique_rep_ids = sorted(set(rep_ids))
        l2_signature = hash_items(unique_rep_ids)
        group_to_l2_signature[group_id] = l2_signature
        signature_to_groups[l2_signature].append(group_id)
        sample_seq_row = next(row for row in seq_rows if row["group_id"] == group_id)
        l2_group_rows.append(
            {
                "group_id": group_id,
                "representative_entry_key": sample_seq_row["representative_entry_key"],
                "representative_pdb_id": sample_seq_row["representative_pdb_id"],
                "mmseqs95_chain_cluster_representatives": json.dumps(unique_rep_ids),
                "mmseqs95_chain_cluster_count": len(unique_rep_ids),
                "l2_cluster_signature": l2_signature,
            }
        )

    l2_signature_group_sizes = {sig: len(groups) for sig, groups in signature_to_groups.items()}
    l2_manifest_rows: List[dict] = []
    for row in l1_manifest_rows:
        group_id = row["group_id"]
        l2_signature = group_to_l2_signature[group_id]
        l2_manifest_rows.append(
            {
                **row,
                "mmseqs95_l2_cluster_signature": l2_signature,
                "mmseqs95_l2_cluster_group_count": l2_signature_group_sizes[l2_signature],
            }
        )

    write_tsv(
        outdir / "mmseqs95_chain_clusters.tsv",
        chain_cluster_rows,
        fieldnames=[
            "record_id",
            "group_id",
            "representative_entry_key",
            "representative_pdb_id",
            "sequence_index",
            "chain_hash",
            "sequence_length",
            "mmseqs95_chain_cluster_representative",
            "mmseqs95_chain_cluster_size",
        ],
    )
    write_tsv(
        outdir / "l2_groups_mmseqs95.tsv",
        l2_group_rows,
        fieldnames=[
            "group_id",
            "representative_entry_key",
            "representative_pdb_id",
            "mmseqs95_chain_cluster_representatives",
            "mmseqs95_chain_cluster_count",
            "l2_cluster_signature",
        ],
    )
    write_tsv(
        outdir / "l2_manifest_mmseqs95.tsv",
        l2_manifest_rows,
        fieldnames=list(l2_manifest_rows[0].keys()) if l2_manifest_rows else [],
    )

    summary = {
        "l1_dir": str(l1_dir),
        "l2_input_dir": str(l2_input_dir),
        "mmseqs95_min_seq_id": args.min_seq_id,
        "mmseqs95_coverage": args.coverage,
        "input_chain_sequences": len(seq_rows),
        "unique_chain_clusters": len(set(member_to_rep.values())),
        "input_groups": len(group_to_cluster_reps),
        "unique_l2_cluster_signatures": len(signature_to_groups),
        "groups_in_merged_l2_clusters": sum(
            len(groups) for groups in signature_to_groups.values() if len(groups) > 1
        ),
        "largest_l2_cluster_signature_size": max(
            (len(groups) for groups in signature_to_groups.values()),
            default=0,
        ),
        "largest_l2_clusters": [
            {
                "l2_cluster_signature": signature,
                "group_count": len(groups),
                "group_ids": sorted(groups)[:50],
            }
            for signature, groups in sorted(
                signature_to_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )[:20]
        ],
        "outputs": {
            "mmseqs_cluster_tsv": str(cluster_tsv),
            "chain_clusters": str(outdir / "mmseqs95_chain_clusters.tsv"),
            "l2_groups": str(outdir / "l2_groups_mmseqs95.tsv"),
            "l2_manifest": str(outdir / "l2_manifest_mmseqs95.tsv"),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
