#!/usr/bin/env python
"""Audit BioLip/evaluation leakage with MMseqs connected components.

Unlike pairwise easy-search, this combines every exact training and evaluation
sequence in one graph and runs MMseqs ``cluster-mode 1``. A mixed component can
therefore expose transitive homology paths, not only a direct train-test hit.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.audit_biolip_sequence_leakage import (
    _load_mapping,
    _read_manifest,
    find_nested_feature,
    load_cached_sequence,
    read_split_file,
    sequence_from_residue_names,
    write_fasta,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--pdbbind-data-root", type=Path, required=True)
    parser.add_argument("--pdbbind-split-file", type=Path, required=True)
    parser.add_argument("--coach-esm-dir", type=Path, required=True)
    parser.add_argument("--holo-esm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, _ = _read_manifest(args.pilot_root / "pilot_manifest.tsv")
    records: list[tuple[str, str]] = []
    metadata: dict[str, dict] = {}

    def add_record(category: str, source_id: str, sequence: str, **extra) -> None:
        record_id = f"s{len(records):06d}"
        records.append((record_id, sequence))
        metadata[record_id] = {
            "category": category,
            "source_id": source_id,
            "length": len(sequence),
            **extra,
        }

    selected_chains = sorted({(row["pdb_id"].lower(), row["protein_chain"]) for row in rows})
    for pdb_id, chain in selected_chains:
        path = args.pilot_root / "_chain_cache" / f"{pdb_id}_{chain}_protein.pt"
        data = _load_mapping(path)
        add_record(
            "biolip_train",
            f"{pdb_id}_{chain}",
            sequence_from_residue_names(data["residue_names"]),
            pdb_id=pdb_id,
            chain=chain,
            feature_path=str(path),
        )

    splits = read_split_file(args.pdbbind_split_file)
    pdbbind_esm = args.pdbbind_data_root / "esm2_latents"
    missing_targets = []
    for split, category in (("val", "pdbbind_val"), ("test", "pdbbind_test")):
        for sample_id in splits.get(split, []):
            path = find_nested_feature(pdbbind_esm, sample_id)
            if path is None:
                missing_targets.append({"category": category, "source_id": sample_id})
                continue
            add_record(category, sample_id, load_cached_sequence(path), feature_path=str(path))

    for category, root in (("coach420", args.coach_esm_dir), ("holo4k", args.holo_esm_dir)):
        for path in sorted(root.glob("*.pt")):
            add_record(category, path.stem, load_cached_sequence(path), feature_path=str(path))

    fasta = output / "combined_sequences.fasta"
    prefix = output / "connected"
    tmp = output / "mmseqs_tmp"
    write_fasta(records, fasta)
    command = [
        args.mmseqs,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        str(args.min_seq_id),
        "-c",
        str(args.coverage),
        "--cov-mode",
        str(args.cov_mode),
        "--cluster-mode",
        "1",
        "--threads",
        str(args.threads),
    ]
    subprocess.run(command, check=True)

    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    members_by_rep: dict[str, list[str]] = defaultdict(list)
    with cluster_tsv.open() as handle:
        for line in handle:
            representative, member = line.rstrip().split("\t")
            members_by_rep[representative].append(member)

    eval_categories = {"pdbbind_val", "pdbbind_test", "coach420", "holo4k"}
    mixed = []
    leaked_train_chains = set()
    leaked_train_pdbs = set()
    mixed_by_eval = Counter()
    for representative, members in members_by_rep.items():
        categories = {metadata[member]["category"] for member in members}
        overlap = categories & eval_categories
        if "biolip_train" not in categories or not overlap:
            continue
        train_members = [member for member in members if metadata[member]["category"] == "biolip_train"]
        eval_members = [member for member in members if metadata[member]["category"] in eval_categories]
        leaked_train_chains.update(train_members)
        leaked_train_pdbs.update(metadata[member]["pdb_id"] for member in train_members)
        for category in overlap:
            mixed_by_eval[category] += 1
        mixed.append(
            {
                "representative": representative,
                "size": len(members),
                "categories": dict(Counter(metadata[member]["category"] for member in members)),
                "train_members": [metadata[member] for member in train_members],
                "evaluation_members": [metadata[member] for member in eval_members],
            }
        )

    category_counts = Counter(row["category"] for row in metadata.values())
    summary = {
        "thresholds": {
            "min_sequence_identity": args.min_seq_id,
            "coverage": args.coverage,
            "mmseqs_cov_mode": args.cov_mode,
            "mmseqs_cluster_mode": 1,
            "cluster_mode_name": "connected_component",
        },
        "source_manifest": str((args.pilot_root / "pilot_manifest.tsv").resolve()),
        "sequence_counts": dict(category_counts),
        "total_sequences": len(records),
        "total_components": len(members_by_rep),
        "mixed_train_evaluation_components": len(mixed),
        "mixed_components_by_evaluation_category": dict(mixed_by_eval),
        "leaked_biolip_chains": len(leaked_train_chains),
        "leaked_biolip_pdbs_group_safe": len(leaked_train_pdbs),
        "missing_evaluation_targets": missing_targets,
        "command": command,
    }
    (output / "connected_cluster_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (output / "mixed_components.json").write_text(json.dumps(mixed, indent=2, sort_keys=True))
    (output / "exclude_connected_component_pdb_ids.txt").write_text(
        "\n".join(sorted(leaked_train_pdbs)) + ("\n" if leaked_train_pdbs else "")
    )
    with (output / "sequence_metadata.tsv").open("w", newline="") as handle:
        fields = ["record_id", "category", "source_id", "length", "pdb_id", "chain", "feature_path"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record_id, row in metadata.items():
            writer.writerow({field: record_id if field == "record_id" else row.get(field, "") for field in fields})

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
