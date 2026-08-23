#!/usr/bin/env python3
"""Prepare a leakage-screened BioLiP + PDBbind-train manifest.

This is a preflight step: it does not copy feature tensors. Candidate protein
sequences are screened directly against PDBbind val/test, COACH420, and HOLO4K.
Rows sharing a protein sequence are retained when they describe distinct
protein-ligand samples; their multiplicity is reported for sampler design.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_biolip_sequence_leakage import (
    EVAL_CATEGORIES,
    find_nested_feature,
    load_cached_sequence,
    read_split_file,
    sequence_from_residue_names,
    write_fasta,
    _load_mapping,
    _read_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--pdbbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-splits", type=Path, required=True)
    parser.add_argument("--coach-esm-dir", type=Path, required=True)
    parser.add_argument("--holo-esm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = read_split_file(args.pdbbind_splits)
    all_pdbbind_ids = {value.lower() for values in splits.values() for value in values}

    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()

    # PDBbind train remains authoritative for direct PDB/sample collisions.
    for sample_id in splits.get("train", []):
        path = find_nested_feature(args.pdbbind_root / "esm2_latents", sample_id)
        if path is None:
            missing.append({"source": "pdbbind", "sample_id": sample_id})
            continue
        sequence = load_cached_sequence(path)
        candidates.append(
            {
                "source": "pdbbind",
                "sample_id": sample_id,
                "pdb_id": sample_id.lower(),
                "protein_chain": "",
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_path": str(path),
            }
        )
        seen_sample_ids.add(sample_id)

    manifest_rows, _ = _read_manifest(args.biolip_root / "pilot_manifest.tsv")
    direct_conflict_rows = 0
    for row in manifest_rows:
        pdb_id = row["pdb_id"].lower()
        sample_id = row.get("sample_id") or row.get("complex_id")
        if not sample_id:
            raise KeyError("BioLiP manifest requires sample_id or complex_id")
        if pdb_id in all_pdbbind_ids or sample_id in seen_sample_ids:
            direct_conflict_rows += 1
            continue
        chain = row["protein_chain"]
        path = args.biolip_root / "_chain_cache" / f"{pdb_id}_{chain}_protein.pt"
        data = _load_mapping(path)
        sequence = sequence_from_residue_names(data["residue_names"])
        candidates.append(
            {
                "source": "biolip",
                "sample_id": sample_id,
                "pdb_id": pdb_id,
                "protein_chain": chain,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_path": str(path),
            }
        )
        seen_sample_ids.add(sample_id)

    target_rows: list[dict[str, Any]] = []

    def add_target(category: str, source_id: str, path: Path) -> None:
        sequence = load_cached_sequence(path)
        target_rows.append(
            {
                "category": category,
                "source_id": source_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_path": str(path),
            }
        )

    for split, category in (("val", "pdbbind_val"), ("test", "pdbbind_test")):
        for sample_id in splits.get(split, []):
            path = find_nested_feature(args.pdbbind_root / "esm2_latents", sample_id)
            if path is None:
                missing.append({"source": category, "sample_id": sample_id})
                continue
            add_target(category, sample_id, path)
    for category, root in (("coach420", args.coach_esm_dir), ("holo4k", args.holo_esm_dir)):
        for path in sorted(root.glob("*.pt")):
            add_target(category, path.stem, path)

    query_ids: dict[str, dict[str, Any]] = {}
    query_records = []
    # Search unique sequences once, then apply exclusions to every associated row.
    sequence_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        sequence_to_rows[row["sequence"]].append(row)
    for index, sequence in enumerate(sorted(sequence_to_rows)):
        query_id = f"q{index:06d}"
        query_records.append((query_id, sequence))
        query_ids[query_id] = {"sequence": sequence, "rows": sequence_to_rows[sequence]}

    target_ids: dict[str, dict[str, Any]] = {}
    target_records = []
    for index, row in enumerate(target_rows):
        target_id = f"t{index:06d}"
        target_records.append((target_id, row["sequence"]))
        target_ids[target_id] = row

    query_fasta = out / "merged_train_unique_sequences.fasta"
    target_fasta = out / "evaluation_sequences.fasta"
    hits_path = out / "mmseqs_hits.tsv"
    write_fasta(query_records, query_fasta)
    write_fasta(target_records, target_fasta)
    command = [
        args.mmseqs, "easy-search", str(query_fasta), str(target_fasta),
        str(hits_path), str(out / "mmseqs_tmp"),
        "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode), "--threads", str(args.threads),
        "--max-seqs", "10000", "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    subprocess.run(command, check=True)

    excluded_queries: set[str] = set()
    exclusions_by_category: Counter[str] = Counter()
    if hits_path.is_file():
        for line in hits_path.read_text().splitlines():
            query_id, target_id, *_ = line.split("\t")
            excluded_queries.add(query_id)
            exclusions_by_category[target_ids[target_id]["category"]] += 1

    excluded_sequences = {query_ids[qid]["sequence"] for qid in excluded_queries}
    retained = [row for row in candidates if row["sequence"] not in excluded_sequences]
    excluded = [row for row in candidates if row["sequence"] in excluded_sequences]

    fields = [
        "source", "sample_id", "pdb_id", "protein_chain", "sequence_length",
        "sequence_sha256", "sequence_multiplicity", "sequence_path",
    ]
    import hashlib
    multiplicity = Counter(row["sequence"] for row in retained)

    def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                sequence = row["sequence"]
                output = {key: row.get(key, "") for key in fields}
                output["sequence_sha256"] = hashlib.sha256(sequence.encode()).hexdigest()
                output["sequence_multiplicity"] = multiplicity.get(sequence, 0)
                writer.writerow(output)

    write_rows(out / "retained_merged_train.tsv", retained)
    write_rows(out / "excluded_homologous_train.tsv", excluded)
    (out / "missing_sequences.json").write_text(json.dumps(missing, indent=2))
    source_before = Counter(row["source"] for row in candidates)
    source_after = Counter(row["source"] for row in retained)
    repeated_groups = [size for size in multiplicity.values() if size > 1]
    summary = {
        "thresholds": {
            "min_sequence_identity": args.min_seq_id,
            "coverage": args.coverage,
            "mmseqs_cov_mode": args.cov_mode,
        },
        "candidate_rows": len(candidates),
        "candidate_unique_sequences": len(sequence_to_rows),
        "retained_rows": len(retained),
        "retained_unique_sequences": len(multiplicity),
        "excluded_rows_by_pairwise_homology": len(excluded),
        "direct_biolip_pdb_or_sample_conflict_rows": direct_conflict_rows,
        "source_rows_before": dict(source_before),
        "source_rows_after": dict(source_after),
        "evaluation_targets": Counter(row["category"] for row in target_rows),
        "raw_hits_by_evaluation_category": dict(exclusions_by_category),
        "retained_repeated_sequence_groups": len(repeated_groups),
        "retained_max_rows_per_sequence": max(multiplicity.values(), default=0),
        "missing_sequences": len(missing),
        "command": command,
        "claim": "Every retained train sequence has no MMseqs hit at the configured pairwise SI/coverage threshold.",
    }
    summary["evaluation_targets"] = dict(summary["evaluation_targets"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
