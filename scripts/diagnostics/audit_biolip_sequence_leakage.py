#!/usr/bin/env python3
"""Audit BioLip pilot chains against internal and external evaluation proteins.

The audit reconstructs BioLip chain sequences from the exact V4 chain sidecars,
reads cached sequences for PDBbind/COACH420/HOLO4K, runs MMseqs easy-search, and
excludes an entire BioLip PDB group when any of its selected chains is homologous
to an evaluation target. Direct PDB-ID filtering remains a separate prerequisite.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ASH": "D",
    "GLH": "E",
    "CYX": "C",
    "CYM": "C",
    "LYN": "K",
}
YEAR_BUCKETS = ("1981-2000", "2001-2010", "2011-2019")
EVAL_CATEGORIES = ("pdbbind_val", "pdbbind_test", "coach420", "holo4k")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping, got {type(value).__name__}")
    return value


def normalize_sequence(value: Any) -> str:
    if isinstance(value, str):
        sequence = value
    elif isinstance(value, Sequence):
        sequence = "".join(str(token) for token in value)
    else:
        raise TypeError(f"unsupported sequence type: {type(value).__name__}")
    return "".join(sequence.split()).replace(":", "").replace("-", "").upper()


def sequence_from_residue_names(names: Sequence[Any]) -> str:
    return "".join(AA3_TO_1.get(str(name).upper(), "X") for name in names)


def read_split_file(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            result[parts[0]].append(parts[1])
    return dict(result)


def find_nested_feature(root: Path, sample_id: str) -> Path | None:
    for bucket in YEAR_BUCKETS:
        path = root / bucket / sample_id / f"{sample_id}_protein.pt"
        if path.is_file():
            return path
    return None


def load_cached_sequence(path: Path) -> str:
    data = _load_mapping(path)
    if "sequence" in data:
        sequence = normalize_sequence(data["sequence"])
    elif "residue_name_1" in data:
        sequence = normalize_sequence(data["residue_name_1"])
    elif "residue_names" in data:
        sequence = sequence_from_residue_names(data["residue_names"])
    else:
        raise KeyError(f"{path}: no sequence or residue-name metadata")
    if len(sequence) < 20:
        raise ValueError(f"{path}: sequence is unexpectedly short ({len(sequence)})")
    return sequence


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> None:
    with path.open("w") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def normalize_identity(value: str) -> float:
    identity = float(value)
    return identity / 100.0 if identity > 1.0 else identity


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


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
    parser.add_argument("--query-coverage", type=float, default=0.80)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_rows, pilot_fields = _read_manifest(args.pilot_root / "pilot_manifest.tsv")

    chain_to_pdb: dict[str, str] = {}
    query_records: list[tuple[str, str]] = []
    query_metadata: list[dict[str, Any]] = []
    selected_chains = sorted({(row["pdb_id"].lower(), row["protein_chain"]) for row in pilot_rows})
    for index, (pdb_id, chain) in enumerate(selected_chains):
        path = args.pilot_root / "_chain_cache" / f"{pdb_id}_{chain}_protein.pt"
        data = _load_mapping(path)
        sequence = sequence_from_residue_names(data["residue_names"])
        query_id = f"q{index:06d}"
        query_records.append((query_id, sequence))
        chain_to_pdb[query_id] = pdb_id
        query_metadata.append(
            {
                "query_id": query_id,
                "pdb_id": pdb_id,
                "chain": chain,
                "length": len(sequence),
                "x_fraction": sequence.count("X") / len(sequence),
                "feature_path": str(path),
            }
        )

    splits = read_split_file(args.pdbbind_split_file)
    target_records: list[tuple[str, str]] = []
    target_metadata: dict[str, dict[str, Any]] = {}
    missing_targets: list[dict[str, str]] = []

    def add_target(category: str, source_id: str, path: Path) -> None:
        target_id = f"t{len(target_records):06d}"
        sequence = load_cached_sequence(path)
        target_records.append((target_id, sequence))
        target_metadata[target_id] = {
            "category": category,
            "source_id": source_id,
            "length": len(sequence),
            "feature_path": str(path),
        }

    pdbbind_esm_root = args.pdbbind_data_root / "esm2_latents"
    for split, category in (("val", "pdbbind_val"), ("test", "pdbbind_test")):
        for sample_id in splits.get(split, []):
            path = find_nested_feature(pdbbind_esm_root, sample_id)
            if path is None:
                missing_targets.append({"category": category, "source_id": sample_id})
                continue
            add_target(category, sample_id, path)

    for category, root in (("coach420", args.coach_esm_dir), ("holo4k", args.holo_esm_dir)):
        for path in sorted(root.glob("*.pt")):
            add_target(category, path.stem, path)

    query_fasta = output_dir / "biolip_queries.fasta"
    target_fasta = output_dir / "evaluation_targets.fasta"
    raw_hits = output_dir / "mmseqs_hits.tsv"
    mmseqs_tmp = output_dir / "mmseqs_tmp"
    write_fasta(query_records, query_fasta)
    write_fasta(target_records, target_fasta)

    command = [
        args.mmseqs,
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(raw_hits),
        str(mmseqs_tmp),
        "--min-seq-id",
        str(args.min_seq_id),
        "-c",
        str(args.query_coverage),
        "--cov-mode",
        "2",
        "--threads",
        str(args.threads),
        "--max-seqs",
        "1000",
        "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    subprocess.run(command, check=True)

    best: dict[tuple[str, str], dict[str, Any]] = {}
    hit_count = 0
    if raw_hits.is_file():
        with raw_hits.open() as handle:
            for line in handle:
                query_id, target_id, identity, alnlen, qcov, tcov, evalue, bits = line.rstrip().split("\t")
                target = target_metadata[target_id]
                category = str(target["category"])
                hit = {
                    "query_id": query_id,
                    "target_id": target_id,
                    "category": category,
                    "source_id": target["source_id"],
                    "identity": normalize_identity(identity),
                    "alignment_length": int(alnlen),
                    "query_coverage": normalize_identity(qcov),
                    "target_coverage": normalize_identity(tcov),
                    "evalue": float(evalue),
                    "bits": float(bits),
                }
                key = (query_id, category)
                previous = best.get(key)
                if previous is None or (hit["identity"], hit["bits"]) > (
                    previous["identity"],
                    previous["bits"],
                ):
                    best[key] = hit
                hit_count += 1

    best_path = output_dir / "best_hit_per_query_category.tsv"
    best_fields = [
        "query_id",
        "pdb_id",
        "chain",
        "category",
        "source_id",
        "identity",
        "alignment_length",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
    ]
    query_meta_by_id = {row["query_id"]: row for row in query_metadata}
    with best_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=best_fields, delimiter="\t")
        writer.writeheader()
        for key in sorted(best):
            hit = dict(best[key])
            query_meta = query_meta_by_id[hit["query_id"]]
            hit["pdb_id"] = query_meta["pdb_id"]
            hit["chain"] = query_meta["chain"]
            writer.writerow({field: hit[field] for field in best_fields})

    scope_categories = {
        "pdbbind_val": {"pdbbind_val"},
        "pdbbind_test": {"pdbbind_test"},
        "pdbbind_internal": {"pdbbind_val", "pdbbind_test"},
        "coach420": {"coach420"},
        "holo4k": {"holo4k"},
        "external_benchmarks": {"coach420", "holo4k"},
        "all_evaluation": set(EVAL_CATEGORIES),
    }
    scope_results: dict[str, dict[str, Any]] = {}
    for scope, categories in scope_categories.items():
        excluded_queries = {
            query_id
            for (query_id, category), hit in best.items()
            if category in categories
            and hit["identity"] >= args.min_seq_id
            and hit["query_coverage"] >= args.query_coverage
        }
        excluded_pdbs = {chain_to_pdb[query_id] for query_id in excluded_queries}
        retained = [row for row in pilot_rows if row["pdb_id"].lower() not in excluded_pdbs]
        exclusion_path = output_dir / f"exclude_sequence_homolog_pdb_ids_{scope}.txt"
        exclusion_path.write_text(
            "\n".join(sorted(excluded_pdbs)) + ("\n" if excluded_pdbs else "")
        )
        retained_path = output_dir / f"retained_pilot_manifest_{scope}.tsv"
        with retained_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pilot_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(retained)
        scope_results[scope] = {
            "categories": sorted(categories),
            "excluded_chains": len(excluded_queries),
            "excluded_pdbs_group_safe": len(excluded_pdbs),
            "retained_pairs": len(retained),
            "retained_pdbs": len({row["pdb_id"].lower() for row in retained}),
            "retained_chains": len(
                {(row["pdb_id"].lower(), row["protein_chain"]) for row in retained}
            ),
        }

    # Backward-compatible aliases always mean the strict union of every
    # evaluation target. Scope-specific files above prevent accidental reuse.
    all_scope = scope_results["all_evaluation"]
    (output_dir / "exclude_sequence_homolog_pdb_ids.txt").write_text(
        (output_dir / "exclude_sequence_homolog_pdb_ids_all_evaluation.txt").read_text()
    )
    (output_dir / "retained_pilot_manifest.tsv").write_text(
        (output_dir / "retained_pilot_manifest_all_evaluation.tsv").read_text()
    )

    category_summary = {}
    for category in EVAL_CATEGORIES:
        identities = [
            hit["identity"]
            for (query_id, hit_category), hit in best.items()
            if hit_category == category
        ]
        category_summary[category] = {
            "targets": sum(1 for row in target_metadata.values() if row["category"] == category),
            "queries_with_hit": len(identities),
            "queries_ge_0_30": sum(value >= 0.30 for value in identities),
            "queries_ge_0_50": sum(value >= 0.50 for value in identities),
            "queries_ge_0_90": sum(value >= 0.90 for value in identities),
            "best_identity_p50": quantile(identities, 0.50),
            "best_identity_p90": quantile(identities, 0.90),
            "best_identity_max": max(identities, default=None),
        }

    summary = {
        "thresholds": {
            "min_sequence_identity": args.min_seq_id,
            "min_query_coverage": args.query_coverage,
            "mmseqs_cov_mode": 2,
        },
        "pilot_pairs_before": len(pilot_rows),
        "pilot_pdbs_before": len({row["pdb_id"].lower() for row in pilot_rows}),
        "pilot_chains_before": len(query_records),
        "evaluation_targets": len(target_records),
        "missing_evaluation_targets": len(missing_targets),
        "raw_hits": hit_count,
        "excluded_chains": all_scope["excluded_chains"],
        "excluded_pdbs_group_safe": all_scope["excluded_pdbs_group_safe"],
        "retained_pairs": all_scope["retained_pairs"],
        "retained_pdbs": all_scope["retained_pdbs"],
        "retained_chains": all_scope["retained_chains"],
        "retention_by_scope": scope_results,
        "query_x_fraction": {
            "max": max((row["x_fraction"] for row in query_metadata), default=0.0),
            "chains_above_0_05": sum(row["x_fraction"] > 0.05 for row in query_metadata),
        },
        "by_category": category_summary,
        "command": command,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "missing_targets.json").write_text(
        json.dumps(missing_targets, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
