#!/usr/bin/env python3
"""Build a chain-level SI30 split and a BioLiP-augmented train allowlist.

Protein complexes are graph nodes. An edge joins two complexes when any pair
of polymer chains has alignment identity >= the configured threshold and
either chain is covered by at least the configured fraction. Connected
components are assigned atomically to train/val/test.

COACH420 and HOLO4K homologs are removed before splitting. BioLiP rows are
train-only and are retained only when they are clean against the final
PDBbind val/test chains and both external benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from audit_biolip_sequence_leakage import (
    VALID_SEQUENCE_SYMBOLS,
    YEAR_BUCKETS,
    find_nested_feature,
    normalize_sequence,
    read_split_file,
    write_fasta,
)


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def load_mapping(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping")
    return value


def extract_chains(path: Path) -> dict[str, str]:
    data = load_mapping(path)
    residues = data.get("residue_name_1")
    chains = data.get("chain_id")
    if residues is None or chains is None:
        sequence = normalize_sequence(data.get("sequence", ""))
        if not sequence:
            raise ValueError(f"{path}: no chain metadata or sequence")
        return {"_": sequence}
    if len(residues) != len(chains) or not residues:
        raise ValueError(f"{path}: invalid residue/chain metadata lengths")
    ordered: dict[str, list[str]] = defaultdict(list)
    for chain, residue in zip(chains, residues):
        token = normalize_sequence([residue])
        if len(token) != 1 or token not in VALID_SEQUENCE_SYMBOLS:
            token = "X"
        ordered[str(chain)].append(token)
    result = {chain: "".join(tokens) for chain, tokens in ordered.items() if tokens}
    if not result:
        raise ValueError(f"{path}: no non-empty chains")
    return result


def locate_sample_paths(root: Path, sample_ids: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for sample_id in sample_ids:
        path = find_nested_feature(root, sample_id)
        if path is not None:
            result[sample_id] = path
    return result


def read_biolip_ids(root: Path) -> list[str]:
    ids = []
    for line in (root / "splits.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            ids.append(parts[1])
    return ids


def find_sample_dir_feature(root: Path, sample_id: str) -> Path | None:
    return find_nested_feature(root / "esm2_latents", sample_id)


def unique_chain_records(
    sample_paths: Mapping[str, Path], source: str
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]], dict[str, list[str]]]:
    sequence_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_chains: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(sample_paths):
        for chain_id, sequence in extract_chains(sample_paths[sample_id]).items():
            sequence_members[sequence].append({"sample_id": sample_id, "chain_id": chain_id})
            sample_chains[sample_id].append(sequence)
    records: list[tuple[str, str]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for index, sequence in enumerate(sorted(sequence_members)):
        record_id = f"{source}_{index:06d}"
        records.append((record_id, sequence))
        metadata[record_id] = {
            "source": source,
            "sequence": sequence,
            "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence),
            "members": sequence_members[sequence],
        }
    return records, metadata, sample_chains


def run_search(
    mmseqs: str,
    query_fasta: Path,
    target_fasta: Path,
    output: Path,
    threads: int,
    min_search_identity: float,
) -> list[str]:
    tmp = output.with_suffix("").with_name(output.stem + "_tmp")
    command = [
        mmseqs, "easy-search", str(query_fasta), str(target_fasta), str(output), str(tmp),
        "--min-seq-id", str(min_search_identity), "-c", "0.0", "--cov-mode", "0",
        "--alignment-mode", "3", "--seq-id-mode", "0", "-e", "1000", "-s", "7.5",
        "--max-seqs", "100000", "--threads", str(threads), "--format-output",
        "query,target,nident,alnlen,fident,qlen,tlen,qcov,tcov,evalue,bits",
    ]
    subprocess.run(command, check=True)
    return command


def parse_violations(
    path: Path, min_identity: float, min_coverage: float
) -> list[dict[str, Any]]:
    result = []
    if not path.is_file():
        return result
    fields = (
        "query", "target", "nident", "alnlen", "fident", "qlen", "tlen",
        "qcov", "tcov", "evalue", "bits",
    )
    for line in path.read_text().splitlines():
        values = dict(zip(fields, line.split("\t")))
        nident, alnlen = int(values["nident"]), int(values["alnlen"])
        identity = nident / alnlen if alnlen else 0.0
        qcov, tcov = float(values["qcov"]), float(values["tcov"])
        if identity >= min_identity and (qcov >= min_coverage or tcov >= min_coverage):
            values.update(identity=identity, qcov=qcov, tcov=tcov)
            result.append(values)
    return result


def write_fasta_and_meta(
    records: Sequence[tuple[str, str]], metadata: Mapping[str, Any], prefix: Path
) -> None:
    write_fasta(records, prefix.with_suffix(".fasta"))
    with prefix.with_suffix(".metadata.jsonl").open("w") as handle:
        for record_id, _ in records:
            handle.write(json.dumps({"record_id": record_id, **metadata[record_id]}, sort_keys=True) + "\n")


def assign_components(
    sample_ids: Sequence[str], union_find: UnionFind, seed: int
) -> tuple[dict[str, str], list[list[str]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        groups[union_find.find(sample_id)].append(sample_id)
    components = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    targets = {
        "train": round(len(sample_ids) * 0.8),
        "val": round(len(sample_ids) * 0.1),
    }
    targets["test"] = len(sample_ids) - targets["train"] - targets["val"]
    counts = Counter()
    assignment: dict[str, str] = {}
    for component in components:
        split = max(("train", "val", "test"), key=lambda name: targets[name] - counts[name])
        for sample_id in component:
            assignment[sample_id] = split
        counts[split] += len(component)
    return assignment, components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdbbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-ids", type=Path, required=True)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--coach-esm-dir", type=Path, required=True)
    parser.add_argument("--holo-esm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--search-floor", type=float, default=0.28)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old_splits = read_split_file(args.pdbbind_ids)
    pdb_ids = sorted(set().union(*old_splits.values()))
    pdb_paths = locate_sample_paths(args.pdbbind_root / "esm2_latents", pdb_ids)
    if len(pdb_paths) != len(pdb_ids):
        missing = sorted(set(pdb_ids) - set(pdb_paths))
        raise RuntimeError(f"missing {len(missing)} PDBbind sequences: {missing[:20]}")

    coach_paths = {path.stem: path for path in sorted(args.coach_esm_dir.glob("*.pt"))}
    holo_paths = {path.stem: path for path in sorted(args.holo_esm_dir.glob("*.pt"))}
    if len(coach_paths) != 420 or len(holo_paths) < 4500:
        raise RuntimeError(f"incomplete external targets: coach={len(coach_paths)} holo={len(holo_paths)}")

    pdb_records, pdb_meta, pdb_sample_chains = unique_chain_records(pdb_paths, "pdb")
    external_paths = {f"coach:{key}": value for key, value in coach_paths.items()}
    external_paths.update({f"holo:{key}": value for key, value in holo_paths.items()})
    ext_records, ext_meta, _ = unique_chain_records(external_paths, "ext")
    write_fasta_and_meta(pdb_records, pdb_meta, out / "pdbbind_chains")
    write_fasta_and_meta(ext_records, ext_meta, out / "external_chains")

    ext_hits_path = out / "pdbbind_vs_external.tsv"
    commands = [run_search(
        args.mmseqs, out / "pdbbind_chains.fasta", out / "external_chains.fasta",
        ext_hits_path, args.threads, args.search_floor,
    )]
    ext_hits = parse_violations(ext_hits_path, args.min_seq_id, args.min_coverage)
    external_excluded: set[str] = set()
    for hit in ext_hits:
        for member in pdb_meta[hit["query"]]["members"]:
            external_excluded.add(member["sample_id"])

    eligible_ids = sorted(set(pdb_ids) - external_excluded)
    eligible_sequences = {seq for pid in eligible_ids for seq in pdb_sample_chains[pid]}
    eligible_records = [(rid, seq) for rid, seq in pdb_records if seq in eligible_sequences]
    write_fasta(eligible_records, out / "eligible_pdbbind_chains.fasta")

    internal_hits_path = out / "pdbbind_all_vs_all.tsv"
    commands.append(run_search(
        args.mmseqs, out / "eligible_pdbbind_chains.fasta", out / "eligible_pdbbind_chains.fasta",
        internal_hits_path, args.threads, args.search_floor,
    ))
    internal_hits = parse_violations(internal_hits_path, args.min_seq_id, args.min_coverage)
    union_find = UnionFind(eligible_ids)
    for hit in internal_hits:
        left = {row["sample_id"] for row in pdb_meta[hit["query"]]["members"] if row["sample_id"] in union_find.parent}
        right = {row["sample_id"] for row in pdb_meta[hit["target"]]["members"] if row["sample_id"] in union_find.parent}
        for left_id in left:
            for right_id in right:
                union_find.union(left_id, right_id)

    assignment, components = assign_components(eligible_ids, union_find, args.seed)
    split_ids = {name: sorted(pid for pid, split in assignment.items() if split == name) for name in ("train", "val", "test")}
    split_path = out / "pdbbind_chain_si30_splits.txt"
    with split_path.open("w") as handle:
        handle.write("# Chain-level connected-component split\n")
        handle.write(f"# predicate: identity >= {args.min_seq_id} and (qcov >= {args.min_coverage} or tcov >= {args.min_coverage})\n")
        handle.write(f"# external-homolog PDBbind complexes excluded: {len(external_excluded)}\n")
        for name in ("train", "val", "test"):
            for pid in split_ids[name]:
                handle.write(f"{name}\t{pid}\n")

    # BioLiP is train-only. Filter every row by its chain sequence.
    biolip_ids = read_biolip_ids(args.biolip_root)
    bio_paths = locate_sample_paths(args.biolip_root / "esm2_latents", biolip_ids)
    if len(bio_paths) != len(biolip_ids):
        missing = sorted(set(biolip_ids) - set(bio_paths))
        raise RuntimeError(f"missing {len(missing)} BioLiP sequences: {missing[:20]}")
    bio_records, bio_meta, _ = unique_chain_records(bio_paths, "bio")
    eval_ids = split_ids["val"] + split_ids["test"]
    eval_sequences = {seq for pid in eval_ids for seq in pdb_sample_chains[pid]}
    eval_records = [(rid, seq) for rid, seq in pdb_records if seq in eval_sequences]
    bio_targets = ext_records + [(f"eval_{index:06d}", seq) for index, (_, seq) in enumerate(eval_records)]
    bio_target_meta = dict(ext_meta)
    for index, (_, sequence) in enumerate(eval_records):
        bio_target_meta[f"eval_{index:06d}"] = {"source": "pdbbind_eval", "sequence": sequence}
    write_fasta_and_meta(bio_records, bio_meta, out / "biolip_chains")
    write_fasta(bio_targets, out / "biolip_filter_targets.fasta")
    bio_hits_path = out / "biolip_vs_evaluation.tsv"
    commands.append(run_search(
        args.mmseqs, out / "biolip_chains.fasta", out / "biolip_filter_targets.fasta",
        bio_hits_path, args.threads, args.search_floor,
    ))
    bio_hits = parse_violations(bio_hits_path, args.min_seq_id, args.min_coverage)
    bio_excluded: set[str] = set()
    for hit in bio_hits:
        for member in bio_meta[hit["query"]]["members"]:
            bio_excluded.add(member["sample_id"])
    retained_bio = sorted(set(biolip_ids) - bio_excluded)
    (out / "biolip_train_ids.txt").write_text("\n".join(retained_bio) + "\n")
    (out / "pdbbind_external_excluded_ids.txt").write_text("\n".join(sorted(external_excluded)) + "\n")
    (out / "biolip_excluded_ids.txt").write_text("\n".join(sorted(bio_excluded)) + "\n")

    old_lookup = {pid: split for split, ids in old_splits.items() for pid in ids}
    movement = Counter(f"{old_lookup[pid]}->{assignment[pid]}" for pid in eligible_ids)
    summary = {
        "predicate": {
            "minimum_alignment_identity": args.min_seq_id,
            "coverage_rule": "qcov_or_tcov",
            "minimum_coverage": args.min_coverage,
            "search_identity_floor": args.search_floor,
        },
        "pdbbind": {
            "input_complexes": len(pdb_ids),
            "unique_chains": len(pdb_records),
            "external_homolog_complexes_excluded": len(external_excluded),
            "eligible_complexes": len(eligible_ids),
            "components": len(components),
            "largest_component": max(map(len, components), default=0),
            "split_counts": {name: len(ids) for name, ids in split_ids.items()},
            "movement_from_old_split": dict(movement),
        },
        "biolip": {
            "input_samples": len(biolip_ids),
            "excluded_samples": len(bio_excluded),
            "retained_train_samples": len(retained_bio),
        },
        "hits": {
            "pdbbind_external_violating_alignments": len(ext_hits),
            "pdbbind_internal_violating_alignments": len(internal_hits),
            "biolip_evaluation_violating_alignments": len(bio_hits),
        },
        "commands": commands,
        "mmseqs_version": subprocess.run([args.mmseqs, "version"], capture_output=True, text=True, check=True).stdout.strip(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
