#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


@dataclass
class EntryRecord:
    entry_key: str
    pdb_id: str
    bucket: str
    entry_dir: str
    protein_path: str
    pocket_path: str
    all_chain_lengths: Dict[str, int]
    contact_chain_lengths: Dict[str, int]
    all_chain_hashes: Dict[str, str]
    contact_chain_hashes: Dict[str, str]

    @property
    def n_all_chains(self) -> int:
        return len(self.all_chain_lengths)

    @property
    def n_contact_chains(self) -> int:
        return len(self.contact_chain_lengths)

    @property
    def all_signature(self) -> str:
        return signature_from_hashes(self.all_chain_hashes.values())

    @property
    def contact_signature(self) -> str:
        return signature_from_hashes(self.contact_chain_hashes.values())

    @property
    def contact_unique_signature(self) -> str:
        return signature_from_hashes(set(self.contact_chain_hashes.values()))

    @property
    def longest_contact_chain_len(self) -> int:
        return max(self.contact_chain_lengths.values(), default=0)

    @property
    def longest_all_chain_len(self) -> int:
        return max(self.all_chain_lengths.values(), default=0)


def signature_from_hashes(hashes: Iterable[str]) -> str:
    items = sorted(hashes)
    if not items:
        return ""
    return hashlib.sha1("|".join(items).encode()).hexdigest()


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


def parse_chain_sequences(pdb_path: Path) -> Dict[str, str]:
    residues_by_chain: Dict[str, List[str]] = defaultdict(list)
    seen_residues: Dict[str, set] = defaultdict(set)
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = (resseq, icode)
            if residue_key in seen_residues[chain_id]:
                continue
            seen_residues[chain_id].add(residue_key)
            residues_by_chain[chain_id].append(AA3_TO_1.get(resname, "X"))
    return {chain: "".join(seq) for chain, seq in residues_by_chain.items() if seq}


def parse_chain_ids_from_pocket(pdb_path: Path) -> List[str]:
    chain_ids = []
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or "_"
            if chain_id not in seen:
                seen.add(chain_id)
                chain_ids.append(chain_id)
    return chain_ids


def iter_entries(root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    for protein_path in root.rglob("*_protein.pdb"):
        if protein_path.name.endswith(".bak"):
            continue
        base = protein_path.name[:-12]
        pocket_path = protein_path.with_name(f"{base}_pocket.pdb")
        if pocket_path.exists():
            yield protein_path.parent, protein_path, pocket_path


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        items = list(items)
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def group_counts(groups: Dict[str, List[EntryRecord]]) -> Dict[str, int]:
    dup_groups = [members for members in groups.values() if len(members) > 1]
    return {
        "unique_signatures": len(groups),
        "duplicate_groups": len(dup_groups),
        "entries_in_duplicate_groups": sum(len(members) for members in dup_groups),
        "truly_unique_entries": sum(1 for members in groups.values() if len(members) == 1),
        "largest_group_size": max((len(members) for members in groups.values()), default=0),
    }


def write_entry_table(path: Path, records: List[EntryRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "entry_key",
                "pdb_id",
                "bucket",
                "entry_dir",
                "n_all_chains",
                "n_contact_chains",
                "all_chain_lengths",
                "contact_chain_lengths",
                "all_signature",
                "contact_signature",
                "contact_unique_signature",
                "all_chain_hashes",
                "contact_chain_hashes",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.entry_key,
                    record.pdb_id,
                    record.bucket,
                    record.entry_dir,
                    record.n_all_chains,
                    record.n_contact_chains,
                    json.dumps(record.all_chain_lengths, sort_keys=True),
                    json.dumps(record.contact_chain_lengths, sort_keys=True),
                    record.all_signature,
                    record.contact_signature,
                    record.contact_unique_signature,
                    json.dumps(record.all_chain_hashes, sort_keys=True),
                    json.dumps(record.contact_chain_hashes, sort_keys=True),
                ]
            )


def write_group_table(
    path: Path,
    groups: Dict[str, List[EntryRecord]],
    mode: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "mode",
                "group_id",
                "group_size",
                "pdb_ids",
                "entry_dirs",
                "contact_chain_lengths",
                "all_chain_lengths",
            ]
        )
        for group_id, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(members) < 2:
                continue
            writer.writerow(
                [
                    mode,
                    group_id,
                    len(members),
                    ",".join(sorted(m.pdb_id for m in members)),
                    ",".join(sorted(m.entry_dir for m in members)),
                    json.dumps(
                        {m.pdb_id: m.contact_chain_lengths for m in members},
                        sort_keys=True,
                    ),
                    json.dumps(
                        {m.pdb_id: m.all_chain_lengths for m in members},
                        sort_keys=True,
                    ),
                ]
            )


def build_records(root: Path) -> List[EntryRecord]:
    records: List[EntryRecord] = []
    for entry_dir, protein_path, pocket_path in iter_entries(root):
        all_chain_seqs = parse_chain_sequences(protein_path)
        pocket_chain_ids = parse_chain_ids_from_pocket(pocket_path)
        contact_chain_seqs = {
            chain_id: all_chain_seqs[chain_id]
            for chain_id in pocket_chain_ids
            if chain_id in all_chain_seqs
        }
        if not contact_chain_seqs:
            contact_chain_seqs = all_chain_seqs
        rel_parts = protein_path.relative_to(root).parts
        bucket = rel_parts[0] if rel_parts else ""
        records.append(
            EntryRecord(
                entry_key=str(entry_dir.relative_to(root)),
                pdb_id=protein_path.stem.replace("_protein", ""),
                bucket=bucket,
                entry_dir=str(entry_dir.relative_to(root)),
                protein_path=str(protein_path),
                pocket_path=str(pocket_path),
                all_chain_lengths={k: len(v) for k, v in all_chain_seqs.items()},
                contact_chain_lengths={k: len(v) for k, v in contact_chain_seqs.items()},
                all_chain_hashes={k: seq_hash(v) for k, v in all_chain_seqs.items()},
                contact_chain_hashes={k: seq_hash(v) for k, v in contact_chain_seqs.items()},
            )
        )
    return records


def build_signature_groups(records: List[EntryRecord], attr: str) -> Dict[str, List[EntryRecord]]:
    groups: Dict[str, List[EntryRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, attr)].append(record)
    return groups


def build_chain_overlap_components(
    records: List[EntryRecord],
    min_chain_len: int,
) -> Dict[str, List[EntryRecord]]:
    uf = UnionFind(record.entry_key for record in records)
    hash_to_records: Dict[str, List[EntryRecord]] = defaultdict(list)
    for record in records:
        for chain_id, chain_hash in record.contact_chain_hashes.items():
            if record.contact_chain_lengths.get(chain_id, 0) < min_chain_len:
                continue
            hash_to_records[chain_hash].append(record)
    for members in hash_to_records.values():
        if len(members) < 2:
            continue
        first = members[0].entry_key
        for member in members[1:]:
            uf.union(first, member.entry_key)
    components: Dict[str, List[EntryRecord]] = defaultdict(list)
    for record in records:
        components[uf.find(record.entry_key)].append(record)
    return components


def write_summary(path: Path, summary: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--min-chain-len",
        type=int,
        default=80,
        help="Minimum contact-chain length for chain-overlap components",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records = build_records(root)
    records.sort(key=lambda r: r.entry_key)

    by_all_signature = build_signature_groups(records, "all_signature")
    by_contact_signature = build_signature_groups(records, "contact_signature")
    by_contact_unique_signature = build_signature_groups(records, "contact_unique_signature")
    chain_components = build_chain_overlap_components(records, min_chain_len=args.min_chain_len)

    summary = {
        "dataset_root": str(root),
        "total_entries": len(records),
        "min_chain_len_for_components": args.min_chain_len,
        "all_chain_signature": group_counts(by_all_signature),
        "contact_chain_signature": group_counts(by_contact_signature),
        "contact_unique_chain_signature": group_counts(by_contact_unique_signature),
        "contact_chain_overlap_components": group_counts(chain_components),
        "largest_all_signature_group": sorted(
            (
                {
                    "group_id": gid,
                    "size": len(members),
                    "pdb_ids": sorted(m.pdb_id for m in members),
                }
                for gid, members in by_all_signature.items()
            ),
            key=lambda x: (-x["size"], x["group_id"]),
        )[:20],
        "largest_contact_signature_group": sorted(
            (
                {
                    "group_id": gid,
                    "size": len(members),
                    "pdb_ids": sorted(m.pdb_id for m in members),
                }
                for gid, members in by_contact_signature.items()
            ),
            key=lambda x: (-x["size"], x["group_id"]),
        )[:20],
        "largest_contact_unique_signature_group": sorted(
            (
                {
                    "group_id": gid,
                    "size": len(members),
                    "pdb_ids": sorted(m.pdb_id for m in members),
                }
                for gid, members in by_contact_unique_signature.items()
            ),
            key=lambda x: (-x["size"], x["group_id"]),
        )[:20],
        "largest_chain_overlap_components": sorted(
            (
                {
                    "component_id": gid,
                    "size": len(members),
                    "pdb_ids": sorted(m.pdb_id for m in members),
                }
                for gid, members in chain_components.items()
            ),
            key=lambda x: (-x["size"], x["component_id"]),
        )[:20],
        "contact_chain_count_distribution": Counter(r.n_contact_chains for r in records),
        "all_chain_count_distribution": Counter(r.n_all_chains for r in records),
    }

    write_entry_table(outdir / "entry_table.tsv", records)
    write_group_table(outdir / "all_chain_signature_groups.tsv", by_all_signature, "all_chain_signature")
    write_group_table(
        outdir / "contact_chain_signature_groups.tsv",
        by_contact_signature,
        "contact_chain_signature",
    )
    write_group_table(
        outdir / "contact_unique_chain_signature_groups.tsv",
        by_contact_unique_signature,
        "contact_unique_chain_signature",
    )
    write_group_table(
        outdir / "contact_chain_overlap_components.tsv",
        chain_components,
        "contact_chain_overlap_component",
    )
    write_summary(outdir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
