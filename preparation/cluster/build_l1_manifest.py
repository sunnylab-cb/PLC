#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pdbbind_exact_multiligand_analysis import (
    parse_chain_ids_from_pocket,
    parse_chain_sequences,
    seq_hash,
    signature_from_hashes,
)


def read_delimited_table(path: Path) -> List[dict]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def value_present(value: str | None) -> bool:
    text = normalize_text(value)
    return text not in {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "."}


def safe_float(value: str | None) -> float | None:
    if not value_present(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def method_priority(method: str | None) -> int:
    text = normalize_text(method).upper()
    if not text:
        return 99
    if "X-RAY" in text or "DIFFRACTION" in text:
        return 0
    if "ELECTRON MICROSCOPY" in text or "CRYO-EM" in text:
        return 1
    if "NMR" in text:
        return 2
    return 50


def fetch_rcsb_entry_metadata(pdb_id: str, timeout: int) -> dict:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return {}
    exptl = data.get("exptl", [])
    methods = [item.get("method", "") for item in exptl if item.get("method")]
    resolution_candidates = data.get("rcsb_entry_info", {}).get("resolution_combined", [])
    resolution = None
    if resolution_candidates:
        try:
            resolution = min(float(x) for x in resolution_candidates if x is not None)
        except ValueError:
            resolution = None
    return {
        "experimental_method": "; ".join(methods),
        "resolution_angstrom": resolution,
        "metadata_source": "rcsb",
    }


def load_or_fetch_rcsb_metadata(
    pdb_ids: Iterable[str],
    cache_path: Path | None,
    fetch_enabled: bool,
    timeout: int,
) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    if fetch_enabled:
        changed = False
        for pdb_id in sorted(set(pdb_ids)):
            if pdb_id in cache:
                continue
            cache[pdb_id] = fetch_rcsb_entry_metadata(pdb_id, timeout=timeout)
            changed = True
        if changed and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle, indent=2, sort_keys=True)
    return cache


def build_label_index(
    label_table: Path | None,
    label_key: str,
    label_columns: List[str],
) -> Dict[str, dict]:
    if label_table is None:
        return {}
    rows = read_delimited_table(label_table)
    index: Dict[str, dict] = {}
    for row in rows:
        key = normalize_text(row.get(label_key))
        if not key:
            continue
        present_count = sum(1 for col in label_columns if value_present(row.get(col)))
        numeric_values = [safe_float(row.get(col)) for col in label_columns]
        best_numeric = None
        for value in numeric_values:
            if value is None:
                continue
            if best_numeric is None or value > best_numeric:
                best_numeric = value
        index[key] = {
            "label_present_count": present_count,
            "label_best_numeric": best_numeric,
            "label_values": {col: row.get(col, "") for col in label_columns},
        }
    return index


@dataclass
class ManifestEntry:
    entry_key: str
    pdb_id: str
    bucket: str
    protein_path: str
    pocket_path: str
    all_chain_sequences: Dict[str, str]
    contact_chain_sequences: Dict[str, str]
    all_chain_lengths: Dict[str, int]
    contact_chain_lengths: Dict[str, int]
    all_chain_hashes: Dict[str, str]
    contact_chain_hashes: Dict[str, str]

    @property
    def all_signature(self) -> str:
        return signature_from_hashes(self.all_chain_hashes.values())

    @property
    def contact_signature(self) -> str:
        return signature_from_hashes(self.contact_chain_hashes.values())

    @property
    def contact_unique_chain_signature(self) -> str:
        return signature_from_hashes(set(self.contact_chain_hashes.values()))

    @property
    def contact_unique_chain_hashes(self) -> List[str]:
        return sorted(set(self.contact_chain_hashes.values()))

    @property
    def n_all_chains(self) -> int:
        return len(self.all_chain_sequences)

    @property
    def n_contact_chains(self) -> int:
        return len(self.contact_chain_sequences)

    @property
    def n_contact_unique_chains(self) -> int:
        return len(self.contact_unique_chain_hashes)

    @property
    def total_all_residues(self) -> int:
        return sum(self.all_chain_lengths.values())

    @property
    def total_contact_residues(self) -> int:
        return sum(self.contact_chain_lengths.values())

    @property
    def total_contact_unique_residues(self) -> int:
        unique_hash_to_length = {}
        for chain_id, chain_hash in self.contact_chain_hashes.items():
            unique_hash_to_length.setdefault(chain_hash, self.contact_chain_lengths[chain_id])
        return sum(unique_hash_to_length.values())

    @property
    def representative_contact_sequences(self) -> List[Tuple[str, str]]:
        seq_by_hash = {}
        for chain_id, seq in self.contact_chain_sequences.items():
            seq_by_hash.setdefault(seq_hash(seq), seq)
        return sorted(seq_by_hash.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)


def iter_entries(root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    for protein_path in root.rglob("*_protein.pdb"):
        if protein_path.name.endswith(".bak"):
            continue
        base = protein_path.name[:-12]
        pocket_path = protein_path.with_name(f"{base}_pocket.pdb")
        if pocket_path.exists():
            yield protein_path.parent, protein_path, pocket_path


def build_manifest_entries(root: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    for entry_dir, protein_path, pocket_path in iter_entries(root):
        all_chain_sequences = parse_chain_sequences(protein_path)
        pocket_chain_ids = parse_chain_ids_from_pocket(pocket_path)
        contact_chain_sequences = {
            chain_id: all_chain_sequences[chain_id]
            for chain_id in pocket_chain_ids
            if chain_id in all_chain_sequences
        }
        if not contact_chain_sequences:
            contact_chain_sequences = all_chain_sequences
        rel_parts = protein_path.relative_to(root).parts
        bucket = rel_parts[0] if rel_parts else ""
        entries.append(
            ManifestEntry(
                entry_key=str(entry_dir.relative_to(root)),
                pdb_id=protein_path.stem.replace("_protein", ""),
                bucket=bucket,
                protein_path=str(protein_path),
                pocket_path=str(pocket_path),
                all_chain_sequences=all_chain_sequences,
                contact_chain_sequences=contact_chain_sequences,
                all_chain_lengths={k: len(v) for k, v in all_chain_sequences.items()},
                contact_chain_lengths={k: len(v) for k, v in contact_chain_sequences.items()},
                all_chain_hashes={k: seq_hash(v) for k, v in all_chain_sequences.items()},
                contact_chain_hashes={k: seq_hash(v) for k, v in contact_chain_sequences.items()},
            )
        )
    entries.sort(key=lambda x: x.entry_key)
    return entries


def build_groups(entries: List[ManifestEntry]) -> Dict[str, List[ManifestEntry]]:
    groups: Dict[str, List[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.contact_unique_chain_signature].append(entry)
    return groups


def build_rank_payload(
    entry: ManifestEntry,
    label_info: dict,
    rcsb_info: dict,
) -> dict:
    label_present_count = int(label_info.get("label_present_count", 0))
    label_best_numeric = label_info.get("label_best_numeric")
    method = rcsb_info.get("experimental_method", "")
    resolution = rcsb_info.get("resolution_angstrom")
    resolution_sort = float(resolution) if resolution is not None else math.inf
    payload = {
        "label_present_count": label_present_count,
        "label_best_numeric": label_best_numeric,
        "experimental_method": method,
        "experimental_method_priority": method_priority(method),
        "resolution_angstrom": resolution,
        "n_all_chains": entry.n_all_chains,
        "n_contact_chains": entry.n_contact_chains,
        "total_all_residues": entry.total_all_residues,
        "total_contact_residues": entry.total_contact_residues,
        "entry_key": entry.entry_key,
    }
    payload["rank_key"] = (
        -label_present_count,
        -1 if label_best_numeric is not None else 0,
        -(label_best_numeric if label_best_numeric is not None else float("-inf")),
        payload["experimental_method_priority"],
        resolution_sort,
        entry.n_all_chains,
        entry.n_contact_chains,
        entry.total_all_residues,
        entry.total_contact_residues,
        entry.entry_key,
    )
    return payload


def select_representatives(
    groups: Dict[str, List[ManifestEntry]],
    label_index: Dict[str, dict],
    label_key: str,
    rcsb_metadata: Dict[str, dict],
) -> Tuple[Dict[str, ManifestEntry], Dict[str, dict]]:
    representatives: Dict[str, ManifestEntry] = {}
    rank_payloads: Dict[str, dict] = {}
    for group_id, members in groups.items():
        scored = []
        for entry in members:
            lookup_key = entry.entry_key if label_key == "entry_key" else entry.pdb_id
            label_info = label_index.get(lookup_key, {})
            rcsb_info = rcsb_metadata.get(entry.pdb_id, {})
            payload = build_rank_payload(entry, label_info=label_info, rcsb_info=rcsb_info)
            scored.append((payload["rank_key"], entry, payload))
        scored.sort(key=lambda x: x[0])
        _, rep_entry, rep_payload = scored[0]
        representatives[group_id] = rep_entry
        rank_payloads[group_id] = rep_payload
    return representatives, rank_payloads


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(path: Path, records: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for idx in range(0, len(sequence), 80):
                handle.write(sequence[idx:idx + 80] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build L1 manifest grouped by contact_unique_chain_signature."
    )
    parser.add_argument("--dataset-root", required=True, help="PDBbind-style dataset root")
    parser.add_argument("--outdir", required=True, help="Output directory for manifest files")
    parser.add_argument(
        "--label-table",
        default=None,
        help="Optional CSV/TSV with affinity or label completeness columns",
    )
    parser.add_argument(
        "--label-key",
        choices=["entry_key", "pdb_id"],
        default="entry_key",
        help="Key column expected in --label-table",
    )
    parser.add_argument(
        "--label-columns",
        default="",
        help="Comma-separated columns used to prefer representative entries",
    )
    parser.add_argument(
        "--rcsb-cache-json",
        default=None,
        help="Optional JSON cache path for RCSB entry metadata",
    )
    parser.add_argument(
        "--fetch-rcsb-metadata",
        action="store_true",
        help="Fetch experimental method and resolution from RCSB",
    )
    parser.add_argument(
        "--rcsb-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for each RCSB metadata request",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    label_columns = [col.strip() for col in args.label_columns.split(",") if col.strip()]
    label_index = build_label_index(
        label_table=Path(args.label_table).resolve() if args.label_table else None,
        label_key=args.label_key,
        label_columns=label_columns,
    )

    entries = build_manifest_entries(dataset_root)
    groups = build_groups(entries)
    rcsb_metadata = load_or_fetch_rcsb_metadata(
        pdb_ids=(entry.pdb_id for entry in entries),
        cache_path=Path(args.rcsb_cache_json).resolve() if args.rcsb_cache_json else None,
        fetch_enabled=args.fetch_rcsb_metadata,
        timeout=args.rcsb_timeout,
    )
    representatives, rank_payloads = select_representatives(
        groups=groups,
        label_index=label_index,
        label_key=args.label_key,
        rcsb_metadata=rcsb_metadata,
    )

    manifest_rows = []
    member_rows = []
    group_rows = []
    sequence_rows = []
    fasta_records = []

    for group_id, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rep_entry = representatives[group_id]
        rep_payload = rank_payloads[group_id]
        rep_metadata = rcsb_metadata.get(rep_entry.pdb_id, {})
        member_keys = [member.entry_key for member in sorted(members, key=lambda x: x.entry_key)]
        member_pdb_ids = [member.pdb_id for member in sorted(members, key=lambda x: x.entry_key)]
        group_rows.append(
            {
                "group_id": group_id,
                "group_size": len(members),
                "representative_entry_key": rep_entry.entry_key,
                "representative_pdb_id": rep_entry.pdb_id,
                "representative_bucket": rep_entry.bucket,
                "representative_experimental_method": rep_payload["experimental_method"],
                "representative_resolution_angstrom": rep_payload["resolution_angstrom"],
                "representative_label_present_count": rep_payload["label_present_count"],
                "contact_unique_chain_count": rep_entry.n_contact_unique_chains,
                "member_entry_keys": json.dumps(member_keys, sort_keys=False),
                "member_pdb_ids": json.dumps(member_pdb_ids, sort_keys=False),
                "representative_rank_payload": json.dumps(
                    {
                        k: v
                        for k, v in rep_payload.items()
                        if k != "rank_key"
                    },
                    sort_keys=True,
                ),
            }
        )
        for member in sorted(members, key=lambda x: x.entry_key):
            member_rows.append(
                {
                    "group_id": group_id,
                    "entry_key": member.entry_key,
                    "pdb_id": member.pdb_id,
                    "bucket": member.bucket,
                    "is_representative": int(member.entry_key == rep_entry.entry_key),
                }
            )
            member_metadata = rcsb_metadata.get(member.pdb_id, {})
            manifest_rows.append(
                {
                    "entry_key": member.entry_key,
                    "pdb_id": member.pdb_id,
                    "bucket": member.bucket,
                    "group_id": group_id,
                    "group_size": len(members),
                    "is_group_representative": int(member.entry_key == rep_entry.entry_key),
                    "representative_entry_key": rep_entry.entry_key,
                    "representative_pdb_id": rep_entry.pdb_id,
                    "all_signature": member.all_signature,
                    "contact_signature": member.contact_signature,
                    "contact_unique_chain_signature": member.contact_unique_chain_signature,
                    "n_all_chains": member.n_all_chains,
                    "n_contact_chains": member.n_contact_chains,
                    "n_contact_unique_chains": member.n_contact_unique_chains,
                    "total_all_residues": member.total_all_residues,
                    "total_contact_residues": member.total_contact_residues,
                    "total_contact_unique_residues": member.total_contact_unique_residues,
                    "all_chain_lengths": json.dumps(member.all_chain_lengths, sort_keys=True),
                    "contact_chain_lengths": json.dumps(member.contact_chain_lengths, sort_keys=True),
                    "all_chain_hashes": json.dumps(member.all_chain_hashes, sort_keys=True),
                    "contact_chain_hashes": json.dumps(member.contact_chain_hashes, sort_keys=True),
                    "contact_unique_chain_hashes": json.dumps(member.contact_unique_chain_hashes),
                    "representative_rank_payload": json.dumps(
                        {
                            k: v
                            for k, v in rep_payload.items()
                            if k != "rank_key"
                        },
                        sort_keys=True,
                    ),
                    "experimental_method": member_metadata.get("experimental_method", ""),
                    "resolution_angstrom": member_metadata.get("resolution_angstrom", ""),
                    "metadata_source": member_metadata.get("metadata_source", ""),
                }
            )
        for seq_index, (chain_hash, sequence) in enumerate(rep_entry.representative_contact_sequences, start=1):
            sequence_rows.append(
                {
                    "group_id": group_id,
                    "representative_entry_key": rep_entry.entry_key,
                    "representative_pdb_id": rep_entry.pdb_id,
                    "sequence_index": seq_index,
                    "chain_hash": chain_hash,
                    "sequence_length": len(sequence),
                    "sequence": sequence,
                }
            )
            fasta_records.append(
                (
                    f"{group_id}|{rep_entry.entry_key}|seq{seq_index}|{chain_hash}|len{len(sequence)}",
                    sequence,
                )
            )

    summary = {
        "dataset_root": str(dataset_root),
        "total_entries": len(entries),
        "total_groups": len(groups),
        "duplicate_groups": sum(1 for members in groups.values() if len(members) > 1),
        "entries_in_duplicate_groups": sum(len(members) for members in groups.values() if len(members) > 1),
        "representative_selection": {
            "label_table": str(Path(args.label_table).resolve()) if args.label_table else None,
            "label_key": args.label_key,
            "label_columns": label_columns,
            "fetch_rcsb_metadata": args.fetch_rcsb_metadata,
            "rcsb_cache_json": str(Path(args.rcsb_cache_json).resolve()) if args.rcsb_cache_json else None,
            "ranking_policy": [
                "prefer entries with more populated label columns",
                "prefer entries with numeric label values when available",
                "prefer X-ray over EM over NMR over unknown methods",
                "prefer better (smaller) resolution when available",
                "prefer fewer total chains",
                "prefer fewer contacting chains",
                "prefer smaller total residue count",
                "tie-break by entry_key",
            ],
        },
        "largest_groups": [
            {
                "group_id": group_id,
                "group_size": len(members),
                "representative_entry_key": representatives[group_id].entry_key,
                "representative_pdb_id": representatives[group_id].pdb_id,
            }
            for group_id, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]
        ],
    }

    write_tsv(
        outdir / "l1_manifest.tsv",
        manifest_rows,
        fieldnames=[
            "entry_key",
            "pdb_id",
            "bucket",
            "group_id",
            "group_size",
            "is_group_representative",
            "representative_entry_key",
            "representative_pdb_id",
            "all_signature",
            "contact_signature",
            "contact_unique_chain_signature",
            "n_all_chains",
            "n_contact_chains",
            "n_contact_unique_chains",
            "total_all_residues",
            "total_contact_residues",
            "total_contact_unique_residues",
            "all_chain_lengths",
            "contact_chain_lengths",
            "all_chain_hashes",
            "contact_chain_hashes",
            "contact_unique_chain_hashes",
            "representative_rank_payload",
            "experimental_method",
            "resolution_angstrom",
            "metadata_source",
        ],
    )
    write_tsv(
        outdir / "l1_group_members.tsv",
        member_rows,
        fieldnames=["group_id", "entry_key", "pdb_id", "bucket", "is_representative"],
    )
    write_tsv(
        outdir / "l1_groups.tsv",
        group_rows,
        fieldnames=[
            "group_id",
            "group_size",
            "representative_entry_key",
            "representative_pdb_id",
            "representative_bucket",
            "representative_experimental_method",
            "representative_resolution_angstrom",
            "representative_label_present_count",
            "contact_unique_chain_count",
            "member_entry_keys",
            "member_pdb_ids",
            "representative_rank_payload",
        ],
    )
    write_tsv(
        outdir / "l1_representative_sequences.tsv",
        sequence_rows,
        fieldnames=[
            "group_id",
            "representative_entry_key",
            "representative_pdb_id",
            "sequence_index",
            "chain_hash",
            "sequence_length",
            "sequence",
        ],
    )
    write_fasta(outdir / "l1_representative_sequences.fasta", fasta_records)
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
