#!/usr/bin/env python3
"""Build a BioLip training-readiness and multi-site supervision ledger.

This phase is metadata-only by design. It loads the existing pair cache once,
indexes already extracted assets by filename, and writes strict clean-organic
candidate manifests. It does not modify BioLip data or load every tensor.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from preparation.ligand.ligand_biolip_types import (  # noqa: E402
    LIGAND_TYPES,
    LigandRoute,
    classify_ligand_type,
)


PAIR_FIELDS = (
    "pair_id",
    "feature_stem",
    "pdb_id",
    "protein_chain",
    "binding_site_id",
    "ligand_id",
    "ligand_resseq",
    "heavy_atoms_meta",
    "binding_residue_count",
    "binding_residues",
    "site_signature",
    "route_meta",
    "primary_type_meta",
    "mapping_status",
    "mapping_complete",
    "has_fixed_receptor",
    "has_protein_v4",
    "has_esm2_chain",
    "has_ligand_v6",
    "pdbbind_split",
    "metadata_eligible",
    "assets_ready_except_ligand",
    "current_feature_ready",
    "strict_no_pdbbind_ready",
    "exclusion_reasons",
)


def _iter_file_names(root: Path, suffix: str = "") -> Iterator[str]:
    if not root.is_dir():
        return
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_file() and (not suffix or entry.name.endswith(suffix)):
                yield entry.name


def _index_assets(biolip_root: Path) -> Dict[str, set]:
    protein_ids = {
        name[: -len("_protein.pt")].lower()
        for name in _iter_file_names(biolip_root / "protein_features", "_protein.pt")
    }
    esm_keys = set()
    for name in _iter_file_names(biolip_root / "esm2_features", "_esm2.pt"):
        bundle = name[: -len("_esm2.pt")]
        if "_" not in bundle:
            continue
        pdb_id, chain = bundle.split("_", 1)
        esm_keys.add((pdb_id.lower(), chain))

    fixed_ids = set()
    fixed_root = biolip_root / "fixed_receptors"
    if fixed_root.is_dir():
        with os.scandir(fixed_root) as entries:
            for entry in entries:
                if entry.is_dir() and (Path(entry.path) / "receptor_fixed.cif").is_file():
                    fixed_ids.add(entry.name.lower())

    ligand_keys = set()
    ligand_root = biolip_root / "ligand_features"
    if ligand_root.is_dir():
        with os.scandir(ligand_root) as pdb_entries:
            for pdb_entry in pdb_entries:
                if not pdb_entry.is_dir():
                    continue
                pdb_id = pdb_entry.name.lower()
                with os.scandir(pdb_entry.path) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith(".pt"):
                            ligand_keys.add((pdb_id, entry.name[:-3]))

    return {
        "protein_ids": protein_ids,
        "esm_keys": esm_keys,
        "fixed_ids": fixed_ids,
        "ligand_keys": ligand_keys,
    }


def _load_pdbbind_splits(path: Path | None) -> Dict[str, str]:
    if path is None or not path.is_file():
        return {}
    split_by_id: Dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            split_by_id[fields[1].lower()] = fields[0].lower()
    return split_by_id


def _binding_residues(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[\s,;]+", str(value).strip())
    return sorted({item.strip() for item in raw if item.strip()})


def _primary_type(type_vector) -> str:
    for index in (0, 1, 2, 4, 5, 6, 7, 8, 12, 13):
        if index < len(type_vector) and float(type_vector[index]) > 0.5:
            return LIGAND_TYPES[index]
    return "unknown_or_unclassified"


def _pair_row(
    pair: Mapping,
    assets: Mapping[str, set],
    pdbbind_splits: Mapping[str, str],
    min_heavy: int,
    max_heavy: int,
    min_binding_residues: int,
) -> Dict[str, object]:
    meta = pair.get("meta") or {}
    pdb_id = str(pair.get("pdb_id") or meta.get("pdb_id") or "").lower()
    chain = str(pair.get("chain") or meta.get("protein_chain") or "")
    site = str(pair.get("bs_id") or meta.get("binding_site_id") or "")
    ligand_id = str(pair.get("lig_id") or meta.get("ligand_id") or "").upper()
    resseq = str(pair.get("resseq") or meta.get("ligand_resseq") or "")
    stem = f"{pdb_id}_{chain}_{site}_{ligand_id}"
    pair_id = f"{stem}_{resseq}" if resseq else stem

    try:
        heavy = int(meta.get("ligand_heavy_atom_count") or 0)
    except (TypeError, ValueError):
        heavy = 0
    residues = _binding_residues(meta.get("binding_residues"))
    signature = "|".join(residues)

    ligand_type, route, _ = classify_ligand_type(meta, None)
    route_name = LigandRoute(int(route)).name.lower()
    primary_type = _primary_type(ligand_type)

    mapping_status = str(meta.get("mapping_status") or "")
    mapping_complete = all(
        meta.get(key) is True
        for key in (
            "heavy_atom_mapping_complete",
            "heavy_atom_mapping_injective",
            "heavy_atom_mapping_surjective",
        )
    )

    has_fixed = pdb_id in assets["fixed_ids"]
    has_protein = pdb_id in assets["protein_ids"]
    has_esm = (pdb_id, chain) in assets["esm_keys"]
    has_ligand = (pdb_id, stem) in assets["ligand_keys"]
    pdbbind_split = pdbbind_splits.get(pdb_id, "")

    reasons = []
    if route != LigandRoute.REGULAR_ORGANIC:
        reasons.append(f"route:{route_name}")
    if heavy < min_heavy or heavy > max_heavy:
        reasons.append("heavy_atom_range")
    if len(residues) < min_binding_residues:
        reasons.append("too_few_binding_residues")
    if mapping_status != "exact_match" or not mapping_complete:
        reasons.append("mapping_not_exact_complete")
    if not pdb_id or not chain or not site or not ligand_id:
        reasons.append("missing_pair_identity")

    metadata_eligible = not reasons
    assets_ready_except_ligand = metadata_eligible and has_fixed and has_protein and has_esm
    current_feature_ready = assets_ready_except_ligand and has_ligand
    strict_no_pdbbind_ready = current_feature_ready and not pdbbind_split

    asset_reasons = []
    if metadata_eligible:
        if not has_fixed:
            asset_reasons.append("missing_fixed_receptor")
        if not has_protein:
            asset_reasons.append("missing_protein_v4")
        if not has_esm:
            asset_reasons.append("missing_esm2_chain")
        if not has_ligand:
            asset_reasons.append("missing_ligand_v6")

    return {
        "pair_id": pair_id,
        "feature_stem": stem,
        "pdb_id": pdb_id,
        "protein_chain": chain,
        "binding_site_id": site,
        "ligand_id": ligand_id,
        "ligand_resseq": resseq,
        "heavy_atoms_meta": heavy,
        "binding_residue_count": len(residues),
        "binding_residues": " ".join(residues),
        "site_signature": signature,
        "route_meta": route_name,
        "primary_type_meta": primary_type,
        "mapping_status": mapping_status,
        "mapping_complete": int(mapping_complete),
        "has_fixed_receptor": int(has_fixed),
        "has_protein_v4": int(has_protein),
        "has_esm2_chain": int(has_esm),
        "has_ligand_v6": int(has_ligand),
        "pdbbind_split": pdbbind_split,
        "metadata_eligible": int(metadata_eligible),
        "assets_ready_except_ligand": int(assets_ready_except_ligand),
        "current_feature_ready": int(current_feature_ready),
        "strict_no_pdbbind_ready": int(strict_no_pdbbind_ready),
        "exclusion_reasons": ";".join(reasons + asset_reasons),
    }


def _open_tsv_gz(path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    handle = gzip.open(tmp_path, "wt", encoding="utf-8", newline="", compresslevel=1)
    return tmp_path, handle


def _write_group_manifest(
    path: Path,
    groups: Mapping[tuple[str, str], MutableMapping[str, object]],
) -> Dict[str, int]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    counts = Counter()
    with tmp.open("w", newline="") as handle:
        fields = (
            "pdb_id",
            "protein_chain",
            "eligible_pairs",
            "distinct_binding_site_ids",
            "distinct_site_signatures",
            "current_ready_pairs",
            "strict_ready_pairs",
            "is_multisite",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for (pdb_id, chain), group in sorted(groups.items()):
            n_signatures = len(group["signatures"])
            is_multisite = n_signatures >= 2
            counts["groups"] += 1
            counts["multisite_groups"] += int(is_multisite)
            counts["pairs_in_multisite_groups"] += int(group["pairs"]) if is_multisite else 0
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "protein_chain": chain,
                    "eligible_pairs": group["pairs"],
                    "distinct_binding_site_ids": len(group["site_ids"]),
                    "distinct_site_signatures": n_signatures,
                    "current_ready_pairs": group["current_ready"],
                    "strict_ready_pairs": group["strict_ready"],
                    "is_multisite": int(is_multisite),
                }
            )
    tmp.replace(path)
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--pairs-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdbbind-splits", type=Path, default=None)
    parser.add_argument("--min-heavy-atoms", type=int, default=6)
    parser.add_argument("--max-heavy-atoms", type=int, default=128)
    parser.add_argument("--min-binding-residues", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 scans the full cache")
    args = parser.parse_args()

    started = time.time()
    root = args.biolip_root.resolve()
    cache_path = (args.pairs_cache or root / "pairs_cache_v6.pkl").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    print(f"Indexing BioLip assets under {root}", flush=True)
    assets = _index_assets(root)
    asset_counts = {key: len(value) for key, value in assets.items()}
    print(f"Asset counts: {asset_counts}", flush=True)

    print(f"Loading pair cache: {cache_path}", flush=True)
    with cache_path.open("rb") as handle:
        pairs = pickle.load(handle)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    print(f"Loaded {len(pairs):,} pairs", flush=True)

    pdbbind_splits = _load_pdbbind_splits(args.pdbbind_splits)
    counters = Counter()
    route_counts = Counter()
    exclusion_counts = Counter()
    unique_pdbs = set()
    unique_chains = set()
    groups: Dict[tuple[str, str], MutableMapping[str, object]] = defaultdict(
        lambda: {
            "pairs": 0,
            "site_ids": set(),
            "signatures": set(),
            "current_ready": 0,
            "strict_ready": 0,
        }
    )

    all_path = args.output_dir / "biolip_pairs_metadata.tsv.gz"
    ready_path = args.output_dir / "biolip_current_feature_ready.tsv"
    extract_path = args.output_dir / "biolip_needs_ligand_extraction.tsv"
    all_tmp, all_handle = _open_tsv_gz(all_path)
    ready_tmp = ready_path.with_suffix(".tsv.tmp")
    extract_tmp = extract_path.with_suffix(".tsv.tmp")

    with all_handle, ready_tmp.open("w", newline="") as ready_handle, extract_tmp.open(
        "w", newline=""
    ) as extract_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=PAIR_FIELDS, delimiter="\t")
        ready_writer = csv.DictWriter(ready_handle, fieldnames=PAIR_FIELDS, delimiter="\t")
        extract_writer = csv.DictWriter(extract_handle, fieldnames=PAIR_FIELDS, delimiter="\t")
        all_writer.writeheader()
        ready_writer.writeheader()
        extract_writer.writeheader()

        for index, pair in enumerate(pairs, 1):
            row = _pair_row(
                pair,
                assets,
                pdbbind_splits,
                args.min_heavy_atoms,
                args.max_heavy_atoms,
                args.min_binding_residues,
            )
            all_writer.writerow(row)
            counters["pairs"] += 1
            route_counts[str(row["route_meta"])] += 1
            unique_pdbs.add(str(row["pdb_id"]))
            unique_chains.add((str(row["pdb_id"]), str(row["protein_chain"])))
            for reason in str(row["exclusion_reasons"]).split(";"):
                if reason:
                    exclusion_counts[reason] += 1

            for key in (
                "metadata_eligible",
                "assets_ready_except_ligand",
                "current_feature_ready",
                "strict_no_pdbbind_ready",
            ):
                counters[key] += int(row[key])
            counters["direct_pdbbind_overlap"] += int(bool(row["pdbbind_split"]))

            if int(row["metadata_eligible"]):
                group = groups[(str(row["pdb_id"]), str(row["protein_chain"]))]
                group["pairs"] += 1
                group["site_ids"].add(str(row["binding_site_id"]))
                group["signatures"].add(str(row["site_signature"]))
                group["current_ready"] += int(row["current_feature_ready"])
                group["strict_ready"] += int(row["strict_no_pdbbind_ready"])

            if int(row["current_feature_ready"]):
                ready_writer.writerow(row)
            elif int(row["assets_ready_except_ligand"]):
                extract_writer.writerow(row)

            if index % 100_000 == 0:
                elapsed = time.time() - started
                print(
                    f"[{index:,}/{len(pairs):,}] {index / max(elapsed, 1):.0f} pairs/s "
                    f"metadata_eligible={counters['metadata_eligible']:,} "
                    f"current_ready={counters['current_feature_ready']:,}",
                    flush=True,
                )

    all_tmp.replace(all_path)
    ready_tmp.replace(ready_path)
    extract_tmp.replace(extract_path)
    group_counts = _write_group_manifest(args.output_dir / "biolip_multisite_groups.tsv", groups)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 2),
        "biolip_root": str(root),
        "pairs_cache": str(cache_path),
        "filters": {
            "route": "regular_organic",
            "min_heavy_atoms": args.min_heavy_atoms,
            "max_heavy_atoms": args.max_heavy_atoms,
            "min_binding_residues": args.min_binding_residues,
            "mapping": "exact_match + complete/injective/surjective",
            "strict_ready_excludes_direct_pdbbind_ids": True,
        },
        "asset_index_counts": asset_counts,
        "counts": {
            **dict(counters),
            "unique_pdbs": len(unique_pdbs),
            "unique_pdb_chains": len(unique_chains),
            **group_counts,
        },
        "route_counts": dict(route_counts.most_common()),
        "exclusion_counts": dict(exclusion_counts.most_common()),
        "pdbbind_split_ids": len(pdbbind_splits),
        "outputs": {
            "all_pairs": str(all_path),
            "current_feature_ready": str(ready_path),
            "needs_ligand_extraction": str(extract_path),
            "multisite_groups": str(args.output_dir / "biolip_multisite_groups.tsv"),
        },
        "next_qc": (
            "Load tensors only for current_feature_ready rows; slice protein V4 to protein_chain, "
            "verify ESM2/residue alignment and finite 109D ligand tensors, then generate per-pair labels."
        ),
    }
    summary_tmp = args.output_dir / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_tmp.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
