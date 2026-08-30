#!/usr/bin/env python3
"""Materialize a symlink-only PDBbind + BioLiP data root from audited IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch

from audit_biolip_sequence_leakage import YEAR_BUCKETS, read_split_file


FEATURE_LAYOUT = (
    ("complete_features_v4_nma", "complete_features_v4_nma"),
    ("esm2_latents", "esm2_latents"),
    ("ligand_features_v5", "ligand_features_v5"),
)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def find_sample_dir(root: Path, feature_dir: str, sample_id: str) -> tuple[str, Path] | None:
    for bucket in YEAR_BUCKETS:
        path = root / feature_dir / bucket / sample_id
        if path.is_dir():
            return bucket, path.resolve()
    return None


def link_sample(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting symlink: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")
    os.symlink(source, destination, target_is_directory=True)


def select_labels(path: Path, ids: Iterable[str]) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted(set(ids) - set(data))
    if missing:
        raise RuntimeError(f"{path}: missing {len(missing)} labels: {missing[:20]}")
    return {sample_id: data[sample_id] for sample_id in ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--pdbbind-root", type=Path, required=True)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--include-esm3",
        action="store_true",
        help="Require and link normalized ESM3 features for both data sources.",
    )
    args = parser.parse_args()

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = read_split_file(args.split_root / "pdbbind_chain_si30_splits.txt")
    bio_ids = read_ids(args.split_root / "biolip_train_ids.txt")
    pdb_ids = sorted(set().union(*splits.values()))
    collision = set(pdb_ids) & set(bio_ids)
    if collision:
        raise RuntimeError(f"sample ID collision: {sorted(collision)[:20]}")

    feature_layout = list(FEATURE_LAYOUT)
    if args.include_esm3:
        feature_layout.append(("esm3_latents_norm", "esm3_latents_norm"))

    provenance = []
    missing_features = []
    for source_name, root, sample_ids in (
        ("pdbbind", args.pdbbind_root, pdb_ids),
        ("biolip", args.biolip_root, bio_ids),
    ):
        for sample_id in sample_ids:
            for output_feature, pdb_feature in feature_layout:
                source_feature = pdb_feature
                if source_name == "biolip" and output_feature == "complete_features_v4_nma":
                    source_feature = "complete_features_v4"
                found = find_sample_dir(root, source_feature, sample_id)
                if found is None:
                    missing_features.append(
                        {"source": source_name, "sample_id": sample_id, "feature": source_feature}
                    )
                    continue
                bucket, source = found
                destination = output / output_feature / bucket / sample_id
                link_sample(source, destination)
                provenance.append(
                    {
                        "source": source_name,
                        "sample_id": sample_id,
                        "feature": output_feature,
                        "path": str(source),
                    }
                )
    if missing_features:
        (output / "missing_features.json").write_text(json.dumps(missing_features, indent=2))
        raise RuntimeError(f"missing {len(missing_features)} required feature directories")

    pdb_labels = select_labels(args.pdbbind_root / "residue_ligand_distances_dcenter.pt", pdb_ids)
    bio_labels = select_labels(args.biolip_root / "residue_ligand_distances_dcenter.pt", bio_ids)
    overlap = set(pdb_labels) & set(bio_labels)
    if overlap:
        raise RuntimeError(f"label collision: {sorted(overlap)[:20]}")
    labels = {**pdb_labels, **bio_labels}
    torch.save(labels, output / "residue_ligand_distances_dcenter.pt")

    with (output / "splits.txt").open("w") as handle:
        handle.write("# Audited chain-SI30 split with BioLiP train-only augmentation\n")
        for sample_id in splits.get("train", []):
            handle.write(f"train\t{sample_id}\n")
        for sample_id in bio_ids:
            handle.write(f"train\t{sample_id}\n")
        for split in ("val", "test"):
            for sample_id in splits.get(split, []):
                handle.write(f"{split}\t{sample_id}\n")

    summary = {
        "pdbbind": {name: len(values) for name, values in splits.items()},
        "biolip_train": len(bio_ids),
        "merged_train": len(splits.get("train", [])) + len(bio_ids),
        "labels": len(labels),
        "feature_links": len(provenance),
        "include_esm3": args.include_esm3,
        "source_split_root": str(args.split_root.resolve()),
    }
    (output / "provenance.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in provenance)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
