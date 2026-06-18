#!/usr/bin/env python3
"""Analyze whether same-protein assignment negatives are safe and frequent.

The proposed V11m15 assignment losses need same-protein hard negatives.  This
script checks whether the current training split has exact residue-order groups
where a logit-region loss can reuse another ligand pocket mask on the current
sample without cross-structure alignment.

Grouping is intentionally conservative:
  group_key = residue_names + chain_ids + residue_indices + insertion_codes

Only samples in the same group have masks compared.  Candidate negatives are
then filtered by contact-mask overlap and pocket centroid distance to avoid
obvious false negatives.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml


YEAR_RANGES = ("1981-2000", "2001-2010", "2011-2019")


def _load_split_ids(splits_path: Path, split: str) -> List[str]:
    ids: List[str] = []
    with splits_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) >= 2 and parts[0] == split:
                ids.append(parts[1])
            elif len(parts) == 1 and split == "all":
                ids.append(parts[0])
    if not ids:
        raise RuntimeError(f"No ids found for split={split!r} in {splits_path}")
    return ids


def _resolve(path: Optional[str], root: Path) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def _find_protein_file(data_root: Path, nma_dir: str, protein_dir: str, pdb_id: str) -> Optional[Path]:
    for base_dir in (nma_dir, protein_dir):
        root = data_root / base_dir
        for yr in YEAR_RANGES:
            d = root / yr / pdb_id
            for name in (f"{pdb_id}_protein_conf0.pt", f"{pdb_id}_protein.pt"):
                p = d / name
                if p.exists():
                    return p
    return None


def _labels_from_distance(distance_data, contact_thresh: float, near_thresh: float) -> torch.Tensor:
    if isinstance(distance_data, dict):
        if "distances_vdw_adjusted" in distance_data:
            d = distance_data["distances_vdw_adjusted"].float()
            labels = torch.zeros_like(d, dtype=torch.long)
            labels[d < 0.5] = 0
            labels[(d >= 0.5) & (d < 4.5)] = 1
            labels[d >= 4.5] = 2
            return labels
        d = distance_data["distances"].float()
    else:
        d = distance_data.float()
    labels = torch.full_like(d, 2, dtype=torch.long)
    labels[d < contact_thresh] = 0
    labels[(d >= contact_thresh) & (d < near_thresh)] = 1
    return labels


def _group_key(prot: Dict) -> str:
    residue_names = [str(x) for x in prot.get("residue_names", [])]
    chain_ids = [str(x) for x in prot.get("chain_ids", [])]
    residue_indices = [str(x) for x in prot.get("residue_indices", [])]
    insertion_codes = [str(x) for x in prot.get("insertion_codes", [])]
    n = min(len(residue_names), len(chain_ids), len(residue_indices), len(insertion_codes))
    if n == 0:
        n = len(residue_names)
        chain_ids = [""] * n
        residue_indices = [str(i) for i in range(n)]
        insertion_codes = [""] * n
    parts = [
        f"{chain_ids[i]}:{residue_indices[i]}:{insertion_codes[i]}:{residue_names[i]}"
        for i in range(n)
    ]
    return "|".join(parts)


def _contact_mask(labels: torch.Tensor, n: int) -> np.ndarray:
    return (labels[:n].cpu().numpy() == 0)


def _centroid(coords: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() == 0:
        return None
    return coords[mask].mean(axis=0)


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {k: float("nan") for k in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    return {
        "min": float(vals.min()),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--jaccard-thresh", type=float, default=0.25)
    ap.add_argument("--centroid-min-dist", type=float, default=7.0)
    ap.add_argument("--overlap-count-thresh", type=int, default=4)
    args = ap.parse_args()

    config_path = Path(args.config)
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    data_cfg = cfg.get("data", {})
    data_root = _resolve(data_cfg.get("data_root", "data"), Path.cwd())
    assert data_root is not None
    distances_path = _resolve(data_cfg.get("distances_file"), Path.cwd()) or (data_root / "residue_ligand_distances_vdw.pt")
    nma_dir = data_cfg.get("nma_features_dir", "complete_features_v3_nma")
    protein_dir = data_cfg.get("protein_features_dir", "complete_features_v3")
    contact_thresh = float(data_cfg.get("label_contact_thresh", 5.0))
    near_thresh = float(data_cfg.get("label_near_thresh", 8.0))

    ids = _load_split_ids(Path(args.splits), args.split)
    if args.max_samples > 0:
        ids = ids[: args.max_samples]
    distances = torch.load(distances_path, map_location="cpu", weights_only=False)

    samples = []
    missing = []
    for pdb_id in ids:
        if pdb_id not in distances:
            missing.append((pdb_id, "distance"))
            continue
        protein_file = _find_protein_file(data_root, nma_dir, protein_dir, pdb_id)
        if protein_file is None:
            missing.append((pdb_id, "protein"))
            continue
        prot = torch.load(protein_file, map_location="cpu", weights_only=False)
        labels = _labels_from_distance(distances[pdb_id], contact_thresh, near_thresh)
        n = min(int(prot.get("n_residues", len(labels))), len(labels), len(prot["ca_coords"]))
        mask = _contact_mask(labels, n)
        coords = prot["ca_coords"][:n].float().cpu().numpy()
        samples.append({
            "pdb_id": pdb_id,
            "group_key": _group_key(prot),
            "n_residues": n,
            "n_contact": int(mask.sum()),
            "mask": mask,
            "centroid": _centroid(coords, mask),
        })

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["group_key"]].append(sample)

    candidate_counts = []
    valid_counts = []
    jaccards = []
    centroid_dists = []
    overlap_counts = []
    group_sizes = []
    valid_sample_ids = []
    invalid_reason_counts = defaultdict(int)

    for group in groups.values():
        if len(group) < 2:
            continue
        group_sizes.append(len(group))
        for src in group:
            if src["n_contact"] == 0 or src["centroid"] is None:
                invalid_reason_counts["zero_positive_contact"] += 1
                continue
            n_candidates = 0
            n_valid = 0
            for neg in group:
                if neg["pdb_id"] == src["pdb_id"]:
                    continue
                if neg["n_contact"] == 0 or neg["centroid"] is None:
                    invalid_reason_counts["negative_zero_contact"] += 1
                    continue
                n_candidates += 1
                inter = int(np.logical_and(src["mask"], neg["mask"]).sum())
                union = int(np.logical_or(src["mask"], neg["mask"]).sum())
                jac = inter / max(union, 1)
                dist = float(np.linalg.norm(src["centroid"] - neg["centroid"]))
                jaccards.append(jac)
                centroid_dists.append(dist)
                overlap_counts.append(inter)
                if jac > args.jaccard_thresh:
                    invalid_reason_counts["jaccard_masked"] += 1
                    continue
                if dist < args.centroid_min_dist:
                    invalid_reason_counts["centroid_masked"] += 1
                    continue
                if inter >= args.overlap_count_thresh:
                    invalid_reason_counts["overlap_count_masked"] += 1
                    continue
                n_valid += 1
            candidate_counts.append(n_candidates)
            valid_counts.append(n_valid)
            if n_valid > 0:
                valid_sample_ids.append(src["pdb_id"])

    n_samples = len(samples)
    n_groups = len(groups)
    multi_groups = [g for g in groups.values() if len(g) >= 2]
    result = {
        "config": str(config_path),
        "splits": str(Path(args.splits)),
        "split": args.split,
        "data_root": str(data_root),
        "distances_file": str(distances_path),
        "n_requested_ids": len(ids),
        "n_loaded_samples": n_samples,
        "n_missing": len(missing),
        "n_groups": n_groups,
        "n_multi_sample_exact_groups": len(multi_groups),
        "samples_in_multi_sample_exact_groups": int(sum(len(g) for g in multi_groups)),
        "fraction_samples_in_multi_sample_exact_groups": float(sum(len(g) for g in multi_groups) / max(n_samples, 1)),
        "samples_with_any_same_group_candidate": int(sum(1 for c in candidate_counts if c > 0)),
        "samples_with_valid_masked_negative": len(valid_sample_ids),
        "fraction_samples_with_valid_masked_negative": float(len(valid_sample_ids) / max(n_samples, 1)),
        "group_size_distribution": _quantiles(group_sizes),
        "candidate_count_distribution": _quantiles(candidate_counts),
        "valid_negative_count_distribution": _quantiles(valid_counts),
        "jaccard_distribution_all_candidates": _quantiles(jaccards),
        "centroid_distance_distribution_all_candidates": _quantiles(centroid_dists),
        "overlap_count_distribution_all_candidates": _quantiles(overlap_counts),
        "invalid_reason_counts": dict(invalid_reason_counts),
        "example_valid_sample_ids": valid_sample_ids[:50],
        "example_missing": missing[:50],
        "mask_policy": {
            "jaccard_thresh": args.jaccard_thresh,
            "centroid_min_dist": args.centroid_min_dist,
            "overlap_count_thresh": args.overlap_count_thresh,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
