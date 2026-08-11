#!/usr/bin/env python3
"""Materialize a leakage-filtered BioLip pilot for the existing V11 dataset.

The source BioLip feature tree remains read-only. One chain-level protein
sidecar is written per PDB-chain, while per-pair PDBbind-compatible directories
contain symlinks to the chain sidecar, chain ESM2, and ligand V6 tensor. Distance
labels use the same residue-heavy-atom to ligand-heavy-atom d_min definition as
``residue_ligand_distances_dcenter.pt`` (the filename is a historical misnomer).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


YEAR_BUCKET = "2001-2010"
RESIDUE_TENSOR_KEYS = (
    "physchem_features",
    "spatial_scalar_features",
    "spatial_scalar_raw_features",
    "spatial_vector_features",
    "ca_coords",
)
RESIDUE_LIST_KEYS = (
    "residue_names",
    "residue_indices",
    "insertion_codes",
    "chain_ids",
)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping, got {type(value).__name__}")
    return value


def _keys(data: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    chains = data.get("chain_ids", data.get("chain_id"))
    indices = data.get("residue_indices", data.get("residue_index"))
    icodes = data.get("insertion_codes", data.get("insertion_code"))
    if chains is None or indices is None:
        raise KeyError("residue chain/index metadata missing")
    if icodes is None:
        icodes = [""] * len(chains)
    if not (len(chains) == len(indices) == len(icodes)):
        raise ValueError("residue metadata lengths differ")
    return [
        (str(chain), int(index), str(icode or "").strip())
        for chain, index, icode in zip(chains, indices, icodes)
    ]


def slice_protein_chain(
    protein: Mapping[str, Any],
    parsed_receptor: Mapping[str, Any],
    chain: str,
) -> dict[str, Any]:
    """Slice full-receptor V4 tensors and attach exactly aligned all-atom data."""
    protein_keys = _keys(protein)
    parsed_keys = _keys(parsed_receptor)
    parsed_index = {key: index for index, key in enumerate(parsed_keys)}
    if len(parsed_index) != len(parsed_keys):
        raise ValueError("duplicate residue keys in parsed receptor")

    positions = [index for index, key in enumerate(protein_keys) if key[0] == chain]
    if not positions:
        raise ValueError(f"chain {chain!r} absent from V4 protein tensor")
    chain_keys = [protein_keys[index] for index in positions]
    missing = [key for key in chain_keys if key not in parsed_index]
    if missing:
        raise ValueError(f"{len(missing)} V4 residues absent from receptor parse: {missing[:5]}")

    index_tensor = torch.tensor(positions, dtype=torch.long)
    out = dict(protein)
    n_full = len(protein_keys)
    for key in RESIDUE_TENSOR_KEYS:
        value = protein.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != n_full:
            raise ValueError(f"{key}: expected first dimension {n_full}")
        out[key] = value.index_select(0, index_tensor).clone()
    for key in RESIDUE_LIST_KEYS:
        value = protein.get(key)
        if value is None or len(value) != n_full:
            raise ValueError(f"{key}: expected length {n_full}")
        out[key] = [value[index] for index in positions]

    parsed_positions = [parsed_index[key] for key in chain_keys]
    parsed_coords = parsed_receptor.get("all_atom_coords")
    parsed_names = parsed_receptor.get("all_atom_names")
    if parsed_coords is None or len(parsed_coords) != len(parsed_keys):
        raise ValueError("parsed receptor all_atom_coords missing or misaligned")
    if parsed_names is None or len(parsed_names) != len(parsed_keys):
        raise ValueError("parsed receptor all_atom_names missing or misaligned")
    out["all_atom_coords"] = [
        torch.as_tensor(parsed_coords[index], dtype=torch.float32).clone()
        for index in parsed_positions
    ]
    out["all_atom_names"] = [list(parsed_names[index]) for index in parsed_positions]
    out["n_residues"] = len(positions)
    out["selected_chain"] = chain
    out["source_full_receptor_n_residues"] = n_full
    out["sidecar_schema"] = "biolip_v11_chain_v1"
    return out


def ligand_heavy_coords(ligand: Mapping[str, Any]) -> torch.Tensor:
    for key in ("coords_original", "coords_heavy_current"):
        value = ligand.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == 3:
            coords = value.float()
            if coords.numel() and bool(torch.isfinite(coords).all()):
                return coords

    coords = ligand.get("coords")
    features = ligand.get("atom_features")
    if not isinstance(coords, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError("ligand has no usable heavy-atom coordinates")
    if coords.shape[0] != features.shape[0] or features.shape[1] < 11:
        raise ValueError("ligand coords/atom_features are incompatible")
    atom_type = features[:, :11].argmax(dim=-1)
    heavy = coords[atom_type != 9].float()  # V5 atom type index 9 is hydrogen.
    if not heavy.numel() or not bool(torch.isfinite(heavy).all()):
        raise ValueError("derived ligand heavy coordinates are empty/non-finite")
    return heavy


def min_heavy_atom_distances(
    residue_coords: Sequence[torch.Tensor],
    ligand_coords: torch.Tensor,
) -> torch.Tensor:
    distances = []
    ligand_coords = ligand_coords.float()
    for coords in residue_coords:
        coords = torch.as_tensor(coords, dtype=torch.float32)
        if coords.ndim != 2 or coords.shape[1] != 3 or not coords.numel():
            raise ValueError("residue has no valid heavy-atom coordinates")
        distances.append(torch.cdist(coords, ligand_coords).min())
    result = torch.stack(distances).float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("non-finite residue-ligand distances")
    return result


def contact_geometry_stats(d_min: torch.Tensor) -> dict[str, int | float]:
    """Summarize whether a ligand has a usable contact label on this chain."""
    if d_min.ndim != 1 or not d_min.numel() or not bool(torch.isfinite(d_min).all()):
        raise ValueError("residue-ligand distances must be a finite non-empty vector")
    return {
        "n_contact_lt5": int((d_min < 5.0).sum()),
        "n_near_5to8": int(((d_min >= 5.0) & (d_min < 8.0)).sum()),
        "min_distance": float(d_min.min()),
    }


def stable_group_split(
    rows: Sequence[Mapping[str, str]],
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign whole PDB structures to train/val using a stable hash."""
    groups = sorted({str(row["pdb_id"]).lower() for row in rows})
    if len(groups) < 2:
        raise ValueError("at least two PDB groups are required for a split")
    threshold = int(max(0.0, min(1.0, val_fraction)) * (2**64 - 1))
    split = {}
    for group in groups:
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        value = int.from_bytes(digest[:8], "big")
        split[group] = "val" if value <= threshold else "train"
    if len(set(split.values())) < 2:
        split[groups[0]] = "val"
        split[groups[-1]] = "train"
    return split


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def _ensure_symlink(source: Path, target: Path, overwrite: bool) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"target exists with different content: {target}")
        target.unlink()
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(source)
    os.replace(tmp, target)


def _select_rows(
    manifest: Path,
    excluded_ids: set[str],
    max_pairs: int,
) -> list[dict[str, str]]:
    rows = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("strict_no_pdbbind_ready") != "1":
                continue
            if row["pdb_id"].lower() in excluded_ids:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (row["pdb_id"], row["protein_chain"], row["pair_id"]))
    stems = [row["feature_stem"] for row in rows]
    if len(stems) != len(set(stems)):
        duplicates = [stem for stem, count in Counter(stems).items() if count > 1]
        raise ValueError(f"feature_stem is not unique: {duplicates[:10]}")
    return rows[:max_pairs] if max_pairs > 0 else rows


def filter_rows_by_ligand_qc(
    rows: Sequence[Mapping[str, str]],
    source_root: Path,
    required_route: int | None,
    min_completeness: float,
) -> tuple[list[Mapping[str, str]], Counter]:
    """Filter using tensor-derived V6 metadata rather than stale pair metadata."""
    if required_route is None and min_completeness <= 0.0:
        return list(rows), Counter()

    retained = []
    dropped = Counter()
    for row in rows:
        pdb_id = str(row["pdb_id"]).lower()
        stem = str(row["feature_stem"])
        path = source_root / "ligand_features" / pdb_id / f"{stem}.pt"
        if not path.is_file():
            dropped["missing_ligand_tensor"] += 1
            continue
        ligand = _load_mapping(path)
        route = int(ligand.get("ligand_route", -1))
        completeness = float(ligand.get("feature_completeness_ratio", 0.0))
        if required_route is not None and route != required_route:
            dropped[f"ligand_route_{route}"] += 1
            continue
        if completeness < min_completeness:
            dropped["feature_completeness"] += 1
            continue
        retained.append(row)
    return retained, dropped


def _receptor_path(root: Path, pdb_id: str, protein: Mapping[str, Any]) -> Path:
    fixed = root / "fixed_receptors" / pdb_id / "receptor_fixed.cif"
    original = root / "shared_receptors" / pdb_id / "receptor_full.cif"
    preferred = fixed if str(protein.get("cif_source", "fixed")) == "fixed" else original
    if preferred.is_file():
        return preferred
    fallback = original if preferred == fixed else fixed
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"no receptor CIF for {pdb_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-pdb-ids", type=Path, action="append", default=[])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--required-ligand-route", type=int)
    parser.add_argument("--min-feature-completeness", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from preparation.protein.biolip_protein_struct_extract import parse_cif_protein

    source_root = args.biolip_root.resolve()
    output_root = args.output_root.resolve()
    excluded_ids: set[str] = set()
    for path in args.exclude_pdb_ids:
        excluded_ids.update(line.strip().lower() for line in path.read_text().splitlines() if line.strip())
    rows = _select_rows(args.manifest, excluded_ids, args.max_pairs)
    rows, ligand_filter_counts = filter_rows_by_ligand_qc(
        rows,
        source_root=source_root,
        required_route=args.required_ligand_route,
        min_completeness=args.min_feature_completeness,
    )
    if not rows:
        raise RuntimeError("no BioLip rows remain after filtering")
    split_by_pdb = stable_group_split(rows, args.val_fraction, args.seed)

    chain_cache_root = output_root / "_chain_cache"
    distances: dict[str, dict[str, Any]] = {}
    output_rows = []
    geometry_failures = []
    current_pdb = None
    current_protein: Mapping[str, Any] | None = None
    current_parsed: Mapping[str, Any] | None = None
    chain_sidecars: dict[tuple[str, str], Path] = {}

    for index, row in enumerate(rows, 1):
        pdb_id = row["pdb_id"].lower()
        chain = row["protein_chain"]
        sample_id = row["feature_stem"]
        protein_source = source_root / "protein_features" / f"{pdb_id}_protein.pt"
        esm_source = source_root / "esm2_features" / f"{pdb_id}_{chain}_esm2.pt"
        ligand_source = source_root / "ligand_features" / pdb_id / f"{sample_id}.pt"

        if current_pdb != pdb_id:
            current_protein = _load_mapping(protein_source)
            receptor = _receptor_path(source_root, pdb_id, current_protein)
            current_parsed = parse_cif_protein(receptor)
            current_pdb = pdb_id
        assert current_protein is not None and current_parsed is not None

        chain_key = (pdb_id, chain)
        sidecar = chain_sidecars.get(chain_key)
        if sidecar is None:
            sidecar = chain_cache_root / f"{pdb_id}_{chain}_protein.pt"
            if args.overwrite or not sidecar.is_file():
                chain_data = slice_protein_chain(current_protein, current_parsed, chain)
                esm = _load_mapping(esm_source)
                if _keys(chain_data) != _keys(esm):
                    raise ValueError(f"{pdb_id}_{chain}: sidecar/ESM2 residue keys differ")
                _atomic_torch_save(chain_data, sidecar)
            chain_sidecars[chain_key] = sidecar

        chain_data = _load_mapping(sidecar)
        ligand = _load_mapping(ligand_source)
        d_min = min_heavy_atom_distances(chain_data["all_atom_coords"], ligand_heavy_coords(ligand))
        geometry = contact_geometry_stats(d_min)
        n_contact = int(geometry["n_contact_lt5"])
        n_near = int(geometry["n_near_5to8"])
        if n_contact < 1:
            geometry_failures.append(
                {
                    "sample_id": sample_id,
                    "pdb_id": pdb_id,
                    "protein_chain": chain,
                    "pair_id": row.get("pair_id", ""),
                    "min_distance": f"{float(geometry['min_distance']):.6f}",
                    "n_contact_lt5": n_contact,
                    "n_near_5to8": n_near,
                    "reason": "no_residue_within_5A",
                }
            )
            print(
                f"[{index:,}/{len(rows):,}] skipped {sample_id}: "
                f"no residue within 5 A (min={float(geometry['min_distance']):.3f} A)",
                flush=True,
            )
            continue

        pair_dir = output_root / "complete_features_v4" / YEAR_BUCKET / sample_id
        _ensure_symlink(sidecar, pair_dir / f"{sample_id}_protein.pt", args.overwrite)
        _ensure_symlink(
            esm_source,
            output_root / "esm2_latents" / YEAR_BUCKET / sample_id / f"{sample_id}_protein.pt",
            args.overwrite,
        )
        _ensure_symlink(
            ligand_source,
            output_root / "ligand_features_v5" / YEAR_BUCKET / sample_id / f"{sample_id}_ligand.pt",
            args.overwrite,
        )
        distances[sample_id] = {
            "distances": d_min,
            "n_residues": len(d_min),
            "min_distance": float(d_min.min()),
            "max_distance": float(d_min.max()),
            "mean_distance": float(d_min.mean()),
            "source": "BioLip ligand coords_original; residue/ligand heavy-atom d_min",
        }
        output_rows.append(
            {
                **row,
                "sample_id": sample_id,
                "split": split_by_pdb[pdb_id],
                "n_chain_residues": len(d_min),
                "n_contact_lt5": n_contact,
                "n_near_5to8": n_near,
                "min_distance": f"{float(d_min.min()):.6f}",
            }
        )
        if index % 100 == 0:
            print(f"[{index:,}/{len(rows):,}] materialized", flush=True)

    if not output_rows:
        raise RuntimeError("no BioLip rows passed geometry QC")

    _atomic_torch_save(distances, output_root / "residue_ligand_distances_dcenter.pt")
    split_lines = [f"{row['split']}\t{row['sample_id']}" for row in output_rows]
    _atomic_text("\n".join(split_lines) + "\n", output_root / "splits.txt")

    manifest_path = output_root / "pilot_manifest.tsv"
    fieldnames = list(output_rows[0])
    tmp_manifest = manifest_path.with_suffix(f".tsv.{os.getpid()}.tmp")
    with tmp_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(tmp_manifest, manifest_path)

    geometry_failure_path = output_root / "geometry_qc_failures.tsv"
    geometry_failure_fields = [
        "sample_id",
        "pdb_id",
        "protein_chain",
        "pair_id",
        "min_distance",
        "n_contact_lt5",
        "n_near_5to8",
        "reason",
    ]
    tmp_geometry = geometry_failure_path.with_suffix(f".tsv.{os.getpid()}.tmp")
    with tmp_geometry.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=geometry_failure_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(geometry_failures)
    os.replace(tmp_geometry, geometry_failure_path)

    split_counts = Counter(row["split"] for row in output_rows)
    summary = {
        "source_manifest": str(args.manifest.resolve()),
        "source_biolip_root": str(source_root),
        "output_root": str(output_root),
        "year_bucket": YEAR_BUCKET,
        "excluded_direct_pdb_ids": len(excluded_ids),
        "ligand_tensor_filter": {
            "required_route": args.required_ligand_route,
            "min_feature_completeness": args.min_feature_completeness,
            "dropped": dict(sorted(ligand_filter_counts.items())),
        },
        "pairs": len(output_rows),
        "geometry_qc": {
            "dropped": len(geometry_failures),
            "failure_manifest": str(geometry_failure_path),
        },
        "unique_pdbs": len({row["pdb_id"] for row in output_rows}),
        "unique_pdb_chains": len({(row["pdb_id"], row["protein_chain"]) for row in output_rows}),
        "split_counts": dict(split_counts),
        "split_group": "pdb_id",
        "label_definition": "residue/ligand heavy-atom d_min; contact<5A, near=5-8A",
        "contact_count": {
            "min": min(int(row["n_contact_lt5"]) for row in output_rows),
            "mean": sum(int(row["n_contact_lt5"]) for row in output_rows) / len(output_rows),
            "max": max(int(row["n_contact_lt5"]) for row in output_rows),
        },
    }
    _atomic_text(json.dumps(summary, indent=2, sort_keys=True), output_root / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
