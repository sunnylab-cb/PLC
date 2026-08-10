#!/usr/bin/env python3
"""Validate BioLip V4, chain-level ESM2, and ligand tensors per pair.

Protein V4 files contain every receptor chain, while BioLip ESM2 files contain
one selected chain. Alignment therefore uses the explicit residue key
``(chain_id, residue_index, insertion_code)`` rather than tensor length alone.
Ligand atom tensors can include hydrogens, so manifest heavy-atom counts are
checked against ``num_heavy_atoms`` instead of ``atom_features.shape[0]``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


def _load(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping, got {type(value).__name__}")
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _residue_keys(
    chains: Sequence[Any],
    indices: Sequence[Any],
    insertion_codes: Sequence[Any],
) -> list[tuple[str, int, str]]:
    return [
        (str(chain), int(index), str(code or "").strip())
        for chain, index, code in zip(chains, indices, insertion_codes)
    ]


def _check_finite(name: str, value: Any, issues: list[str]) -> None:
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        issues.append(f"nonfinite:{name}")


def _check_shape(
    name: str,
    value: Any,
    expected_tail: tuple[int, ...],
    expected_first: int,
    issues: list[str],
) -> None:
    if not isinstance(value, torch.Tensor):
        issues.append(f"missing_tensor:{name}")
        return
    if value.ndim != len(expected_tail) + 1:
        issues.append(f"shape:{name}:{tuple(value.shape)}")
        return
    if value.shape[0] != expected_first or tuple(value.shape[1:]) != expected_tail:
        issues.append(f"shape:{name}:{tuple(value.shape)}")
    _check_finite(name, value, issues)


def validate_row(row: Mapping[str, str], biolip_root: Path) -> dict[str, Any]:
    pdb_id = row["pdb_id"].lower()
    chain = row["protein_chain"]
    stem = row["feature_stem"]
    paths = {
        "protein": biolip_root / "protein_features" / f"{pdb_id}_protein.pt",
        "esm2": biolip_root / "esm2_features" / f"{pdb_id}_{chain}_esm2.pt",
        "ligand": biolip_root / "ligand_features" / pdb_id / f"{stem}.pt",
    }
    issues: list[str] = []
    warnings: list[str] = []
    for kind, path in paths.items():
        if not path.is_file():
            issues.append(f"missing_file:{kind}")
    if issues:
        return {
            "pair_id": row.get("pair_id", stem),
            "passed": False,
            "issues": issues,
            "warnings": warnings,
            "paths": {key: str(value) for key, value in paths.items()},
        }

    try:
        protein = _load(paths["protein"])
        esm2 = _load(paths["esm2"])
        ligand = _load(paths["ligand"])
    except Exception as error:  # A corrupt torch archive must be reported per row.
        issues.append(f"load_error:{type(error).__name__}:{error}")
        return {
            "pair_id": row.get("pair_id", stem),
            "passed": False,
            "issues": issues,
            "warnings": warnings,
            "paths": {key: str(value) for key, value in paths.items()},
        }

    physchem = protein.get("physchem_features")
    n_protein = int(physchem.shape[0]) if isinstance(physchem, torch.Tensor) else 0
    _check_shape("physchem_features", physchem, (62,), n_protein, issues)
    _check_shape(
        "spatial_scalar_features",
        protein.get("spatial_scalar_features"),
        (89,),
        n_protein,
        issues,
    )
    _check_shape(
        "spatial_vector_features",
        protein.get("spatial_vector_features"),
        (8, 3),
        n_protein,
        issues,
    )
    _check_shape("ca_coords", protein.get("ca_coords"), (3,), n_protein, issues)

    prot_chains = _as_list(protein.get("chain_ids"))
    prot_indices = _as_list(protein.get("residue_indices"))
    prot_icodes = _as_list(protein.get("insertion_codes"))
    if not (len(prot_chains) == len(prot_indices) == len(prot_icodes) == n_protein):
        issues.append("protein_metadata_length")
        protein_keys: list[tuple[str, int, str]] = []
    else:
        protein_keys = _residue_keys(prot_chains, prot_indices, prot_icodes)
    chain_positions = [index for index, key in enumerate(protein_keys) if key[0] == chain]
    chain_keys = [protein_keys[index] for index in chain_positions]
    if not chain_keys:
        issues.append("target_chain_absent_from_protein")

    embeddings = esm2.get("embeddings")
    n_esm = int(embeddings.shape[0]) if isinstance(embeddings, torch.Tensor) else 0
    _check_shape("esm2_embeddings", embeddings, (1280,), n_esm, issues)
    esm_chains = _as_list(esm2.get("chain_id"))
    esm_indices = _as_list(esm2.get("residue_index"))
    esm_icodes = _as_list(esm2.get("insertion_code"))
    if not (len(esm_chains) == len(esm_indices) == len(esm_icodes) == n_esm):
        issues.append("esm2_metadata_length")
        esm_keys: list[tuple[str, int, str]] = []
    else:
        esm_keys = _residue_keys(esm_chains, esm_indices, esm_icodes)
    if chain_keys != esm_keys:
        issues.append("protein_esm2_residue_key_mismatch")

    protein_names = _as_list(protein.get("residue_names"))
    esm_names = _as_list(esm2.get("residue_name_3"))
    if len(protein_names) == n_protein and len(esm_names) == n_esm and chain_positions:
        chain_names = [str(protein_names[index]).upper() for index in chain_positions]
        esm_names = [str(name).upper() for name in esm_names]
        mismatches = sum(left != right for left, right in zip(chain_names, esm_names))
        if mismatches:
            warnings.append(f"residue_name_mismatches:{mismatches}")

    atom_features = ligand.get("atom_features")
    n_atoms = int(atom_features.shape[0]) if isinstance(atom_features, torch.Tensor) else 0
    _check_shape("atom_features", atom_features, (109,), n_atoms, issues)
    _check_shape("ligand_coords", ligand.get("coords"), (3,), n_atoms, issues)
    edge_index = ligand.get("edge_index")
    edge_attr = ligand.get("edge_attr")
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        issues.append("shape:edge_index")
        n_edges = 0
    else:
        n_edges = int(edge_index.shape[1])
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n_atoms):
            issues.append("edge_index_out_of_range")
    _check_shape("edge_attr", edge_attr, (6,), n_edges, issues)

    try:
        heavy_meta = int(row.get("heavy_atoms_meta") or 0)
        heavy_tensor = int(ligand.get("num_heavy_atoms") or 0)
    except (TypeError, ValueError):
        heavy_meta = heavy_tensor = 0
        issues.append("invalid_heavy_atom_count")
    if heavy_tensor != heavy_meta:
        issues.append(f"heavy_atom_mismatch:{heavy_tensor}!={heavy_meta}")
    if int(ligand.get("num_atoms") or n_atoms) != n_atoms:
        issues.append("ligand_num_atoms_mismatch")
    for key, expected in (
        ("pdb_id", pdb_id),
        ("protein_chain", chain),
        ("binding_site_id", row.get("binding_site_id", "")),
        ("ligand_id", row.get("ligand_id", "")),
    ):
        actual = str(ligand.get(key, ""))
        if actual.lower() != str(expected).lower():
            issues.append(f"ligand_identity:{key}:{actual}!={expected}")

    return {
        "pair_id": row.get("pair_id", stem),
        "pdb_id": pdb_id,
        "protein_chain": chain,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "n_protein_residues": n_protein,
        "n_chain_residues": len(chain_keys),
        "n_esm2_residues": n_esm,
        "n_ligand_atoms_with_h": n_atoms,
        "n_ligand_heavy_atoms": heavy_tensor,
        "n_ligand_edges": n_edges,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _percentile(values: Iterable[int], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 2)


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100, help="0 checks every selected row")
    parser.add_argument(
        "--strict-no-pdbbind",
        action="store_true",
        help="Only inspect rows with strict_no_pdbbind_ready=1.",
    )
    args = parser.parse_args()

    selected: list[dict[str, str]] = []
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if args.strict_no_pdbbind and row.get("strict_no_pdbbind_ready") != "1":
                continue
            selected.append(row)
            if args.max_samples > 0 and len(selected) >= args.max_samples:
                break

    results = []
    issue_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for index, row in enumerate(selected, 1):
        result = validate_row(row, args.biolip_root)
        results.append(result)
        issue_counts.update(issue.split(":", 1)[0] for issue in result["issues"])
        warning_counts.update(warning.split(":", 1)[0] for warning in result["warnings"])
        if index % 100 == 0:
            print(f"Checked {index:,}/{len(selected):,}", flush=True)

    passed = sum(bool(result["passed"]) for result in results)
    summary = {
        "biolip_root": str(args.biolip_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "strict_no_pdbbind": args.strict_no_pdbbind,
        "samples_checked": len(results),
        "samples_passed": passed,
        "samples_failed": len(results) - passed,
        "issue_counts": dict(issue_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "distributions": {
            key: _distribution([int(result[key]) for result in results if key in result])
            for key in (
                "n_protein_residues",
                "n_chain_residues",
                "n_ligand_atoms_with_h",
                "n_ligand_heavy_atoms",
                "n_ligand_edges",
            )
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    tmp.replace(args.output_json)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return int(not results or passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
