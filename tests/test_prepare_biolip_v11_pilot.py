import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostics.prepare_biolip_v11_pilot import (
    filter_rows_by_ligand_qc,
    ligand_heavy_coords,
    min_heavy_atom_distances,
    slice_protein_chain,
    stable_group_split,
)


def test_slice_protein_chain_uses_explicit_residue_keys():
    protein = {
        "physchem_features": torch.arange(4 * 62).reshape(4, 62).float(),
        "spatial_scalar_features": torch.zeros(4, 89),
        "spatial_scalar_raw_features": torch.zeros(4, 89),
        "spatial_vector_features": torch.zeros(4, 8, 3),
        "ca_coords": torch.zeros(4, 3),
        "residue_names": ["ALA", "GLY", "SER", "THR"],
        "residue_indices": [10, 11, 1, 2],
        "insertion_codes": ["", "A", "", ""],
        "chain_ids": ["A", "A", "B", "B"],
        "n_residues": 4,
    }
    parsed = {
        "chain_ids": ["B", "A", "A", "B"],
        "residue_indices": [1, 11, 10, 2],
        "insertion_codes": ["", "A", "", ""],
        "all_atom_coords": [
            np.ones((2, 3)) * 1,
            np.ones((2, 3)) * 11,
            np.ones((2, 3)) * 10,
            np.ones((2, 3)) * 2,
        ],
        "all_atom_names": [["N", "CA"]] * 4,
    }

    result = slice_protein_chain(protein, parsed, "A")

    assert result["n_residues"] == 2
    assert result["residue_indices"] == [10, 11]
    assert torch.all(result["all_atom_coords"][0] == 10)
    assert torch.all(result["all_atom_coords"][1] == 11)
    assert result["physchem_features"].shape == (2, 62)


def test_ligand_heavy_coords_prefers_original_and_distance_is_dmin():
    ligand = {
        "coords_original": torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        "coords": torch.randn(4, 3),
        "atom_features": torch.randn(4, 109),
    }
    residues = [
        torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        torch.tensor([[6.0, 0.0, 0.0]]),
    ]

    coords = ligand_heavy_coords(ligand)
    distances = min_heavy_atom_distances(residues, coords)

    assert coords.shape == (2, 3)
    assert torch.allclose(distances, torch.tensor([1.0, 3.0]))


def test_stable_group_split_never_splits_same_pdb():
    rows = [
        {"pdb_id": "1abc", "protein_chain": "A", "feature_stem": "a"},
        {"pdb_id": "1abc", "protein_chain": "B", "feature_stem": "b"},
        {"pdb_id": "2def", "protein_chain": "A", "feature_stem": "c"},
        {"pdb_id": "3ghi", "protein_chain": "A", "feature_stem": "d"},
    ]

    first = stable_group_split(rows, val_fraction=0.5, seed=7)
    second = stable_group_split(rows, val_fraction=0.5, seed=7)

    assert first == second
    assert set(first) == {"1abc", "2def", "3ghi"}
    assert set(first.values()) == {"train", "val"}


def test_ligand_tensor_filter_uses_v6_route_and_completeness(tmp_path):
    feature_root = tmp_path / "ligand_features" / "1abc"
    feature_root.mkdir(parents=True)
    torch.save(
        {"ligand_route": 5, "feature_completeness_ratio": 1.0},
        feature_root / "keep.pt",
    )
    torch.save(
        {"ligand_route": 3, "feature_completeness_ratio": 1.0},
        feature_root / "metal.pt",
    )
    torch.save(
        {"ligand_route": 5, "feature_completeness_ratio": 0.5},
        feature_root / "partial.pt",
    )
    rows = [
        {"pdb_id": "1abc", "feature_stem": "keep"},
        {"pdb_id": "1abc", "feature_stem": "metal"},
        {"pdb_id": "1abc", "feature_stem": "partial"},
        {"pdb_id": "1abc", "feature_stem": "missing"},
    ]

    retained, dropped = filter_rows_by_ligand_qc(
        rows,
        source_root=tmp_path,
        required_route=5,
        min_completeness=0.8,
    )

    assert [row["feature_stem"] for row in retained] == ["keep"]
    assert dropped == {
        "ligand_route_3": 1,
        "feature_completeness": 1,
        "missing_ligand_tensor": 1,
    }
