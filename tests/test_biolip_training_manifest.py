import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostics.build_biolip_training_manifest import _index_assets, _pair_row
from scripts.diagnostics.inspect_biolip_feature_alignment import validate_row

import torch


def _meta(ligand_id="BEN", heavy_atoms=9):
    return {
        "pdb_id": "3gy7",
        "protein_chain": "A",
        "binding_site_id": "BS02",
        "ligand_id": ligand_id,
        "ligand_resseq": "1",
        "ligand_heavy_atom_count": heavy_atoms,
        "binding_residues": "D189 S190 Q192",
        "mapping_status": "exact_match",
        "heavy_atom_mapping_complete": True,
        "heavy_atom_mapping_injective": True,
        "heavy_atom_mapping_surjective": True,
    }


def test_biolip_manifest_marks_clean_current_assets_ready(tmp_path):
    (tmp_path / "protein_features").mkdir()
    (tmp_path / "esm2_features").mkdir()
    (tmp_path / "fixed_receptors" / "3gy7").mkdir(parents=True)
    (tmp_path / "ligand_features" / "3gy7").mkdir(parents=True)
    (tmp_path / "protein_features" / "3gy7_protein.pt").touch()
    (tmp_path / "esm2_features" / "3gy7_A_esm2.pt").touch()
    (tmp_path / "fixed_receptors" / "3gy7" / "receptor_fixed.cif").touch()
    (tmp_path / "ligand_features" / "3gy7" / "3gy7_A_BS02_BEN.pt").touch()

    assets = _index_assets(tmp_path)
    row = _pair_row(
        {
            "pdb_id": "3gy7",
            "chain": "A",
            "bs_id": "BS02",
            "lig_id": "BEN",
            "resseq": "1",
            "meta": _meta(),
        },
        assets,
        {},
        min_heavy=6,
        max_heavy=128,
        min_binding_residues=3,
    )

    assert row["route_meta"] == "regular_organic"
    assert row["metadata_eligible"] == 1
    assert row["current_feature_ready"] == 1
    assert row["strict_no_pdbbind_ready"] == 1


def test_biolip_manifest_excludes_cofactor_and_direct_pdbbind_overlap(tmp_path):
    for directory in ("protein_features", "esm2_features", "fixed_receptors", "ligand_features"):
        (tmp_path / directory).mkdir()
    assets = _index_assets(tmp_path)
    pair = {
        "pdb_id": "3gy7",
        "chain": "A",
        "bs_id": "BS02",
        "lig_id": "ATP",
        "resseq": "1",
        "meta": _meta(ligand_id="ATP", heavy_atoms=31),
    }
    row = _pair_row(
        pair,
        assets,
        {"3gy7": "test"},
        min_heavy=6,
        max_heavy=128,
        min_binding_residues=3,
    )

    assert row["route_meta"] == "no_conformer_biological"
    assert row["pdbbind_split"] == "test"
    assert row["metadata_eligible"] == 0
    assert "route:no_conformer_biological" in row["exclusion_reasons"]


def test_biolip_tensor_qc_aligns_full_receptor_to_selected_chain(tmp_path):
    (tmp_path / "protein_features").mkdir()
    (tmp_path / "esm2_features").mkdir()
    (tmp_path / "ligand_features" / "3gy7").mkdir(parents=True)
    protein = {
        "physchem_features": torch.zeros(4, 62),
        "spatial_scalar_features": torch.zeros(4, 89),
        "spatial_vector_features": torch.zeros(4, 8, 3),
        "ca_coords": torch.zeros(4, 3),
        "chain_ids": ["A", "A", "B", "B"],
        "residue_indices": [10, 11, 1, 2],
        "insertion_codes": ["", "A", "", ""],
        "residue_names": ["ALA", "GLY", "SER", "THR"],
    }
    esm2 = {
        "embeddings": torch.zeros(2, 1280),
        "chain_id": ["A", "A"],
        "residue_index": [10, 11],
        "insertion_code": ["", "A"],
        "residue_name_3": ["ALA", "GLY"],
    }
    ligand = {
        "atom_features": torch.zeros(5, 109),
        "coords": torch.zeros(5, 3),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        "edge_attr": torch.zeros(4, 6),
        "num_atoms": 5,
        "num_heavy_atoms": 3,
        "pdb_id": "3gy7",
        "protein_chain": "A",
        "binding_site_id": "BS02",
        "ligand_id": "BEN",
    }
    torch.save(protein, tmp_path / "protein_features" / "3gy7_protein.pt")
    torch.save(esm2, tmp_path / "esm2_features" / "3gy7_A_esm2.pt")
    torch.save(ligand, tmp_path / "ligand_features" / "3gy7" / "3gy7_A_BS02_BEN.pt")

    result = validate_row(
        {
            "pair_id": "3gy7_A_BS02_BEN_1",
            "feature_stem": "3gy7_A_BS02_BEN",
            "pdb_id": "3gy7",
            "protein_chain": "A",
            "binding_site_id": "BS02",
            "ligand_id": "BEN",
            "heavy_atoms_meta": "3",
        },
        tmp_path,
    )

    assert result["passed"] is True
    assert result["n_protein_residues"] == 4
    assert result["n_chain_residues"] == 2
    assert result["n_esm2_residues"] == 2
    assert result["n_ligand_atoms_with_h"] == 5
    assert result["n_ligand_heavy_atoms"] == 3
