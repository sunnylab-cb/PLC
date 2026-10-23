import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer.benchmark_integrity import (
    align_distances, align_embedding, checked_ligand_features,
    residue_keys, safe_tensor_load, legacy_tag_axis, verify_tag_distances,
)
from scripts.infer.infer_v11_benchmark import BenchmarkV11Dataset, benchmark_collate, preflight_dataset
from src.models.pocket_gvp_fusion_v11 import PocketGVPFusionV11
from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13


def metadata(numbers):
    return {"chain_ids": ["A"] * len(numbers), "residue_indices": numbers,
            "insertion_codes": [""] * len(numbers), "residue_names": ["ALA"] * len(numbers)}


def test_missing_middle_residue_is_not_a_prefix_alignment():
    src = {**metadata([1, 3]), "embeddings": torch.tensor([[10.], [30.]])}
    aligned, valid, qc = align_embedding(src, metadata([1, 2, 3]), "ESM2")
    assert aligned[:, 0].tolist() == [10., 0., 30.]
    assert valid.tolist() == [True, False, True]
    assert qc["positional_mismatches"] == 1


def test_equal_length_reordering_and_insertion_codes():
    src = {**metadata([1, 1]), "insertion_codes": ["B", "A"],
           "embeddings": torch.tensor([[20.], [10.]])}
    ref = {**metadata([1, 1]), "insertion_codes": ["A", "B"]}
    aligned, valid, _ = align_embedding(src, ref, "ESM3")
    assert aligned[:, 0].tolist() == [10., 20.]
    assert valid.all()


def test_duplicate_keys_and_identity_mismatches_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        residue_keys(metadata([1, 1]))
    src = {**metadata([1]), "residue_names": ["GLY"], "embeddings": torch.zeros(1, 4)}
    with pytest.raises(ValueError, match="identity mismatch"):
        align_embedding(src, metadata([1]), "ESM2")


def test_duplicate_aux_keys_need_unique_geometry_and_sequence_evidence():
    src = {**metadata([1, 1]), "embeddings": torch.tensor([[10.], [20.]]),
           "residue_name_1": ["S", "D"], "residue_name_3": ["ASP", "ASP"],
           "atom37_positions": torch.zeros(2, 37, 3), "atom37_mask": torch.zeros(2, 37, dtype=torch.bool)}
    src["atom37_mask"][:, 1] = True
    src["atom37_positions"][1, 1, 0] = 5.
    ref = {**metadata([1]), "residue_names": ["SER"], "ca_coords": torch.zeros(1, 3)}
    aligned, valid, qc = align_embedding(src, ref, "ESM3")
    assert aligned.item() == 10. and valid.all()
    assert qc["duplicate_key_matches_resolved"] == 1 and qc["source_name3_conflicts"] == 1
    src["atom37_positions"][1, 1, 0] = 0.
    with pytest.raises(ValueError, match="ambiguous CA"):
        align_embedding(src, ref, "ESM3")
    src["atom37_positions"][1, 1, 0] = 5.
    src["residue_name_1"][0] = "D"
    with pytest.raises(ValueError, match="sequence identity disagrees"):
        align_embedding(src, ref, "ESM3")


def test_aux_geometry_can_rekey_but_does_not_match_different_chains():
    src = {**metadata([99]), "embeddings": torch.tensor([[10.]]), "residue_name_1": ["A"],
           "atom37_positions": torch.zeros(1, 37, 3), "atom37_mask": torch.ones(1, 37, dtype=torch.bool)}
    ref = {**metadata([1]), "ca_coords": torch.zeros(1, 3)}
    aligned, valid, qc = align_embedding(src, ref, "ESM3")
    assert valid.all() and aligned.item() == 10. and qc["coordinate_rekeyed"] == 1
    ref["chain_ids"] = ["B"]
    with pytest.raises(ValueError, match="no shared residue CA"):
        align_embedding(src, ref, "ESM3")


def test_labels_reorder_by_verified_tag_axis():
    result = align_distances(torch.tensor([8., 3.]), metadata([2, 1]), metadata([1, 2]))
    assert result.tolist() == [3., 8.]
    with pytest.raises(ValueError, match="missing from tag axis"):
        align_distances(torch.ones(1), metadata([1]), metadata([1, 2]))


def test_sasa_repair_is_explicit_bounded_and_does_not_mutate_source():
    x = torch.ones(2, 109)
    x[0, 38] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        checked_ligand_features(x, "error", "ligand")
    y, positions = checked_ligand_features(x, "sasa_zero", "ligand")
    assert positions == [[0, 38]] and y[0, 38] == 0 and torch.isnan(x[0, 38])
    x[1, 33] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        checked_ligand_features(x, "sasa_zero", "ligand")


def test_restricted_loader_accepts_legacy_numpy_metadata(tmp_path):
    path = tmp_path / "cache.pt"
    torch.save({"x": torch.ones(2), "ids": np.array(["A", "B"]), "n": np.int64(2)}, path)
    assert safe_tensor_load(path)["n"] == 2


def test_tag_axis_reconstruction_uses_legacy_sort_and_includes_no_ca_residues(tmp_path):
    path = tmp_path / "demo" / "1abc.pdb"
    path.parent.mkdir()
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  CA  GLY A   3A      3.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      3  N   ALA A   2       2.000   0.000   0.000  1.00 20.00           N  \n"
        "END\n"
    )
    ds = BenchmarkV11Dataset.__new__(BenchmarkV11Dataset)
    ds.dataset, ds.tag_protein_root = "demo", tmp_path
    meta = ds._tag_metadata("1abc")
    assert residue_keys(meta) == [("A", 1, ""), ("A", 2, ""), ("A", 3, "A")]
    assert len(meta["source_sha256"]) == 64


def test_tag_distance_verification_rejects_equal_length_wrong_order(tmp_path):
    from rdkit import Chem

    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00 20.00           C  \n"
        "END\n"
    )
    sdf = tmp_path / "ligand.sdf"
    mol = Chem.MolFromSmiles("C")
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (1., 0., 0.))
    mol.AddConformer(conf)
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(mol)
    axis = legacy_tag_axis(pdb)
    qc = verify_tag_distances(torch.tensor([1., 2.]), axis, [sdf])
    assert qc["distance_verified"]
    with pytest.raises(ValueError, match="geometry mismatch"):
        verify_tag_distances(torch.tensor([2., 1.]), axis, [sdf])


def test_esm3_global_pool_excludes_missing_rows_even_with_projection_bias():
    model = PocketGVPFusionV11.__new__(PocketGVPFusionV11)
    nn.Module.__init__(model)
    model.use_esm3 = True
    model.config = {}
    model.esm3_fusion = "global_prior"
    model.esm3_input_norm = nn.Identity()
    model.esm3_proj = nn.Linear(1, 1)
    with torch.no_grad():
        model.esm3_proj.weight.fill_(1)
        model.esm3_proj.bias.fill_(10)
    model.esm3_alpha_max = 1.
    model.esm3_raw_alpha = nn.Parameter(torch.tensor(0.))
    model.eval()
    base = torch.zeros(1, 2, 1)
    emb = torch.tensor([[[2.], [0.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert torch.all(model._apply_esm3_fusion(base, emb, mask) == 5.5)
    assert torch.all(model._apply_esm3_fusion(base, emb, mask, torch.tensor([[True, False]])) == 6.)
    assert torch.equal(model._apply_esm3_fusion(base, emb, mask, ~mask), base)


def test_preflight_reports_all_failures_without_treating_them_as_skipped(tmp_path):
    from types import SimpleNamespace

    class BrokenDataset:
        dataset = "demo"
        keys = ["bad1", "good", "bad2"]

        def __len__(self):
            return len(self.keys)

        def _protein_key(self, key):
            return key

        def __getitem__(self, index):
            if index != 1:
                raise ValueError("injected failure")
            return {"n_residues": 2, "input_qc": {"ligand_repairs": []}}

    result = preflight_dataset(BrokenDataset(), SimpleNamespace(batch_size=2, num_workers=0), tmp_path)
    assert result["status"] == "failed"
    assert result["n_checked"] == 3 and result["n_failed"] == 2 and result["n_passed"] == 1
    assert [row["sample_key"] for row in result["errors"]] == ["bad1", "bad2"]


def test_v11m13_mask_forwarding_preserves_all_valid_predictions():
    model = PocketGVPFusionV11m13({
        "d_model": 32, "d_esm": 16, "d_esm3": 8, "d_gvp_internal": 16,
        "n_heads": 4, "n_transformer_layers": 1, "n_cross_attention_layers": 1,
        "n_ligand_self_attn_layers": 1, "gvp_k_neighbors": 2,
        "local_delta_dim": 8, "local_delta_hidden": 16,
        "use_unimol_ligand_input": False, "use_global_delta_gate": False,
        "use_esm3": True, "esm3_fusion": "global_prior",
        "use_v12_coattention_block": True,
    }).eval()
    inputs = {
        "esm_embeddings": torch.randn(1, 4, 16), "esm3_embeddings": torch.randn(1, 4, 8),
        "protein_physchem": torch.zeros(1, 4, 62),
        "protein_spatial_scalar": torch.zeros(1, 4, 89),
        "protein_spatial_vector": torch.zeros(1, 4, 8, 3),
        "protein_gvp_vectors": torch.randn(1, 4, 4, 3),
        "protein_coords": torch.randn(1, 4, 3),
        "protein_mask": torch.ones(1, 4, dtype=torch.bool),
        "chain_ids": torch.zeros(1, 4, dtype=torch.long),
        "ligand_x": torch.randn(3, 109), "ligand_pos": torch.randn(3, 3),
        "ligand_edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "ligand_edge_attr": torch.zeros(4, 6), "ligand_batch": torch.zeros(3, dtype=torch.long),
        "ligand_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        old = model(**inputs)["logits"]
        masked = model(**inputs, esm_valid_mask=inputs["protein_mask"],
                       esm3_valid_mask=inputs["protein_mask"])["logits"]
    assert torch.equal(old, masked)


@pytest.mark.parametrize("coverage,expected", [("protein", 3), ("legacy_budget", 2)])
@pytest.mark.parametrize("cohort_normalization", [False, True])
def test_dataset_keeps_canonical_axis_and_collates_masks(tmp_path, coverage, expected, cohort_normalization):
    root = tmp_path
    for name in ("protein/demo", "esm/demo", "ligand/demo"):
        (root / name).mkdir(parents=True)
    prot = {**metadata([1, 2, 3]), "physchem_features": torch.zeros(3, 62),
            "spatial_scalar_features": torch.zeros(3, 89),
            "spatial_vector_features": torch.zeros(3, 8, 3), "ca_coords": torch.zeros(3, 3),
            "all_atom_coords": [torch.zeros(3, 3) for _ in range(3)]}
    prot["spatial_scalar_raw_features"] = torch.full((3, 89), 7.)
    prot["spatial_scalar_raw_features"][:, 62] = 100.
    torch.save(prot, root / "protein/demo/1abc.pt")
    torch.save({**metadata([1, 3]), "embeddings": torch.tensor([[1.], [3.]])}, root / "esm/demo/1abc.pt")
    torch.save({"atom_features": torch.ones(2, 109), "coords": torch.zeros(2, 3),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, 6)}, root / "ligand/demo/1abc_LIG_A1.pt")
    torch.save({"1abc_LIG_A1": {**metadata([1, 2, 3]), "distances": torch.tensor([3., 9., 4.])}}, root / "tags.pt")
    ds = BenchmarkV11Dataset(root, "demo", root / "tags.pt", root / "protein", root / "esm",
                            None, root / "ligand", tag_mode="per_ligand",
                            residue_alignment="keys", residue_coverage=coverage,
                            spatial_norm=dict(mean=torch.ones(89), std=torch.full((89,), 2.)) if cohort_normalization else None,
                            spatial_clamp_dims=[62, 76, 78, 79])
    sample = ds[0]
    assert sample["n_residues"] == expected
    assert sample["esm"][:, 0].tolist() == [1., 0., 3.][:expected]
    assert sample["labels"].tolist() == [0, 2, 0][:expected]
    assert bool((sample["protein_spatial_scalar"][:, 0] == (3. if cohort_normalization else 0.)).all())
    assert bool((sample["protein_spatial_scalar"][:, 62] == (5. if cohort_normalization else 0.)).all())
    batch = benchmark_collate([sample])
    assert batch["esm_valid_mask"][0].tolist() == [True, False, True][:expected]
    assert batch["input_qc"][0]["tag_axis"]["verified"]
