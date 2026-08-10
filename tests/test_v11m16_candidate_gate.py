import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pocket_gvp_fusion_v11m16 import (
    PocketGVPFusionV11m16,
    gaussian_candidate_membership,
    select_spatial_candidate_centers,
)


def test_spatial_candidate_centers_apply_nms_and_padding_mask():
    scores = torch.tensor([[4.0, 3.0, 2.0, 10.0]])
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, True, False]])

    indices, centers, valid = select_spatial_candidate_centers(
        scores, coords, mask, num_candidates=3, min_distance=5.0
    )

    assert indices.tolist() == [[0, 2, 0]]
    assert valid.tolist() == [[True, True, False]]
    assert centers[0, :2, 0].tolist() == [0.0, 20.0]
    assert torch.equal(centers[0, 2], torch.zeros(3))


def test_gaussian_membership_respects_protein_and_candidate_masks():
    coords = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    centers = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    membership = gaussian_candidate_membership(
        coords,
        centers,
        protein_mask=torch.tensor([[True, True, False]]),
        candidate_valid=torch.tensor([[True, False]]),
        sigma=2.0,
    )

    assert membership.shape == (1, 2, 3)
    assert membership[0, 0, 0].item() == pytest.approx(1.0)
    assert 0.0 < membership[0, 0, 1].item() < 1.0
    assert membership[0, 0, 2].item() == 0.0
    assert torch.count_nonzero(membership[0, 1]).item() == 0


def test_candidate_rank_path_is_bounded_and_trainable():
    model = PocketGVPFusionV11m16(
        {
            "d_model": 32,
            "d_esm": 16,
            "d_gvp_internal": 16,
            "n_heads": 4,
            "n_transformer_layers": 1,
            "n_cross_attention_layers": 1,
            "n_ligand_self_attn_layers": 1,
            "local_delta_dim": 8,
            "local_delta_hidden": 16,
            "use_global_delta_gate": False,
            "use_candidate_rank_gate": True,
            "candidate_num_sites": 2,
            "candidate_min_center_distance": 5.0,
            "candidate_membership_sigma": 3.0,
            "candidate_embed_dim": 8,
            "candidate_gate_alpha_init": 0.02,
            "candidate_gate_alpha_max": 0.10,
        }
    )
    batch_size, length, atoms = 2, 5, 3
    logits = torch.zeros(batch_size, length, 3)
    outputs = {
        "logits": logits,
        "h_prot_global": torch.randn(batch_size, length, 32),
        "h_ligand_context": torch.randn(batch_size, atoms + 1, 32),
    }
    spatial = torch.zeros(batch_size, length, 89)
    spatial[:, :, 46] = torch.tensor([[5.0, 4.0, 1.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [12.0, 0.0, 0.0], [18.0, 0.0, 0.0], [24.0, 0.0, 0.0]],
        ]
    )
    protein_mask = torch.ones(batch_size, length, dtype=torch.bool)
    ligand_mask = torch.ones(batch_size, atoms, dtype=torch.bool)

    result = model._candidate_rank_forward(
        outputs, spatial, coords, protein_mask, ligand_mask
    )
    alpha = float(result["candidate_gate_alpha"])

    assert result["candidate_rank_weights"].shape == (batch_size, 2)
    assert torch.allclose(result["candidate_rank_weights"].sum(dim=-1), torch.ones(batch_size))
    assert abs(alpha) <= 0.10
    assert torch.allclose(result["logits"][..., 0].mean(dim=-1), torch.zeros(batch_size), atol=1e-6)

    loss = result["candidate_rank_logits"].square().mean() + result["logits"][0, 0, 0]
    loss.backward()
    assert model.assignment_candidate_gate_raw_alpha.grad is not None
    assert model.assignment_candidate_lig_proj[1].weight.grad is not None
    assert model.assignment_candidate_pocket_proj[1].weight.grad is not None
