import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pocket_gvp_fusion_v11m4 import PocketGVPFusionV11m4
from src.models.pocket_gvp_fusion_v11m4t import PocketGVPFusionV11m4t
from src.models.pocket_gvp_fusion_v11m6 import PocketGVPFusionV11m6


def _minimal_gate_wrapper(use_gate: bool) -> PocketGVPFusionV11m4t:
    model = PocketGVPFusionV11m4t.__new__(PocketGVPFusionV11m4t)
    nn.Module.__init__(model)
    model.use_global_delta_gate = use_gate
    model.pair_delta_logit_scale = nn.Parameter(torch.tensor(0.5))
    model.pair_delta_logit_scale_max = 1.0
    if use_gate:
        model._compute_delta_gate = lambda **kwargs: torch.ones(1, 1)
    return model


def _fake_parent_forward(self, **kwargs):
    base = torch.zeros(1, 2, 3)
    delta = torch.ones_like(base)
    scale = self._delta_scale()
    return {
        "base_logits": base,
        "delta_logits": delta,
        "pair_delta_scale": scale.detach(),
        "logits": base + scale * delta,
    }


@pytest.mark.parametrize("use_gate", [False, True])
def test_delta_scale_keeps_gradient_through_gate_wrapper(monkeypatch, use_gate):
    monkeypatch.setattr(PocketGVPFusionV11m4, "forward", _fake_parent_forward)
    model = _minimal_gate_wrapper(use_gate)
    dummy = torch.zeros(1, 1)

    outputs = model(
        esm_embeddings=dummy,
        protein_physchem=dummy,
        protein_spatial_scalar=dummy,
        protein_spatial_vector=dummy,
        protein_gvp_vectors=dummy,
        protein_coords=dummy,
        protein_mask=torch.ones(1, 1, dtype=torch.bool),
        ligand_x=dummy,
        ligand_pos=dummy,
        ligand_edge_index=torch.zeros(2, 0, dtype=torch.long),
        ligand_edge_attr=dummy,
        ligand_batch=torch.zeros(1, dtype=torch.long),
        ligand_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    outputs["logits"].sum().backward()

    assert model.pair_delta_logit_scale.grad is not None
    assert float(model.pair_delta_logit_scale.grad) > 0.0


@pytest.mark.parametrize("trainable", [False, True])
def test_local_delta_scale_trainability_is_configurable(trainable):
    model = PocketGVPFusionV11m6(
        {
            "d_model": 32,
            "d_esm": 16,
            "d_gvp_internal": 16,
            "n_heads": 4,
            "n_transformer_layers": 1,
            "n_cross_attn_layers": 1,
            "n_ligand_self_attn_layers": 1,
            "local_delta_dim": 8,
            "local_delta_hidden": 16,
            "local_delta_trainable_logit_scale": trainable,
            "use_global_delta_gate": False,
        }
    )

    assert model.local_delta_logit_scale.requires_grad is trainable
    assert model.config["local_delta_trainable_logit_scale"] is trainable
    assert float(model._delta_scale().detach()) == pytest.approx(
        float(torch.tanh(torch.tensor(0.5)))
    )
