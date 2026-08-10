import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import train_v11m3
from scripts.train.train_v11m3 import accumulate_validation_aux_metrics


def test_validation_aux_metrics_are_sample_weighted_and_detached():
    sums = {}
    accumulate_validation_aux_metrics(
        sums,
        {"rank_acc": torch.tensor(0.5, requires_grad=True), "alpha": 0.02},
        sample_count=2,
    )
    accumulate_validation_aux_metrics(
        sums,
        {"rank_acc": torch.tensor(1.0), "alpha": 0.04},
        sample_count=1,
    )

    assert sums["rank_acc"] / 3 == pytest.approx(2.0 / 3.0)
    assert sums["alpha"] / 3 == pytest.approx(0.08 / 3.0)


def test_validation_aux_metrics_skip_nonfinite_values():
    sums = {}
    accumulate_validation_aux_metrics(
        sums,
        {"finite": 1.0, "nan": float("nan"), "inf": torch.tensor(float("inf"))},
        sample_count=4,
    )

    assert sums == {"finite": 4.0}


def test_evaluate_reports_validation_assignment_metrics(monkeypatch):
    class DummyModel(nn.Module):
        def forward(self, **kwargs):
            protein_mask = kwargs["protein_mask"]
            return {
                "logits": torch.zeros(
                    *protein_mask.shape, 3, device=protein_mask.device
                )
            }

        def compute_assignment_auxiliary_loss(self, outputs, labels, mask, batch):
            zero = outputs["logits"].new_tensor(0.0)
            return zero, {"candidate_rank_acc": batch["candidate_rank_acc"]}

    def fake_inputs(batch, device, non_blocking=True):
        batch_size, length = batch["protein_mask"].shape
        return {
            "esm_embeddings": None,
            "protein_physchem": None,
            "protein_spatial_scalar": None,
            "protein_spatial_vector": None,
            "protein_gvp_vectors": None,
            "protein_coords": None,
            "protein_mask": batch["protein_mask"].to(device),
            "chain_ids": None,
            "ligand_x": None,
            "ligand_pos": None,
            "ligand_edge_index": None,
            "ligand_edge_attr": None,
            "ligand_batch": torch.arange(batch_size, device=device),
            "ligand_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=device),
        }

    monkeypatch.setattr(train_v11m3, "prepare_model_inputs_v11", fake_inputs)
    monkeypatch.setattr(train_v11m3, "get_batch_unimol", lambda *args, **kwargs: None)
    batches = [
        {
            "pdb_ids": ["a", "b"],
            "labels": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "protein_mask": torch.ones(2, 3, dtype=torch.bool),
            "candidate_rank_acc": torch.tensor(0.5),
        },
        {
            "pdb_ids": ["c"],
            "labels": torch.tensor([[0, 1, 2]]),
            "protein_mask": torch.ones(1, 3, dtype=torch.bool),
            "candidate_rank_acc": torch.tensor(1.0),
        },
    ]

    metrics = train_v11m3.evaluate(
        model=DummyModel(),
        dataloader=batches,
        class_weights=torch.ones(3),
        gamma=2.0,
        device=torch.device("cpu"),
    )

    assert metrics["assignment_candidate_rank_acc"] == pytest.approx(2.0 / 3.0)
