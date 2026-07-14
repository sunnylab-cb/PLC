import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.metrics import (
    contact_gate_predictions,
    compute_binary_ranking_metrics,
    compute_contact_probability_metrics,
    find_best_f1_threshold,
)


def test_contact_gate_preserves_best_non_contact_class():
    probabilities = torch.tensor(
        [
            [0.60, 0.10, 0.30],
            [0.49, 0.40, 0.11],
            [0.49, 0.12, 0.39],
        ]
    )
    predictions = contact_gate_predictions(probabilities, threshold=0.5)
    assert predictions.tolist() == [0, 1, 2]


def test_binary_ranking_and_exact_best_f1_gate():
    scores = torch.tensor([0.9, 0.8, 0.4, 0.1])
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    ranking = compute_binary_ranking_metrics(scores, targets)
    assert ranking["auprc"] == pytest.approx(5.0 / 6.0)
    assert ranking["auroc"] == pytest.approx(0.75)

    best = find_best_f1_threshold(scores, targets)
    assert best["threshold"] == pytest.approx(0.4)
    assert best["f1"] == pytest.approx(0.8)


def test_tied_scores_are_evaluated_as_one_threshold_group():
    scores = torch.tensor([0.5, 0.5])
    targets = torch.tensor([1, 0], dtype=torch.bool)

    ranking = compute_binary_ranking_metrics(scores, targets)
    best = find_best_f1_threshold(scores, targets)
    assert ranking["auprc"] == pytest.approx(0.5)
    assert ranking["auroc"] == pytest.approx(0.5)
    assert best["threshold"] == pytest.approx(0.5)
    assert best["f1"] == pytest.approx(2.0 / 3.0)


def test_contact_metrics_use_global_ranking_and_macro_top_l():
    sample_scores = [
        torch.tensor([0.9, 0.8, 0.4, 0.1]),
        torch.tensor([0.7, 0.6]),
    ]
    sample_targets = [
        torch.tensor([1, 0, 1, 0], dtype=torch.bool),
        torch.tensor([0, 1], dtype=torch.bool),
    ]

    metrics = compute_contact_probability_metrics(
        sample_scores,
        sample_targets,
        top_l_divisors=(2,),
    )
    assert metrics["n_residues_c0_metrics"] == 6
    assert metrics["n_positive_c0_metrics"] == 3
    assert metrics["top_l2_precision_c0_macro"] == pytest.approx(0.25)
    assert metrics["top_l2_recall_c0_macro"] == pytest.approx(0.25)
    assert metrics["top_l2_hit_rate_c0_macro"] == pytest.approx(0.5)
    assert 0.0 <= metrics["best_gate_c0"] <= 1.0


def test_ranking_is_undefined_without_both_classes():
    ranking = compute_binary_ranking_metrics(
        torch.tensor([0.8, 0.2]),
        torch.tensor([0, 0], dtype=torch.bool),
    )
    assert math.isnan(ranking["auprc"])
    assert math.isnan(ranking["auroc"])
