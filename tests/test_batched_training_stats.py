import ast
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.pocket_local_spatial_scan import PocketLocalSpatialScan
from src.utils.batched_training_stats import compute_metrics_batched, local_gradient_report_batched


ROOT = Path(__file__).resolve().parents[1]


def legacy_functions():
    tree = ast.parse((ROOT / "scripts/train/train_v11m3.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("compute_metrics", "compute_gradient_stats")]
    ns = dict(torch=torch, nn=nn, DDP=DDP, Dict=dict, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_stats", "exec"), ns)
    tree = ast.parse((ROOT / "scripts/train/train_local_spatial_scan.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_gradient_stats")
    ns["_original_grad_stats"] = ns["compute_gradient_stats"]
    exec(compile(ast.Module(body=[node], type_ignores=[]), "legacy_local_stats", "exec"), ns)
    return ns["compute_metrics"], ns["compute_gradient_stats"]


@pytest.mark.parametrize("mask_mode", ["random", "empty", "full"])
@pytest.mark.parametrize("n_classes", [3, 5])
def test_classification_counts_match_exactly(mask_mode, n_classes):
    torch.manual_seed(73)
    logits = torch.randn(3, 17, n_classes, requires_grad=True)
    labels = torch.randint(n_classes, (3, 17))
    mask = torch.rand(3, 17) > .4
    if mask_mode != "random":
        mask.fill_(mask_mode == "full")
    labels[~mask] = -100
    before = torch.get_rng_state().clone()
    legacy, _ = legacy_functions()
    assert compute_metrics_batched(logits, labels, mask, n_classes) == legacy(logits, labels, mask, n_classes)
    assert logits.grad is None and torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("variant", ["S0_local_gvp", "S1_cond_gvp", "T1_cond_triangle"])
@pytest.mark.parametrize("grad_mode", ["finite", "missing", "zero", "nan", "inf"])
def test_gradient_report_matches_legacy_and_is_read_only(variant, grad_mode):
    torch.manual_seed(14)
    model = PocketLocalSpatialScan(dict(scan_variant=variant))
    for i, p in enumerate(model.parameters()):
        if grad_mode != "missing" and i % 5 != 3:
            p.grad = torch.zeros_like(p) if grad_mode == "zero" else torch.randn_like(p)
    if grad_mode in ("nan", "inf"):
        next(model.parameters()).grad.flatten()[0] = float(grad_mode)
    saved = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    state = torch.get_rng_state().clone()
    _, legacy = legacy_functions()
    expected = legacy(model)
    norm, actual = local_gradient_report_batched(model)
    assert actual.keys() == expected.keys()
    for key in expected:
        if math.isnan(expected[key]):
            assert math.isnan(actual[key])
        else:
            assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=1e-8)
    if grad_mode in ("nan", "inf"):
        assert math.isnan(norm)
    else:
        reference = sum(g.float().norm(2).item() ** 2 for g in saved if g is not None) ** .5
        assert norm == pytest.approx(reference, rel=1e-7)
    for p, g in zip(model.parameters(), saved):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, g, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(state, torch.get_rng_state())


def test_nonlocal_model_is_rejected():
    with pytest.raises(ValueError, match="LocalScan-only"):
        local_gradient_report_batched(nn.Linear(2, 3))
