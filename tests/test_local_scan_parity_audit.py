import pytest
import torch

from scripts.diagnostics import audit_local_scan_parity as audit
from src.models.pocket_local_spatial_scan import PocketLocalSpatialScan


def test_state_difference_keeps_original_tolerance():
    left = {"weight": torch.tensor([1.0, 0.0])}
    assert audit.state_difference(left, left)["original_tolerance_pass"]
    changed = {"weight": torch.tensor([1.001, 0.0])}
    result = audit.state_difference(left, changed)
    assert not result["original_tolerance_pass"]
    assert result["mismatched_elements"] == 1
    assert result["worst_tensor"] == "weight"
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.state_difference(left, {"weight": torch.tensor([float("nan"), 0.0])})


def fixture():
    torch.manual_seed(31)
    model = PocketLocalSpatialScan(dict(scan_variant="S0_local_gvp"))
    for p in model.parameters():
        p.grad = torch.ones_like(p) * .01
    logits = torch.randn(2, 5, 3)
    labels = torch.arange(10).reshape(2, 5) % 3
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, -1] = False
    return model, logits, labels, mask


def test_shared_tensor_audit_passes_without_mutation():
    assert audit.check_shared_tensors(*fixture())["passed"]


def test_shared_tensor_audit_detects_gradient_mutation(monkeypatch):
    original = audit.local_gradient_report_batched

    def corrupt(model):
        result = original(model)
        next(model.parameters()).grad.add_(.01)
        return result

    monkeypatch.setattr(audit, "local_gradient_report_batched", corrupt)
    with pytest.raises(AssertionError):
        audit.check_shared_tensors(*fixture())
