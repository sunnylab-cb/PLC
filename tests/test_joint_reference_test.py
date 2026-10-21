import ast
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.eval_joint_reference_test import (
    POLICY, checkpoint_gate, donor_indices, summarize, validate_logits,
)


def test_shuffle_is_source_preserving_bijective_and_has_no_fixed_points():
    ids = [f"s{i}" for i in range(15)]
    sources = {sid: "pdbbind" if i < 5 else "biolip" for i, sid in enumerate(ids)}
    donors = donor_indices(ids, sources)
    assert donors == donor_indices(ids, sources)
    assert sorted(donors) == list(range(len(ids)))
    assert all(i != j and sources[ids[i]] == sources[ids[j]] for i, j in enumerate(donors))


def test_singleton_source_cannot_be_shuffled():
    with pytest.raises(ValueError, match="At least two"):
        donor_indices(["a", "b"], {"a": "pdbbind", "b": "biolip"})


def test_locked_checkpoint_and_gate():
    ck = dict(epoch=24, val_contact_gate=.553555, val_metrics=dict(best_gate_c0=.553555))
    assert checkpoint_gate(ck) == .553555
    with pytest.raises(ValueError, match="epoch25"):
        checkpoint_gate(dict(ck, epoch=33))
    with pytest.raises(ValueError, match="differs"):
        checkpoint_gate(dict(ck, val_contact_gate=.5))


def test_nonfinite_valid_residues_fail_instead_of_changing_metric_population():
    z = torch.zeros(3, 3)
    labels = torch.tensor([0, 1, 2])
    validate_logits(z, labels, "ok")
    z[1, 2] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_logits(z, labels, "bad")


def test_metrics_use_only_prespecified_thresholds():
    rows = [dict(scores=torch.tensor([.8, .6, .4, .1]),
                 labels=torch.tensor([0, 1, 0, 2]), argmax=torch.tensor([0, 0, 2, 2]))]
    result = summarize(rows, .7)
    assert result["n_residues"] == 4
    assert result["at_0_5_global"]["threshold"] == .5
    assert result["at_val_gate_global"]["threshold"] == .7
    assert result["at_0_5_global"]["f1"] == .5
    assert result["at_val_gate_global"]["f1"] == pytest.approx(2/3)
    assert result["at_val_gate_global"]["mcc"] == pytest.approx(1 / 3**.5)
    assert "best_gate" not in result
    rows[0]["scores"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize(rows, .7)


def test_reference_runner_has_no_threshold_search_or_optimization_calls():
    source = (ROOT / "scripts/diagnostics/eval_joint_reference_test.py").read_text()
    tree = ast.parse(source)
    call_names = {getattr(n.func, "id", getattr(n.func, "attr", ""))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not call_names & {"find_best_f1_threshold", "compute_contact_probability_metrics",
                             "backward", "optimizer", "AdamW", "train"}
    assert POLICY["reference_only"]
    assert POLICY["test_driven_iteration_allowed"] is False
    assert POLICY["test_used_for_selection"] is False
