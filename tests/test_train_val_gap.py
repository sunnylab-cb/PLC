import pytest
import torch

from scripts.diagnostics.eval_train_val_gap import (
    common_losses, expected_calibration_error, stratified_ids, summarize, tensor_digest,
)


TRAINING = dict(class_weights=[9., 7., 1.], focal_gamma=2., focal_weight=.7,
                tversky_weight=.3, tversky_alpha=.3, tversky_beta=.7, tversky_smooth=1.)


def test_selection_is_stratified_stable_and_disjoint():
    ids = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(4)]
    sources = {s: s[0] for s in ids}
    selected = stratified_ids(ids, sources, 6)
    assert selected == stratified_ids(ids[::-1], sources, 6)
    assert sum(s[0] == "a" for s in selected) == 4
    assert len(set(selected)) == 6
    assert set(selected) <= set(ids)
    assert stratified_ids(ids, sources, len(ids)) == sorted(ids)


@pytest.mark.parametrize("n", [0, -1, 3])
def test_bad_subset_rejected(n):
    with pytest.raises(ValueError):
        stratified_ids(["a", "b"], {"a": "x", "b": "y"}, n)


def test_shared_loss_formula():
    z = torch.zeros(3, 3)
    y = torch.tensor([0, 1, 2])
    loss = common_losses(z, y, TRAINING)
    focal = torch.log(torch.tensor(3.)) * (2 / 3)**2 * (17 / 3)
    tv = 1 - (1 / 3 + 1) / (1 / 3 + .3 * 2 / 3 + .7 * 2 / 3 + 1)
    assert loss["focal"] == pytest.approx(float(focal), abs=1e-6)
    assert loss["tversky"] == pytest.approx(tv, abs=1e-6)
    assert loss["total"] == pytest.approx(.7 * focal + .3 * tv, abs=1e-6)


def test_perfect_teacher_and_wrong_teacher_differ():
    y = torch.tensor([0, 1, 2])
    good = dict(sample_id="good", labels=y, logits=torch.eye(3) * 20)
    bad = dict(sample_id="bad", labels=y, logits=torch.roll(torch.eye(3), 1, 1) * 20)
    g = summarize([good], .7, TRAINING)
    b = summarize([bad], .7, TRAINING)
    assert g["accuracy_global"] == 1 and b["accuracy_global"] == 0
    assert g["contact_argmax_global"]["f1"] == 1
    assert g["wrong_confidence_mean"] is None
    assert b["wrong_confidence_ge_0_9_fraction"] == 1
    assert g["nll_global"] < b["nll_global"]
    assert g["entropy_T1_global"] < g["entropy_T2_global"] < g["entropy_T4_global"]


def test_no_train_threshold_tuning_and_no_padding():
    row = dict(sample_id="one", labels=torch.tensor([0, 2]),
               logits=torch.tensor([[1., 0., 0.], [0., 0., 2.]]))
    low, high = summarize([row], .3, TRAINING), summarize([row], .99, TRAINING)
    assert low["contact_at_checkpoint_val_gate_global"]["f1"] == 1
    assert high["contact_at_checkpoint_val_gate_global"]["f1"] == 0
    assert high["contact_at_checkpoint_val_gate_global"]["threshold"] == .99
    assert low["n_residues"] == 2 and low["class_counts"] == [1, 0, 1]
    assert low["nll_by_true_class"]["1"] is None
    assert "best_gate" not in low


def test_nonfinite_valid_logits_fail_closed():
    with pytest.raises(ValueError):
        summarize([dict(labels=torch.tensor([0]), logits=torch.full((1, 3), float("nan")))], .5, TRAINING)


def test_calibration_bins_include_one():
    assert expected_calibration_error(torch.tensor([0., 1.]), torch.tensor([False, True])) == 0
    assert expected_calibration_error(torch.tensor([1.]), torch.tensor([False])) == 1


def test_axis_digest_detects_reorder():
    x = torch.arange(9).reshape(3, 3).float()
    assert tensor_digest(x) == tensor_digest(x.clone())
    assert tensor_digest(x) != tensor_digest(x.flip(0))
