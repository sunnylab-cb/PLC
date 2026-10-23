import numpy as np
import pytest
import torch

from scripts.benchmark.eval_holo_chain_context import (
    POLICY, chain_indices, split_sample, stitch, conditional_scores, check_sample,
)
from scripts.diagnostics.recompute_joint_topk_reference import score_row


def sample():
    return dict(pdb_id="1abc_L1", n_residues=4, n_atoms=4,
        residue_keys=[("A", "1", ""), ("a", "1", ""), ("A", "1", "A"), ("a", "2", "")],
        esm=torch.arange(12).reshape(4, 3), chain_ids=torch.tensor([0, 1, 0, 1]),
        labels=torch.tensor([0, 2, 0, 2]), raw_distances=torch.tensor([2., 10., 3., 11.]),
        gvp_coords=torch.zeros(4, 3), esm_valid_mask=torch.tensor([True, False, True, True]),
        ligand_features=torch.arange(16).reshape(4, 4), ligand_coords=torch.zeros(4, 3),
        ligand_edge_index=torch.tensor([[0, 1], [1, 0]]))


def test_all_chains_including_zero_contact_noncontiguous_case_sensitive():
    s = sample()
    parts = split_sample(s)
    assert [c for c, _, _ in parts] == ["A", "a"]
    assert [idx.tolist() for _, idx, _ in parts] == [[0, 2], [1, 3]]
    assert parts[1][2]["labels"].tolist() == [2, 2]
    for _, idx, part in parts:
        assert part["n_residues"] == 2 and part["n_atoms"] == 4
        assert part["chain_ids"].tolist() == [0, 0]
        assert torch.equal(part["esm"], s["esm"][idx])
        assert torch.equal(part["esm_valid_mask"], s["esm_valid_mask"][idx])
        for key in ("ligand_features", "ligand_coords", "ligand_edge_index"):
            assert part[key] is s[key]
    altered = dict(s, labels=torch.tensor([2, 0, 2, 0]))
    assert [idx.tolist() for _, idx, _ in split_sample(altered)] == [[0, 2], [1, 3]]


def test_stitch_restores_axis_and_probability_simplex():
    probs = np.array([[.1, .2, .7], [.3, .1, .6], [.4, .2, .4], [.2, .3, .5]], dtype=np.float32)
    pieces = [(np.array([1, 3]), probs[[1, 3]]), (np.array([0, 2]), probs[[0, 2]])]
    np.testing.assert_array_equal(stitch(4, pieces), probs)
    for bad in (pieces[:1], pieces + pieces[:1], [(np.array([0, 0]), probs[:2])],
                [(np.arange(4), np.ones((4, 3)))], [(np.arange(4), probs * np.nan)]):
        with pytest.raises(ValueError):
            stitch(4, bad)


def test_duplicate_keys_rejected_and_single_residue_chain_allowed():
    with pytest.raises(ValueError, match="Duplicate"):
        chain_indices([["A", "1", ""], ["A", "1", ""]])
    assert list(chain_indices([["A", "1", ""]])["A"]) == [0]


def test_global_budget_not_twenty_per_chain():
    scores = np.linspace(.99, .01, 50)
    y = np.arange(50) < 10
    assert score_row(scores, y, .55)["top20"]["predicted"] == 20
    assert sum(score_row(scores[idx], y[idx], .55)["top20"]["predicted"]
               for idx in (np.arange(25), np.arange(25, 50))) == 40
    assert "global Top20" in POLICY["primary"]
    assert not POLICY["test_driven_iteration_allowed"]


def test_conditional_comparison_crops_full_and_chain_to_identical_axes():
    scores = np.array([.9, .7, .9, .1], dtype=np.float32)
    r = dict(full=scores, chain=scores, libre=scores, target=np.array([True, False, True, False]),
             keys=np.asarray(sample()["residue_keys"]))
    out = conditional_scores({"1abc_L1": r}, .55)
    assert out["oracle_binding_chain"]
    assert out["n_all_chain_pairs"] == 2 and out["n_positive_chain_pairs"] == 1
    assert out["models"]["full"] == out["models"]["chain"]
    assert out["zero_contact_chains"]["chain"]["fpr"] == .5
    assert out["zero_contact_chains"]["chain"]["predicted"] == 1


def test_axis_checks_fail_closed():
    s = sample()
    baseline = dict(labels=s["labels"].numpy(), ca_coords=s["gvp_coords"].numpy(),
                    residue_keys=np.asarray(s["residue_keys"]))
    check_sample(s, baseline)
    with pytest.raises(ValueError, match="residue keys"):
        check_sample(s, dict(baseline, residue_keys=baseline["residue_keys"][::-1]))
