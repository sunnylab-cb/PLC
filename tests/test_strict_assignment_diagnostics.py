import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.strict_assignment_common import read_strict_splits, quantiles
from scripts.diagnostics.strict_val_candidates import generate_candidates, evaluate_candidates, fps_seeds
from scripts.diagnostics.audit_strict_train_negatives import pair_status, rigid_errors, random_batch_hit_probability
from scripts.diagnostics.strict_val_hybrid_candidates import build_pools, rank_candidates, evaluate_pool


def protein():
    rng = np.random.default_rng(3)
    return np.vstack((rng.normal(size=(5, 3)), rng.normal(size=(5, 3)) + [30, 0, 0]))


def pair():
    ca = protein()
    return (dict(ca=ca, mask=np.arange(10) < 5, geometry_key="a", smiles="C"),
            dict(ca=ca.copy(), mask=np.arange(10) >= 5, geometry_key="a", smiles="N"))


def test_whole_protein_candidates_recover_regions_outside_contact_mask():
    ca = protein()
    p = np.array([.9] * 5 + [.1] * 5)
    active, target = np.arange(10) < 5, np.arange(10) >= 5
    old = generate_candidates(ca, p, active, "components_gate", 8., 16)
    new = generate_candidates(ca, p, active, "prob_nms", 8., 16)
    assert evaluate_candidates(ca, target, active, old)["oracle_f1"] == 0
    assert evaluate_candidates(ca, target, active, new)["oracle_f1"] == 1
    assert evaluate_candidates(ca, target, active, new)["selected_f1"] == 0
    assert evaluate_candidates(ca, ~target, active, new)["selected_f1"] == 1


def test_fps_generation_is_independent_of_ligand_probabilities():
    ca = protein()
    a = generate_candidates(ca, np.linspace(0, 1, 10), np.zeros(10, bool), "fps", 8., 8)
    b = generate_candidates(ca, np.linspace(1, 0, 10), np.ones(10, bool), "fps", 8., 8)
    assert {tuple(x) for x in a} == {tuple(x) for x in b}
    assert len(fps_seeds(ca, 100)) == len(ca)


def test_no_contact_targets_have_explicit_null_coverage():
    ca = protein()
    row = evaluate_candidates(ca, np.zeros(10, bool), np.zeros(10, bool), [])
    assert row["candidate_hit_dcc4"] is None
    assert row["candidate_hit_iou025"] is None
    assert row["oracle_f1"] == 0


def test_negative_audit_invariant_to_rigid_coordinate_frames():
    a, b = pair()
    before = pair_status(a, b)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    b["ca"] = b["ca"] @ rotation + [200, 300, -40]
    b["geometry_key"] = "rotated"
    after = pair_status(a, b)
    assert before["valid_conservative"] and after["valid_conservative"]
    assert np.isclose(before["center_min"], after["center_min"])
    assert after["max_ca_error"] < 1e-10


def test_identical_chemistry_is_not_a_hard_negative():
    a, b = pair()
    b["smiles"] = a["smiles"]
    result = pair_status(a, b)
    assert result["geometry_verified"] and not result["valid_conservative"]
    assert result["chemistry_relation"] == "same"


def test_unknown_chemistry_not_promoted_to_valid():
    a, b = pair()
    b["smiles"] = None
    assert not pair_status(a, b)["valid_conservative"]


def test_coordinate_distortion_not_safe_for_direct_mask_loss():
    a, b = pair()
    b["ca"][7] += [0, 20, 0]
    b["geometry_key"] = "distorted"
    result = pair_status(a, b)
    assert result["status"] == "distinct_mask_candidate"
    assert not result["valid_conservative"]


def test_high_overlap_and_zero_contact_are_not_negatives():
    a, b = pair()
    b["mask"] = a["mask"].copy()
    assert pair_status(a, b)["status"] == "high_overlap"
    b["mask"][:] = False
    assert pair_status(a, b)["status"] == "zero_contact"


def test_reflection_not_a_rigid_rotation():
    ca = protein()
    _, error = rigid_errors(ca, ca * [-1, 1, 1])
    assert error > .05


def test_split_overlap_duplicate_or_empty_fails_closed(tmp_path):
    path = tmp_path / "splits.txt"
    for content in ("train a\nval a\ntest c\n", "train a\ntrain a\nval b\ntest c\n", "train a\nval b\n"):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_strict_splits(path)
    path.write_text("train a\nval b\ntest c\n")
    assert read_strict_splits(path)["val"] == ["b"]


def test_negative_count_quantiles_include_zero_samples():
    q = quantiles([0] * 90 + [1] * 10)
    assert q["p50"] == 0 and q["mean"] == .1


def test_dataset_coverage_is_not_random_batch_coverage():
    assert np.isclose(random_batch_hit_probability(101, 1, 11), .1)
    assert random_batch_hit_probability(101, 0, 11) == 0
    assert random_batch_hit_probability(101, 100, 11) == 1


def test_internal_axis_shift_is_rejected():
    import torch
    from scripts.diagnostics.export_strict_val_predictions import validate_axis

    prot = {"ca_coords": torch.zeros(2, 3), "chain_ids": ["A", "A"], "residue_indices": [1, 2]}
    esm = {"embeddings": torch.zeros(2, 4), "chain_ids": ["A", "A"], "residue_indices": [2, 1]}
    with pytest.raises(ValueError, match="positional axis shift"):
        validate_axis(prot, esm, torch.ones(2))


def test_label_overlay_is_bound_to_both_original_values_and_protein_axis():
    import torch
    from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, reference_digest
    from scripts.diagnostics.strict_assignment_common import digest_object

    protein = {"ca_coords": torch.zeros(2, 3), "residue_names": ["ALA", "GLY"],
               "chain_ids": ["D", "D"], "residue_indices": [1, 2]}
    raw = torch.tensor([99., 88., 3., 7.])
    overlay = {"x": {"reference_sha256": reference_digest(protein), "raw_distances_sha256": digest_object(raw.tolist()),
                     "source_indices": [2, 3], "verification": {"distance_verified": True}}}
    assert aligned_values("x", protein, raw, overlay).tolist() == [3., 7.]
    with pytest.raises(ValueError, match="provenance mismatch"):
        aligned_values("x", protein, raw + 1., overlay)
    protein["residue_indices"] = [3, 4]
    with pytest.raises(ValueError, match="provenance mismatch"):
        aligned_values("x", protein, raw, overlay)


def test_label_overlay_rejects_duplicate_or_unverified_indices():
    import torch
    from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, reference_digest
    from scripts.diagnostics.strict_assignment_common import digest_object

    protein = {"ca_coords": torch.zeros(2, 3), "residue_names": ["ALA", "GLY"],
               "chain_ids": ["D", "D"], "residue_indices": [1, 2]}
    raw = torch.tensor([99., 3., 7.])
    overlay = {"x": {"reference_sha256": reference_digest(protein), "raw_distances_sha256": digest_object(raw.tolist()),
                     "source_indices": [1, 1], "verification": {"distance_verified": True}}}
    with pytest.raises(ValueError, match="Invalid overlay"):
        aligned_values("x", protein, raw, overlay)
    overlay["x"]["verification"]["distance_verified"] = False
    with pytest.raises(ValueError, match="Unverified"):
        aligned_values("x", protein, raw, overlay)


def test_hybrid_preserves_original_first_and_boundary_pool_is_subset():
    ca = protein()
    p = np.array([.9] * 5 + [.1] * 5)
    active = p > .5
    pools = build_pools(ca, p, active)
    for candidates in pools.values():
        assert np.array_equal(candidates[0], np.flatnonzero(active))
        assert len({tuple(c) for c in candidates}) == len(candidates)
    assert all(active[c].all() for c in pools["boundary_preserving"])
    assert any((~active[c]).any() for c in pools["raw_union"])


def test_soft_f1_proxy_does_not_just_prefer_tiny_confident_fragments():
    p = np.array([.99, .8, .8, .8, .1])
    candidates = [np.array([0]), np.arange(4)]
    assert rank_candidates(p, candidates, "top20_mean")[0] == 0
    selected, scores = rank_candidates(p, candidates, "soft_f1_proxy")
    assert selected == 1
    assert np.isclose(scores[1], 2 * p[:4].sum() / (4 + p.sum()))


def test_hybrid_empty_and_ties_retain_original():
    ca = protein()
    pools = build_pools(ca, np.zeros(10), np.zeros(10, bool))
    for candidates in pools.values():
        assert len(candidates[0]) == 0
        for method in ("top20_mean", "soft_f1_proxy"):
            selected, scores = rank_candidates(np.zeros(10), candidates, method)
            assert selected == 0 and np.isfinite(scores).all()


def test_hybrid_selection_is_label_independent_and_oracle_retains_base():
    ca = protein()
    p = np.array([.9] * 5 + [.1] * 5)
    active = p > .5
    candidates = build_pools(ca, p, active)["hybrid_union"]
    selected, _ = rank_candidates(p, candidates, "soft_f1_proxy")
    for target in (active, ~active, np.zeros(10, bool)):
        result = evaluate_pool(ca, target, active, candidates, selected)
        assert result["oracle_f1"] >= result["baseline_f1"]
        assert rank_candidates(p, candidates, "soft_f1_proxy")[0] == selected


@pytest.mark.parametrize("p", [[np.nan], [np.inf], [-.1], [1.1]])
def test_hybrid_rank_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError, match="probabilities"):
        rank_candidates(p, [np.array([0])], "soft_f1_proxy")
