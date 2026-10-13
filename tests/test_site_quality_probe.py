import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.site_quality_features import candidate_inputs, candidate_targets, pack_candidates, STAT_NAMES
from scripts.diagnostics.strict_assignment_common import write_json
from scripts.infer.benchmark_integrity import sha256_file
from scripts.train.train_site_quality_probe import read_cache, fit_stats, make_batch, evaluation, main
from src.models.site_quality_head import SiteQualityHead, quality_loss, select_with_fallback


def test_candidate_features_are_target_free_and_geometry_frame_invariant():
    rng = np.random.default_rng(44)
    ca = rng.normal(size=(20, 3)) * 4
    p = np.linspace(0, 1, 20)
    active = p > .4
    candidates, stats = candidate_inputs(ca, p, active, np.ones(20, bool))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated, rotated_stats = candidate_inputs(ca @ rotation + [10, -20, 5], p, active, np.ones(20, bool))
    assert [tuple(c) for c in candidates] == [tuple(c) for c in rotated]
    np.testing.assert_allclose(stats, rotated_stats, atol=1e-5)
    assert np.array_equal(candidates[0], np.flatnonzero(active))
    before = stats.copy()
    a, _, _ = candidate_targets(candidates, active)
    b, _, _ = candidate_targets(candidates, ~active)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(before, stats)
    indices, offsets = pack_candidates(candidates)
    assert all(np.array_equal(indices[offsets[i]:offsets[i+1]], c) for i, c in enumerate(candidates))


def test_empty_original_and_unobserved_tail_are_explicit():
    ca = np.arange(30).reshape(10, 3)
    observed = np.arange(10) < 5
    candidates, stats = candidate_inputs(ca, np.zeros(10), np.zeros(10, bool), observed)
    assert candidates[0].size == 0 and stats[0, -1] == 1
    assert np.isfinite(stats).all()
    assert any(len(c) and stats[i, 2] < 1 for i, c in enumerate(candidates))


@pytest.mark.parametrize("mode", ["stats_only", "protein_site", "ligand_site"])
def test_controls_query_dependence_and_gradient_health(mode):
    torch.manual_seed(44)
    model = SiteQualityHead(mode, 16, np.zeros(len(STAT_NAMES)), np.ones(len(STAT_NAMES)))
    p, q, s = torch.randn(2, 3, 16), torch.randn(2, 16), torch.randn(2, 3, len(STAT_NAMES))
    model.eval()
    scores = model(p, q, s)
    other = model(p, q.flip(0), s)
    if mode == "ligand_site":
        assert not torch.equal(scores, other)
    else:
        assert torch.equal(scores, other)
    loss = quality_loss(scores, torch.rand(2, 3), torch.ones(2, 3, dtype=torch.bool), torch.ones(2))
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())


def test_padding_does_not_change_loss_and_candidate_loss_is_sample_balanced():
    scores = torch.tensor([[.2, .5], [.1, .9]], requires_grad=True)
    target = torch.tensor([[.3, .6], [.2, .8]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    weights = torch.tensor([.5, 1.5])
    loss = quality_loss(scores, target, valid, weights)
    padded = quality_loss(torch.cat([scores, torch.ones(2, 3)], 1),
                          torch.cat([target, torch.zeros(2, 3)], 1),
                          torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], 1), weights)
    assert torch.allclose(loss, padded)


def test_selection_margin_ties_padding_and_nonfinite():
    s = torch.tensor([[.50, .51, 1.], [.5, .54, 1.], [.5, .5, 1.]])
    valid = torch.tensor([[True, True, False]] * 3)
    assert select_with_fallback(s, valid).tolist() == [0, 1, 0]
    assert select_with_fallback(s, valid, margin=0).tolist() == [1, 1, 0]
    s[0, 1] = float("nan")
    with pytest.raises(ValueError):
        select_with_fallback(s, valid)


def create_cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    split = tmp_path / "splits.txt"
    split.write_text("train t1\ntrain t2\ntrain t3\ntrain t4\nval v1\nval v2\nval v3\ntest forbidden\n")
    manifest = []
    rng = np.random.default_rng(44)
    for subset, names in (("train", ["t1", "t2", "t3", "t4"]), ("val", ["v1", "v2", "v3"])):
        (root / subset).mkdir()
        for sid in names:
            stats = rng.normal(size=(3, len(STAT_NAMES))).astype(np.float32)
            stats[:, -1] = [1, 0, 0]
            path = root / subset / f"{sid}.npz"
            np.savez_compressed(path, pocket=rng.normal(size=(3, 16)).astype(np.float16),
                                ligand=rng.normal(size=16).astype(np.float16), stats=stats,
                                quality_f1=np.array([.5, 1., 0.], np.float32), sizes=np.array([4, 4, 3]),
                                true_positives=np.array([2, 4, 0]), n_true=np.array(4),
                                sample_id=np.array(sid), subset=np.array(subset),
                                axis_key=np.array("group" if sid in ["t1", "t2"] else sid), protein_key=np.array(sid))
            manifest.append({"sample_id": sid, "subset": subset, "sha256": sha256_file(path)})
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", {"complete": True, "smoke": True, "split_sha256": sha256_file(split),
                "manifest_sha256": sha256_file(root / "manifest.json"), "stats_names": list(STAT_NAMES),
                "counts": {"train": 4, "val": 3}})
    return root, split


def test_cache_split_hash_smoke_separation_and_train_only_normalization(tmp_path):
    root, split = create_cache(tmp_path)
    with pytest.raises(ValueError):
        read_cache(root, split, smoke=False)
    rows, _ = read_cache(root, split, smoke=True)
    mean, std, weights = fit_stats(rows["train"])
    assert weights[0] == weights[1] and np.isclose(weights[2], weights[0] * 2)
    for r in rows["val"]:
        r["stats"] += 1000
    mean2, std2, _ = fit_stats(rows["train"])
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
    with (root / "train/t1.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Changed cache"):
        read_cache(root, split, smoke=True)


def test_real_training_cli_synthetic_smoke_and_label_independent_selection(tmp_path, monkeypatch):
    root, split = create_cache(tmp_path)
    out = tmp_path / "heads"
    monkeypatch.setattr(sys, "argv", ["train_site_quality_probe.py", "--cache", str(root), "--splits", str(split),
                        "--output", str(out), "--smoke", "--epochs", "2", "--batch-size", "2", "--device", "cpu"])
    main()
    result = json.loads((out / "summary.json").read_text())
    assert len(result["results"]) == 5 and result["split"] == "val" and result["smoke"]
    rows, _ = read_cache(root, split, smoke=True)
    mean, std, _ = fit_stats(rows["train"])
    model = SiteQualityHead("ligand_site", 16, mean, std)
    before, detail = evaluation(model, rows["val"], "cpu", 2, .02)
    for r in rows["val"]:
        r["quality_f1"] = 1 - r["quality_f1"]
    after, changed = evaluation(model, rows["val"], "cpu", 2, .02)
    assert [r["selected"] for r in detail] == [r["selected"] for r in changed]
