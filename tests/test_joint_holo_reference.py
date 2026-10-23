import json
import pytest
import torch

from scripts.benchmark.eval_joint_holo_reference import POLICY, overlap_by_pdb
from scripts.infer.infer_v11_benchmark import _load_cohort_norm, _normalize_spatial_raw, _normalize_ligand_features


def stats_file(tmp_path):
    obj = dict(fit_scope="this cohort TRAIN only; unique source inputs", train_keys=["pdbbind:1abc"])
    for name, dim in (("ligand", 109), ("spatial", 89)):
        obj[name] = dict(mean=[2.] * dim, std=[.5] * dim)
    obj["spatial"]["std"][0] = 0.
    obj["spatial"]["std"][1] = 1e-9
    obj["ligand"]["norm_dims"] = [33, 34, 35]
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(obj))
    return path, obj


def test_joint_stats_match_training_recipe_and_preserve_sources(tmp_path):
    path, stats = stats_file(tmp_path)
    ligand, spatial = _load_cohort_norm(path)
    raw = torch.arange(178).reshape(2, 89).float()
    original = raw.clone()
    result = _normalize_spatial_raw(dict(spatial_scalar_raw_features=raw,
                                        spatial_scalar_features=torch.full_like(raw, -999)), spatial)
    mean, std = torch.tensor(stats["spatial"]["mean"]), torch.tensor(stats["spatial"]["std"])
    std[std < 1e-8] = 1.
    torch.testing.assert_close(result, (raw - mean) / std)
    torch.testing.assert_close(raw, original)
    x = torch.ones(3, 109)
    normalized = _normalize_ligand_features(x, ligand)
    assert bool((normalized[:, 33:36] == -2).all())
    assert bool((normalized[:, :33] == 1).all()) and bool((x == 1).all())
    with pytest.raises(ValueError, match="Raw spatial"):
        _normalize_spatial_raw(dict(spatial_scalar_features=raw), spatial)
    bad = raw.clone()
    bad[0, 4] = float('nan')
    with pytest.raises(ValueError, match="non-finite"):
        _normalize_spatial_raw(dict(spatial_scalar_raw_features=bad), spatial)


def test_bad_stats_fail_closed(tmp_path):
    path, stats = stats_file(tmp_path)
    stats["ligand"]["std"][2] = -1
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="shape/std"):
        _load_cohort_norm(path)
    path, stats = stats_file(tmp_path)
    stats["fit_scope"] = "benchmark fitted"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="TRAIN-only"):
        _load_cohort_norm(path)


def test_overlap_counts_are_not_a_sequence_certificate(tmp_path):
    p = tmp_path / "split.tsv"
    p.write_text("split\tpdb_id\ntrain\t1abc\ntrain\t1abc\nval\t2xyz\ntest\t3abc\n")
    audit = overlap_by_pdb(p, dict(coach420=["1abcA_LIG", "1abcA_OTH", "4xxxB_LIG"],
                                  holo4k=["1abc_LIG", "2xyz_LIG", "3abc_LIG"]))
    assert audit["counts"]["coach420"]["train"]["shared_protein_keys"] == 1
    assert audit["counts"]["coach420"]["train"]["shared_pair_rows"] == 2
    assert audit["counts"]["holo4k"]["val"]["shared_pair_rows"] == 1
    assert not audit["external_isolation_guaranteed"]
    assert not POLICY["coach420_isolated"] and not POLICY["holo4k_isolated"]
    assert not POLICY["test_driven_iteration_allowed"] and not POLICY["overlap_filter"]
    assert POLICY["max_atoms"] == -1 and POLICY["primary"] == "top20"
