import pytest
import torch

from scripts.benchmark.analyze_holo_nonshared import (
    POLICY, make_cohorts, add_protein_means, same_numbers,
)
from scripts.benchmark.run_libre_joint_test import compare


def manifest(tmp_path, extra=""):
    path = tmp_path / "split.tsv"
    path.write_text("split\tpdb_id\ntrain\t1abc\ntrain\t1ABC\nval\t2xyz\ntest\t3def\n" + extra)
    return path


def test_metadata_cohorts_preserve_whole_proteins_and_exclude_val(tmp_path):
    ids = ["1abc_L1", "1abc_L2", "2xyz_L1", "3def_L1", "4ghi_L1", "4ghi_L2"]
    result = make_cohorts(manifest(tmp_path), ids)
    c = result["cohorts"]
    assert c["all_matched"]["sample_ids"] == ids
    assert c["no_train_pdb"]["sample_ids"] == ids[2:]
    assert c["no_train_val_pdb"]["sample_ids"] == ids[3:]
    assert c["no_internal_pdb"]["sample_ids"] == ids[4:]
    assert c["no_train_val_pdb"]["n_protein_keys"] == 2
    assert result == make_cohorts(manifest(tmp_path), list(reversed(ids)))
    assert POLICY["primary_cohort"] == "no_train_val_pdb"
    assert not POLICY["si30_certified"] and not POLICY["libre_train_isolation_certified"]
    assert not POLICY["test_driven_iteration_allowed"]


def test_internal_accessions_must_not_cross_splits(tmp_path):
    with pytest.raises(ValueError, match="crosses train/val"):
        make_cohorts(manifest(tmp_path, "val\t1ABC\n"), ["1abc_L1"])


@pytest.mark.parametrize("ids", [[], ["1abc_L1", "1abc_L1"], ["bad_L1"]])
def test_invalid_ids_fail_closed(tmp_path, ids):
    with pytest.raises(ValueError):
        make_cohorts(manifest(tmp_path), ids)


def test_exclusion_can_leave_empty_group(tmp_path):
    c = make_cohorts(manifest(tmp_path), ["1abc_L1"])["cohorts"]
    assert c["no_train_val_pdb"]["n_samples"] == 0
    assert c["all_matched"]["n_samples"] == 1


def test_protein_mean_not_ligand_row_mean_and_full_control():
    rows = [dict(sample_id=sid, source="holo4k", scores=torch.tensor(p), labels=torch.tensor([0, 2]))
            for sid, p in [("1abc_L1", [.9, .1]), ("1abc_L2", [.9, .1]), ("2abc_L1", [.1, .9])]]
    lib = {r["sample_id"]: r["scores"].numpy() for r in rows}
    result = compare(rows, lib, .5)
    add_protein_means(result, rows, lib, .5)
    r = result["models"]["plc"]["all"]["rules"]["native_threshold"]
    assert r["sample_macro"]["f1"] == pytest.approx(2 / 3)
    assert r["protein_mean_of_ligand_mean"]["f1"] == pytest.approx(.5)
    assert r["n_proteins"] == 2
    same_numbers(result["models"], result["models"])
    with pytest.raises(ValueError, match="metric changed"):
        same_numbers(dict(f1=.4), dict(f1=.5))
    with pytest.raises(ValueError, match="fields changed"):
        same_numbers(dict(f1=.5), dict(mcc=.5))
