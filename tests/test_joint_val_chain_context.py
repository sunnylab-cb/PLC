import numpy as np
import pytest
import torch

from scripts.diagnostics.eval_joint_val_chain_context import (
    POLICY, val_members, axis_digest, axis_spec, stratum_names, split_val_sample,
    summarize_records, ValAxisDataset,
)
from src.contact_labels import fixed5_protocol


def sample():
    n = 6
    keys = [["A", str(i), ""] if i < 3 else ["B", str(i), ""] for i in range(n)]
    y = torch.tensor([0, 0, 0, 2, 2, 2])
    item = dict(pdb_id="1abc", n_residues=n, n_atoms=n, labels=y, residue_keys=keys,
        chain_ids=torch.tensor([0, 0, 0, 1, 1, 1]), esm=torch.ones(n, 3),
        protein_rsa_raw=torch.arange(n).reshape(n, 1).float(), protein_rsa_mask=torch.ones(n, 1),
        ligand_features=torch.ones(n, 109), ligand_edge_index=torch.tensor([[0, 1], [1, 0]]))
    label = dict(contact_label_protocol=fixed5_protocol(), residue_keys=keys, n_residues=n,
                 distances=torch.tensor([1., 2., 3., 10., 11., 12.]), contact_mask=y == 0)
    member = dict(source="pdbbind", protein_id="1abc", component_id="10")
    return item, label, member


def test_manifest_selects_only_val(tmp_path):
    path = tmp_path / "split.tsv"
    path.write_text("split\tsample_key\tsource\tsample_id\tpdb_id\tcomponent_id\n"
                    "val\tpdbbind:1abc\tpdbbind\t1abc\t1abc\t10\n"
                    "test\tbiolip:2def_A\tbiolip\t2def_A\t2def\t20\n")
    groups = dict(train=["3ghi"], val=["1abc"], test=["2def_A"])
    assert list(val_members(path, groups)) == ["1abc"]
    with pytest.raises(ValueError, match="overlaps"):
        val_members(path, dict(groups, test=["1abc"]))
    with pytest.raises(ValueError, match="Incomplete"):
        val_members(path, dict(groups, val=["4jkl"]))


def test_axis_specs_strata_and_all_chain_rsa_slicing():
    item, label, member = sample()
    spec = axis_spec(label, member)
    assert spec["n_chains"] == 2 and spec["n_contact_chains"] == 1
    assert "pdbbind/multichain" in stratum_names(spec)
    parts = split_val_sample(item)
    assert parts[1][2]["labels"].tolist() == [2, 2, 2]
    assert parts[1][2]["protein_rsa_raw"].flatten().tolist() == [3., 4., 5.]
    assert parts[1][2]["ligand_features"] is item["ligand_features"]
    assert axis_digest(item["residue_keys"], item["labels"]) == spec["axis_sha256"]
    assert axis_digest(item["residue_keys"][::-1], item["labels"]) != spec["axis_sha256"]
    with pytest.raises(ValueError, match="label axis"):
        axis_digest(item["residue_keys"], np.array([256, 0, 0, 2, 2, 2]))
    truncated = axis_spec(label, member, cap=3)
    assert truncated["n_chains"] == 1 and truncated["n_residues"] == 3
    with pytest.raises(ValueError, match=">=3"):
        axis_spec(label, member, cap=2)


def test_production_wrapper_validates_membership_and_chain_axis():
    item, label, member = sample()

    class Base:
        pdb_ids = ["1abc"]
        distances_dict = {"1abc": label}
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return dict(item)

    spec = axis_spec(label, member)
    dataset = ValAxisDataset(Base(), {"1abc": spec})
    assert len(dataset) == 1 and dataset[0]["residue_keys"] == item["residue_keys"]
    item["chain_ids"] = torch.zeros(6, dtype=torch.long)
    with pytest.raises(ValueError, match="chain IDs"):
        dataset[0]
    with pytest.raises(ValueError, match="coverage"):
        ValAxisDataset(Base(), {"other": spec})


def test_summary_retains_both_fixed_threshold_mcc_and_pooled():
    item, label, member = sample()
    spec = axis_spec(label, member)
    record = dict(spec=spec, labels=item["labels"].numpy(), keys=np.asarray(item["residue_keys"]),
        full=np.array([.9, .7, .6, .2, .1, .05]), chain=np.array([.9, .7, .6, .8, .1, .05]))
    summary = summarize_records([record], .65)
    r = summary["results"]["all"]
    for mode in ("full", "chain"):
        for rule in ("at_0_5", "at_val_gate", "top20"):
            for agg in ("sample_macro", "pooled", "protein_mean_of_ligand_mean"):
                assert "mcc" in r[mode]["rules"][rule][agg]
    assert r["full"]["rules"]["at_0_5"]["sample_macro"]["mcc"] == 1.
    assert r["full"]["rules"]["at_val_gate"]["sample_macro"]["mcc"] < 1.
    assert r["full"]["zero_contact_chains"]["at_val_gate"]["predicted"] == 0
    assert r["chain"]["zero_contact_chains"]["at_val_gate"]["predicted"] == 1
    assert summary["chain_minus_full"]["all"]["at_0_5"]["sample_macro"]["mcc"] < 0
    assert not POLICY["test_forward"] and not POLICY["tune_threshold"]
    assert POLICY["split"] == "val" and POLICY["max_atoms"] == 256
