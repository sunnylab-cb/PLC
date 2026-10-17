import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/diagnostics"))
from prepare_internal_gfgf import make_config, split_members, source_smoke_ids


def baseline():
    return dict(model=dict(protein_hybrid_pattern="GFGF", use_compile=False,
            use_local54_residue_chem_compat=True, use_v12_coattention_block=True),
        training=dict(init_from_checkpoint=None, freeze_v11_base=False, batch_size=16, learning_rate=.0001),
        experiment=dict(seed=44, name="old", output_dir="old"),
        data=dict(data_root="old", split_file="old", distances_file="old", max_seq_len=1800,
                  max_atoms=256, atom_truncation_policy="alternating_ends"))


def test_only_paths_change_and_input_not_mutated(tmp_path):
    base = baseline()
    before = copy.deepcopy(base)
    result = make_config(base, tmp_path, tmp_path/"dataset")
    assert base == before
    assert result["model"] == base["model"] and result["training"] == base["training"]
    assert result["experiment"]["seed"] == 44
    assert result["data"]["max_atoms"] == 256
    assert result["data"]["distances_file"] == str(tmp_path/"dataset/residue_ligand_distances_dcenter.pt")


@pytest.mark.parametrize("section,key,value", [
    ("training", "init_from_checkpoint", "old.pt"), ("training", "resume", "old.pt"),
    ("training", "freeze_v11_base", True), ("model", "use_compile", True),
    ("model", "protein_hybrid_pattern", "FFFF"), ("experiment", "seed", 42),
    ("data", "distances_file_original", "legacy.pt"),
])
def test_no_resume_freezing_compile_or_legacy_label_fallback(tmp_path, section, key, value):
    base = baseline()
    base[section][key] = value
    with pytest.raises(ValueError):
        make_config(base, tmp_path, tmp_path/"dataset")


def test_joint_recipe_changes_only_identity_and_data_paths(tmp_path):
    base = baseline()
    cfg = make_config(base, tmp_path, tmp_path/"dataset", "pdbbind_biolip_joint")
    assert cfg["model"] == base["model"]
    assert cfg["training"] == base["training"]
    assert cfg["experiment"]["name"] == "pdbbind_biolip_joint_global_si30_GFGF_scratch_seed44"


def joint_members(tmp_path):
    rows = [("train", "pdbbind", "p1"), ("train", "biolip", "b1"),
            ("train", "biolip", "b2"), ("val", "pdbbind", "p2"), ("test", "biolip", "b3")]
    path = tmp_path/"split.tsv"
    path.write_text("split\tsource\tsample_id\tsample_key\n" + "".join(
        f"{s}\t{source}\t{sid}\t{source}:{sid}\n" for s, source, sid in rows))
    groups = {s:[sid for t, _, sid in rows if s == t] for s in ("train", "val", "test")}
    return path, groups


def test_joint_stats_membership_uses_real_source_keys(tmp_path):
    path, groups = joint_members(tmp_path)
    members = split_members(path, groups, "pdbbind_biolip_joint")
    assert {members[sid][0] for sid in groups["train"]} == {"pdbbind:p1", "biolip:b1", "biolip:b2"}
    ids = source_smoke_ids(groups, members, "train", 3)
    assert {members[sid][1] for sid in ids[:2]} == {"biolip", "pdbbind"}
    assert set(ids).issubset(groups["train"])
    with pytest.raises(ValueError):
        split_members(path, groups, "pdbbind_only")


@pytest.mark.parametrize("corruption", ["wrong_prefix", "holdout_to_train", "duplicate_id"])
def test_joint_membership_rejects_source_leakage_and_collisions(tmp_path, corruption):
    path, groups = joint_members(tmp_path)
    if corruption == "wrong_prefix":
        path.write_text(path.read_text().replace("biolip:b1", "pdbbind:b1"))
    elif corruption == "holdout_to_train":
        groups["train"].append(groups["val"].pop())
    else:
        groups["test"].append(groups["train"][0])
    with pytest.raises(ValueError):
        split_members(path, groups, "pdbbind_biolip_joint")
