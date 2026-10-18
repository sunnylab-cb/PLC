import copy
from pathlib import Path
import sys
import json
import types

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


def test_validate_dataset_keeps_source_mapping_after_label_loop(tmp_path, monkeypatch):
    import torch
    import prepare_internal_gfgf as module
    path, groups = joint_members(tmp_path)
    (tmp_path / "split").mkdir()
    path.rename(tmp_path / "split/candidate_split.tsv")
    data = tmp_path / "dataset"
    data.mkdir()
    counts = {key: len(ids) for key, ids in groups.items()}
    protocol = module.fixed5_protocol()
    si = dict(passed=True, contract=dict(edge_rule="global_only", pool_sha256="hash"),
              cross_set_audit=dict(edges=0, max_global_si=.2), cohort="pdbbind_biolip_joint",
              external_benchmarks_used=False, counts=counts, split_sha256="hash", by_source={})
    prep = dict(preprocessing_passed=True, cohort=si["cohort"], contact_label_protocol=protocol,
                training_split_sha256="hash", stats_sha256="hash", labels_sha256="hash",
                split_sha256="hash", pool_sha256="hash", dataset_root=str(data),
                labels_file=str(data / "labels.pt"), dataset_counts=counts)
    for p, obj in [(tmp_path / "split/summary.json", si),
                   (data / "preprocessing_summary.json", prep),
                   (data / "train_only_stats.json", dict(split_sha256="hash",
                    train_keys=["pdbbind:p1", "biolip:b1", "biolip:b2"]))]:
        p.write_text(json.dumps(obj))
    read = []
    class Loader:
        def __init__(self, pdb_ids, **kwargs):
            self.pdb_ids = pdb_ids
            self.distances_dict = {sid: dict(contact_label_protocol=protocol) for sid in pdb_ids}

        def __getitem__(self, index):
            read.append(self.pdb_ids[index])
            return dict(labels=torch.zeros(3, dtype=torch.long))

    monkeypatch.setitem(sys.modules, "src.data.pocket_dataset_v8", types.SimpleNamespace(PocketDatasetV8=Loader))
    monkeypatch.setattr(module, "sha256", lambda path: "hash")
    monkeypatch.setattr(module, "read_strict_splits", lambda path: groups)
    monkeypatch.setattr(module, "decode_contact_labels", lambda label: (torch.zeros(3), None))
    cfg = dict(data=dict(distances_file=str(data / "labels.pt"), max_seq_len=1800))
    qc, _ = module.validate_dataset(tmp_path, cfg, tmp_path, si["cohort"])
    assert qc["passed"] and qc["minimum_contact_count_after_residue_cap"] == dict(train=3, val=3, test=3)
    assert set(read) == set(groups["train"] + groups["val"])
    assert not set(read).intersection(groups["test"])
