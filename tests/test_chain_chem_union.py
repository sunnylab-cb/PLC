import copy
import inspect

import pytest
import torch
from rdkit import Chem

from src.data.chain_chem_union import (
    UnionBuilder, chemical_identity, geometry_digest, union_sample, TASK_ID,
)
from scripts.diagnostics.prepare_chain_chem_union import make_configs


def record(sid="copy_b", distances=(2., 9., 6., 10.), chain="A", split="train", chemistry="mol"):
    keys = [(chain, i + 1, "") for i in range(4)]
    xyz = torch.arange(12).reshape(4, 3).float()
    return dict(sample_id=sid, source="biolip", pdb_id="1abc", chain=chain, split=split,
                component_id="component1", residue_keys=keys, indices=list(range(4)), parent_n=4,
                distances=torch.tensor(distances), geometry_sha256=geometry_digest(keys, xyz),
                chemistry=dict(status="exact", key=chemistry, smiles="CC"))


def test_union_updates_contact_near_and_distance_aux_and_representative():
    a = record()
    b = record("copy_a", (9., 3., 4., 7.))
    builder = UnionBuilder(minimum_contacts=3)
    builder.add(a)
    builder.add(b)
    manifest = builder.finish()
    assert manifest["task_id"] == TASK_ID
    row, = manifest["rows"]
    assert row["representative"] == "copy_a"
    assert row["members"] == ["copy_a", "copy_b"] and row["eligible"]
    assert row["distances"].tolist() == [2., 3., 4., 7.]
    # Atom axis deliberately has the SAME length as the residue axis.
    ligand = torch.randn(4, 109)
    parent = dict(pdb_id="copy_a", n_residues=4, gvp_coords=torch.arange(12).reshape(4, 3).float(),
                  ligand_features=ligand, labels_original=torch.ones(4).long(),
                  protein_rsa_raw=torch.arange(4)[:, None].float())
    sample = union_sample(parent, row, 1800)
    assert sample["labels"].tolist() == [0, 0, 0, 1]
    assert sample["raw_distances"].tolist() == [2., 3., 4., 7.]
    assert sample["ligand_features"] is ligand
    assert "labels_original" not in sample


def test_no_cross_chain_structure_source_chemistry_or_geometry_merge():
    builder = UnionBuilder()
    variants = [record(), record("copy2", chain="B"), record("copy3", chemistry="other"),
                dict(record("copy4"), pdb_id="2def"), dict(record("copy5"), source="pdbbind"),
                dict(record("copy6"), geometry_sha256="changed_coordinates")]
    for row in variants:
        builder.add(row)
    assert len(builder.finish()["rows"]) == 6
    with pytest.raises(ValueError, match="crosses frozen splits"):
        builder.add(record("other_split", split="val"))
    with pytest.raises(ValueError, match="Duplicate"):
        builder.add(record())


def test_unknown_chemistry_keeps_singletons():
    builder = UnionBuilder()
    for sid in ("a", "b"):
        builder.add(dict(record(sid), chemistry=dict(status="unresolved_singleton", reason="missing SDF")))
    assert len(builder.finish()["rows"]) == 2


def test_geometry_and_axis_guard():
    row = record()
    builder = UnionBuilder()
    builder.add(row)
    row = builder.finish()["rows"][0]
    parent = dict(pdb_id="copy_b", n_residues=4, gvp_coords=torch.zeros(4, 3))
    with pytest.raises(ValueError, match="geometry"):
        union_sample(parent, row, 1800)
    parent["gvp_coords"] = torch.arange(12).reshape(4, 3).float()
    parent["residue_keys"] = list(reversed(row["residue_keys"]))
    with pytest.raises(ValueError, match="keys changed"):
        union_sample(parent, row, 1800)


def test_later_chain_survives_parent_prefix_cap_and_eligibility_after_union():
    builder = UnionBuilder(cap=3, minimum_contacts=3)
    r = record(distances=(1., 2., 3., 4.), chain="B")
    r.update(indices=[4, 5, 6, 7], parent_n=8)
    builder.add(r)
    row, = builder.finish()["rows"]
    parent = dict(pdb_id="copy_b", n_residues=8,
        gvp_coords=torch.cat((torch.zeros(4, 3), torch.arange(12).reshape(4, 3).float())),
        esm=torch.arange(8)[:, None], ligand_features=torch.ones(8, 109))
    sample = union_sample(parent, row, 3)
    assert sample["esm"].flatten().tolist() == [4, 5, 6]
    assert row["n_contact_full"] == 4 and row["n_contact"] == 3
    assert row["eligible"] and sample["ligand_features"].shape[0] == 8


def test_thresholds_remain_strict_five_and_eight_and_zero_chains_retained():
    builder = UnionBuilder()
    builder.add(record(distances=(5., 8., 4.999, 7.999)))
    builder.add(record("zero", chain="B", distances=(10., 10., 10., 10.)))
    rows = builder.finish()["rows"]
    assert len(rows) == 2 and all(not r["eligible"] for r in rows)
    row = next(r for r in rows if r["chain"] == "A")
    parent = dict(pdb_id="copy_b", n_residues=4, gvp_coords=torch.arange(12).reshape(4, 3).float())
    assert union_sample(parent, row, 1800)["labels"].tolist() == [1, 2, 0, 1]


def write_sdf(path, smiles):
    with Chem.SDWriter(str(path)) as writer:
        writer.write(Chem.MolFromSmiles(smiles))


def test_exact_chemistry_stereo_charge_and_atom_order(tmp_path):
    identities = []
    for i, smiles in enumerate(("CCO", "OCC", "C[C@H](O)F", "C[C@@H](O)F", "CC[O-]")):
        path = tmp_path / f"{i}.sdf"
        write_sdf(path, smiles)
        identities.append(chemical_identity(path)["key"])
    assert identities[0] == identities[1]
    assert identities[2] != identities[3]
    assert identities[0] != identities[4]
    with pytest.raises(ValueError, match="count disagree"):
        chemical_identity(tmp_path / "0.sdf", 20)


def test_regularization_arms_change_one_knob(tmp_path):
    base = dict(model=dict(scan_variant="G0_small_global", dropout=.15),
                experiment=dict(seed=44), training=dict(weight_decay=.05), data={})
    configs = make_configs(base, tmp_path)
    for name, expected in (("union_base", (.05, .15)), ("union_wd010", (.1, .15)), ("union_drop020", (.05, .2))):
        c = configs[name]
        assert (c["training"]["weight_decay"], c["model"]["dropout"]) == expected
        assert c["training"]["init_from_checkpoint"] is None and not c["training"]["freeze_v11_base"]
        assert c["training"]["early_stopping_metric"] == "f1_c0"
        assert c["data"]["distances_file_original"] is None
        assert not c["model"]["use_compile"]
    assert "chain_union_manifest" not in base


def test_dataset_view_preserves_membership_and_replaces_all_targets(tmp_path, monkeypatch):
    from src.data.chain_chem_union import ChainChemUnionDataset
    import src.data.pocket_dataset_v8 as production
    builder = UnionBuilder()
    builder.add(record("parent", (1., 2., 3., 9.)))
    manifest = builder.finish()
    row = manifest["rows"][0]
    path = tmp_path / "union.pt"
    torch.save(manifest, path)

    class FakeParent:
        def __init__(self, pdb_ids, split, **kwargs):
            assert kwargs["max_seq_len"] > 1800 and not kwargs["use_nma"]
            self.pdb_ids = pdb_ids
            self.distances_dict = {"parent": dict(residue_keys=row["residue_keys"])}
        def __len__(self):
            return len(self.pdb_ids)
        def __getitem__(self, index):
            return dict(pdb_id=self.pdb_ids[index], n_residues=4,
                gvp_coords=torch.arange(12).reshape(4, 3).float(), labels=torch.zeros(4).long())

    monkeypatch.setattr(production, "PocketDatasetV8", FakeParent)
    ds = ChainChemUnionDataset(path, [row["sample_id"]], "train", max_seq_len=1800,
                              nma_n_confs=1, nma_max_conf=0)
    assert len(ds) == 1 and ds[0]["labels"].tolist() == [0, 0, 0, 2]
    with pytest.raises(ValueError, match="Wrong split"):
        ChainChemUnionDataset(path, [row["sample_id"]], "val")
    with pytest.raises(ValueError, match="conf0"):
        ChainChemUnionDataset(path, [row["sample_id"]], "train", nma_n_confs=3)


def test_train_subset_is_label_independent_source_proportional():
    from scripts.diagnostics.eval_g0_train_val_gap import stratified_train_ids
    meta = {f"x{i}": dict(source="pdbbind" if i < 20 else "biolip", split="train") for i in range(100)}
    meta["heldout"] = dict(source="pdbbind", split="test")
    selected = stratified_train_ids(meta, 30)
    assert len(selected) == 30 and "heldout" not in selected
    assert sum(meta[sid]["source"] == "pdbbind" for sid in selected) == 6
    reversed_meta = dict(reversed(list(meta.items())))
    assert stratified_train_ids(reversed_meta, 30) == selected


def test_custom_holo_metrics_keep_zero_chains_and_fixed_mcc():
    import numpy as np
    from scripts.benchmark.eval_chain_chem_union import summarize
    records = [
        (dict(eligible=True, n_contact=3, pdb_id="1abc"), np.array([.9, .8, .7, .1]), np.array([1, 1, 1, 0], dtype=bool)),
        (dict(eligible=False, n_contact=0, pdb_id="1abc"), np.array([.8, .1]), np.zeros(2, dtype=bool)),
    ]
    out = summarize(records, .75)
    assert out["all_queries"]["rules"]["at_0_5"]["n_samples"] == 2
    assert out["zero_contact"]["rules"]["at_0_5"]["n_predicted"] == 1
    assert out["eligible_ge3"]["rules"]["at_0_5"]["sample_macro"]["mcc"] == 1.
    assert out["eligible_ge3"]["rules"]["at_val_gate"]["sample_macro"]["mcc"] < 1.
    assert out["zero_contact"]["auprc"] is None
