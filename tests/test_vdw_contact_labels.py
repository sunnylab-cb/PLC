import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/diagnostics"))
from internal_contact_geometry import count_contacts, ligand_elements, protein_elements
from internal_cohort_qc import cif_atoms
from src.contact_labels import contact_protocol, decode_contact_labels, fixed5_protocol, validate_contact_axis


def label_fixture(minimum=2):
    # One residue atom can make two distinct ligand atom-pair contacts.
    result = count_contacts(
        [[[0., 0., 0.]], [[6., 0., 0.]], [[20., 0., 0.]]],
        [["C"], ["C"], ["C"]], [[2., 0., 0.], [0., 2., 0.]], ["C", "C"], minimum)
    result.update(n_residues=3, residue_keys=[("A", i, "") for i in range(1, 4)])
    return result


def test_vdw_threshold_depends_on_both_elements_and_includes_boundary():
    cc = count_contacts([[[0, 0, 0]]], [["C"]], [[3.9, 0, 0]], ["C"])
    no = count_contacts([[[0, 0, 0]]], [["N"]], [[3.9, 0, 0]], ["O"])
    assert cc["atom_contact_counts"].tolist() == [1]
    assert no["atom_contact_counts"].tolist() == [0]
    assert cc["contact_mask_casp1"].tolist() == [True]
    assert cc["contact_mask_biolip2"].tolist() == [False]
    outside = count_contacts([[[0, 0, 0]]], [["C"]], [[3.90001, 0, 0]], ["C"])
    assert outside["atom_contact_counts"].tolist() == [0]


def test_two_pairs_not_two_different_protein_atoms_and_keep_physical_distances():
    entry = label_fixture()
    assert entry["atom_contact_counts"].tolist() == [2, 0, 0]
    labels, raw = decode_contact_labels(entry)
    assert labels.tolist() == [0, 1, 2]
    assert raw.tolist() == [2., 4., 18.]
    assert not torch.equal(raw, entry["vdw_surface_gap"])


def test_one_pair_is_near_not_contact_in_biolip_mode_even_below_old_cutoff():
    entry = label_fixture()
    entry["atom_contact_counts"][0] = 1
    entry["contact_mask"][0] = False
    entry["contact_mask_biolip2"][0] = False
    assert decode_contact_labels(entry)[0].tolist() == [1, 1, 2]
    entry["contact_label_protocol"] = contact_protocol(1)
    entry["contact_mask"][0] = True
    assert decode_contact_labels(entry)[0].tolist() == [0, 1, 2]


def test_corrupt_protocol_mask_and_axis_fail_closed():
    entry = label_fixture()
    entry["contact_mask"][0] = False
    with pytest.raises(ValueError, match="disagrees"):
        decode_contact_labels(entry)
    entry = label_fixture()
    entry["contact_label_protocol"]["tolerance_angstrom"] = 5.5
    with pytest.raises(ValueError, match="protocol"):
        decode_contact_labels(entry)
    entry = label_fixture()
    data = dict(chain_id=["A"]*3, residue_index=[1, 2, 3], insertion_code=[""]*3)
    validate_contact_axis(entry, data, data)
    bad = dict(data, residue_index=[2, 1, 3])
    with pytest.raises(ValueError, match="residue-key"):
        validate_contact_axis(entry, data, bad)


def test_fixed5_is_strict_center_distance_and_ignores_vdw_pair_minimum():
    distances = torch.tensor([0., 4.999, 5., 5.5, 7.999, 8.])
    entry = dict(contact_label_protocol=fixed5_protocol(), distances=distances,
                 contact_mask=distances < 5., n_residues=6,
                 residue_keys=[("A", i, "") for i in range(6)])
    labels, raw = decode_contact_labels(entry)
    assert labels.tolist() == [0, 0, 1, 1, 1, 2]
    assert torch.equal(raw, distances)
    entry["contact_mask"][2] = True
    with pytest.raises(ValueError, match="fixed 5A"):
        decode_contact_labels(entry)
    entry["contact_mask"][2] = False
    entry["contact_label_protocol"]["contact_dmin_angstrom"] = 5.5
    with pytest.raises(ValueError, match="protocol"):
        decode_contact_labels(entry)


def test_explicit_element_resolution_never_calls_ca_calcium_or_other_carbon():
    assert protein_elements("ALA", ["N", "CA", "C", "O", "CB"]) == ["N", "C", "C", "O", "C"]
    assert protein_elements("MSE", ["SE"]) == ["Se"]
    with pytest.raises(ValueError, match="element unavailable"):
        protein_elements("ALA", ["ZN"])
    with pytest.raises(ValueError, match="Duplicate"):
        protein_elements("ALA", ["CA", "CA"])
    x = torch.zeros(2, 109)
    x[0, 0], x[1, 9] = 1, 1
    ligand = dict(atom_features=x)
    assert ligand_elements(ligand, np.zeros((1, 3)))[0] == ["C"]
    x[0, 0], x[0, 10] = 0, 1
    with pytest.raises(ValueError, match="no radius fallback"):
        ligand_elements(ligand, np.zeros((1, 3)))


def test_source_sdf_element_and_native_coordinate_order_are_checked(tmp_path):
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CB")
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, (0, 0, 0))
    conf.SetAtomPosition(1, (1, 0, 0))
    mol.AddConformer(conf)
    path = tmp_path/"native.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    x = torch.zeros(2, 109)
    x[0, 0], x[1, 10] = 1, 1
    ligand = dict(atom_features=x, ligand_path=str(path))
    assert ligand_elements(ligand, np.array([[0, 0, 0], [1, 0, 0]]))[0] == ["C", "B"]
    with pytest.raises(ValueError, match="order or coordinates"):
        ligand_elements(ligand, np.array([[1, 0, 0], [0, 0, 0]]))


def test_cif_alternate_conformers_do_not_double_atom_contacts(tmp_path):
    import gemmi
    st = gemmi.Structure()
    model, chain, residue = gemmi.Model("1"), gemmi.Chain("A"), gemmi.Residue()
    residue.name, residue.seqid = "ALA", gemmi.SeqId(1, " ")
    for alt, occupancy, x in (("A", .7, 0), ("B", .3, .1)):
        atom = gemmi.Atom()
        atom.name, atom.element = "CA", gemmi.Element("C")
        atom.altloc, atom.occ, atom.pos = alt, occupancy, gemmi.Position(x, 0, 0)
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    path = tmp_path/"alternate.pdb"
    st.write_pdb(str(path))
    data = cif_atoms(str(path), "A")[("A", 1, "")]
    assert len(data[2]) == 1
    assert data[3] == ["C"]


def test_biolip_chain_qc_preserves_atom_geometry_needed_by_loader(tmp_path):
    import gemmi
    from internal_cohort_qc import qc_sample
    st, model, chain = gemmi.Structure(), gemmi.Model("1"), gemmi.Chain("A")
    for i in range(1, 4):
        residue, atom = gemmi.Residue(), gemmi.Atom()
        residue.name, residue.seqid = "ALA", gemmi.SeqId(i, " ")
        atom.name, atom.element, atom.pos = "CA", gemmi.Element("C"), gemmi.Position(i, 0, 0)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    cif = tmp_path/"shared_receptors/1abc/receptor_full.cif"
    cif.parent.mkdir(parents=True)
    st.make_mmcif_document().write_file(str(cif))
    protein = dict(cif_source="original", physchem_features=torch.zeros(4, 62),
        spatial_scalar_raw_features=torch.zeros(4, 89), spatial_scalar_features=torch.zeros(4, 89),
        spatial_vector_features=torch.zeros(4, 8, 3),
        ca_coords=torch.tensor([[1., 0, 0], [2., 0, 0], [3., 0, 0], [100., 0, 0]]),
        residue_names=["ALA"]*4, residue_indices=[1, 2, 3, 1], insertion_codes=[""]*4,
        chain_ids=["A", "A", "A", "B"])
    esm = dict(embeddings=torch.zeros(3, 1280), residue_name_1=["A"]*3, residue_index=[1, 2, 3],
               chain_id=["A"]*3, insertion_code=[""]*3)
    identity = dict(pdb_id="1abc", protein_chain="A", binding_site_id="BS01", ligand_id="LIG")
    x = torch.zeros(6, 109)
    x[:, 0] = 1
    ligand = dict(**identity, atom_features=x, coords=torch.zeros(6, 3), coords_original=torch.zeros(6, 3),
        num_heavy_atoms=6, ligand_route=5, x_group_mask=torch.ones(6, 14), feature_completeness_ratio=1.,
        edge_index=torch.empty(2, 0, dtype=torch.long), edge_attr=torch.empty(0, 6))
    paths = {}
    for name, value in (("v4_path", protein), ("esm_path", esm), ("ligand_path", ligand)):
        path = tmp_path/(name+".pt")
        torch.save(value, path)
        paths[name] = str(path)
    sample = dict(source="biolip", sample_id="1abc_A_BS01_LIG", pdb_id="1abc", selected_chain="A",
                  biolip_root=str(tmp_path), ligand_metadata=dict(identity, heavy_atoms_meta="6"), **paths)
    result = qc_sample(("biolip:1abc_A_BS01_LIG", sample, str(tmp_path/"qc")))
    assert result["error"] is None
    sliced = torch.load(result["aligned_protein_path"], weights_only=False)
    assert sliced["chain_ids"] == ["A"]*3
    assert len(sliced["all_atom_coords"]) == 3
    assert sliced["all_atom_names"] == [["CA"]]*3
    assert result["n_contact_biolip2"] == 3


@pytest.mark.parametrize("contact_rule", ["vdw", "fixed5"])
def test_small_cohort_end_to_end_pool_split_stats_and_production_loader(tmp_path, monkeypatch, contact_rule):
    import prepare_internal_codataset_sequences as prepare
    import run_internal_pairwise_split as split
    import preprocess_internal_cohort as preprocess
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    data, pool, si, out = (tmp_path/s for s in ("data", "pool", "si", "dataset"))
    ids = tmp_path/"ids.txt"
    ids.write_text("train\t1abc\nval\t2abc\ntest\t3abc\n")
    for sid, aa3, aa1 in (("1abc", "ALA", "A"), ("2abc", "ARG", "R"), ("3abc", "TRP", "W")):
        protein = dict(physchem_features=torch.zeros(3, 62), spatial_scalar_raw_features=torch.zeros(3, 89),
            spatial_scalar_features=torch.zeros(3, 89), spatial_vector_features=torch.zeros(3, 8, 3),
            ca_coords=torch.tensor([[1., 0, 0], [2., 0, 0], [3., 0, 0]]), residue_names=[aa3]*3,
            residue_indices=[1, 2, 3], insertion_codes=[""]*3, chain_ids=["A"]*3,
            all_atom_coords=[np.array([[float(i), 0, 0]]) for i in (1, 2, 3)], all_atom_names=[["CA"]]*3)
        esm = dict(embeddings=torch.zeros(3, 1280), residue_name_1=[aa1]*3, residue_index=[1, 2, 3],
                   chain_id=["A"]*3, insertion_code=[""]*3)
        x = torch.zeros(6, 109)
        x[:, 0] = 1
        ligand = dict(atom_features=x, coords=torch.zeros(6, 3), coords_original=torch.zeros(6, 3),
            num_heavy_atoms=6, has_original_coords=True, edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 6))
        for folder, suffix, content in (("complete_features_v4_nma", "_protein_conf0.pt", protein),
                ("esm2_latents", "_protein.pt", esm), ("ligand_features_v5_ori", "_ligand.pt", ligand)):
            path = data/folder/"2001-2010"/sid/(sid+suffix)
            path.parent.mkdir(parents=True)
            torch.save(content, path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--pdbbind-root", str(data), "--pdbbind-ids", str(ids),
                                      "--output-dir", str(pool), "--workers", "1",
                                      "--contact-rule", contact_rule])
    prepare.main()
    rows = [json.loads(s) for s in (pool/"samples.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    # Regression: a sequence-loop variable previously overwrote each sample's QC.
    assert all(row["qc"]["aligned_protein_path"] for row in rows)
    assert all(row["qc"]["n_contact"] == 3 for row in rows)
    if contact_rule == "vdw":
        assert all(row["qc"]["n_contact_biolip2"] == 3 for row in rows)
    else:
        assert all("n_contact_biolip2" not in row["qc"] for row in rows)
    monkeypatch.setattr(sys, "argv", ["split", "--pool-dir", str(pool), "--output-dir", str(si),
                                      "--workers", "1", "--chunk-size", "2"])
    split.main()
    monkeypatch.setattr(sys, "argv", ["prep", "--pool-dir", str(pool), "--split-dir", str(si),
                                      "--output-dir", str(out)])
    preprocess.main()
    summary = json.loads((out/"preprocessing_summary.json").read_text())
    assert summary["contact_label_protocol"] == (contact_protocol(2) if contact_rule == "vdw" else fixed5_protocol())
    assert Path(summary["labels_file"]).name == ("residue_ligand_contacts_vdw.pt" if contact_rule == "vdw"
                                                else "residue_ligand_distances_dcenter.pt")
    dataset = PocketDatasetV8(data_root=str(out), pdb_ids=["1abc"],
        protein_features_dir="complete_features_v4_nma", nma_features_dir="complete_features_v4_nma",
        ligand_features_dir="ligand_features_v5",
        distances_file=summary["labels_file"], label_contact_thresh=0.01, use_nma=False)
    sample = dataset[0]
    assert sample["labels"].tolist() == [0, 0, 0]
    assert sample["raw_distances"].tolist() == [1., 2., 3.]
