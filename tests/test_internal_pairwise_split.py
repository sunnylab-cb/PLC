import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/diagnostics"))
from internal_cohort_qc import heavy_distances, ligand_check, qc_sample, residue_axis
from prepare_internal_codataset_sequences import bio_eligible, chain_sequences, digest, save_json
from run_internal_pairwise_split import (
    UnionFind, alignment_stats, assign_components, check_saved_chunk, is_edge,
    iter_pairs, load_pool, make_aligners, pair_at, pair_stats, run_stage,
)
from preprocess_internal_cohort import Moments, fit_training_stats, norm_dims


def test_every_pair_exactly_once_across_chunk_boundaries():
    for n in range(2, 35):
        expected = [(i, j) for i in range(n) for j in range(i+1, n)]
        assert [pair_at(k, n) for k in range(len(expected))] == expected
        actual = []
        for k in range(0, len(expected), 7):
            actual += list(iter_pairs(k, min(k+7, len(expected)), n))
        assert actual == expected
    with pytest.raises(IndexError):
        pair_at(3, 3)


def test_global_and_local_contract_at_exact_thresholds():
    assert is_edge((.30, 0, 0, 0))
    assert is_edge((.29, .30, .80, .1))
    assert is_edge((.29, .30, .1, .80))
    assert not is_edge((.299, .299, 1, 1))
    assert not is_edge((.1, .9, .79, .79))


def test_biopython_identity_is_not_score_and_keeps_gap_columns():
    global_a, local_a = make_aligners()
    a = "ACDEFGHIKLMNPQRSTVWY"
    assert pair_stats(a, a, (global_a, local_a)) == (1., 1., 1., 1.)
    shorter = "ACDEFGHI"
    g, local, qc, tc = pair_stats(shorter, "WWWWWWWWWW"+shorter+"YYYYYYYYYY", (global_a, local_a))
    assert g < .3
    assert local == 1
    assert max(qc, tc) == 1
    assert is_edge((g, local, qc, tc))
    stats = alignment_stats(global_a.align("ACDE", "ACDDE")[0], "ACDE", "ACDDE")
    assert stats == (.8, 1., 1.)


def test_rare_amino_acids_only_alias_scores_not_identity():
    stats = pair_stats("UUUU", "CCCC", make_aligners())
    assert stats[:2] == (0., 0.)


def test_transitive_components_and_same_source_independence():
    keys = list("abcdefg")
    uf = UnionFind(keys)
    uf.union("a", "b")
    uf.union("b", "c")
    assignment, groups = assign_components(dict.fromkeys(keys), uf, 42)
    assert any(set(g) == set("abc") for g in groups)
    assert len({assignment[x] for x in "abc"}) == 1
    assert set(assignment.values()) == {"train", "val", "test"}
    # A joint-pool bridge must not be inherited by an independent PDB-only graph.
    solo = UnionFind(keys)
    assert solo.find("a") != solo.find("c")


def test_giant_component_fails_closed_instead_of_splitting_family():
    uf = UnionFind("abc")
    uf.union("a", "b")
    uf.union("b", "c")
    with pytest.raises(ValueError, match="three separable"):
        assign_components(dict.fromkeys("abc"), uf, 42)


def test_explicit_residue_chain_metadata_required():
    with pytest.raises(ValueError, match="missing"):
        chain_sequences({"sequence": "ACDE"})
    data = dict(residue_name_1=["A", "C", "D"], chain_id=["A", "B", "A"],
                residue_index=[1, 1, 2], insertion_code=["", "", ""])
    assert chain_sequences(data)[0] == {"A": "AD", "B": "C"}
    data["residue_index"][2] = 1
    with pytest.raises(ValueError, match="duplicate"):
        chain_sequences(data)
    with pytest.raises(ValueError, match="duplicate"):
        residue_axis(data)


def test_legacy_biolip_chemistry_metadata_gate():
    row = dict(route_meta="regular_organic", mapping_status="exact_match", mapping_complete="1",
               heavy_atoms_meta="40", binding_residue_count="3", pdb_id="1abc",
               protein_chain="A", ligand_id="LIG", binding_site_id="BS01")
    assert bio_eligible(row)
    for field, bad in [("heavy_atoms_meta", "129"), ("binding_residue_count", "2"),
                       ("route_meta", "pure_metal_ion"), ("mapping_complete", "0")]:
        assert not bio_eligible(dict(row, **{field:bad}))


def ligand_fixture():
    features = torch.zeros(6,109)
    features[:, 0] = 1
    return dict(atom_features=features, coords=torch.zeros(6,3),
        coords_original=torch.zeros(6,3), edge_index=torch.empty(2,0,dtype=torch.long),
        edge_attr=torch.empty(0,6), num_heavy_atoms=6, has_original_coords=True)


def test_reject_normalized_ligand_and_native_coord_absence():
    ligand = ligand_fixture()
    assert ligand_check(ligand, {"source":"pdbbind"})[1] == 6
    ligand["feature_normalization"] = {"old_all_data_stats":True}
    with pytest.raises(ValueError, match="raw"):
        ligand_check(ligand, {"source":"pdbbind"})
    del ligand["feature_normalization"], ligand["coords_original"]
    with pytest.raises(KeyError):
        ligand_check(ligand, {"source":"pdbbind"})


def test_contact_geometry_not_metadata_counts(tmp_path):
    n = 3
    protein = dict(physchem_features=torch.zeros(n,62), spatial_scalar_raw_features=torch.zeros(n,89),
        spatial_vector_features=torch.zeros(n,8,3), ca_coords=torch.zeros(n,3),
        residue_names=["ALA"]*n, residue_indices=[1,2,3], insertion_codes=[""]*n, chain_ids=["A"]*n,
        all_atom_coords=[np.array([[float(x),0,0]]) for x in (1,2,9)], all_atom_names=[["CA"]]*n)
    esm = dict(embeddings=torch.zeros(n,1280), residue_name_1=["A"]*n, residue_index=[1,2,3],
        insertion_code=[""]*n, chain_id=["A"]*n)
    p, e, l = [tmp_path/f"{name}.pt" for name in ("p","e","l")]
    for path, value in ((p,protein),(e,esm),(l,ligand_fixture())):
        torch.save(value, path)
    sample = dict(source="pdbbind", sample_id="1abc", selected_chain=None,
                  v4_path=str(p), esm_path=str(e), ligand_path=str(l))
    result = qc_sample(("pdbbind:1abc",sample,str(tmp_path)))
    assert "fewer_than_3_actual_contact_tags:2" in result["error"]
    assert not (tmp_path/"labels/1abc.pt").exists()
    assert torch.equal(heavy_distances(protein["all_atom_coords"],np.zeros((6,3))),torch.tensor([1.,2.,9.]))


def test_full_enumeration_resume_integrity_and_fresh_cross_audit(tmp_path):
    seqs = ["AAAAAAAA", "RRRRRRRR", "WWWWWWWW", "DDDDDDDD"]
    contract = dict(identity=.3,coverage=.8,test=True)
    result, tasks = run_stage(seqs, None, contract, tmp_path/"all", 2, 2)
    assert result["evaluated"] == 6
    assert result["edges"] == 0
    assert run_stage(seqs,None,contract,tmp_path/"all",2,2)[0] == result
    audit, _ = run_stage(seqs,["train","train","val","test"],contract,tmp_path/"audit",2,2)
    assert audit["evaluated"] == 5
    assert sorted(audit["by_set_pair"].values()) == [1,2,2]
    with pytest.raises(ValueError, match="changed"):
        run_stage(seqs,None,dict(contract,identity=.4),tmp_path/"all",2,2)
    edge = tmp_path/"all/chunk_0000000.edges.tsv.gz"
    edge.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        check_saved_chunk(tmp_path/"all/chunk_0000000.json",result["contract_hash"],0,2)


def test_train_only_stats_ignore_holdout_and_differ_between_cohorts(tmp_path):
    p, l = tmp_path/"p.pt",tmp_path/"l.pt"
    torch.save({"spatial_scalar_raw_features":torch.full((3,89),2.)},p)
    torch.save({"atom_features":torch.full((6,109),4.)},l)
    train = dict(ligand_path=str(l),v4_path=str(p),selected_chain=None,
                 qc=dict(aligned_protein_path=str(p),axis_sha256="a"))
    # No file paths: touching any holdout during fitting would raise immediately.
    samples = {"train":train,"held_out":{}}
    first = fit_training_stats(samples,{"train":"train","held_out":"val"})
    assert first["spatial"]["mean"] == [2.]*89
    assert first["ligand"]["mean"] == [4.]*109
    lp = tmp_path/"other.pt"
    torch.save({"atom_features":torch.full((6,109),8.)},lp)
    second = fit_training_stats({"new_train":dict(train,ligand_path=str(lp))},{"new_train":"train"})
    assert second["ligand"]["mean"] == [8.]*109


def test_moments_float64_and_legacy_selective_normalization():
    a = Moments(3)
    x = np.array([[0.,1.,0.3],[1.,1.,8.1],[0.,1.,17.4]])
    a.update(x[:1])
    a.update(x[1:])
    s = a.report(ddof=1)
    assert np.allclose(s["mean"],x.mean(0))
    assert np.allclose(s["std"],x.std(0,ddof=1))
    assert norm_dims(s) == [2]
    with pytest.raises(ValueError):
        a.update(np.full((1,3),np.nan))
