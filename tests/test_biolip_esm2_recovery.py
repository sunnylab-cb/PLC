from pathlib import Path
import argparse
import json
import sys

import gemmi
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics import biolip_esm2_recovery as recovery


def residue(name="ALA", number=1, insertion=" ", has_ca=True, x=1):
    r = gemmi.Residue()
    r.name = name
    r.seqid = gemmi.SeqId(number, insertion)
    a = gemmi.Atom()
    a.name = "CA" if has_ca else "N"
    a.element = gemmi.Element("C")
    a.pos = gemmi.Position(x, 2, 3)
    r.add_atom(a)
    return r


def model_with_chain(name="AAA", residues=None):
    m = gemmi.Model("1")
    c = gemmi.Chain(name)
    for r in residues if residues is not None else [residue(), residue("MSE", 1, "A")]:
        c.add_residue(r)
    m.add_chain(c)
    return m


def test_cif_axis_retains_long_chain_and_insertions():
    axis = recovery.cif_chain_axis(model_with_chain(), "AAA")
    assert axis["keys"] == [("AAA", 1, ""), ("AAA", 1, "A")]
    assert axis["sequence"] == "AM"
    assert axis["ca_coords"] == [[1, 2, 3], [1, 2, 3]]


def test_cif_axis_does_not_require_pdb_chain_capacity():
    model = gemmi.Model("1")
    for i in range(100):
        c = gemmi.Chain(str(i))
        c.add_residue(residue())
        model.add_chain(c)
    assert recovery.cif_chain_axis(model, "99")["keys"] == [("99", 1, "")]


def test_duplicate_residue_keys_are_not_arbitrarily_deduplicated():
    model = model_with_chain(residues=[residue(), residue("GLY")])
    with pytest.raises(ValueError, match="duplicate_residue_keys"):
        recovery.cif_chain_axis(model, "AAA")


def test_duplicate_chain_names_are_rejected():
    model = model_with_chain()
    model.add_chain(gemmi.Chain("AAA"))
    with pytest.raises(ValueError, match="duplicate_chain_identifier"):
        recovery.cif_chain_axis(model, "AAA")


def test_missing_nonprotein_and_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="chain_absent"):
        recovery.cif_chain_axis(model_with_chain(), "B")
    for rs in ([residue("HOH")], [residue(has_ca=False)], [residue("SEP")]):
        with pytest.raises(ValueError, match="no_v4_compatible"):
            recovery.cif_chain_axis(model_with_chain(residues=rs), "AAA")
    with pytest.raises(ValueError, match="nonfinite"):
        recovery.cif_chain_axis(model_with_chain(residues=[residue(x=float("nan"))]), "AAA")


def test_v4_alignment_requires_coordinates_and_identity():
    axis = recovery.cif_chain_axis(model_with_chain(), "AAA")
    protein = dict(chain_ids=["AAA", "AAA"], residue_indices=[1, 1],
                   insertion_codes=["", "A"], residue_names=["ALA", "MET"],
                   ca_coords=torch.tensor(axis["ca_coords"]))
    recovery.compare_v4(axis, protein, "AAA")
    protein["residue_names"][0] = "GLY"
    with pytest.raises(ValueError, match="identity"):
        recovery.compare_v4(axis, protein, "AAA")
    protein["residue_names"][0] = "ALA"
    protein["ca_coords"][0, 0] += 1
    with pytest.raises(ValueError, match="ca_mismatch"):
        recovery.compare_v4(axis, protein, "AAA")


def test_latent_cache_provenance_and_shape():
    data = dict(sequence="AM", checkpoint_sha256="abc", embeddings=torch.ones(2, 1280))
    row = dict(sequence="AM")
    recovery.validate_latent(data, row, "abc")
    with pytest.raises(ValueError, match="provenance"):
        recovery.validate_latent(data, row, "bad")
    with pytest.raises(ValueError, match="provenance"):
        recovery.validate_latent(data, dict(sequence="AG"), "abc")
    data["embeddings"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="shape/values"):
        recovery.validate_latent(data, row, "abc")


def test_audit_raw_cif_without_v4_is_quarantined(tmp_path):
    root, expansion = tmp_path / "biolip", tmp_path / "expansion"
    p = root / "fixed_receptors/1abc/receptor_fixed.cif"
    p.parent.mkdir(parents=True)
    st = gemmi.Structure()
    st.add_model(model_with_chain())
    st.make_mmcif_document().write_file(str(p))
    request = dict(bundle="1abc_AAA", chain="AAA", candidate_pairs=2)
    rows, inputs = recovery.audit_protein(("1abc", [request], str(root), str(expansion)))
    assert rows[0]["status"] == "sequence_ready"
    assert rows[0]["v4_alignment"] == "missing_v4"
    assert rows[0]["training_approved"] is False
    assert len(inputs) == 1


def test_audit_missing_chain_remains_explicit(tmp_path):
    rows, _ = recovery.audit_protein(("1abc", [dict(bundle="1abc_A", chain="A")],
                                      str(tmp_path / "bio"), str(tmp_path / "exp")))
    assert rows[0]["status"] == "unresolved"
    assert rows[0]["training_approved"] is False
    assert len(rows[0]["attempts"]) == 2


def test_plan_counts_candidates_and_seals_axis_manifest(tmp_path, monkeypatch):
    class SerialPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, tasks):
            return map(fn, tasks)

    monkeypatch.setattr(recovery, "ProcessPoolExecutor", SerialPool)
    bio, exp, out = (tmp_path / p for p in ("bio", "expansion", "recovery"))
    (exp / "plan").mkdir(parents=True)
    out.mkdir()
    cif = bio / "fixed_receptors/1abc/receptor_fixed.cif"
    cif.parent.mkdir(parents=True)
    st = gemmi.Structure()
    st.add_model(model_with_chain())
    st.make_mmcif_document().write_file(str(cif))
    for name in ("esm_missing_chains.tsv", "candidates.tsv"):
        (exp / "plan" / name).write_text("pdb_id\tprotein_chain\n1abc\tAAA\n2abc\tB\n")
    (exp / "plan/summary.json").write_text(json.dumps(dict(
        contract=dict(root=str(bio)), output_sha256={n:recovery.digest(exp / "plan" / n)
            for n in ("esm_missing_chains.tsv", "candidates.tsv")})))
    for stage, bundle in (("esm_ready", "1abc_AAA"), ("esm_new", "2abc_B")):
        (exp / f"{stage}_summary.json").write_text(json.dumps(dict(
            failures=[dict(bundle=bundle, error="missing V4")], checkpoint_sha256="unused")))
    args = argparse.Namespace(expansion=exp, output=out, biolip_root=bio, workers=1)
    recovery.plan(args)
    summary = json.loads((out / "plan_summary.json").read_text())
    assert summary["requested_chains"] == 2
    assert summary["counts"] == {"sequence_ready": 1, "unresolved": 1}
    assert summary["candidate_pairs_by_status"] == {"sequence_ready": 1, "unresolved": 1}
    assert summary["training_approved"] is False
    recovery.plan(args)
    with (out / "axes.jsonl").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValueError, match="plan changed"):
        recovery.plan(args)
