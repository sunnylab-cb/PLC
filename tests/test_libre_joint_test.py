import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.benchmark import run_libre_joint_test as run


def axis_sha(keys, names):
    return hashlib.sha256(json.dumps([[*k, a] for k, a in zip(keys, names)],
                                     separators=(",", ":")).encode()).hexdigest()


def test_chainwise_alignment_preserves_noncontiguous_axis_and_rare_aa():
    keys = [("A", 1, ""), ("B", 3, ""), ("A", 2, "A"), ("B", 4, "")]
    segments = run.make_segments(keys, {"A": "AU", "B": "CD"}, axis_sha(keys, "ACUD"), 3)
    assert segments == [dict(sequence="AX", axis=[0, 2]), dict(sequence="CD", axis=[1, 3])]
    p = run.merge_probabilities(segments, [np.array([.1, .2]), np.array([.3, .4])], 3)
    np.testing.assert_allclose(p, [.1, .3, .2])


def test_long_chain_windows_cover_original_cap():
    keys = [("A", i, "") for i in range(2100)]
    segments = run.make_segments(keys, {"A": "A" * 2100}, axis_sha(keys, "A" * 2100), 1800)
    assert len(segments) == 2 and all(len(s["axis"]) == 1500 for s in segments)
    values = [np.full(1500, .2), np.full(1500, .8)]
    p = run.merge_probabilities(segments, values, 1800)
    np.testing.assert_allclose(p[:600], .2)
    np.testing.assert_allclose(p[600:1500], .5)
    np.testing.assert_allclose(p[1500:], .8)


def test_axis_hash_chain_and_missing_coverage_fail_closed():
    keys = [("A", 1, ""), ("A", 2, "")]
    with pytest.raises(ValueError, match="hash"):
        run.make_segments(keys, {"A": "CC"}, axis_sha(keys, "AA"), 2)
    with pytest.raises(ValueError, match="Chain"):
        run.make_segments(keys, {"B": "AA"}, axis_sha(keys, "AA"), 2)
    with pytest.raises(ValueError, match="Uncovered"):
        run.merge_probabilities([dict(axis=[0])], [np.array([.9])], 2)
    with pytest.raises(ValueError, match="Invalid segment"):
        run.merge_probabilities([dict(axis=[0])], [np.array([np.nan])], 1)


def label_data():
    d = torch.tensor([4.999, 5., 7.999, 8.])
    return dict(distances=d, contact_mask=d < 5,
                contact_label_protocol=dict(contact_dmin_angstrom=5., comparison="<", heavy_atoms_only=True))


def test_exact_native_five_angstrom_target():
    label = label_data()
    assert run.validate_target(label, [0, 1, 1, 2]) == 4
    with pytest.raises(ValueError, match="Reference target"):
        run.validate_target(label, [0, 0, 1, 2])
    label["contact_label_protocol"]["comparison"] = "<="
    with pytest.raises(ValueError, match="frozen"):
        run.validate_target(label, [0, 1, 1, 2])


def test_native_thresholds_and_equal_budget_metrics():
    rows = [dict(sample_id="a", source="pdbbind", scores=torch.tensor([.6, .7, .1]),
                 labels=torch.tensor([0, 0, 2], dtype=torch.uint8)),
            dict(sample_id="b", source="biolip", scores=torch.tensor([.2, .8, .9]),
                 labels=torch.tensor([0, 1, 2], dtype=torch.uint8))]
    predictions = {r["sample_id"]: r["scores"].numpy() for r in rows}
    result = run.compare(rows, predictions, .55)
    libre = result["models"]["libre"]["all"]
    plc = result["models"]["plc"]["all"]
    assert libre["rules"]["native_threshold"]["n_predicted"] == 3
    assert plc["rules"]["native_threshold"]["n_predicted"] == 4
    assert libre["rules"]["top20"]["n_predicted"] == 6
    assert libre["rules"]["top20"] == plc["rules"]["top20"]
    assert result["libre_minus_plc"]["all"]["top20"]["pooled"]["f1"] == 0
    assert "at_val_gate" not in libre["rules"]
    assert libre["auprc"] == plc["auprc"]


def test_prepare_native_chemistry_and_no_silent_repair(tmp_path, monkeypatch):
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    sdf = tmp_path / "native.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
    # Read rounded SDF coordinates, not pre-write full precision coordinates.
    native = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    xyz = torch.tensor(np.asarray(native.GetConformer().GetPositions())[:3], dtype=torch.float32)
    ligand_path = tmp_path / "ligand.pt"
    torch.save(dict(ligand_path=str(sdf), coords_original=xyz), ligand_path)
    keys = [("A", i, "") for i in range(4)]
    label = label_data()
    label["residue_keys"] = keys
    labels_path = tmp_path / "labels.pt"
    torch.save(label, labels_path)
    qc = dict(error=None, labels_path=str(labels_path), labels_sha256=run.sha(labels_path),
              chains={"A": "ACDE"}, axis_sha256=axis_sha(keys, "ACDE"))
    sample = dict(sample_id="example", source="biolip", pdb_id="test",
                  qc=qc, ligand_path=str(ligand_path))
    monkeypatch.setattr(run, "graph_from_smiles", lambda s: SimpleNamespace(
        x=torch.ones(3, 40), edge_index=torch.tensor([[0, 1], [1, 0]])), raising=False)
    result, error = run.prepare_one((sample, [0, 1, 1, 2], str(tmp_path)))
    assert error is None and result["n_residues"] == 4 and result["smiles"] == "CCO"
    assert result["graph_x"].shape == (3, 40)
    torch.save(dict(ligand_path=str(sdf), coords_original=xyz+1), ligand_path)
    result, error = run.prepare_one((sample, [0, 1, 1, 2], str(tmp_path)))
    assert result is None and error["stage"] == "ligand" and "geometry" in error["error"]


def test_no_training_search_or_overlap_filter():
    tree = ast.parse(Path(run.__file__).read_text())
    calls = {getattr(n.func, "id", getattr(n.func, "attr", ""))
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"backward", "AdamW", "find_best_f1_threshold", "train_test_split"}
    assert run.POLICY["training_overlap_accepted"]
    assert not run.POLICY["overlap_filter"]
    assert not run.POLICY["select_best_k"]
    assert not run.POLICY["test_driven_iteration_allowed"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "load":
            assert any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                       for k in node.keywords)
