from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
gemmi = pytest.importorskip("gemmi")
from preparation.protein import biolip_protein_struct_extract as feature


def make_cif(path):
    st = gemmi.Structure()
    model = gemmi.Model("1")
    for name, offset in (("AAA", 0), ("A", 10)):
        chain = gemmi.Chain(name)
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(12, "A")
        for atom_name, element, dx in (("N", "N", -1), ("CA", "C", 0), ("C", "C", 1), ("O", "O", 2)):
            atom = gemmi.Atom()
            atom.name, atom.element = atom_name, gemmi.Element(element)
            atom.pos = gemmi.Position(offset + dx, 0, 0)
            residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
    st.add_model(model)
    st.make_mmcif_document().write_file(str(path))


def test_aliases_bijective_preserve_existing_short_names():
    aliases = feature._temporary_chain_aliases(["AAA", "A", "SSS", "B", "AAA"])
    assert aliases["A"] == "A" and aliases["B"] == "B"
    assert len(set(aliases.values())) == 4
    assert all(len(value) == 1 for value in aliases.values())
    with pytest.raises(ValueError, match="capacity"):
        feature._temporary_chain_aliases([f"chain{i}" for i in range(63)])


def test_temporary_pdb_does_not_mutate_source_axis(tmp_path):
    path = tmp_path / "input.cif"
    make_cif(path)
    before = path.read_bytes()
    temp, aliases = feature._write_temp_pdb_from_cif(path, return_mapping=True)
    try:
        st = gemmi.read_structure(temp)
        assert [c.name for c in st[0]] == [aliases["AAA"], "A"]
        assert [c[0].seqid.num for c in st[0]] == [12, 12]
        assert [c[0].seqid.icode for c in st[0]] == ["A", "A"]
        assert path.read_bytes() == before
        assert feature.parse_cif_protein(path)["chain_ids"] == ["AAA", "A"]
    finally:
        Path(temp).unlink()


def test_extractor_uses_aliases_but_saves_original_axis(tmp_path, monkeypatch):
    fixed = tmp_path / "fixed"
    (fixed / "1abc").mkdir(parents=True)
    make_cif(fixed / "1abc/receptor_fixed.cif")
    expected = feature._temporary_chain_aliases(["AAA", "A"])
    chains = [expected["AAA"], "A"]
    def phys(path, sequence, names, indices, got_chains):
        assert got_chains == chains
        return np.zeros((2, 62), dtype=np.float32)
    def geom(path, xyz, indices, got_chains):
        assert got_chains == chains
        return np.zeros((2, 20), dtype=np.float32)
    def vectors(path, info):
        assert info["chain_ids"] == chains
        return np.zeros((2, 8, 3), dtype=np.float32)
    monkeypatch.setattr(feature, "extract_physicochemical_features", phys)
    monkeypatch.setattr(feature, "extract_structural_geometry_features", geom)
    for name in ("extract_surface_features", "compute_pocketness_features", "compute_flexibility_features", "compute_environmental_features"):
        monkeypatch.setattr(feature, name, lambda *a: np.zeros((2, 3), dtype=np.float32))
    # Unit-test axis routing without requiring the optional full V4 stack locally.
    monkeypatch.setattr(feature, "extract_physchem_features_v4", lambda *a: np.zeros((2, 62), dtype=np.float32), raising=False)
    monkeypatch.setattr(feature, "extract_spatial_scalar_features_v4", lambda *a: (np.zeros((2, 89), dtype=np.float32), np.zeros((2, 89), dtype=np.float32)), raising=False)
    monkeypatch.setattr(feature, "extract_spatial_vector_features_v4", vectors, raising=False)
    monkeypatch.setattr(feature, "get_spatial_scalar_feature_names_v4", lambda: [str(i) for i in range(89)], raising=False)
    output = tmp_path / "features"
    ok, message = feature.extract_protein_features("1abc", fixed, tmp_path, output)
    assert ok, message
    data = torch.load(output / "1abc_protein.pt", weights_only=False)
    assert data["chain_ids"] == ["AAA", "A"]
    assert data["residue_indices"] == [12, 12]
    assert data["insertion_codes"] == ["A", "A"]
    assert data["temporary_pdb_chain_aliases"] == expected
