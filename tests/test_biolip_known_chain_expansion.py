import pytest

from scripts.diagnostics.extract_biolip_known_chain_expansion import verify_manifest


def row(**updates):
    r = dict(feature_stem="1abc_A_BS01_ATP", known_train_chain_key="1",
             known_sequence_matches="True", heavy_atoms_meta="31", binding_residue_count="4")
    r.update(updates)
    return r


def test_verified_known_chain_is_preparable():
    verify_manifest([row()])


@pytest.mark.parametrize("rows", [[], [row(), row()], [row(known_sequence_matches="False")],
    [row(known_train_chain_key="0")], [row(heavy_atoms_meta="1")]])
def test_unverified_or_wrong_scope_rejected(rows):
    with pytest.raises(ValueError):
        verify_manifest(rows)
