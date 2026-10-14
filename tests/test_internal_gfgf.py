import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/diagnostics"))
from prepare_internal_gfgf import make_config


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
