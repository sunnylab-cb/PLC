import pytest
import torch

from scripts.infer.infer_v11_internal_ligdep import detect_v11_version
from src.data.pocket_dataset_v8 import _atom_crop_bounds


def test_negative_max_atoms_disables_truncation():
    assert _atom_crop_bounds(513, -1, "alternating_ends") == (0, 513)


def test_zero_max_atoms_is_rejected():
    with pytest.raises(ValueError, match="max_atoms=0"):
        _atom_crop_bounds(12, 0, "head")


def test_version_suffix_preserves_v11m13_identity_without_edge_parameters():
    checkpoint = {
        "model_state_dict": {
            "local54_compat_alpha": torch.tensor(0.0),
        }
    }
    config = {
        "experiment": {
            "name": "pocket_gvp_fusion_seed44_chain_si30",
            "version_suffix": "v11m13",
        }
    }

    assert detect_v11_version(checkpoint, config) == "v11m13"
