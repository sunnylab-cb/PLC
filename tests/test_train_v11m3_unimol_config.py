import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train.train_v11m3 import config_needs_unimol


def test_unimol_cache_remains_required_by_default():
    assert config_needs_unimol({})
    assert config_needs_unimol({"use_unimol_ligand_input": True})


def test_zero_unimol_fallback_overrides_all_unimol_consumers():
    config = {
        "allow_zero_unimol_fallback": True,
        "use_unimol_ligand_input": True,
        "use_unimol_memory_kv": True,
        "n_co_attn_blocks": 2,
    }

    assert not config_needs_unimol(config)
