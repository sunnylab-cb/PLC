import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostics.audit_biolip_sequence_leakage import (
    normalize_identity,
    normalize_sequence,
    quantile,
    sequence_from_residue_names,
)


def test_sequence_normalization_and_modified_residues():
    assert normalize_sequence(["A", "C", ":", "D", "-"]) == "ACD"
    assert sequence_from_residue_names(["ALA", "MSE", "HIE", "UNK"]) == "AMHX"


def test_identity_accepts_fraction_or_percent():
    assert normalize_identity("0.42") == 0.42
    assert normalize_identity("42.0") == 0.42


def test_quantile_handles_empty_and_endpoints():
    assert quantile([], 0.5) is None
    assert quantile([0.1, 0.5, 0.9], 0.0) == 0.1
    assert quantile([0.1, 0.5, 0.9], 1.0) == 0.9
