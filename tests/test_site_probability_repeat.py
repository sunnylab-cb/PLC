import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.diagnose_site_probability_repeat import probability_difference, repeat_summary


def test_near_gate_difference_without_flip():
    row = probability_difference([.454, 0], [.403, 0], [True, False], [True, True], .391)
    assert row['max_abs'] == pytest.approx(.051)
    assert row['flips'] == 0
    assert row['f1'] == pytest.approx(2 / 3)
    assert row['mean_abs_observed'] == pytest.approx(.051)
    assert row['mean_abs_full'] == pytest.approx(.0255)


def test_all_pairwise_not_just_reference_error():
    summary = repeat_summary([np.array([.4]), np.array([.35]), np.array([.45])], [True], [True], .4)
    assert summary['max_pairwise_abs'] == pytest.approx(.1)
    assert summary['max_pairwise_flips'] == 1
    assert summary['max_pairwise_f1_delta'] == 1


def test_unobserved_tail_never_predicted():
    row = probability_difference([.6, 1.], [.6, 0.], [True, False], [True, False], .5)
    assert row['flips'] == 0
    assert row['f1'] == 1


def test_nonfinite_and_mismatch_rejected():
    with pytest.raises(ValueError):
        probability_difference([np.nan], [0.], [True], [True], .5)
    with pytest.raises(ValueError):
        probability_difference([0., 0.], [0.], [True], [True], .5)
    with pytest.raises(ValueError):
        repeat_summary([[.5]], [True], [True], .5)
