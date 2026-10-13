from types import SimpleNamespace

import pytest

from scripts.diagnostics.audit_libre_transfer_logs import summarize_events


def event(step, value, time):
    return SimpleNamespace(step=step, value=value, wall_time=time)


def test_duplicate_logging_is_not_an_extra_epoch():
    result = summarize_events([event(0, 0.4, 1), event(0, 0.4, 2),
                               event(1, 0.6, 3)])
    assert result["n_events"] == 3
    assert result["n_unique_steps"] == 2
    assert result["mean_over_unique_steps"] == pytest.approx(0.5)
    assert result["max_duplicate_value_spread"] == 0


def test_conflict_is_retained_and_last_wall_time_wins():
    result = summarize_events([event(2, 0.8, 3), event(2, 0.5, 1),
                               event(1, 0.4, 2)])
    assert result["series"] == [[1, 0.4], [2, 0.8]]
    assert result["duplicate_steps"] == 1
    assert result["max_duplicate_value_spread"] == pytest.approx(0.3)


def test_empty_series():
    assert summarize_events([]) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_not_silently_averaged(value):
    with pytest.raises(ValueError, match="Non-finite"):
        summarize_events([event(0, value, 1)])
