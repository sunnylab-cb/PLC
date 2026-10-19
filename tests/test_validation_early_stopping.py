import ast
import copy
import math
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.early_stopping import ValidationEarlyStopping


def test_default_monitor_is_f1_and_loss_does_not_reset_patience():
    stop = ValidationEarlyStopping()
    assert stop.update(dict(f1_c0=.5, loss=.8))
    assert not stop.update(dict(f1_c0=.49, loss=.4))
    assert stop.bad_epochs == 1
    assert stop.update(dict(f1_c0=.6, loss=math.nan))
    assert stop.best == .6 and stop.bad_epochs == 0


@pytest.mark.parametrize("value", [.5, .49, math.nan, math.inf, -math.inf])
def test_ties_regressions_and_nonfinite_f1_count_as_no_improvement(value):
    stop = ValidationEarlyStopping()
    stop.update(dict(f1_c0=.5))
    assert not stop.update(dict(f1_c0=value))
    assert stop.best == .5 and stop.bad_epochs == 1


def test_explicit_loss_monitor_remains_available():
    stop = ValidationEarlyStopping("loss")
    assert stop.update(dict(loss=.7))
    assert not stop.update(dict(loss=.8))
    assert stop.update(dict(loss=.6))
    assert stop.best == .6 and stop.bad_epochs == 0


def test_unknown_metric_fails_fast():
    with pytest.raises(ValueError, match="early_stopping_metric"):
        ValidationEarlyStopping("F1_typo")


def test_matching_monitor_restores_complete_patience_state():
    stop = ValidationEarlyStopping()
    stop.update(dict(f1_c0=.6))
    stop.update(dict(f1_c0=.59))
    other = ValidationEarlyStopping()
    assert other.restore(dict(early_stopping_state=stop.state_dict(), patience_counter=20))
    assert other.best == .6 and other.bad_epochs == 1


@pytest.mark.parametrize("state", [None, dict(metric="loss", best=.5, bad_epochs=29)])
def test_legacy_or_switched_monitor_never_reuses_loss_patience(state):
    checkpoint = dict(best_val_f1_c0=.59, val_metrics=dict(f1_c0=.61), patience_counter=29)
    if state is not None:
        checkpoint["early_stopping_state"] = state
    stop = ValidationEarlyStopping()
    assert not stop.restore(checkpoint)
    assert stop.best == .61 and stop.bad_epochs == 0


def trainer_source(patched):
    source = (ROOT / "scripts/train/train_v11m3.py").read_text()
    if patched:
        tree = ast.parse((ROOT / "scripts/train/train_v11m4.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_patch_v11m3_source_for_v11m4")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "patcher", "exec"), namespace)
        source = namespace[function.name](source)
    compile(source, "trainer", "exec")
    return source


@pytest.mark.parametrize("patched", [False, True])
def test_real_checkpoint_block_saves_only_two_bests_with_current_stop_state(patched, tmp_path):
    source = trainer_source(patched)
    block = source.split("            # Update stopping state BEFORE saving", 1)[1]
    block = block.split("            # Early stopping", 1)[0]
    block = textwrap.dedent(block.split("\n", 1)[1])
    saved = {}

    def save(state, path):
        saved[path.name] = copy.deepcopy(state)

    stateful = SimpleNamespace(state_dict=lambda: {"weight": 1})
    env = dict(math=math, torch=SimpleNamespace(save=save), model=stateful,
               optimizer=stateful, scheduler=stateful, world_size=1,
               scaler=None, exp_dir=tmp_path, config={}, train_metrics={},
               print_rank0=lambda *args: None, early_stopping=ValidationEarlyStopping(),
               best_val_loss=math.inf, best_val_f1_c0=-math.inf,
               train_cfg=dict(save_every=120, early_stopping_patience=30))
    metrics = [(.5, .7), (.6, math.nan), (.59, .4), (.61, .8)]
    for epoch, (f1, loss) in enumerate(metrics):
        env.update(epoch=epoch, val_metrics=dict(f1_c0=f1, loss=loss,
                   best_gate_c0=.53, auprc_c0_global=.65, auroc_c0_global=.95))
        exec(block, env)
        latest = saved["checkpoint_latest.pt"]
        assert latest["early_stopping_state"] == env["early_stopping"].state_dict()
        assert latest["patience_counter"] == env["early_stopping"].bad_epochs
        assert latest["best_val_f1_c0"] == env["best_val_f1_c0"]
        if epoch == 1:
            assert saved["best_model.pt"]["epoch"] == 0
            assert saved["best_f1_c0_model.pt"]["epoch"] == 1
        if epoch == 2:
            assert latest["patience_counter"] == 1
            assert saved["best_model.pt"]["epoch"] == 2
    assert set(saved) == {"checkpoint_latest.pt", "best_model.pt", "best_f1_c0_model.pt"}
    assert saved["best_model.pt"]["val_metrics"]["loss"] == .4
    best = saved["best_f1_c0_model.pt"]
    assert best["val_metrics"]["f1_c0"] == .61
    assert best["val_contact_gate"] == .53
    assert best["val_metrics"]["auprc_c0_global"] == .65
    assert best["early_stopping_state"]["bad_epochs"] == 0
