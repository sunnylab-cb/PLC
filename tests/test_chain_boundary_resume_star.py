"""Resume preserves scientific recipe, restored schedule and epoch logging."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts.train.train_chain_boundary_resume_star import checkpoint_info, resume_config, restore_optimizer
from scripts.train.train_chainwise_esm_complex import install


def checkpoint():
    return dict(epoch=13, model_state_dict={}, optimizer_state_dict=dict(
        state={0: {}}, param_groups=[dict(lr=0.000098)]),
        scheduler_state_dict=dict(last_epoch=14, _last_lr=[0.000098]),
        scaler_state_dict=dict(scale=65536), early_stopping_state=dict(metric="f1_c0", best=0.58, bad_epochs=0),
        best_val_f1_c0=0.58, config={}, val_metrics=dict(f1_c0=0.58))


def test_best_epoch_resume_not_latest_replay():
    info = checkpoint_info(checkpoint())
    assert (info["saved_epoch"], info["resume_epoch"]) == (14, 15)
    assert not info["rng_state_present"]


@pytest.mark.parametrize("change", [
    lambda c: c.pop("optimizer_state_dict"),
    lambda c: c["scheduler_state_dict"].update(last_epoch=0),
    lambda c: c["scheduler_state_dict"].update(_last_lr=[0.1]),
    lambda c: c["early_stopping_state"].update(metric="loss"),
    lambda c: c["early_stopping_state"].update(bad_epochs=3),
    lambda c: c.update(best_val_f1_c0=0.7),
])
def test_incomplete_or_inconsistent_resume_fails(change):
    c = checkpoint()
    change(c)
    with pytest.raises(ValueError):
        checkpoint_info(c)


def base_config():
    return dict(experiment=dict(seed=44, name="original", output_dir="/hpc/out"),
                model=dict(scan_variant="G0_small_global", use_unimol_ligand_input=False),
                training=dict(learning_rate=1e-4, early_stopping_metric="f1_c0"),
                data=dict(nma_max_conf=0, data_root="/hpc/data", esm_features_dir="/hpc/esm",
                          split_file="/hpc/data/splits.txt", distances_file="/hpc/data/distances.pt"),
                scan_metadata_file="/hpc/meta.tsv")


def test_relocation_changes_only_paths_and_experiment(tmp_path):
    base = base_config()
    c = copy.deepcopy(base)
    c["model"].update(gvp_chain_boundary_fix=True, coattn_chain_position_fix=False)
    cfg = resume_config(c, base, tmp_path / "core", tmp_path / "resume", "gvp_boundary")
    assert cfg["model"] == c["model"] and cfg["training"] == c["training"]
    assert cfg["data"]["data_root"] == str(tmp_path / "core/data")
    assert cfg["scan_metadata_file"] == str(tmp_path / "core/prepared/metadata.tsv")
    c["training"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="scientific recipe"):
        resume_config(c, base, tmp_path / "core", tmp_path / "resume", "gvp_boundary")


def test_optimizer_schedule_resume_matches_next_update():
    t = dict(weight_decay=0.01, learning_rate=1e-4, betas=[0.9, 0.999], warmup_epochs=5,
             num_epochs=120, min_lr=1e-6)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW([dict(params=[model.weight], weight_decay=0.01),
                            dict(params=[model.bias], weight_decay=0.)], lr=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=115, eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[5])
    x = torch.ones(3, 2)
    for _ in range(14):
        opt.zero_grad(); model(x).square().sum().backward(); opt.step(); sched.step()
    other = copy.deepcopy(model)
    saved = copy.deepcopy(dict(optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict()))
    restored, rsched = restore_optimizer(other, dict(training=t), saved, torch.optim.AdamW)
    assert restored.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    for m, o, s in ((model, opt, sched), (other, restored, rsched)):
        o.zero_grad(); m(x).square().sum().backward(); o.step(); s.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sched.get_last_lr() == rsched.get_last_lr()


@pytest.mark.parametrize("start", [0, 14, 19])
def test_stratified_val_steps_continue_from_checkpoint(tmp_path, monkeypatch, start):
    class Writer:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.steps = []
        def add_scalar(self, tag, value, step):
            self.steps.append(step)
        def flush(self):
            pass
    module = ModuleType("train_local_spatial_scan")
    module.source_metrics = lambda records, gate: {"pdbbind_count": len(records)}
    module.base = SimpleNamespace(SummaryWriter=Writer)
    monkeypatch.setitem(sys.modules, "train_local_spatial_scan", module)
    old_path = list(sys.path)
    try:
        trainer = install(tmp_path, training_code=tmp_path, start_epoch=start)
        writer = trainer.base.SummaryWriter(tmp_path)
        trainer.source_metrics([({"n_chains": 2}, None, None)], 0.5)
        row = json.loads((tmp_path / "fullcomplex_val_strata.jsonl").read_text())
        assert row["val_pass"] == start + 1
        assert set(writer.steps) == {start}
    finally:
        sys.path[:] = old_path


def test_negative_epoch_offset_fails():
    with pytest.raises(ValueError, match="nonnegative"):
        install(None, start_epoch=-1)
