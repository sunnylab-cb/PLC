"""KD arithmetic, leakage/alignment guards, RNG compatibility and trainer wiring."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.losses.self_distillation import (
    TeacherCache, add_self_distillation, attach_distillation, object_digest,
    residue_kl, scheduled_weight, tensor_digest, tracked_source_dropout,
)
from src.models.pocket_gvp_fusion_v11m8 import PocketGVPFusionV11m8
from scripts.diagnostics.prepare_self_distill import make_configs, VARIANTS


def fixture_cache():
    cfg = dict(data={"max_seq_len": 3}, model={"n_classes": 3})
    splits = dict(train=["a", "b"], val=["v"], test=["t"])
    expected = dict(split_sha256="split", label_sha256="label", teacher_checkpoint_sha256="teacher")
    batch = dict(pdb_ids=["a", "b"], protein_coords=torch.arange(18).reshape(2, 3, 3).float(),
                 chain_ids=torch.zeros(2, 3, dtype=torch.long), labels=torch.tensor([[0, 2, 0], [0, 1, 2]]),
                 protein_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    rows = []
    for i, key in enumerate(batch["pdb_ids"]):
        valid = batch["protein_mask"][i]
        y = batch["labels"][i][valid]
        rows.append(dict(sample_id=key, logits=torch.randn(len(y), 3), labels=y,
            residue_axis_sha256=tensor_digest(batch["protein_coords"][i][valid], batch["chain_ids"][i][valid], y)))
    payload = dict(format_version=1, split="train", split_sha256="split", label_sha256="label", n_rows=2, rows=rows,
        teacher=dict(checkpoint_sha256="teacher", data_config=cfg["data"], model_config_sha256=object_digest(cfg["model"])))
    return cfg, splits, expected, batch, payload


def test_cache_aligned_batch_and_roundtrip(tmp_path):
    cfg, splits, expected, batch, payload = fixture_cache()
    path = tmp_path / "teacher.pt"
    torch.save(payload, path)
    cache = TeacherCache.load(path, cfg, splits, expected)
    result = cache.aligned_logits(batch, "cpu")
    assert result.shape == (2, 3, 3) and not result.requires_grad
    torch.testing.assert_close(result[0, :2], payload["rows"][0]["logits"])
    assert not result[0, 2].count_nonzero()
    permuted = {k: ["b", "a"] if k == "pdb_ids" else v.flip(0) for k, v in batch.items()}
    torch.testing.assert_close(cache.aligned_logits(permuted, "cpu"), result.flip(0))


@pytest.mark.parametrize("corruption", ["val", "missing", "duplicate", "split", "teacher", "data", "model", "nan", "overlap"])
def test_cache_rejects_invalid_inputs(corruption):
    cfg, splits, expected, _, payload = fixture_cache()
    if corruption == "val":
        payload["split"] = "val"
    elif corruption == "missing":
        payload["rows"].pop()
    elif corruption == "duplicate":
        payload["rows"][1] = payload["rows"][0]
    elif corruption == "split":
        payload["split_sha256"] = "wrong"
    elif corruption == "teacher":
        payload["teacher"]["checkpoint_sha256"] = "wrong"
    elif corruption == "data":
        payload["teacher"]["data_config"] = {"max_seq_len": 2}
    elif corruption == "model":
        payload["teacher"]["model_config_sha256"] = "wrong"
    elif corruption == "nan":
        payload["rows"][0]["logits"][0, 0] = float("nan")
    elif corruption == "overlap":
        splits["val"] = ["a"]
    with pytest.raises(ValueError):
        TeacherCache(payload, cfg, splits, expected)


@pytest.mark.parametrize("field", ["protein_coords", "chain_ids", "labels", "pdb_ids"])
def test_alignment_fails_closed(field):
    cfg, splits, expected, batch, payload = fixture_cache()
    cache = TeacherCache(payload, cfg, splits, expected)
    if field == "pdb_ids":
        batch[field][0] = "v"
    else:
        batch[field][0, 0] += 1
    with pytest.raises(ValueError):
        cache.aligned_logits(batch, "cpu")


def test_kl_formula_detach_padding_dropout_and_class_sums():
    s = torch.randn(3, 4, 3)
    t = torch.randn(3, 4, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    eligible = torch.tensor([True, True, False])
    s[~(mask & eligible[:, None])] = float("nan")
    s.requires_grad_()
    y = torch.tensor([[0, 2, 0, 0], [0, 1, 2, 0], [1, 1, 1, 1]])
    loss, stats = residue_kl(s, t, mask, eligible, 2.0, y)
    expected = sum(F.kl_div((s[i, mask[i]] / 2).log_softmax(-1),
                   (t[i, mask[i]].detach() / 2).softmax(-1), reduction="batchmean") * 4 for i in (0, 1)) / 3
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(sum(stats[f"true_class{c}_contribution"] for c in range(3)), loss.detach())
    loss.backward()
    assert torch.isfinite(s.grad).all() and not s.grad[~(mask & eligible[:, None])].count_nonzero()
    assert t.grad is None and stats["dropped_rows"] == 1


def test_all_dropped_safe_zero_gradient():
    s = torch.full((2, 3, 3), float("nan"), requires_grad=True)
    loss, _ = residue_kl(s, s.detach(), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 2)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(s.grad).all() and not s.grad.count_nonzero()


def test_schedule():
    assert [scheduled_weight(e, .1) for e in range(5)] == [0] * 5
    assert [scheduled_weight(e, .1) for e in range(5, 10)] == pytest.approx([.02, .04, .06, .08, .1])
    assert scheduled_weight(100, .1) == .1


@pytest.mark.parametrize("probs", [(0, 0, 0), (.03, .04, .04), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_dropout_exact_rng_and_arithmetic(probs):
    m = SimpleNamespace(training=True, ligand_source_dropout_p=probs[0],
                        ligand_dropout_hidden_p=probs[1], ligand_dropout_local54_p=probs[2], _sd_dropout_calls=0)
    h, l = torch.randn(200, 3, 4), torch.randn(200, 3, 54)
    torch.manual_seed(44)
    reference = PocketGVPFusionV11m8._apply_source_dropout(m, h, l)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(44)
    tracked = tracked_source_dropout(m, h, l)
    assert torch.equal(rng, torch.get_rng_state()) and m._sd_dropout_calls == 1
    for expected, value in zip(reference, tracked):
        torch.testing.assert_close(expected, value, rtol=0, atol=0)
    expected_keep = reference[0].abs().sum((1, 2)).bool() & reference[1].abs().sum((1, 2)).bool()
    assert torch.equal(expected_keep, m._sd_ligand_keep)


def test_zero_lambda_identity_no_rng_or_cache_access():
    m = torch.nn.Linear(2, 3)
    attach_distillation(m, None, dict(weight=0, warmup_epochs=5, ramp_epochs=5, temperature=2))
    hard = torch.tensor(3., requires_grad=True)
    rng = torch.get_rng_state().clone()
    actual = add_self_distillation(m, None, None, None, None, hard, 20, None, 1, False)
    assert actual is hard and torch.equal(rng, torch.get_rng_state())
    actual.backward()
    assert hard.grad == 1
    assert set(m.state_dict()) == {"weight", "bias"}
    m.eval()
    with pytest.raises(RuntimeError, match="validation/test"):
        add_self_distillation(m, None, None, None, None, hard, 20, None, 1, False)


def test_only_train_loss_changed_not_validation():
    from scripts.train import train_self_distill as sd
    source = sd.trainer.shared._patch_v11m3_source_for_v11m4(
        (Path(__file__).parents[1] / "scripts/train/train_v11m3.py").read_text())
    modified = sd.distillation_epoch_ast(source)
    additions = [n for n in ast.walk(modified) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "add_self_distillation"]
    assert len(additions) == 1

    class RemoveHook(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "add_self_distillation":
                return None
            return node

    original, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "train_epoch"]
    assert ast.dump(RemoveHook().visit(modified).body[0]) == ast.dump(original)
    original_eval, original_train, original_create = sd.trainer.base.evaluate, sd.trainer.base.train_epoch, sd.trainer.base.create_model
    try:
        base = sd.install_training_hook()
        assert base.evaluate is original_eval and base.train_epoch is not original_train
    finally:
        sd.trainer.base.train_epoch, sd.trainer.base.create_model = original_train, original_create


def test_config_only_changes_kd_and_output(tmp_path):
    path = Path(__file__).parents[1] / "tmp/libre_transfer_12964718/baseline/config.yaml"
    if not path.exists():
        pytest.skip("Full config present on the deployment host")
    base = yaml.safe_load(path.read_text())
    before = copy.deepcopy(base)
    configs = make_configs(base, tmp_path, "/tmp/train_only.pt")
    assert set(configs) == set(VARIANTS) and base == before
    for name, cfg in configs.items():
        assert cfg.pop("self_distillation")["weight"] == VARIANTS[name]
        cfg["experiment"] = base["experiment"]
        assert cfg == base


def test_diagnostic_state_digest_handles_scalar_and_bfloat16():
    from scripts.diagnostics.diagnose_self_distill_control import state_hash
    state = {"scalar": torch.tensor(3), "matrix": torch.randn(2, 3).bfloat16()}
    assert state_hash(state) == state_hash({k: v.clone() for k, v in state.items()})
    other = {**state, "scalar": torch.tensor(4)}
    assert state_hash(other) != state_hash(state)


def test_diagnostic_exposes_differences_instead_of_passing():
    from scripts.diagnostics.diagnose_self_distill_control import state_difference
    a = {"p": torch.tensor([1., 2.]), "b": torch.tensor(3)}
    b = {"p": torch.tensor([1., 2.1]), "b": torch.tensor(3)}
    diff = state_difference(a, b)
    assert diff["changed_tensors"] == 1
    assert diff["max_abs"] == pytest.approx(.1)
    assert diff["largest"][0]["name"] == "p"
    assert state_difference(a, a)["max_abs"] == 0


@pytest.mark.parametrize("cache_ready", [False, True])
def test_retry_submission_preserves_smoke_dependency(tmp_path, monkeypatch, cache_ready):
    import sys
    from scripts.diagnostics import submit_self_distill as submit
    checks, commands = [], []
    monkeypatch.setattr(submit, "verify_prepared", lambda root, **kw: checks.append(kw))

    def sbatch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=str(100 + len(commands)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    option = ["--cache-ready"] if cache_ready else ["--cache-job", "99"]
    monkeypatch.setattr(sys, "argv", ["submit", "--experiment", str(tmp_path), *option])
    submit.main()
    assert checks == [{"check_cache": cache_ready}]
    assert any(c == "--dependency=afterok:99" for c in commands[0]) is not cache_ready
    assert "--dependency=afterok:101" in commands[1]
    assert "--array=0-1%2" in commands[1]
    assert all("--kill-on-invalid-dep=yes" in cmd for cmd in commands)


def control_report():
    names = ("base_cold", "base_warm1", "base_warm2", "kd_zero")
    report = dict(complete=True, runs=[], comparisons={})
    for n in names:
        report["runs"].append(dict(name=n, initial_state_sha256="same", start_rng={"cpu": "x", "cuda": "y"},
            end_rng={"cpu": "x", "cuda": "z"}, metrics=dict(loss=1.), hard_loss_identity_checks=int(n == "kd_zero")))
    for i, a in enumerate(names):
        for b in names[i+1:]:
            report["comparisons"][f"{a}_vs_{b}"] = dict(logits=dict(rms=.001, max_abs=.006),
                state=dict(max_abs=.0002), argmax_disagreement=.001)
    return report


def test_repeatability_gate_accepts_matched_base_noise():
    from scripts.diagnostics.diagnose_self_distill_control import check_control_equivalence
    assert check_control_equivalence(control_report(), 1e-4)["passed"]


@pytest.mark.parametrize("corruption", ["initial", "rng", "identity", "extra_drift", "unstable_base"])
def test_repeatability_gate_still_fails_on_real_changes(corruption):
    from scripts.diagnostics.diagnose_self_distill_control import check_control_equivalence
    r = control_report()
    if corruption == "initial":
        r["runs"][-1]["initial_state_sha256"] = "different"
    elif corruption == "rng":
        r["runs"][-1]["end_rng"]["cuda"] = "different"
    elif corruption == "identity":
        r["runs"][-1]["hard_loss_identity_checks"] = 0
    elif corruption == "extra_drift":
        r["comparisons"]["base_warm2_vs_kd_zero"]["logits"]["rms"] = .004
    else:
        for v in r["comparisons"].values():
            v["logits"]["rms"] = .01
    with pytest.raises(RuntimeError):
        check_control_equivalence(r, 1e-4)


def test_dropout_smoke_exact_on_cpu_and_rng_preserved():
    from scripts.diagnostics.smoke_self_distill import check_dropout_arithmetic
    before = torch.get_rng_state().clone()
    assert check_dropout_arithmetic("cpu")["passed"]
    assert torch.equal(before, torch.get_rng_state())


def test_repeatability_loss_floor_handles_bf16_aggregate_roundoff():
    from scripts.diagnostics.diagnose_self_distill_control import check_control_equivalence
    r = control_report()
    for row, loss in zip(r["runs"], [1.092152, 1.092071, 1.092080, 1.091979]):
        row["metrics"]["loss"] = loss
    assert check_control_equivalence(r, 1e-4)["passed"]
    r["runs"][-1]["metrics"]["loss"] = 1.095
    with pytest.raises(RuntimeError, match="loss_abs"):
        check_control_equivalence(r, 1e-4)
