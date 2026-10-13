#!/usr/bin/env python3
"""Slurm-only diagnosis of lambda-zero smoke, including repeated base controls.

Read the existing frozen snapshot through PLC_FROZEN_CODE. Never edit it or
write a smoke-pass artifact. A diagnostic exit 0 does NOT authorize training.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import yaml

CODE = Path(os.environ.get("PLC_FROZEN_CODE", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts/train"))
from scripts.diagnostics.prepare_self_distill import verify_prepared, sha256, VARIANTS


def rng_hashes():
    return {name: hashlib.sha256(t.numpy().tobytes()).hexdigest() for name, t in
            (("cpu", torch.get_rng_state()), ("cuda", torch.cuda.get_rng_state()))}


def state_hash(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_difference(a, b):
    d = (a.double() - b.double()).abs()
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rms=float(d.square().mean().sqrt()), unequal_fraction=float((a != b).float().mean()))


def state_difference(a, b):
    rows = []
    for key in a:
        if not torch.equal(a[key], b[key]):
            rows.append(dict(name=key, **tensor_difference(a[key], b[key])))
    return dict(changed_tensors=len(rows), max_abs=max((r["max_abs"] for r in rows), default=0),
                largest=sorted(rows, key=lambda r: -r["max_abs"])[:15])


def write_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run_controls(cfg, batch, output):
    from scripts.diagnostics.smoke_self_distill import sd, seed, train_steps

    if cfg["self_distillation"]["weight"] != 0:
        raise ValueError("Control audit must use lambda zero")
    original_epoch = sd.trainer.base.train_epoch
    patched = sd.install_training_hook().train_epoch
    report = dict(diagnostic_only=True, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  runs=[], comparisons={}, complete=False)
    states, logits = {}, {}
    for name in ("base_cold", "base_warm1", "base_warm2", "kd_zero"):
        seed()
        start = time.monotonic()
        kd = name == "kd_zero"
        model = (sd.create_model if kd else sd.trainer.create_model)(cfg, torch.device("cuda"))
        row = dict(name=name, initial_state_sha256=state_hash(model.state_dict()), start_rng=rng_hashes())

        def capture(module, args, output):
            logits[name] = output["logits"].detach().float().cpu()[batch["protein_mask"].bool()]

        hook = model.register_forward_hook(capture)
        original_add_loss = sd.trainer.base.add_self_distillation
        identity_calls = []

        def checked_zero_loss(*args, **kwargs):
            value = original_add_loss(*args, **kwargs)
            if value is not args[5]:
                raise RuntimeError("Lambda zero changed the hard-loss graph")
            identity_calls.append(True)
            return value

        try:
            sd.trainer.base.add_self_distillation = checked_zero_loss
            metrics, scalars = train_steps(model, cfg, batch, patched if kd else original_epoch, 1)
        finally:
            sd.trainer.base.add_self_distillation = original_add_loss
        hook.remove()
        row.update(end_rng=rng_hashes(), metrics=metrics, seconds=time.monotonic()-start,
                   gradient_norms={k: v for k, v in scalars.items() if k.startswith("Gradients/")},
                   hard_loss_identity_checks=len(identity_calls))
        states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        report["runs"].append(row)
        for reference in ("base_cold", "base_warm1", "base_warm2"):
            if reference in states and reference != name:
                report["comparisons"][f"{reference}_vs_{name}"] = dict(
                    state=state_difference(states[reference], states[name]),
                    logits=tensor_difference(logits[reference], logits[name]),
                    argmax_disagreement=float((logits[reference].argmax(-1) != logits[name].argmax(-1)).float().mean()))
        write_report(output, report)
        print("CONTROL_DIAGNOSTIC", json.dumps(row), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report["complete"] = True
    write_report(output, report)
    print("CONTROL_DIAGNOSTIC_COMPLETE", json.dumps(report["comparisons"]), flush=True)
    return report


def check_control_equivalence(report, learning_rate):
    """Exact structural/RNG checks plus a capped empirical repeatability band.

    BF16/scatter training is not bitwise deterministic, including base vs base.
    Do not use near-zero Adam parameter updates as an allclose oracle. Require
    KD0 to remain inside twice the measured base-only variation AND hard caps.
    """
    runs = {r["name"]: r for r in report["runs"]}
    names = ("base_cold", "base_warm1", "base_warm2", "kd_zero")
    if not report["complete"] or set(runs) != set(names):
        raise RuntimeError("Incomplete repeated-control audit")
    for field in ("initial_state_sha256", "start_rng", "end_rng"):
        if any(runs[n][field] != runs[names[0]][field] for n in names[1:]):
            raise RuntimeError(f"KD0/control {field} changed")
    if runs["kd_zero"]["hard_loss_identity_checks"] != 1:
        raise RuntimeError("Missing exact hard-loss graph identity check")

    def measurements(a, b):
        value = report["comparisons"][f"{a}_vs_{b}"]
        return dict(logit_rms=value["logits"]["rms"], logit_max=value["logits"]["max_abs"],
                    argmax=value["argmax_disagreement"], state_max=value["state"]["max_abs"],
                    loss_abs=abs(runs[a]["metrics"]["loss"] - runs[b]["metrics"]["loss"]))

    reference = [measurements(names[i], names[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    kd = [measurements(n, "kd_zero") for n in names[:3]]
    # Aggregate loss/argmax are nonlinear in quantized BF16 logits. A few base
    # repeats cannot estimate their noise to FP32-ULP accuracy. Keep small
    # absolute floors there, while the logit/RNG/graph checks remain primary.
    limits = dict(logit_rms=(1e-6, .005), logit_max=(2e-6, .05), argmax=(.001, .01),
                  state_max=(2e-6, 2.1 * learning_rate), loss_abs=(1e-4, .002))
    statistics = {}
    for metric, (floor, absolute_cap) in limits.items():
        noise = max(r[metric] for r in reference)
        observed = max(r[metric] for r in kd)
        bound = min(absolute_cap, 2.0 * noise + floor)
        if noise > absolute_cap or observed > bound:
            raise RuntimeError(f"Unexplained KD0 drift: {metric}, base={noise}, KD0={observed}, bound={bound}")
        statistics[metric] = dict(base_repeat_max=noise, kd_zero_max=observed, bound=bound)
    return dict(passed=True, criterion="exact initial weights/RNG/loss graph; capped measured base-repeatability band",
                statistics=statistics)


def main():
    from scripts.diagnostics.smoke_self_distill import training_batch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    verify_prepared(args.experiment, check_cache=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Submit on a Slurm GPU allocation")
    torch.set_num_threads(4)
    torch.cuda.init()
    cfg = yaml.safe_load((args.experiment / "prepared" / f"{next(iter(VARIANTS))}.yaml").read_text())
    cfg["self_distillation"]["weight"] = 0.0
    report = run_controls(cfg, training_batch(cfg), args.output)
    report["provenance_sha256"] = sha256(args.experiment / "prepared/provenance.json")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
