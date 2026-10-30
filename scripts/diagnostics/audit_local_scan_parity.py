#!/usr/bin/env python3
"""Diagnose the failed GPU speed gate without weakening it or training a model.

Compare legacy/legacy and legacy/batched ten-step replays, then compare both
diagnostic implementations on identical tensors. No speed claim or automatic
adoption follows from this audit. Existing training jobs are untouched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
from profile_local_scan_speed import setup, case_config, run_train, trainer
from prepare_local_spatial_scan import verify
from prepare_libre_transfer import sha256, source_hashes
from smoke_libre_transfer import ScalarRecorder, training_batch
from src.data.collate import prepare_model_inputs_v11
from src.utils.batched_training_stats import compute_metrics_batched, local_gradient_report_batched


def rng_digest():
    result = {"cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    if torch.cuda.is_available():
        result["cuda"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    return result


def state_difference(left, right, rtol=1e-5, atol=1e-6):
    if left.keys() != right.keys():
        raise ValueError("State dictionaries have different keys")
    count, total, largest, worst = 0, 0, 0.0, None
    for key in left:
        if not torch.isfinite(left[key]).all() or not torch.isfinite(right[key]).all():
            raise ValueError(f"Nonfinite replay state: {key}")
        diff = (left[key] - right[key]).abs()
        count += int((diff > atol + rtol * right[key].abs()).sum())
        total += diff.numel()
        maximum = float(diff.max()) if diff.numel() else 0.0
        if maximum > largest:
            largest, worst = maximum, key
    return dict(max_abs=largest, worst_tensor=worst, mismatched_elements=count,
                total_elements=total, original_tolerance_pass=count == 0,
                rtol=rtol, atol=atol)


def check_shared_tensors(model, logits, labels, mask):
    """Strictly check metric equivalence and no weight/gradient/RNG mutation."""
    weights = {k: p.detach().clone() for k, p in model.named_parameters()}
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    saved_inputs = [x.detach().clone() for x in (logits, labels, mask)]
    before = rng_digest()
    legacy_metrics = trainer._original_compute_metrics(logits, labels, mask)
    fast_metrics = compute_metrics_batched(logits, labels, mask)
    if legacy_metrics != fast_metrics:
        raise AssertionError("Metric dictionaries differ on identical inputs")
    legacy = trainer.compute_gradient_stats(model)
    total, fast = local_gradient_report_batched(model)
    if legacy.keys() != fast.keys():
        raise AssertionError("Gradient statistic keys differ")
    for key in legacy:
        if not math.isclose(legacy[key], fast[key], rel_tol=2e-6, abs_tol=1e-8):
            raise AssertionError(f"Gradient statistic mismatch on shared tensors: {key}")
    expected_total = sum(g.float().norm(2).item()**2 for g in grads.values())**.5
    if not math.isclose(total, expected_total, rel_tol=2e-6, abs_tol=1e-8):
        raise AssertionError("Total gradient norm differs on shared tensors")
    for key, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), weights[key], rtol=0, atol=0)
        if key in grads:
            torch.testing.assert_close(p.grad, grads[key], rtol=0, atol=0)
        elif p.grad is not None:
            raise AssertionError(f"Diagnostic introduced a gradient: {key}")
    for actual, saved in zip((logits, labels, mask), saved_inputs):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    if before != rng_digest():
        raise AssertionError("Diagnostic consumed RNG state")
    return dict(passed=True, metrics_equal=True, buffers_unchanged=True,
                rng_unchanged=True, total_gradient_norm=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--failed-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm GPU allocation")
    verify(args.reference)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    old = json.loads(args.failed_report.read_text())
    if old["reference_manifest_sha256"] != sha256(args.reference / "prepared/provenance.json"):
        raise ValueError("Failed report belongs to a different reference")
    cfg = yaml.safe_load((args.reference / "prepared/S0_local_gvp.yaml").read_text())
    batch = training_batch(cfg, old["train_batches"][0]["ids"])
    hashes = source_hashes(ROOT)
    report = dict(completed=False, speed_optimization_approved=False, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), source_hashes=hashes,
                  failed_report_sha256=sha256(args.failed_report),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  train_ids=batch["pdb_ids"], replays=[], comparisons={}, shared_tensor_check=None,
                  test_forward_used=False, note="Diagnostic only; original speed parity tolerance is unchanged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    states, initial = {}, None
    save()
    for name, fast in (("legacy_a", False), ("legacy_b", False), ("batched_a", True), ("legacy_c", False)):
        config = case_config(cfg, 128, fast)
        model, opt, scaler = setup(config)
        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if initial is None:
            initial = current
        for key in initial:
            torch.testing.assert_close(current[key], initial[key], rtol=0, atol=0)
        start_rng = rng_digest()
        if report["replays"] and start_rng != report["replays"][0]["start_rng"]:
            raise AssertionError("Replay initialization RNG differs")
        writer = ScalarRecorder()
        metrics = run_train(model, opt, scaler, [batch]*10, config, writer)
        if opt.successful_steps != 10:
            raise RuntimeError("AMP skipped an update")
        states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        report["replays"].append(dict(name=name, batched_stats=fast, metrics=metrics,
            scalars=writer.scalars, start_rng=start_rng, end_rng=rng_digest(), updates=opt.successful_steps))
        for other in states:
            if other != name:
                report["comparisons"][other+"__"+name] = state_difference(states[other], states[name])
        print("PARITY_AUDIT_REPLAY", name, json.dumps(report["comparisons"]), flush=True)
        save()
        del model, opt, scaler, writer, current
        gc.collect()
        torch.cuda.empty_cache()
    model, opt, scaler = setup(case_config(cfg, 128, False))
    model.eval()
    inputs = prepare_model_inputs_v11(batch, torch.device("cuda"))
    out = model(**inputs)
    (out["logits"].square().mean() + out["pred_dist"].mean()).backward()
    report["shared_tensor_check"] = check_shared_tensors(model, out["logits"].detach(),
        batch["labels"].to("cuda"), inputs["protein_mask"])
    report["completed"] = True
    report["legacy_self_replay_pass"] = all(v["original_tolerance_pass"] for k, v in
        report["comparisons"].items() if "batched" not in k)
    if hashes != source_hashes(ROOT):
        raise RuntimeError("Audit source changed during execution")
    verify(args.reference)
    save()
    print("PARITY_AUDIT_COMPLETE", json.dumps({k: v for k, v in report.items()
          if k in ("completed", "shared_tensor_check", "legacy_self_replay_pass", "comparisons")}), flush=True)


if __name__ == "__main__":
    main()
