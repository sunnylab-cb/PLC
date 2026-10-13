#!/usr/bin/env python3
"""Actual B16 production loss/backward, lambda-zero control, checkpoint roundtrip."""

import argparse
import copy
import gc
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/train"))
import train_self_distill as sd
from scripts.diagnostics.smoke_libre_transfer import training_batch, CheckedAdamW, ScalarRecorder
from scripts.diagnostics.prepare_self_distill import VARIANTS, sha256, verify_prepared
from scripts.diagnostics.diagnose_self_distill_control import run_controls, check_control_equivalence
from src.losses.self_distillation import tracked_source_dropout
from src.models.pocket_gvp_fusion_v11m8 import PocketGVPFusionV11m8


def seed():
    torch.manual_seed(44)
    random.seed(44)
    np.random.seed(44)


def train_steps(model, cfg, batch, epoch_function, count, epoch=9):
    t = cfg["training"]
    optimizer = CheckedAdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                             betas=tuple(t["betas"]))
    writer = ScalarRecorder()
    reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
    kwargs = {k: v for k, v in t.items() if k in inspect.signature(epoch_function).parameters and k not in reserved}
    kwargs.update(gamma=t["focal_gamma"], writer=writer, log_every=1, scaler=None)
    kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
    kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
    metrics = epoch_function(model=model, dataloader=[batch] * count, optimizer=optimizer,
        device=torch.device("cuda"), epoch=epoch,
        **{**kwargs, "class_weights": torch.tensor(t["class_weights"], device="cuda")})
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("Nonfinite smoke metrics")
    del optimizer
    return metrics, writer.scalars


def release():
    gc.collect()
    torch.cuda.empty_cache()


def check_dropout_arithmetic(device):
    """Exact GPU outputs/draws for the only model-forward instrumentation."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    state = torch.cuda.get_rng_state if device.type == "cuda" else torch.get_rng_state
    with torch.random.fork_rng(devices=devices):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            m = SimpleNamespace(training=True, ligand_source_dropout_p=.03,
                                ligand_dropout_hidden_p=.04, ligand_dropout_local54_p=.04, _sd_dropout_calls=0)
            hidden = torch.ones(128, 3, 8, device=device, dtype=dtype)
            local = torch.ones(128, 3, 54, device=device, dtype=dtype)
            torch.manual_seed(44)
            expected = PocketGVPFusionV11m8._apply_source_dropout(m, hidden, local)
            after = state()
            torch.manual_seed(44)
            actual = tracked_source_dropout(m, hidden, local)
            if not torch.equal(after, state()):
                raise RuntimeError("Tracked dropout changed RNG draws")
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            keep = expected[0].abs().sum((1, 2)).bool() & expected[1].abs().sum((1, 2)).bool()
            if not torch.equal(keep, m._sd_ligand_keep):
                raise RuntimeError("Incorrect ligand-dropout eligibility")
    return dict(passed=True, dtypes=["fp32", "fp16", "bf16"], outputs="bitwise", rng="exact")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    args = ap.parse_args()
    experiment = args.experiment
    manifest = verify_prepared(experiment, check_cache=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm GPU allocation")
    torch.set_num_threads(4)
    configs = {n: yaml.safe_load((experiment / "prepared" / f"{n}.yaml").read_text()) for n in VARIANTS}
    first = next(iter(configs.values()))
    batch = training_batch(first)
    device = torch.device("cuda")
    report = dict(passed=False, provenance_sha256=sha256(experiment / "prepared/provenance.json"),
                  teacher_cache_sha256=sha256(manifest["teacher_cache"]), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, train_pairs=batch["pdb_ids"], runs=[])
    report["dropout_arithmetic"] = check_dropout_arithmetic(device)
    zero_cfg = copy.deepcopy(first)
    zero_cfg["self_distillation"]["weight"] = 0.0
    control = run_controls(zero_cfg, batch, experiment / "smoke_control.json")
    report["lambda_zero_equivalence"] = check_control_equivalence(control, first["training"]["learning_rate"])
    base = sd.trainer.base
    print("LAMBDA_ZERO_EQUIVALENCE_PASSED", flush=True)
    for name, cfg in configs.items():
        seed()
        model = sd.create_model(cfg, device)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        metrics, scalars = train_steps(model, cfg, batch, base.train_epoch, 3)
        if min(scalars.get("KD/weighted_loss", [0])) <= 0 or min(scalars.get("KD/eligible_rows", [0])) <= 0:
            raise RuntimeError("KD disconnected or all rows excluded")
        if scalars["KD/lambda"] != [cfg["self_distillation"]["weight"]] * 3:
            raise RuntimeError("Unexpected ramp schedule")
        if sum(p.numel() for p in model.parameters()) != 84784118:
            raise RuntimeError("Architecture/parameter count changed")
        model.eval()
        from src.data.collate import prepare_model_inputs_v11
        inputs = prepare_model_inputs_v11(batch, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs)
        if not torch.isfinite(outputs["logits"][inputs["protein_mask"].bool()]).all():
            raise RuntimeError("Nonfinite FP16 evaluation logits")
        if model._sd_ligand_keep is not None:
            raise RuntimeError("Training dropout bookkeeping leaked into evaluation")
        checkpoint = experiment / "smoke_roundtrip.pt"
        torch.save({"config": cfg, "model_state_dict": model.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        clone = sd.trainer.PocketGVPFusionV11m13(saved["config"]["model"])
        clone.load_state_dict(saved["model_state_dict"], strict=True)
        if any(k.startswith("_sd") for k in saved["model_state_dict"]):
            raise RuntimeError("Teacher cache must not be stored in student model weights")
        del clone, saved
        checkpoint.unlink()
        report["runs"].append(dict(name=name, metrics=metrics, scalars=scalars,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, seconds=time.monotonic()-start))
        print("SELF_DISTILL_ARM_SMOKE_PASSED", name, json.dumps(report["runs"][-1]), flush=True)
        del model, outputs, inputs
        release()
    verify_prepared(experiment)
    if sha256(manifest["teacher_cache"]) != report["teacher_cache_sha256"]:
        raise RuntimeError("Teacher cache changed during smoke")
    report["passed"] = True
    tmp = experiment / "smoke_pass.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(experiment / "smoke_pass.json")
    print("SELF_DISTILL_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
