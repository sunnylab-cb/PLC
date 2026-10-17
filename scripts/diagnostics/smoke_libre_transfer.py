#!/usr/bin/env python3
"""Slurm GPU smoke: actual patched trainer/loss, BF16 backward, FP16 validation."""

import argparse
import gc
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/train"))
import train_v11m13 as trainer
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
from scripts.diagnostics.prepare_libre_transfer import VARIANTS, sha256, verify_prepared
from scripts.diagnostics.strict_assignment_common import read_strict_splits


class ScalarRecorder:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite smoke scalar: {tag}={value}")
        self.scalars.setdefault(tag, []).append(value)


class CheckedAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("Non-finite training gradient")
        return super().step(closure)


def training_batch(cfg, ids=None):
    data = cfg["data"]
    n = cfg["training"]["batch_size"]
    train_ids = read_strict_splits(data["split_file"])["train"]
    ids = train_ids[:n] if ids is None else ids
    if len(ids) != n or len(set(ids)) != n or not set(ids).issubset(train_ids):
        raise ValueError("Smoke batch must contain distinct TRAIN IDs only")
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in data.items() if k in allowed}
    kwargs.update(pdb_ids=ids, use_nma=False, split="train", verbose=False)
    dataset = PocketDatasetV8(**kwargs)
    if len(dataset) != n:
        raise RuntimeError(f"Expected {n} aligned TRAIN smoke pairs, got {len(dataset)}")
    return pocket_collate_fn_v10([dataset[i] for i in range(n)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    args = p.parse_args()
    experiment = args.experiment
    manifest = verify_prepared(experiment)
    if not torch.cuda.is_available():
        raise RuntimeError("Submit this smoke through a Slurm GPU allocation")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    configs = {n: yaml.safe_load((experiment / "prepared" / f"{n}.yaml").read_text()) for n in VARIANTS}
    batch = training_batch(next(iter(configs.values())))
    inputs = prepare_model_inputs_v11(batch, device)
    valid = inputs["protein_mask"].bool()
    report = dict(passed=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  provenance_sha256=sha256(experiment / "prepared/provenance.json"),
                  batch_shape=list(valid.shape), train_pairs=batch["pdb_ids"], runs=[])
    base_cfg = yaml.safe_load(Path(next(iter(manifest["asset_hashes"]))).read_text())
    base = trainer.PocketGVPFusionV11m13(base_cfg["model"])
    base_parameters = sum(p.numel() for p in base.parameters())
    del base
    for name, cfg in configs.items():
        torch.manual_seed(44); random.seed(44); np.random.seed(44)
        model = trainer.create_model(cfg, device)
        t = cfg["training"]
        optimizer = CheckedAdamW(model.parameters(), lr=t["learning_rate"],
                                 weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
        writer = ScalarRecorder()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Use the real patched epoch function, not an easier surrogate CE loss.
        reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
        kwargs = {k: v for k, v in t.items()
                  if k in inspect.signature(trainer.base.train_epoch).parameters and k not in reserved}
        kwargs.update(gamma=t["focal_gamma"], writer=writer, log_every=1, scaler=None)
        kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
        kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
        metrics = trainer.base.train_epoch(
            model=model, dataloader=[batch] * 3, optimizer=optimizer,
            device=device, epoch=6,
            **{**kwargs, "class_weights": torch.tensor(t["class_weights"], device=device)},
        )
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"Non-finite metrics: {name}")
        if cfg["model"].get("conditional_readout_mode", "none") != "none":
            if max(writer.scalars.get("Gradients/conditional_readout_norm", [0])) <= 0:
                raise RuntimeError("Disconnected conditional readout")
        if name == "plc_residue_classmass":
            if min(writer.scalars.get("ResidueContrast/eligible_anchors", [0])) <= 0:
                raise RuntimeError("Classmass loss was not called by the production trainer")
            if max(writer.scalars["ResidueContrast/weighted_loss"]) <= 0:
                raise RuntimeError("Classmass smoke has no negative-class supervision")
        elif any(k.startswith("ResidueContrast/") for k in writer.scalars):
            raise RuntimeError("Unexpected classmass objective in another arm")
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs)
        if not torch.isfinite(outputs["logits"][valid]).all() or "residue_classmass_hidden" in outputs:
            raise RuntimeError("Invalid validation output or training-only hidden leaked to eval")
        if model._cached_classmass_hidden is not None:
            raise RuntimeError("Training graph retained on model")
        # Checkpoint reconstruction must be strict, including both widened heads.
        checkpoint = experiment / "smoke_roundtrip.pt"
        torch.save({"config": cfg, "model_state_dict": model.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        clone = trainer.PocketGVPFusionV11m13(saved["config"]["model"])
        clone.load_state_dict(saved["model_state_dict"], strict=True)
        del clone, saved
        checkpoint.unlink()
        torch.cuda.synchronize()
        n_params = sum(p.numel() for p in model.parameters())
        expected_extra = 82304 if "readout" in name else 0
        if n_params - base_parameters != expected_extra:
            raise RuntimeError(f"Unexpected parameter count for {name}: {n_params - base_parameters}")
        report["runs"].append(dict(name=name, parameters=n_params, extra_parameters=n_params-base_parameters,
                                   metrics=metrics, scalars=writer.scalars,
                                   seconds=time.perf_counter()-start,
                                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print("ARM_SMOKE_PASSED", name, json.dumps(report["runs"][-1]), flush=True)
        del outputs, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    verify_prepared(experiment)
    report["passed"] = True
    (experiment / "smoke_pass.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LIBRE_TRANSFER_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
