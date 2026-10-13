#!/usr/bin/env python3
"""Slurm GPU smoke + TRAIN-only geometry-mask QC for original-model noise."""

import argparse
from dataclasses import replace
import gc
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/train"))
import train_context_noise as noise_trainer
from scripts.diagnostics.prepare_context_noise import VARIANTS, sha256, verify_prepared
from scripts.diagnostics.smoke_libre_transfer import CheckedAdamW, ScalarRecorder, training_batch
from scripts.diagnostics.strict_assignment_common import read_strict_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
from src.models.protein_context_noise import ContextNoiseOptions, ProteinContextNoise, geometry_protection


def mask_audit(cfg):
    data = cfg["data"]
    ids = read_strict_splits(data["split_file"])["train"]
    selected = [ids[i] for i in np.linspace(0, len(ids) - 1, min(128, len(ids)), dtype=int)]
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in data.items() if k in allowed}
    kwargs.update(pdb_ids=selected, use_nma=False, split="train", verbose=False)
    ds = PocketDatasetV8(**kwargs)
    if len(ds) != len(selected):
        raise RuntimeError("Missing rows in geometry TRAIN audit")
    options = ContextNoiseOptions(**cfg["context_noise"])
    totals = {str(c): dict(count=0, protected=0, protection_sum=0.) for c in range(3)}
    rows = []
    for start in range(0, len(ds), 8):
        batch = pocket_collate_fn_v10([ds[i] for i in range(start, min(start + 8, len(ds)))])
        inputs = prepare_model_inputs_v11(batch, torch.device("cpu"))
        mask, labels = inputs["protein_mask"].bool(), batch["labels"]
        protection = geometry_protection(inputs["protein_spatial_scalar"], inputs["protein_coords"], mask, options)
        for b, key in enumerate(batch["pdb_ids"]):
            values = protection[b, mask[b]]
            rows.append(dict(sample=key, residues=len(values), mean=values.mean().item(),
                             std=values.std(unbiased=False).item(), protected_fraction=(values >= .5).float().mean().item()))
        for c in range(3):
            values = protection[mask & (labels == c)]
            totals[str(c)]["count"] += len(values)
            totals[str(c)]["protected"] += (values >= .5).sum().item()
            totals[str(c)]["protection_sum"] += values.sum().item()
    for value in totals.values():
        value["mean_protection"] = value["protection_sum"] / max(1, value["count"])
        value["coverage_at_half"] = value["protected"] / max(1, value["count"])
    nonconstant = sum(row["std"] > .01 for row in rows)
    # This is a soft geometry heuristic, not a GT oracle or a pocket detector claim.
    passed = len(rows) == len(selected) and nonconstant >= len(rows) * .5 and totals["0"]["count"] > 0
    return dict(passed=passed, train_only=True, n_rows=len(rows), nonconstant_rows=nonconstant,
                classes=totals, rows=rows, label_usage="post-hoc QC only; never candidate construction")


def max_probability_difference(a, b, mask):
    return (a.float().softmax(-1)[mask] - b.float().softmax(-1)[mask]).abs().max().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, type=Path)
    args = ap.parse_args()
    root = args.experiment
    verify_prepared(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Run through a Slurm GPU allocation")
    torch.set_num_threads(4)
    trainer = noise_trainer.install_training_hook()
    device = torch.device("cuda")
    configs = {n: yaml.safe_load((root / "prepared" / f"{n}.yaml").read_text()) for n in VARIANTS}
    cfg0 = next(iter(configs.values()))
    report = dict(passed=False, provenance_sha256=sha256(root / "prepared/provenance.json"),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  teacher_used=False, mask_qc=mask_audit(cfg0), runs=[])
    (root / "mask_qc.json").write_text(json.dumps(report["mask_qc"], indent=2) + "\n")
    print("MASK_QC", json.dumps({k: v for k, v in report["mask_qc"].items() if k != "rows"}), flush=True)
    if not report["mask_qc"]["passed"]:
        raise RuntimeError("TRAIN geometry masks are missing or degenerate")
    batch = training_batch(cfg0)
    inputs = prepare_model_inputs_v11(batch, device)
    valid = inputs["protein_mask"].bool()
    report["train_pairs"] = batch["pdb_ids"]
    report["batch_shape"] = list(valid.shape)
    report["zero_noise_equivalence"] = True
    report["clean_eval_equivalence"] = True
    for name, cfg in configs.items():
        torch.manual_seed(44); random.seed(44); np.random.seed(44)
        model = noise_trainer.create_model(cfg, device)
        controller = model._protein_context_noise
        # Force all smoke steps to exercise noise, not just the 25% production rate.
        controller.options = replace(controller.options, batch_probability=1.)
        original_parameters = sum(p.numel() for p in model.parameters())
        t = cfg["training"]
        optimizer = CheckedAdamW(model.parameters(), lr=t["learning_rate"],
                                 weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
        writer = ScalarRecorder()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
        kwargs = {k: v for k, v in t.items() if k in inspect.signature(trainer.train_epoch).parameters and k not in reserved}
        kwargs.update(gamma=t["focal_gamma"], writer=writer, log_every=1, scaler=None)
        kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
        kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
        metrics = trainer.train_epoch(model=model, dataloader=[batch] * 3, optimizer=optimizer,
            device=device, epoch=6,
            **{**kwargs, "class_weights": torch.tensor(t["class_weights"], device=device)})
        if not all(math.isfinite(v) for v in metrics.values()) or controller.applied != 3 or controller.calls != 3:
            raise RuntimeError("Noise not exercised exactly once per TRAIN forward")
        if min(writer.scalars.get("ContextNoise/relative_noise_rms", [0])) <= 0:
            raise RuntimeError("No noise logged in production trainer")
        if any(tag.startswith(("KD/", "ResidueContrast/")) for tag in writer.scalars):
            raise RuntimeError("Unexpected auxiliary objective in original-model noise trial")
        if max(writer.scalars.get("Gradients/context_noise_protein_context_encoder_norm", [0])) <= 0:
            raise RuntimeError("No protein-context gradients")
        # Inference ignores the controller; original class loads state strictly.
        model.eval()
        before = controller.calls
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            actual = model(**inputs)["logits"]
            del model._protein_context_noise
            expected = model(**inputs)["logits"]
        difference = max_probability_difference(actual, expected, valid)
        if controller.calls != before or not torch.isfinite(actual[valid]).all() or difference > .01:
            raise RuntimeError(f"Clean inference mismatch beyond existing BF16/scatter repeat budget: {difference}")
        clone = noise_trainer.trainer.PocketGVPFusionV11m13(cfg["model"])
        clone.load_state_dict(model.state_dict(), strict=True)
        if sum(p.numel() for p in clone.parameters()) != original_parameters:
            raise RuntimeError("Noise unexpectedly changed model parameters")
        del clone
        # Zero sigma must return the exact hidden object, including RNG state.
        zero = ProteinContextNoise(replace(controller.options, sigma=0.))
        h = torch.randn((*valid.shape, 8), device=device)
        state = torch.cuda.get_rng_state().clone()
        assert zero(h, inputs["protein_spatial_scalar"], inputs["protein_coords"], valid) is h
        assert torch.equal(state, torch.cuda.get_rng_state()) and not zero.generators
        torch.cuda.synchronize()
        row = dict(name=name, parameters=original_parameters, extra_parameters=0,
                   metrics=metrics, scalars=writer.scalars, clean_eval_max_probability_diff=difference,
                   forced_smoke_probability=1., production_probability=.25,
                   seconds=time.perf_counter()-start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        report["runs"].append(row)
        print("NOISE_ARM_SMOKE_PASSED", json.dumps(row), flush=True)
        del actual, expected, model, optimizer, h, controller
        gc.collect(); torch.cuda.empty_cache()
    verify_prepared(root)
    report["passed"] = True
    (root / "smoke_pass.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CONTEXT_NOISE_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
