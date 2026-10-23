#!/usr/bin/env python3
"""Real-data BF16 training/FP16 VAL smoke and 100-step cached-batch profile.

Uses TRAIN/VAL only. Smoke weights are discarded; the full job starts fresh.
Also stress-tests a synthetic cap-sized batch to catch long-protein OOM.
"""
from __future__ import annotations

import argparse
import csv
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
from prepare_local_spatial_scan import VARIANTS, verify
from prepare_libre_transfer import sha256
from smoke_libre_transfer import CheckedAdamW, ScalarRecorder, training_batch
from strict_assignment_common import read_strict_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
import train_local_spatial_scan as trainer


def cap_batch(batch, length=1800):
    """Repeat valid TRAIN rows to stress the unchanged maximum sequence cap."""
    result = dict(batch)
    residue_keys = ("esm_embeddings", "protein_physchem", "protein_spatial_scalar", "protein_spatial_vector",
                    "protein_vectors", "protein_coords", "protein_mask", "chain_ids", "labels", "raw_distances")
    for key in residue_keys:
        if key not in batch:
            continue
        samples = []
        for i in range(len(batch["pdb_ids"])):
            n = int(batch["protein_mask"][i].sum())
            idx = torch.arange(length) % n
            value = batch[key][i, idx].clone()
            if key == "protein_coords":
                value[:, 0] += (torch.arange(length) // n) * 100
            if key == "chain_ids":
                value += (torch.arange(length) // n) * (int(value.max()) + 1)
            samples.append(value)
        result[key] = torch.stack(samples)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    root, name = args.experiment, args.variant
    print("LOCAL_SCAN_SMOKE_PHASE", name, "verify_inputs", flush=True)
    verify(root)
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm GPU allocation is required")
    torch.set_num_threads(4)
    torch.manual_seed(44)
    np.random.seed(44)
    random.seed(44)
    device = torch.device("cuda")
    cfg = yaml.safe_load((root / "prepared" / f"{name}.yaml").read_text())
    with open(cfg["scan_metadata_file"]) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    def balanced(split, n):
        groups = [[r["sample_id"] for r in rows if r["source"] == source and r["split"] == split]
                  for source in ("pdbbind", "biolip")]
        return [groups[i % 2][i // 2] for i in range(n)]
    print("LOCAL_SCAN_SMOKE_PHASE", name, "load_train_batch", flush=True)
    batch = training_batch(cfg, balanced("train", cfg["training"]["batch_size"]))
    model = trainer.create_model(cfg, device)
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Scratch model unexpectedly frozen")
    t = cfg["training"]
    optimizer = CheckedAdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
    reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
    kwargs = {k: v for k, v in t.items() if k in inspect.signature(trainer.base.train_epoch).parameters and k not in reserved}
    kwargs.update(gamma=t["focal_gamma"], scaler=None, log_every=10)
    kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
    kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
    writer = ScalarRecorder()
    def train(batches):
        return trainer.base.train_epoch(model=model, dataloader=batches, optimizer=optimizer,
            device=device, epoch=6, class_weights=torch.tensor(t["class_weights"], device=device), writer=writer, **kwargs)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    print("LOCAL_SCAN_SMOKE_PHASE", name, "profile_train", args.steps, flush=True)
    metrics = train([batch] * args.steps)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not all(math.isfinite(x) for x in metrics.values()):
        raise RuntimeError("Non-finite production TRAIN loss/metrics")
    profile_peak = torch.cuda.max_memory_allocated() / 2**30
    print("LOCAL_SCAN_SMOKE_PHASE", name, "cap_stress", flush=True)
    stress = cap_batch(batch)
    stress_metrics = train([stress])
    if not all(math.isfinite(x) for x in stress_metrics.values()):
        raise RuntimeError("Non-finite cap-sized TRAIN loss/metrics")
    stress_peak = torch.cuda.max_memory_allocated() / 2**30
    del stress
    print("LOCAL_SCAN_SMOKE_PHASE", name, "validation", flush=True)
    ids = balanced("val", 4)
    allowed = inspect.signature(PocketDatasetV8).parameters
    ds = PocketDatasetV8(**{k: v for k, v in cfg["data"].items() if k in allowed},
                         pdb_ids=ids, use_nma=False, split="val", verbose=False)
    if ds.pdb_ids != ids:
        raise RuntimeError("VAL loader dropped smoke rows")
    val_batch = pocket_collate_fn_v10([ds[i] for i in range(len(ds))])
    vm = trainer.base.evaluate(model, [val_batch], class_weights=torch.tensor(t["class_weights"], device=device),
                              gamma=t["focal_gamma"], device=device, use_amp=True)
    # Validation loss can be non-finite under legacy FP16; finite logits/ranking
    # are mandatory. The entrypoint hook enforces this without discarding rows.
    for key in ("f1_c0", "auprc_c0_global", "mcc_c0_at_0_5_global"):
        if not math.isfinite(vm[key]):
            raise RuntimeError(f"Non-finite VAL metric: {key}")
    cls = trainer.shared.PocketGVPFusionV11m13 if name == "G0_small_global" else trainer.PocketLocalSpatialScan
    clone = cls(cfg["model"])
    clone.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    report = dict(passed=True, variant=name, parameters=sum(p.numel() for p in model.parameters()),
        provenance_sha256=sha256(root / "prepared/provenance.json"), gpu=torch.cuda.get_device_name(),
        train_ids=batch["pdb_ids"], val_ids=ids, train_metrics=metrics,
        val_metrics=vm, scalars=writer.scalars, train_shape=list(batch["protein_mask"].shape),
        profile_steps=args.steps, profile_seconds=elapsed, cached_batch_pairs_per_second=len(batch["pdb_ids"]) * args.steps / elapsed,
        peak_allocated_gib=profile_peak, cap_stress_peak_allocated_gib=stress_peak,
        cap_stress_length=1800, test_forward_used=False,
        note="Cached-batch timing excludes data loading; not a full-epoch ETA. Smoke weights discarded.")
    verify(root)
    (root / "smoke").mkdir(exist_ok=True)
    (root / "smoke" / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LOCAL_SCAN_SMOKE_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
