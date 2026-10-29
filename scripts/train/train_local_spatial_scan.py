#!/usr/bin/env python3
"""Scratch LocalScan pilots, reusing the production V11 loss/data/optimizer.

G0 is the width-256 V11m13 control. Source-stratified VAL diagnostics do not
change legacy best-F1 selection. No TEST evaluation or automatic submission.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from functools import wraps
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import train_v11m13 as shared
from src.models.pocket_local_spatial_scan import PocketLocalSpatialScan
from src.metrics.metrics import compute_contact_probability_metrics, compute_binary_metrics_at_threshold
from src.utils.batched_training_stats import compute_metrics_batched

base = shared.base
_original_evaluate = base.evaluate
_original_grad_stats = base.compute_gradient_stats
_original_compute_metrics = base.compute_metrics
_metadata = {}


def create_model(config, device, use_gradient_checkpointing=None, use_compile=None):
    global _metadata
    t = config["training"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("LocalScan pilots must be fully trainable scratch models")
    if use_compile or config["model"].get("use_compile", False):
        raise ValueError("torch.compile is deliberately disabled")
    c = dict(config["model"])
    if use_gradient_checkpointing is not None:
        c["use_gradient_checkpointing"] = use_gradient_checkpointing
    model = (shared.PocketGVPFusionV11m13(c) if c["scan_variant"] == "G0_small_global"
             else PocketLocalSpatialScan(c)).to(device)
    model._batched_local_stats = bool(t.get("batched_local_stats", False))
    if model._batched_local_stats and c["scan_variant"] == "G0_small_global":
        raise ValueError("Batched local statistics are not implemented for G0")
    base.compute_metrics = compute_metrics_batched if model._batched_local_stats else _original_compute_metrics
    if config.get("scan_metadata_file"):
        with open(config["scan_metadata_file"]) as handle:
            _metadata = {r["sample_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    print(f"LOCAL_SCAN_MODEL {c['scan_variant']} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model


def compute_gradient_stats(model):
    stats = _original_grad_stats(model)
    model = getattr(model, "module", model)
    for name in ("spatial_blocks", "spatial_proj", "vector_proj", "local54_proj", "atom_q", "atom_k", "atom_v", "ligand_pool"):
        module = getattr(model, name, None)
        if module is not None:
            squares = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
            if squares:
                stats[f"grad_norm_{name}"] = torch.stack(squares).sum().sqrt().item()
    return stats


def source_metrics(records, gate):
    result = {}
    for source in ("pdbbind", "biolip"):
        subset = [r for r in records if r[0]["source"] == source]
        if not subset:
            continue
        scores, targets = [r[1] for r in subset], [r[2] for r in subset]
        metrics = compute_contact_probability_metrics(scores, targets)
        # One combined VAL-selected gate, not a new gate for each source.
        metrics = {k: v for k, v in metrics.items() if "best_gate" not in k}
        at_gate = compute_binary_metrics_at_threshold(torch.cat(scores), torch.cat(targets), gate)
        metrics.update({f"{k}_c0_at_joint_val_gate_global": at_gate[k] for k in ("f1", "mcc", "precision", "recall")})
        pdbs = defaultdict(list)
        f1s, mccs, top_f1 = [], [], []
        for meta, p, y in subset:
            at_half = compute_binary_metrics_at_threshold(p, y, 0.5)
            f1s.append(at_half["f1"])
            mccs.append(at_half["mcc"])
            pdbs[meta["pdb_id"]].append(at_half["f1"])
            k = min(20, len(p))
            tp = float(y[p.topk(k).indices].sum())
            top_f1.append(2 * tp / max(k + float(y.sum()), 1))
        metrics.update(n_pairs=len(subset), f1_c0_at_0_5_sample_macro=float(np.mean(f1s)),
                       mcc_c0_at_0_5_sample_macro=float(np.mean(mccs)),
                       f1_c0_at_0_5_pdb_balanced=float(np.mean([np.mean(v) for v in pdbs.values()])),
                       top20_f1_c0_sample_macro=float(np.mean(top_f1)))
        result.update({f"{source}_{key}": value for key, value in metrics.items()})
    return result


@wraps(_original_evaluate)
def evaluate(model, dataloader, *args, **kwargs):
    records = []
    current = {}

    class ObservedLoader:
        def __iter__(self):
            for batch in dataloader:
                current["batch"] = batch
                yield batch

        def __len__(self):
            return len(dataloader)

    def collect(module, inputs, outputs):
        batch = current["batch"]
        mask = batch["protein_mask"].bool().cpu()
        logits = outputs["logits"].detach().float().cpu()
        if not torch.isfinite(logits[mask]).all():
            raise RuntimeError("Non-finite valid-residue logits in VAL")
        p = logits.softmax(-1)[..., 0]
        for i, sample_id in enumerate(batch["pdb_ids"]):
            meta = _metadata[sample_id]
            if meta["split"] != "val":
                raise RuntimeError("LocalScan development evaluation is VAL-only")
            records.append((meta, p[i, mask[i]], (batch["labels"][i].cpu()[mask[i]] == 0)))

    if not _metadata:
        raise RuntimeError("Missing source metadata for VAL reporting")
    hook = model.register_forward_hook(collect)
    try:
        result = _original_evaluate(model, ObservedLoader(), *args, **kwargs)
    finally:
        hook.remove()
    result.update(source_metrics(records, result["best_gate_c0"]))
    print("LOCAL_SCAN_VAL", {k: v for k, v in result.items() if k.startswith(("pdbbind_", "biolip_"))}, flush=True)
    return result


base.create_model = create_model
base.compute_gradient_stats = compute_gradient_stats
base.evaluate = evaluate

if __name__ == "__main__":
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in sys.argv):
        raise ValueError("Scratch pilot: --resume is not allowed")
    base.main()
