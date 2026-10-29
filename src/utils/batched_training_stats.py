"""Opt-in, single-transfer diagnostics for the LocalScan speed experiment.

Only statistics change. Gradients, clipping, AMP and optimizer state are never
modified here. The legacy batch-macro metric definitions are retained.
"""

from __future__ import annotations

import torch


LOCAL_VARIANTS = ("S0_local_gvp", "S1_cond_gvp", "T1_cond_triangle")
LOCAL_MODULES = ("spatial_blocks", "spatial_proj", "vector_proj", "local54_proj",
                 "atom_q", "atom_k", "atom_v", "ligand_pool")


@torch.no_grad()
def compute_metrics_batched(logits, labels, mask, n_classes=3):
    """Transfer all integer counts once, without GPU boolean indexing/nonzero."""
    preds = logits.argmax(dim=-1)
    valid = mask.bool()
    counts = [valid.sum(), ((preds == labels) & valid).sum()]
    for c in range(n_classes):
        pred, target = preds == c, labels == c
        counts.extend((((pred & target) & valid).sum(),
                       ((pred & ~target) & valid).sum(),
                       ((~pred & target) & valid).sum()))
    n, correct, *counts = torch.stack(counts).cpu().tolist()
    if n == 0:
        return {"accuracy": 0.0}
    metrics = {"accuracy": correct / n}
    for c in range(n_classes):
        tp, fp, fn = counts[3 * c:3 * c + 3]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        metrics.update({f"precision_c{c}": precision, f"recall_c{c}": recall,
                        f"f1_c{c}": 2 * precision * recall / (precision + recall + 1e-8)})
    return metrics


@torch.no_grad()
def local_gradient_report_batched(model):
    """Return the existing total/module norms with one device-to-host transfer.

Total norm retains the explicit nonfinite-gradient -> NaN diagnostic. Base
module norms retain Python sum-of-norm-squares; LocalScan module norms retain
their FP32 sum-of-squares reduction. Missing-gradient keys remain unchanged.
"""
    model = getattr(model, "module", model)
    if getattr(model, "variant", None) not in LOCAL_VARIANTS:
        raise ValueError("Batched local gradient reporting is LocalScan-only")
    parameters = [p for p in model.parameters() if p.grad is not None]
    by_id = {id(p): i for i, p in enumerate(parameters)}
    norms, native_norms, finite, squares = [], [], [], []
    for p in parameters:
        grad = p.grad.detach()
        values = grad.coalesce().values() if grad.is_sparse else grad
        f32 = values.float()
        norms.append(torch.linalg.vector_norm(f32))
        native_norms.append(norms[-1] if values.dtype == torch.float32 else values.norm(2).float())
        finite.append(torch.isfinite(values).all().float())
        squares.append(f32.square().sum())
    local = {}
    for name in LOCAL_MODULES:
        module = getattr(model, name)
        ids = [by_id[id(p)] for p in module.parameters() if id(p) in by_id]
        if ids:
            local[f"grad_norm_{name}"] = torch.stack([squares[i] for i in ids]).sum().sqrt()
    if parameters:
        packed = torch.stack(norms + native_norms + finite + list(local.values())).cpu().tolist()
        n = len(parameters)
        total_norms, per_param, flags = packed[:n], packed[n:2*n], packed[2*n:3*n]
        total = sum(v ** 2 for v in total_norms) ** 0.5 if all(flags) else float("nan")
        local_values = packed[3*n:]
    else:
        total, per_param, local_values = 0.0, [], []
    stats = {}
    for name in ("esm_proj", "ligand_encoder", "classifier"):
        ids = [by_id[id(p)] for p in getattr(model, name).parameters() if id(p) in by_id]
        stats[f"grad_norm_{name}"] = sum(per_param[i] ** 2 for i in ids) ** 0.5
    if stats["grad_norm_esm_proj"] > 0:
        esm = stats["grad_norm_esm_proj"]
        stats.update(grad_ratio_gvp_pathway=0.0 / esm, grad_ratio_esm_pathway=esm / esm)
    stats.update(zip(local, local_values))
    return total, stats
