#!/usr/bin/env python3
"""MLP/TCN/BiGRU scratch controls, retaining G0's complete ligand decision path."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent), str(ROOT / "scripts/diagnostics")]
import train_local_spatial_scan as production
from prepare_sequence_context_controls import VARIANTS

base = production.base
_create_model = production.create_model
_gradient_stats = production.compute_gradient_stats


def create_model(config, device, use_gradient_checkpointing=None, use_compile=None):
    c = config["model"]
    variant = c["context_variant"]
    if variant not in VARIANTS or c["protein_context_operator"] != VARIANTS[variant]:
        raise ValueError("Sequence context variant/operator mismatch")
    if c["scan_variant"] != "G0_small_global" or c["d_model"] != 256:
        raise ValueError("Controls must retain the G0 width256 V11m13 architecture")
    model = _create_model(config, device, use_gradient_checkpointing, use_compile)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("All parameters must be trainable")
    print(f"SEQUENCE_CONTEXT_MODEL {variant} parameters={sum(p.numel() for p in model.parameters())} "
          f"front_parameters={sum(p.numel() for p in model.protein_context_encoder.parameters())}", flush=True)
    return model


def compute_gradient_stats(model):
    stats = _gradient_stats(model)
    encoder = getattr(model, "module", model).protein_context_encoder
    squares = [p.grad.detach().float().square().sum() for p in encoder.parameters() if p.grad is not None]
    if squares:
        stats["grad_norm_sequence_context"] = torch.stack(squares).sum().sqrt().item()
    for key, value in getattr(encoder, "_last_context_stats", {}).items():
        stats[key] = value.item()
    return stats


base.create_model = create_model
base.compute_gradient_stats = compute_gradient_stats

if __name__ == "__main__":
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in sys.argv):
        raise ValueError("Scratch control: --resume is not allowed")
    base.main()
