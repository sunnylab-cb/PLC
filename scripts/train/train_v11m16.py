#!/usr/bin/env python
"""Training entrypoint for V11m16 candidate-pocket ranking probes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_v11m15 as shared  # noqa: E402
from src.models.pocket_gvp_fusion_v11m16 import PocketGVPFusionV11m16  # noqa: E402


base = shared.base


def create_model(
    config: Dict,
    device: torch.device,
    use_gradient_checkpointing: bool = None,
    use_compile: bool = None,
) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    if use_gradient_checkpointing is not None:
        model_cfg["use_gradient_checkpointing"] = use_gradient_checkpointing
        base.print_rank0(f"Gradient checkpointing: {use_gradient_checkpointing} (from CLI)")
    else:
        model_cfg.setdefault("use_gradient_checkpointing", False)
        base.print_rank0(
            f"Gradient checkpointing: {model_cfg['use_gradient_checkpointing']} (from config/default)"
        )

    # torch.compile has repeatedly timed out for this project; keep the V11m16
    # entrypoint explicit rather than inheriting a potentially stale config flag.
    if use_compile:
        base.print_rank0("Ignoring --compile for V11m16: torch.compile is disabled by policy")
    model_cfg["use_compile"] = False

    base.print_rank0(
        "Creating V11m16 model "
        f"(candidate_rank_gate={model_cfg.get('use_candidate_rank_gate', False)}, "
        f"sites={model_cfg.get('candidate_num_sites', 4)}, "
        f"candidate_aux_weight={model_cfg.get('candidate_aux_weight', 0.02)}, "
        f"protein_grad_scale={model_cfg.get('local_delta_protein_grad_scale', 1.0)})..."
    )
    model = PocketGVPFusionV11m16(model_cfg)
    shared._load_init_checkpoint(model, config)
    shared._apply_freeze_policy(model, config)

    model = model.to(device)
    n_params = sum(param.numel() for param in model.parameters())
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    base.print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    return model


def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    config.setdefault("experiment", {})["version_suffix"] = "v11m16"
    return shared.setup_experiment(config, rank, resume_path)


base.setup_experiment = setup_experiment
base.create_model = create_model


if __name__ == "__main__":
    base.main()
