#!/usr/bin/env python
"""Training entrypoint for V11m14 base-conditioned local chemistry token probes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_v11m13 as previous  # noqa: E402
from src.models.pocket_gvp_fusion_v11m14 import PocketGVPFusionV11m14  # noqa: E402


base = previous.base
setup_experiment = previous.setup_experiment


def _load_init_checkpoint(model: torch.nn.Module, config: Dict) -> None:
    train_cfg = config.get("training", {})
    init_path = train_cfg.get("init_from_checkpoint") or train_cfg.get("init_from_v11m3")
    if not init_path:
        return

    path = Path(init_path)
    if not path.exists():
        raise FileNotFoundError(f"training.init_from_checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    compatible_state = {}
    skipped_shape = []
    for key, value in state.items():
        if key not in current:
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    allowed_missing = (
        "local_delta_",
        "global_delta_gate.",
        "local_ligand_film",
        "local54_chem_",
        "local54_anchor_proj.",
        "local54_group_proj.",
        "local54_interaction_",
        "local54_compat_",
        "local54_edge_context_",
        "local54_basecond_",
        "ligand_context_prior",
        "pocket_",
        "center_aux_",
        "cross_attention_layers.",
    )
    bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    if unexpected or bad_missing:
        raise RuntimeError(
            "Unexpected checkpoint mismatch when initializing V11m14: "
            f"missing={bad_missing[:20]}, unexpected={list(unexpected)[:20]}"
        )
    base.print_rank0(
        f"Initialized V11m14 from {path}; loaded={len(compatible_state)} tensors, "
        f"new/missing={len(missing)}, shape_skipped={len(skipped_shape)}"
    )
    if skipped_shape:
        base.print_rank0(f"Shape-skipped tensors: {skipped_shape[:8]}")


def _apply_freeze_policy(model: torch.nn.Module, config: Dict) -> None:
    train_cfg = config.get("training", {})
    if not bool(train_cfg.get("freeze_v11_base", False)):
        return

    trainable_tokens = tuple(
        train_cfg.get(
            "delta_trainable_tokens",
            [
                "local54_chem_",
                "local54_anchor_proj.",
                "local54_group_proj.",
                "local54_interaction_",
                "local54_compat_",
                "local54_edge_context_",
                "local54_basecond_",
                "ligand_context_prior",
                "pocket_",
                "center_aux_",
                "cross_attention_layers.",
            ],
        )
    )
    n_frozen = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        keep = any(tok in name for tok in trainable_tokens)
        param.requires_grad = keep
        if keep:
            n_trainable += param.numel()
        else:
            n_frozen += param.numel()
    base.print_rank0(f"Freeze policy: frozen={n_frozen:,}, trainable={n_trainable:,}")


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
    elif "use_gradient_checkpointing" in model_cfg:
        base.print_rank0(
            f"Gradient checkpointing: {model_cfg['use_gradient_checkpointing']} (from config)"
        )
    else:
        model_cfg["use_gradient_checkpointing"] = False
        base.print_rank0("Gradient checkpointing: False (default)")

    if use_compile is not None:
        enable_compile = use_compile
        base.print_rank0(f"Torch.compile: {enable_compile} (from CLI)")
    elif "use_compile" in model_cfg:
        enable_compile = model_cfg["use_compile"]
        base.print_rank0(f"Torch.compile: {enable_compile} (from config)")
    else:
        enable_compile = False
        base.print_rank0("Torch.compile: False (default)")

    base.print_rank0(
        "Creating V11m14 model "
        f"(basecond={model_cfg.get('use_local54_basecond_token', False)}, "
        f"l54_kv={model_cfg.get('use_local54_interaction_kv', False)}, "
        f"l54_compat={model_cfg.get('use_local54_residue_chem_compat', False)}, "
        f"coattn={model_cfg.get('use_v12_coattention_block', False)}, "
        f"protein_grad_scale={model_cfg.get('local_delta_protein_grad_scale', 1.0)})..."
    )
    model = PocketGVPFusionV11m14(model_cfg)
    _load_init_checkpoint(model, config)
    _apply_freeze_policy(model, config)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base.print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    if enable_compile:
        base.print_rank0("Compiling model with torch.compile (dynamic=True, mode='reduce-overhead')...")
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        base.print_rank0("Model compiled successfully")
    return model


base.setup_experiment = setup_experiment
base.create_model = create_model


if __name__ == "__main__":
    base.main()
