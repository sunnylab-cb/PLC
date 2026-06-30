#!/usr/bin/env python
"""Training entrypoint for V13m0 residue-atom compatibility bias probes."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_v11m4 as shared  # noqa: E402
from src.models.pocket_gvp_fusion_v13m0 import PocketGVPFusionV13m0  # noqa: E402


base = shared.base


def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    if rank != 0:
        return None
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            exp_dir = resume_path.parent
            base.print_rank0(f"Resuming in original directory: {exp_dir}")
            return exp_dir

    version_tokens = {
        "v6", "v7", "v7m1", "v7m2", "v7m3", "v8", "v9", "v9m2",
        "v10", "v10t", "v10m3", "v10m4", "v10m5", "v10m6", "v10m7", "v10m8",
        "v11", "v11m1", "v11m2", "v11m3", "v11m4", "v11m4t",
        "v11m5", "v11m5t", "v11m6", "v11m6n", "v11m7", "v11m8", "v11m8x",
        "v11m9", "v11m10", "v11m11", "v11m12", "v11m13", "v11m14",
        "v11m15", "v13m0",
        "v13m1", "v13m2",
    }
    raw_exp_name = config["experiment"]["name"]
    version_suffix = config.get("experiment", {}).get("version_suffix", "v13m0")
    exp_tokens = [token for token in raw_exp_name.split("_") if token]
    normalized_tokens = [token for token in exp_tokens if token not in version_tokens]
    exp_name = "_".join(normalized_tokens + [version_suffix])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config["experiment"]["output_dir"]) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    base.print_rank0(f"Created new experiment directory: {exp_dir}")
    return exp_dir


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
        "esm_mlc_",
        "ligand_context_prior",
        "pocket_",
        "center_aux_",
        "cross_attention_layers.",
        "residue_atom_bias_",
        "residue_atom_triangle_",
        "ligand_slot_",
        "local54_q",
        "local54_k",
        "residue_chem_value_enrich_",
    )
    bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    if unexpected or bad_missing:
        raise RuntimeError(
            "Unexpected checkpoint mismatch when initializing V13m0: "
            f"missing={bad_missing[:20]}, unexpected={list(unexpected)[:20]}"
        )
    base.print_rank0(
        f"Initialized V13m0 from {path}; loaded={len(compatible_state)} tensors, "
        f"new/missing={len(missing)}, shape_skipped={len(skipped_shape)}"
    )
    if skipped_shape:
        base.print_rank0(f"Shape-skipped tensors: {skipped_shape[:8]}")


def _apply_freeze_policy(model: torch.nn.Module, config: Dict) -> None:
    train_cfg = config.get("training", {})
    if not bool(train_cfg.get("freeze_v11_base", False)):
        return
    trainable_tokens = tuple(train_cfg.get(
        "delta_trainable_tokens",
        ["residue_atom_bias_"],
    ))
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
        base.print_rank0(f"Gradient checkpointing: {model_cfg['use_gradient_checkpointing']} (from config)")
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
        "Creating V13m0 model "
        f"(res_atom_bias={model_cfg.get('use_residue_atom_bias', False)}, "
        f"triangle_bias={model_cfg.get('use_residue_atom_triangle_bias', False)}, "
        f"slots={model_cfg.get('use_ligand_slots_delta', False)}, "
        f"qk_reparam={model_cfg.get('use_local54_qk_reparam', False)}, "
        f"value_enrich={model_cfg.get('use_residue_chem_value_enrich', False)}, "
        f"coattn={model_cfg.get('use_v12_coattention_block', False)}, "
        f"l54_kv={model_cfg.get('use_local54_interaction_kv', False)}, "
        f"l54_compat={model_cfg.get('use_local54_residue_chem_compat', False)}, "
        f"protein_grad_scale={model_cfg.get('local_delta_protein_grad_scale', 1.0)})..."
    )
    model = PocketGVPFusionV13m0(model_cfg)
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
