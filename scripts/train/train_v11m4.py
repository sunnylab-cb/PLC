#!/usr/bin/env python3
"""
Training entrypoint for Pocket-GVP-Fusion V11m4.

This reuses the mature V11m3 training loop and swaps only:
- model class: PocketGVPFusionV11m4
- experiment suffix: v11m4
- optional initialization from a V11m3 checkpoint via
  training.init_from_checkpoint
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch
import yaml
# ROCm/AIAA local environment can segfault if TensorBoard/TensorFlow is loaded
# before Triton.  Preloading Triton keeps TensorBoard enabled without fallback.
import triton  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pocket_gvp_fusion_v11m4 import PocketGVPFusionV11m4  # noqa: E402


def _load_v11m3_train_module() -> ModuleType:
    """
    Execute train_v11m3.py and reuse its training loop.

    SummaryWriter is imported normally above; TensorFlow CPU feature guard
    messages from TensorBoard are harmless.
    """
    src_path = Path(__file__).with_name("train_v11m3.py")
    source = src_path.read_text()
    source = source.replace(
        "from torch.utils.tensorboard import SummaryWriter",
        "# SummaryWriter injected by train_v11m4.py",
    )
    module = ModuleType("train_v11m3_sanitized")
    module.__dict__["__file__"] = str(src_path)
    module.__dict__["SummaryWriter"] = SummaryWriter
    exec(compile(source, str(src_path), "exec"), module.__dict__)
    return module


base = _load_v11m3_train_module()


def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    """V11m4 experiment directory setup; otherwise identical to V11m3."""
    if rank != 0:
        return None

    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            exp_dir = resume_path.parent
            base.print_rank0(f"Resuming in original directory: {exp_dir}")
            return exp_dir

    raw_exp_name = config["experiment"]["name"]
    version_tokens = {
        "v6", "v7", "v7m1", "v7m2", "v7m3", "v8", "v9", "v9m2",
        "v10", "v10t", "v10m3", "v10m4", "v10m5", "v10m6", "v10m7", "v10m8",
        "v11", "v11m1", "v11m2", "v11m3", "v11m4",
    }
    exp_tokens = [token for token in raw_exp_name.split("_") if token]
    normalized_tokens = [token for token in exp_tokens if token not in version_tokens]
    exp_name = "_".join(normalized_tokens + ["v11m4"])
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
        raise FileNotFoundError(
            f"training.init_from_checkpoint not found: {path}. "
            "V11m4 should normally start from a trained V11m3 checkpoint."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    expected_missing_prefixes = (
        "pair_delta_",
    )
    unexpected_nonempty = list(unexpected)
    missing_unexpected = [
        k for k in missing if not k.startswith(expected_missing_prefixes)
    ]
    if unexpected_nonempty or missing_unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch when initializing V11m4 from V11m3: "
            f"missing={missing_unexpected[:20]}, unexpected={unexpected_nonempty[:20]}"
        )

    base.print_rank0(
        f"Initialized V11m4 from {path}; new delta params={len(missing)} missing keys"
    )


def _apply_freeze_policy(model: torch.nn.Module, config: Dict) -> None:
    train_cfg = config.get("training", {})
    if not bool(train_cfg.get("freeze_v11m3_base", False)):
        return

    trainable_prefixes = tuple(train_cfg.get("delta_trainable_prefixes", ["pair_delta_"]))
    n_frozen = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        keep = name.startswith(trainable_prefixes)
        param.requires_grad = keep
        if keep:
            n_trainable += param.numel()
        else:
            n_frozen += param.numel()
    base.print_rank0(
        f"Freeze policy: train delta only; frozen={n_frozen:,}, trainable={n_trainable:,}"
    )


def create_model(
    config: Dict,
    device: torch.device,
    use_gradient_checkpointing: bool = None,
    use_compile: bool = None,
) -> torch.nn.Module:
    """Create V11m4 model while preserving V11m3 train-loop behavior."""
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
        "Creating V11m4 model "
        f"(local_pair_delta={model_cfg.get('use_local_pair_delta', True)}, "
        f"pair_dim={model_cfg.get('pair_delta_dim', 128)}, "
        f"scale_init={model_cfg.get('pair_delta_logit_scale_init', 0.5)})..."
    )
    model = PocketGVPFusionV11m4(model_cfg)
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
