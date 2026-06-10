#!/usr/bin/env python3
"""
Training entrypoint for Pocket-GVP-Fusion V11m4.

This reuses the mature V11m3 training loop and swaps only:
- model class: PocketGVPFusionV11m4
- experiment suffix: v11m4
- optional initialization via training.init_from_checkpoint. The HPC config
  trains from scratch by default.
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


def _patch_v11m3_source_for_v11m4(source: str) -> str:
    """Patch the reused V11m3 train loop only inside the V11m4 entrypoint."""
    source = source.replace(
        "import time\nimport yaml",
        "import time\nimport math\nimport yaml",
    )

    old_grad_block = """            # ✅ Record gradients BEFORE clipping and optimizer step
            if writer is not None and is_main_process() and (batch_idx + 1) % log_every == 0:
                global_step = epoch * len(dataloader) + batch_idx

                # 梯度统计（在清零之前记录）
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Train/grad_norm', total_norm, global_step)

                # 组件级梯度统计
                grad_stats = compute_gradient_stats(model)
                for key, value in grad_stats.items():
                    writer.add_scalar(f'Gradients/{key}', value, global_step)

            # Gradient clipping
            if gradient_clip > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
"""
    new_grad_block = """            # AMP gradients are scaled after backward. Unscale BEFORE logging so
            # V11m4 gradient metrics are true pre-clipping norms, matching V12.
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            # ✅ Record gradients AFTER unscale, BEFORE clipping and optimizer step
            if writer is not None and is_main_process() and (batch_idx + 1) % log_every == 0:
                global_step = epoch * len(dataloader) + batch_idx

                # 梯度统计（true pre-clipping norm）
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad = p.grad.detach()
                        vals = grad.coalesce()._values() if grad.is_sparse else grad
                        if torch.isfinite(vals).all():
                            param_norm = torch.linalg.vector_norm(vals.float(), ord=2)
                            total_norm += param_norm.item() ** 2
                        else:
                            total_norm = float("nan")
                            break
                total_norm = total_norm ** 0.5 if math.isfinite(total_norm) else total_norm
                writer.add_scalar('Train/grad_norm', total_norm, global_step)

                # 组件级梯度统计
                grad_stats = compute_gradient_stats(model)
                for key, value in grad_stats.items():
                    writer.add_scalar(f'Gradients/{key}', value, global_step)

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
"""
    if old_grad_block not in source:
        raise RuntimeError("Could not patch V11m3 grad logging block for V11m4")
    source = source.replace(old_grad_block, new_grad_block, 1)

    source = source.replace(
        "    best_val_loss = float('inf')\n    patience_counter = 0",
        "    best_val_loss = float('inf')\n    best_val_f1_c0 = -float('inf')\n    best_val_dice_c0 = -float('inf')\n    patience_counter = 0",
        1,
    )
    source = source.replace(
        "            best_val_loss = checkpoint.get('best_val_loss', float('inf'))\n            patience_counter = checkpoint.get('patience_counter', 0)",
        "            best_val_loss = checkpoint.get('best_val_loss', float('inf'))\n            best_val_f1_c0 = checkpoint.get('best_val_f1_c0', -float('inf'))\n            best_val_dice_c0 = checkpoint.get('best_val_dice_c0', -float('inf'))\n            patience_counter = checkpoint.get('patience_counter', 0)",
        1,
    )
    source = source.replace(
        "                'best_val_loss': best_val_loss,\n                'patience_counter': patience_counter,",
        "                'best_val_loss': best_val_loss,\n                'best_val_f1_c0': best_val_f1_c0,\n                'best_val_dice_c0': best_val_dice_c0,\n                'patience_counter': patience_counter,",
    )

    old_best_tail = """            else:
                patience_counter += 1

            # Early stopping
"""
    new_best_tail = """            else:
                patience_counter += 1

            # 4. Save metric-best checkpoints. Val loss can become a poor
            # selector for contact segmentation, so keep explicit F1/Dice bests.
            current_f1_c0 = float(val_metrics.get('f1_c0', -float('inf')))
            if math.isfinite(current_f1_c0) and current_f1_c0 > best_val_f1_c0:
                best_val_f1_c0 = current_f1_c0
                best_f1_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_f1_c0': best_val_f1_c0,
                    'best_val_dice_c0': best_val_dice_c0,
                    'patience_counter': patience_counter,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }
                if scaler is not None:
                    best_f1_checkpoint['scaler_state_dict'] = scaler.state_dict()
                torch.save(best_f1_checkpoint, exp_dir / 'best_f1_c0_model.pt')
                print_rank0(f"✓ Saved best F1_c0 model (f1_c0={best_val_f1_c0:.4f})")

            current_dice_c0 = float(val_metrics.get('dice_c0', -float('inf')))
            if math.isfinite(current_dice_c0) and current_dice_c0 > best_val_dice_c0:
                best_val_dice_c0 = current_dice_c0
                best_dice_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_f1_c0': best_val_f1_c0,
                    'best_val_dice_c0': best_val_dice_c0,
                    'patience_counter': patience_counter,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }
                if scaler is not None:
                    best_dice_checkpoint['scaler_state_dict'] = scaler.state_dict()
                torch.save(best_dice_checkpoint, exp_dir / 'best_dice_c0_model.pt')
                print_rank0(f"✓ Saved best DICE_c0 model (dice_c0={best_val_dice_c0:.4f})")

            # Early stopping
"""
    if old_best_tail not in source:
        raise RuntimeError("Could not patch V11m3 best-metric checkpoint block for V11m4")
    local_delta_anchor = """                # V11m3: Uni-Mol ligand feature contribution diagnostics.
                unimol_stats = compute_unimol_stats(_model, _unimol)
                for key, value in unimol_stats.items():
                    writer.add_scalar(f'Health/{key}', value, global_step)
"""
    local_delta_block = """                # V11m6: local-delta numeric-scale diagnostics for scale-prior probes.
                for key in (
                    'local_delta_score_abs_mean',
                    'local_delta_attn_entropy_mean',
                    'local_delta_logits_abs_mean',
                    'local_delta_scale',
                    'local_delta_qk_scale',
                    'local_ligand_film_alpha',
                    'pocket_virtual_alpha',
                    'pocket_virtual_attn_entropy',
                    'center_aux_scale',
                    'center_aux_abs_mean',
                    'local54_chem_reweight_mean',
                    'local54_chem_adapter_scale',
                    'local_evidence_group_kv_alpha',
                    'local_evidence_compat_alpha',
                    'local_evidence_compat_gate_abs_mean',
                    'local_evidence_compat_read_abs_mean',
                    'local54_interaction_kv_alpha',
                    'local54_compat_alpha',
                    'ligand_context_prior_alpha',
                    'local54_compat_gate_abs_mean',
                    'local54_compat_read_raw_abs_mean',
                    'local54_compat_read_abs_mean',
                    'ligand_context_prior_abs_mean',
                    'bimamba_residue_alpha',
                    'bimamba_residue_delta_abs_mean',
                    'bimamba_ligand_kv_alpha',
                    'bimamba_ligand_delta_abs_mean',
                    'bimamba_ligand_k_abs_mean',
                    'bimamba_ligand_v_abs_mean',
                    'mamba_lig_memory_alpha',
                    'mamba_lig_memory_score_bias',
                    'mamba_lig_memory_attn_max',
                    'mamba_lig_memory_v_abs_mean',
                    'unimol_memory_alpha',
                    'unimol_memory_score_bias',
                    'unimol_memory_v_abs_mean',
                    'unimol_group_gate_alpha',
                    'unimol_group_gate_abs_mean',
                    'unimol_atom_gate_alpha',
                    'unimol_atom_gate_abs_mean',
                    'unimol_residue_compat_alpha',
                    'unimol_residue_compat_abs_mean',
                    'v12_cross_alpha_mean',
                    'v12_cross_delta_norm_mean',
                    'v12_coattn_p2l_gate_mean',
                    'v12_coattn_l2p_gate_mean',
                    'v12_coattn_p2l_delta_norm_mean',
                    'v12_coattn_l2p_delta_norm_mean',
                ):
                    if key in outputs:
                        value = outputs[key]
                        if torch.is_tensor(value):
                            value = value.detach().float().mean().item()
                        writer.add_scalar(f'Health/{key}', float(value), global_step)
"""
    if local_delta_anchor not in source:
        raise RuntimeError("Could not patch V11m3 local-delta diagnostics block for V11m4")
    source = source.replace(local_delta_anchor, local_delta_anchor + local_delta_block, 1)
    return source.replace(old_best_tail, new_best_tail, 1)


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
    source = _patch_v11m3_source_for_v11m4(source)
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
            "Unset this field for scratch V11m4 training."
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
