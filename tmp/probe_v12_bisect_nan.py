#!/usr/bin/env python3
"""
V12 NaN/Inf gradient bisect probe.

Purpose:
  - no_step mode: run backward on each batch with a fresh/unchanged model and no
    optimizer.step(). If this fails, the batch/loss/backward graph itself emits
    non-finite grads.
  - train mode: run normal optimizer steps. If only this fails, previous updates
    poison parameters.
  - branch masks: disable individual CoAttentionBlock sub-branches to isolate
    protein SA / ligand SA / P->L / L->P.
  - loss modes: ce / focal / full to isolate loss-induced gradient spikes.
  - norm mode: optionally replace LayerNorm with RMSNorm for a backend/norm check.

Example:
  env HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \\
    conda run -n AIAA python tmp/probe_v12_bisect_nan.py \\
      --config configs/pocket_gvp_v12.yaml --device 0 --require-idle \\
      --mode no_step --cases all,no_lig_sa,no_p2l --loss-modes ce,focal,full
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v12
from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.models.pocket_gvp_fusion_v12 import (
    PocketGVPFusionV12,
    zero_init_co_attention_residuals,
)
from scripts.train.train_v12 import (
    build_adamw_param_groups,
    compute_combined_loss_v4,
    compute_dist_aux_loss,
    get_batch_unimol,
    load_unimol_cache,
    safe_clip_grad_norm_,
)


class IdentitySelf(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class IdentityCross(nn.Module):
    def forward(self, q, kv, q_mask=None, kv_mask=None):
        if q_mask is not None:
            return q * q_mask.unsqueeze(-1).to(dtype=q.dtype)
        return q


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def check_rocm_idle(device_index: int, max_util: float, max_mem_pct: float) -> None:
    """Best-effort safety check before occupying a GPU."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmemuse"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError(f"rocm-smi check failed; refusing GPU run: {exc}") from exc

    # Typical lines contain: GPU[1] ... GPU use (%) ... GPU memory use (%)
    gpu_line = None
    for line in out.splitlines():
        if f"GPU[{device_index}]" in line:
            gpu_line = line
            break
    if gpu_line is None:
        print(out)
        raise RuntimeError(f"could not find GPU[{device_index}] in rocm-smi output")

    util_match = re.search(r"GPU use \(%\):\s*([0-9]+(?:\.[0-9]+)?)", gpu_line)
    if util_match is None:
        util_match = re.search(r"GPU use.*?:\s*([0-9]+(?:\.[0-9]+)?)", gpu_line)
    if util_match is None:
        print(out)
        raise RuntimeError(f"could not parse utilization from: {gpu_line}")
    util = float(util_match.group(1))

    mem_pct = 0.0
    mem_line = None
    for line in out.splitlines():
        if f"GPU[{device_index}]" in line and "GPU Memory Allocated" in line:
            mem_line = line
            break
    if mem_line is not None:
        mem_match = re.search(r"VRAM%\):\s*([0-9]+(?:\.[0-9]+)?)", mem_line)
        if mem_match is None:
            mem_match = re.search(r"VRAM.*?:\s*([0-9]+(?:\.[0-9]+)?)", mem_line)
        if mem_match is not None:
            mem_pct = float(mem_match.group(1))

    if util > max_util or mem_pct > max_mem_pct:
        raise RuntimeError(
            f"GPU[{device_index}] is busy (util={util}%, mem={mem_pct}%). "
            "Refusing to run probe."
        )
    print(f"[GPU check] GPU[{device_index}] idle enough: util={util}%, mem={mem_pct}%")


def make_loader(
    cfg: Dict,
    n_ids: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    use_nma: bool,
) -> DataLoader:
    data_cfg = cfg["data"]
    train_ids, _, _ = load_splits(data_cfg["split_file"])
    if n_ids > 0:
        train_ids = train_ids[:n_ids]
    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=train_ids,
        max_seq_len=data_cfg["max_seq_len"],
        max_atoms=data_cfg["max_atoms"],
        distances_file=data_cfg.get("distances_file"),
        distances_file_original=data_cfg.get("distances_file_original"),
        ligand_features_dir=data_cfg.get("ligand_features_dir", "ligand_features_v5"),
        protein_features_dir=data_cfg.get("protein_features_dir", "complete_features_v4"),
        nma_features_dir=data_cfg.get("nma_features_dir", "complete_features_v4_nma"),
        nma_n_confs=data_cfg.get("nma_n_confs", 1),
        nma_max_conf=data_cfg.get("nma_max_conf", 9),
        label_contact_thresh=data_cfg.get("label_contact_thresh", 5.0),
        label_near_thresh=data_cfg.get("label_near_thresh", 8.0),
        split="train",
        use_nma=use_nma,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pocket_collate_fn_v10,
        generator=generator,
    )


def replace_layernorm_with_rmsnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            rms = nn.RMSNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    rms.weight.copy_(child.weight.detach())
            setattr(module, name, rms)
        else:
            replace_layernorm_with_rmsnorm(child)


def apply_branch_case(model: nn.Module, case: str) -> None:
    """Disable selected V12 co-attn branches by replacing them with identity."""
    if hasattr(model, "module"):
        model = model.module
    flags = set(case.split("+")) if case else {"all"}
    for block in model.co_attn_blocks:
        if "no_film" in flags:
            block.film = None
        if "no_prot_sa" in flags:
            block.prot_self_attn = IdentitySelf()
            block.prot_dist_bias = None
        if "no_lig_sa" in flags:
            block.lig_self_attn = IdentitySelf()
            block.lig_dist_bias = None
        if "no_p2l" in flags:
            block.prot_to_lig = IdentityCross()
        if "no_l2p" in flags:
            block.lig_to_prot = IdentityCross()
        if "only_p2l" in flags:
            block.prot_self_attn = IdentitySelf()
            block.prot_dist_bias = None
            block.lig_self_attn = IdentitySelf()
            block.lig_dist_bias = None
            block.lig_to_prot = IdentityCross()
        if "only_l2p" in flags:
            block.prot_self_attn = IdentitySelf()
            block.prot_dist_bias = None
            block.lig_self_attn = IdentitySelf()
            block.lig_dist_bias = None
            block.prot_to_lig = IdentityCross()
        if "only_self" in flags:
            block.prot_to_lig = IdentityCross()
            block.lig_to_prot = IdentityCross()
        if "only_cross" in flags:
            block.prot_self_attn = IdentitySelf()
            block.prot_dist_bias = None
            block.lig_self_attn = IdentitySelf()
            block.lig_dist_bias = None


def rezero_coattn_final_projections(model: nn.Module) -> None:
    """Debug helper: keep CoAttentionBlock residual branches exactly closed."""
    if hasattr(model, "module"):
        model = model.module
    with torch.no_grad():
        for block in model.co_attn_blocks:
            zero_init_co_attention_residuals(block)


def first_bad_grad(model: nn.Module) -> List[str]:
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        vals = g.coalesce()._values() if g.is_sparse else g
        if not torch.isfinite(vals).all():
            bad.append(
                f"{name}: nan={torch.isnan(vals).sum().item()} "
                f"inf={torch.isinf(vals).sum().item()} shape={tuple(g.shape)}"
            )
            if len(bad) >= 8:
                break
    return bad


def top_grad_norms(model: nn.Module, n: int = 8) -> List[Tuple[float, str]]:
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        vals = g.coalesce()._values() if g.is_sparse else g
        if torch.isfinite(vals).all():
            rows.append((float(torch.linalg.vector_norm(vals.float()).cpu()), name))
    rows.sort(reverse=True)
    return rows[:n]


def param_group_name(name: str) -> str:
    """Coarse functional group for update/gradient diagnostics."""
    if name.startswith("co_attn_blocks."):
        parts = name.split(".")
        if len(parts) >= 3:
            block = parts[1]
            branch = parts[2]
            if branch in {"film", "prot_self_attn", "lig_self_attn", "prot_to_lig", "lig_to_prot"}:
                suffix = ".".join(parts[3:])
                final = any(tok in suffix for tok in ("attn.w_o", "cross_attn.w_o", "ffn.w_2", "proj.weight", "proj.bias"))
                return f"{block}.{branch}.{'final' if final else 'internal'}"
            if branch in {"prot_dist_bias", "lig_dist_bias"}:
                return f"{block}.{branch}"
        return "co_attn_blocks.other"
    for prefix in (
        "gvp_proj",
        "esm_proj",
        "physchem_encoder",
        "input_norm",
        "ligand_encoder",
        "ligand_self_attn",
        "unimol_to_global",
        "rsa_input_proj",
        "final_norm",
        "classifier_input_norm",
        "classifier",
    ):
        if name.startswith(prefix):
            return prefix
    return name.split(".", 1)[0]


def grouped_grad_stats(model: nn.Module, n: int = 16) -> List[Tuple[float, float, int, str]]:
    """Return top groups by L2 grad norm: (norm, max_abs, n_params, group)."""
    stats: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        vals = p.grad.detach()
        vals = vals.coalesce()._values() if vals.is_sparse else vals
        finite_vals = vals[torch.isfinite(vals)]
        if finite_vals.numel() == 0:
            continue
        group = param_group_name(name)
        entry = stats.setdefault(group, {"sq": 0.0, "max": 0.0})
        vf = finite_vals.float()
        entry["sq"] += float(torch.sum(vf * vf).cpu())
        entry["max"] = max(entry["max"], float(vf.abs().max().cpu()))
        counts[group] = counts.get(group, 0) + 1
    rows = [(v["sq"] ** 0.5, v["max"], counts.get(k, 0), k) for k, v in stats.items()]
    rows.sort(reverse=True)
    return rows[:n]


def tensor_stats(x: torch.Tensor) -> str:
    x_det = x.detach()
    vals = x_det.coalesce()._values() if x_det.is_sparse else x_det
    finite_mask = torch.isfinite(vals)
    n_total = vals.numel()
    n_finite = int(finite_mask.sum().item())
    n_nan = int(torch.isnan(vals).sum().item())
    n_inf = int(torch.isinf(vals).sum().item())
    if n_finite == 0:
        return (
            f"shape={tuple(x_det.shape)} finite=0/{n_total} "
            f"nan={n_nan} inf={n_inf}"
        )
    v = vals[finite_mask].float()
    return (
        f"shape={tuple(x_det.shape)} finite={n_finite}/{n_total} "
        f"nan={n_nan} inf={n_inf} min={v.min().item():.4e} "
        f"max={v.max().item():.4e} mean={v.mean().item():.4e} "
        f"std={v.std(unbiased=False).item():.4e}"
    )


def batch_summary(batch: Dict, inputs: Dict[str, torch.Tensor], labels: torch.Tensor, mask: torch.Tensor) -> str:
    """Compact failure-batch summary for reproducing problematic batches."""
    rows = []
    pdb_ids = batch.get("pdb_ids", [])
    rows.append(f"pdb_ids={list(pdb_ids)[:20]}")
    for key in ("n_residues", "n_atoms"):
        if key in batch and torch.is_tensor(batch[key]):
            vals = batch[key].detach().cpu().tolist()
            rows.append(f"{key}={vals}")
    valid_counts = mask.detach().sum(dim=1).cpu().tolist()
    rows.append(f"protein_valid_counts={valid_counts}")
    if "ligand_mask" in batch and torch.is_tensor(batch["ligand_mask"]):
        rows.append(f"ligand_valid_counts={batch['ligand_mask'].sum(dim=1).detach().cpu().tolist()}")
    valid_labels = labels[mask]
    if valid_labels.numel() > 0:
        hist = torch.bincount(valid_labels.detach().cpu(), minlength=3).tolist()
        rows.append(f"label_hist_valid={hist}")

    feature_keys = (
        "esm_embeddings",
        "protein_physchem",
        "protein_spatial_scalar",
        "protein_spatial_vector",
        "protein_coords",
        "ligand_x",
        "ligand_pos",
        "ligand_edge_attr",
        "unimol_embed",
    )
    for key in feature_keys:
        value = inputs.get(key)
        if torch.is_tensor(value) and value.is_floating_point():
            rows.append(f"{key}: {tensor_stats(value)}")
    return "\n".join(rows)


class NonFiniteTracer:
    """Forward/backward hook tracer for first non-finite module tensor."""

    def __init__(self, model: nn.Module):
        self.bad: List[str] = []
        self.handles = []
        self.norm_inputs: Dict[str, str] = {}
        self.norm_grad_outputs: Dict[str, str] = {}
        for name, module in model.named_modules():
            if not name:
                continue
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                self.handles.append(module.register_forward_pre_hook(self._norm_pre_hook(name)))
                self.handles.append(module.register_full_backward_pre_hook(self._norm_backward_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._hook(name)))
            self.handles.append(module.register_full_backward_hook(self._backward_hook(name)))
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.handles.append(param.register_hook(self._param_hook(name)))

    def _norm_pre_hook(self, name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                self.norm_inputs[name] = tensor_stats(x)
        return hook

    def _norm_backward_pre_hook(self, name: str):
        def hook(_module, grad_output):
            tensors = []
            for i, item in enumerate(grad_output):
                if torch.is_tensor(item):
                    tensors.append(f"grad_output[{i}]={tensor_stats(item)}")
                    if not torch.isfinite(item).all() and not self.bad:
                        context = self.norm_inputs.get(name)
                        suffix = f" last_norm_input={context}" if context is not None else ""
                        self.bad.append(f"{name}.backward_pre.grad_output[{i}]: {tensor_stats(item)}{suffix}")
            if tensors:
                self.norm_grad_outputs[name] = " ".join(tensors)
        return hook

    def _norm_context_for_param(self, param_name: str) -> str:
        module_name = param_name.rsplit(".", 1)[0]
        pieces = []
        input_stats = self.norm_inputs.get(module_name)
        grad_stats = self.norm_grad_outputs.get(module_name)
        if input_stats is not None:
            pieces.append(f"last_norm_input[{module_name}]={input_stats}")
        if grad_stats is not None:
            pieces.append(f"last_norm_grad_output[{module_name}]={grad_stats}")
        return " " + " ".join(pieces) if pieces else ""

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            if self.bad:
                return
            # DistanceBias modules intentionally emit -inf for padded key
            # positions. That is a legal additive attention mask, not a model
            # activation failure. Still flag NaN there.
            allow_inf = "dist_bias" in name
            tensors: List[Tuple[str, torch.Tensor]] = []
            if torch.is_tensor(output):
                tensors.append(("out", output))
            elif isinstance(output, (tuple, list)):
                for i, item in enumerate(output):
                    if torch.is_tensor(item):
                        tensors.append((f"out[{i}]", item))
            elif isinstance(output, dict):
                for key, item in output.items():
                    if torch.is_tensor(item):
                        tensors.append((f"out[{key}]", item))
            for label, tensor in tensors:
                if allow_inf:
                    bad_tensor = torch.isnan(tensor).any()
                else:
                    bad_tensor = ~torch.isfinite(tensor).all()
                if bool(bad_tensor.item()):
                    self.bad.append(f"{name}.{label}: {tensor_stats(tensor)}")
                    return
        return hook

    def _param_hook(self, name: str):
        def hook(grad):
            if self.bad:
                return grad
            if grad is None:
                return grad
            if not torch.isfinite(grad).all():
                self.bad.append(
                    f"param_grad.{name}: {tensor_stats(grad)}"
                    f"{self._norm_context_for_param(name)}"
                )
            return grad
        return hook

    def _backward_hook(self, name: str):
        def hook(_module, grad_input, grad_output):
            if self.bad:
                return
            tensors: List[Tuple[str, torch.Tensor]] = []
            for i, item in enumerate(grad_input):
                if torch.is_tensor(item):
                    tensors.append((f"grad_input[{i}]", item))
            for i, item in enumerate(grad_output):
                if torch.is_tensor(item):
                    tensors.append((f"grad_output[{i}]", item))
            for label, tensor in tensors:
                if not torch.isfinite(tensor).all():
                    norm_input = self.norm_inputs.get(name)
                    context = f" last_norm_input={norm_input}" if norm_input is not None else ""
                    self.bad.append(f"{name}.{label}: {tensor_stats(tensor)}{context}")
                    return
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def first_bad_param(model: nn.Module) -> List[str]:
    bad = []
    for name, p in model.named_parameters():
        vals = p.detach()
        if not torch.isfinite(vals).all():
            bad.append(f"{name}: {tensor_stats(vals)}")
            if len(bad) >= 8:
                break
    return bad


def snapshot_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def max_param_delta(model: nn.Module, before: Dict[str, torch.Tensor]) -> Tuple[str, float]:
    best_name = ""
    best_delta = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            old = before.get(name)
            if old is None:
                continue
            delta = (p.detach() - old).abs().max().item()
            if delta > best_delta:
                best_delta = delta
                best_name = name
    return best_name, best_delta


def grouped_param_delta(model: nn.Module, before: Dict[str, torch.Tensor], n: int = 16) -> List[Tuple[float, float, int, str]]:
    """Return top groups by max absolute parameter delta."""
    stats: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            old = before.get(name)
            if old is None:
                continue
            delta = (p.detach() - old).float()
            finite_delta = delta[torch.isfinite(delta)]
            if finite_delta.numel() == 0:
                continue
            group = param_group_name(name)
            entry = stats.setdefault(group, {"max": 0.0, "l2sq": 0.0})
            entry["max"] = max(entry["max"], float(finite_delta.abs().max().cpu()))
            entry["l2sq"] += float(torch.sum(finite_delta * finite_delta).cpu())
            counts[group] = counts.get(group, 0) + 1
    rows = [(v["max"], v["l2sq"] ** 0.5, counts.get(k, 0), k) for k, v in stats.items()]
    rows.sort(reverse=True)
    return rows[:n]


def freeze_by_pattern(model: nn.Module, pattern: str) -> int:
    """Freeze parameters whose names match a regex or comma-separated regex list."""
    if not pattern:
        return 0
    regexes = [re.compile(p.strip()) for p in pattern.split(",") if p.strip()]
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(r.search(name) for r in regexes):
            p.requires_grad_(False)
            n_frozen += p.numel()
    return n_frozen


def first_bad_inputs(inputs: Dict[str, torch.Tensor]) -> List[str]:
    bad = []
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all():
                bad.append(f"{name}: {tensor_stats(value)}")
                if len(bad) >= 8:
                    break
    return bad


def make_optimizer(model: nn.Module, cfg: Dict, args) -> torch.optim.Optimizer:
    lr = args.lr if args.lr is not None else cfg["training"].get("learning_rate", 5e-5)
    wd = args.weight_decay if args.weight_decay is not None else cfg["training"].get("weight_decay", 0.05)
    betas = tuple(cfg["training"].get("betas", [0.9, 0.999]))
    eps = args.adamw_eps

    if not args.real_param_groups:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)

    param_groups, info = build_adamw_param_groups(
        model,
        wd,
        base_lr=lr,
        zero_open_lr_mult=args.zero_open_lr_mult,
        zero_open_eps=args.zero_open_eps,
    )
    bad_norm_decay = [
        n for n in info["decay_names"]
        if any(tok in n for tok in (".norm", "norm_", "norm1", "norm2", "key_norm"))
    ]
    print(
        f"AdamW groups: decay={info['decay_count']} no_decay={info['no_decay_count']} "
        f"zero_open={info.get('zero_open_count', 0)} "
        f"bad_norm_decay={len(bad_norm_decay)}",
        flush=True,
    )
    if bad_norm_decay:
        print("bad_norm_decay_examples=" + ", ".join(bad_norm_decay[:12]), flush=True)
    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        foreach=False,
    )


def compute_probe_loss(
    loss_mode: str,
    out: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    cfg: Dict,
    batch: Dict,
    device: torch.device,
) -> torch.Tensor:
    logits = out["logits"]
    if loss_mode == "ce":
        flat_logits = logits[mask]
        flat_labels = labels[mask]
        return F.cross_entropy(flat_logits.float(), flat_labels)
    if loss_mode == "weighted_ce":
        flat_logits = logits[mask]
        flat_labels = labels[mask]
        return F.cross_entropy(flat_logits.float(), flat_labels, weight=class_weights.float())
    if loss_mode == "focal":
        loss, _, _ = compute_combined_loss_v4(
            logits=logits,
            labels=labels,
            mask=mask,
            class_weights=class_weights,
            focal_gamma=cfg["training"]["focal_gamma"],
            focal_weight=1.0,
            tversky_weight=0.0,
            weighted_dice_weight=0.0,
            label_smoothing=cfg["training"].get("label_smoothing", 0.05),
        )
        return loss
    if loss_mode == "full":
        tr = cfg["training"]
        loss, _, _ = compute_combined_loss_v4(
            logits=logits,
            labels=labels,
            mask=mask,
            class_weights=class_weights,
            focal_gamma=tr["focal_gamma"],
            focal_weight=tr.get("focal_weight", 0.7),
            tversky_weight=tr.get("tversky_weight", 0.3),
            tversky_alpha=tr.get("tversky_alpha", 0.3),
            tversky_beta=tr.get("tversky_beta", 0.7),
            tversky_smooth=tr.get("tversky_smooth", 1.0),
            weighted_dice_weight=tr.get("weighted_dice_weight", 0.0),
            weighted_dice_smooth=tr.get("weighted_dice_smooth", 1.0),
            label_smoothing=tr.get("label_smoothing", 0.05),
        )
        dist_cfg = tr.get("dist_aux", {})
        if dist_cfg.get("enabled", False) and "dist_pred" in out and "raw_distances" in batch:
            da = compute_dist_aux_loss(
                out["dist_pred"],
                batch["raw_distances"].to(device),
                mask,
                dist_cfg.get("max_dist", 20.0),
            )
            loss = loss + dist_cfg.get("weight", 0.05) * da
        return loss
    raise ValueError(f"unknown loss_mode: {loss_mode}")


def run_case(
    cfg: Dict,
    case: str,
    loss_mode: str,
    norm_mode: str,
    mode: str,
    device: torch.device,
    args,
) -> bool:
    print("\n" + "=" * 100)
    print(f"CASE branch={case} loss={loss_mode} norm={norm_mode} mode={mode} amp={args.amp}")
    print("=" * 100, flush=True)

    seed = args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["use_rsa_input"] = False
    model_cfg["use_sasa_gate"] = False
    model_cfg["debug_asserts"] = True
    if norm_mode == "rmsnorm":
        model_cfg["norm_type"] = "rmsnorm"
    elif norm_mode == "layernorm":
        model_cfg["norm_type"] = "layernorm"
    if args.blocks is not None:
        model_cfg["n_co_attn_blocks"] = args.blocks
    if args.inner_scale is not None:
        model_cfg["co_attn_inner_residual_scale"] = args.inner_scale
    if args.outer_scale is not None:
        model_cfg["co_attn_residual_scale"] = args.outer_scale
    print(
        "model_scales: "
        f"inner={model_cfg.get('co_attn_inner_residual_scale')} "
        f"outer={model_cfg.get('co_attn_residual_scale')} "
        f"use_attn_res={model_cfg.get('use_attn_res')}",
        flush=True,
    )
    if args.dropout is not None:
        model_cfg["dropout"] = args.dropout
        model_cfg["attn_dropout"] = args.dropout
        model_cfg["stochastic_depth_prob"] = 0.0
        model_cfg["classifier_dropout"] = args.dropout
        # V12 has additional stochastic regularizers that are independent from
        # module dropout.  For numerical probes, --dropout 0 must make the whole
        # forward deterministic, otherwise train/no_step comparisons are noisy.
        if args.dropout == 0.0:
            model_cfg["esm_dropout_prob"] = 0.0
            model_cfg["physchem_dropout_prob"] = 0.0
            model_cfg["gvp_dropout_prob"] = 0.0
            model_cfg["coord_noise_std"] = 0.0
    model = PocketGVPFusionV12(model_cfg)
    apply_branch_case(model, case)
    if norm_mode == "rmsnorm":
        replace_layernorm_with_rmsnorm(model)
    if args.freeze_pattern:
        n_frozen = freeze_by_pattern(model, args.freeze_pattern)
        print(f"freeze_pattern={args.freeze_pattern!r} frozen_params={n_frozen}", flush=True)
    model.to(device).train()
    tracer = NonFiniteTracer(model) if args.trace_nonfinite else None

    opt = make_optimizer(model, cfg, args)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    autocast_enabled = args.amp != "none" and device.type == "cuda"
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    if args.class_weights is not None:
        class_weights_values = [float(x) for x in args.class_weights.split(",")]
    else:
        class_weights_values = cfg["training"]["class_weights"]
    if args.focal_gamma is not None:
        cfg["training"]["focal_gamma"] = float(args.focal_gamma)
    class_weights = torch.tensor(class_weights_values, device=device, dtype=torch.float32)
    print(
        f"class_weights={class_weights_values} "
        f"focal_gamma={cfg['training'].get('focal_gamma')}",
        flush=True,
    )
    loader = make_loader(
        cfg,
        n_ids=args.n_ids if args.n_ids is not None else max(args.steps * args.batch_size, args.batch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        seed=seed,
        use_nma=args.use_nma,
    )
    cache = load_unimol_cache(
        cfg["data"].get("data_root", "data"),
        cfg["data"].get("unimol_embeddings_dir", "unimol_embeddings"),
    )

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        opt.zero_grad(set_to_none=True)
        unimol = get_batch_unimol(batch["pdb_ids"], cache, model_cfg.get("d_unimol", 512), device)
        inputs = prepare_model_inputs_v12(batch, device, non_blocking=False, unimol_embed=unimol)
        labels = batch["labels"].to(device)
        mask = batch["protein_mask"].to(device)

        def dump_context(reason: str) -> None:
            print(reason, flush=True)
            print("FAIL_BATCH_CONTEXT", flush=True)
            print(batch_summary(batch, inputs, labels, mask), flush=True)

        bad_inputs = first_bad_inputs(inputs)
        if bad_inputs:
            dump_context(f"FAIL inputs step={step}: {'; '.join(bad_inputs)}")
            if tracer is not None:
                tracer.close()
            return False

        anomaly_ctx = (
            torch.autograd.detect_anomaly(check_nan=True)
            if args.detect_anomaly
            else nullcontext()
        )
        try:
            with anomaly_ctx:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
                    out = model(**inputs)
                    if not args.fp32_loss:
                        loss = compute_probe_loss(loss_mode, out, labels, mask, class_weights, cfg, batch, device)
                if args.fp32_loss:
                    out_for_loss = {
                        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                        for k, v in out.items()
                    }
                    loss = compute_probe_loss(loss_mode, out_for_loss, labels, mask, class_weights, cfg, batch, device)
                if tracer is not None and tracer.bad:
                    dump_context(f"FAIL forward_hook step={step}: {tracer.bad[0]}")
                    tracer.close()
                    return False
                if not finite(out["logits"]):
                    dump_context(f"FAIL forward step={step}: logits non-finite {tensor_stats(out['logits'])}")
                    if tracer is not None:
                        tracer.close()
                    return False
                if not finite(loss):
                    dump_context(f"FAIL loss step={step}: loss={loss.detach().float().cpu().item()}")
                    if tracer is not None:
                        tracer.close()
                    return False
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
        except RuntimeError as exc:
            dump_context(f"FAIL runtime step={step}: {exc}")
            if tracer is not None:
                tracer.close()
            return False
        if tracer is not None and tracer.bad:
            dump_context(f"FAIL backward_hook step={step}: {tracer.bad[0]}")
            print("top_grad:", top_grad_norms(model), flush=True)
            tracer.close()
            return False
        bad = first_bad_grad(model)
        if bad:
            dump_context(f"FAIL backward step={step} loss={loss.detach().float().item():.6g}")
            print("bad_grad:", "; ".join(bad))
            print("top_grad:", top_grad_norms(model))
            print("top_group_grad:", grouped_grad_stats(model))
            bad_param = first_bad_param(model)
            if bad_param:
                print("bad_param:", "; ".join(bad_param))
            if tracer is not None:
                tracer.close()
            return False
        total_norm = safe_clip_grad_norm_(model.named_parameters(), args.grad_clip, error_if_nonfinite=True)
        if step % args.log_every == 0:
            print(f"step={step} loss={loss.detach().float().item():.6g} grad_norm={total_norm:.4e}", flush=True)
            print("top_grad:", top_grad_norms(model), flush=True)
            if args.track_group_stats:
                print("top_group_grad:", grouped_grad_stats(model), flush=True)
        if mode == "train":
            before_step = snapshot_params(model) if args.track_param_delta else None
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if args.rezero_coattn_final:
                rezero_coattn_final_projections(model)
            if before_step is not None:
                delta_name, delta = max_param_delta(model, before_step)
                print(f"param_delta_after_step step={step}: {delta_name} max_abs_delta={delta:.4e}", flush=True)
                print(f"group_delta_after_step step={step}: {grouped_param_delta(model, before_step)}", flush=True)
            bad_param = first_bad_param(model)
            if bad_param:
                print(f"FAIL optimizer_step step={step}: {'; '.join(bad_param)}", flush=True)
                if tracer is not None:
                    tracer.close()
                return False
        else:
            opt.zero_grad(set_to_none=True)

    if tracer is not None:
        tracer.close()
    print("PASS", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pocket_gvp_v12.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--max-gpu-util", type=float, default=5.0)
    parser.add_argument("--max-gpu-mem", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-ids", type=int, default=None)
    parser.add_argument("--use-nma", action="store_true")
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--inner-scale", type=float, default=None)
    parser.add_argument("--outer-scale", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--mode", choices=["no_step", "train"], default="no_step")
    parser.add_argument(
        "--cases",
        default="all,no_lig_sa,no_prot_sa,no_p2l,no_l2p,only_self,only_cross,only_p2l,only_l2p",
    )
    parser.add_argument("--loss-modes", default="ce,weighted_ce,focal,full")
    parser.add_argument("--norm-modes", default="layernorm")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--zero-open-lr-mult", type=float, default=1.0)
    parser.add_argument("--zero-open-eps", type=float, default=None)
    parser.add_argument("--class-weights", default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--rezero-coattn-final", action="store_true")
    parser.add_argument("--track-param-delta", action="store_true")
    parser.add_argument("--track-group-stats", action="store_true")
    parser.add_argument("--freeze-pattern", default="")
    parser.add_argument("--real-param-groups", action="store_true")
    parser.add_argument("--trace-nonfinite", action="store_true")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--fp32-loss", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.require_idle:
            check_rocm_idle(args.device, args.max_gpu_util, args.max_gpu_mem)
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print("device", device, torch.cuda.get_device_name(device) if device.type == "cuda" else "")

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    results = []
    for norm_mode in [x.strip() for x in args.norm_modes.split(",") if x.strip()]:
        for loss_mode in [x.strip() for x in args.loss_modes.split(",") if x.strip()]:
            for case in [x.strip() for x in args.cases.split(",") if x.strip()]:
                ok = run_case(cfg, case, loss_mode, norm_mode, args.mode, device, args)
                results.append((ok, norm_mode, loss_mode, case))

    print("\nSUMMARY")
    for ok, norm_mode, loss_mode, case in results:
        print(f"{'PASS' if ok else 'FAIL'} norm={norm_mode} loss={loss_mode} branch={case}")


if __name__ == "__main__":
    main()
