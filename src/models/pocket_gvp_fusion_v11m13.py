"""
Pocket-GVP-Fusion V11m13: local evidence enhancement probes.

V11m13 keeps the V11m10/V11m8x co-attn trunk and adds one missing local-evidence
path: an explicit edge-conditioned K/V residual for the late residue-to-ligand
local_delta attention.  Existing V11m8/V11m10 switches cover the other two
probe axes:

- local54 group chemistry adapter/reweighting;
- local54 group K/V interaction;
- residue-chem compatibility readout.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m8 import _atanh_ratio
from .pocket_gvp_fusion_v11m10 import PocketGVPFusionV11m10
from .conditional_local_readout import ConditionalLocalReadout, widen_local_head
from ..losses.residue_classmass import residue_classmass_loss


class PocketGVPFusionV11m13(PocketGVPFusionV11m10):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m10.DEFAULT_CONFIG,
        "use_local54_edge_context_kv": False,
        "local54_edge_context_hidden": 96,
        "local54_edge_context_alpha_init": 0.03,
        "local54_edge_context_alpha_max": 0.20,
        "conditional_readout_mode": "none",
        "conditional_readout_dim": 64,
        "residue_classmass_weight": 0.0,
        "residue_classmass_temperature": 0.2,
        "residue_classmass_max_per_class": 32,
        "residue_classmass_seed": 44,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_local54_edge_context_kv = bool(cfg.get("use_local54_edge_context_kv", False))
        self.local54_edge_context_alpha_max = float(
            cfg.get("local54_edge_context_alpha_max", 0.20)
        )
        edge_dim = int(self.config.get("edge_dim", 6)) + 1  # mean edge_attr + degree.
        d_model = int(self.config["d_model"])
        hidden = int(cfg.get("local54_edge_context_hidden", 96))
        if self.use_local54_edge_context_kv:
            self.local54_edge_context_k = nn.Sequential(
                nn.LayerNorm(edge_dim, elementwise_affine=False),
                nn.Linear(edge_dim, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.local_delta_dim, bias=False),
            )
            self.local54_edge_context_v = nn.Sequential(
                nn.LayerNorm(edge_dim, elementwise_affine=False),
                nn.Linear(edge_dim, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, d_model, bias=False),
            )
            nn.init.zeros_(self.local54_edge_context_k[-1].weight)
            nn.init.zeros_(self.local54_edge_context_v[-1].weight)
            init = float(cfg.get("local54_edge_context_alpha_init", 0.03))
            self.local54_edge_context_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local54_edge_context_alpha_max))
            )
        else:
            self.local54_edge_context_k = None
            self.local54_edge_context_v = None
            self.register_buffer(
                "local54_edge_context_raw_alpha",
                torch.tensor(0.0),
                persistent=False,
            )

        self.config["use_local54_edge_context_kv"] = self.use_local54_edge_context_kv
        self._cached_v11m13_edge_context = None
        self._cached_v11m13_stats = {}
        self._cached_classmass_hidden = None
        self.conditional_readout = None
        mode = str(cfg["conditional_readout_mode"])
        if mode not in ("none", "product", "additive"):
            raise ValueError(f"Unknown conditional_readout_mode: {mode}")
        self.residue_classmass_weight = float(cfg["residue_classmass_weight"])
        if self.residue_classmass_weight < 0:
            raise ValueError("residue_classmass_weight must be non-negative")
        if (mode != "none" or self.residue_classmass_weight > 0) and not self.use_local_delta_only:
            raise ValueError("Readout/classmass probes require the existing local-delta path")
        if mode != "none":
            # Extra modules must not advance the shared initialization/training RNG.
            with torch.random.fork_rng(devices=[]):
                width = int(cfg["conditional_readout_dim"])
                self.conditional_readout = ConditionalLocalReadout(d_model, width, mode)
                widen_local_head(self.local_delta_head, width)

    def _classify_local_delta(self, delta_in, protein_mask, ligand_mask):
        if self.conditional_readout is not None:
            d = int(self.config["d_model"])
            valid = protein_mask.bool() & ligand_mask.bool().any(dim=1, keepdim=True)
            z = self.conditional_readout(delta_in[..., :d], delta_in[..., d:2*d], valid)
            self._cached_v11m13_stats["conditional_readout_abs_mean"] = (
                z.detach()[valid].float().abs().mean() if valid.any() else z.new_zeros(())
            )
            delta_in = torch.cat((delta_in, z.to(delta_in.dtype)), dim=-1)
        if self.training and self.residue_classmass_weight > 0:
            hidden = delta_in
            for layer in list(self.local_delta_head.children())[:-1]:
                hidden = layer(hidden)
            self._cached_classmass_hidden = hidden
            return self.local_delta_head[-1](hidden)
        return self.local_delta_head(delta_in)

    def compute_residue_classmass_loss(self, outputs, labels, mask, *, epoch=0, step=0):
        if not self.training or self.residue_classmass_weight == 0:
            return None, {}
        hidden = outputs["residue_classmass_hidden"]
        raw, stats = residue_classmass_loss(
            hidden, labels, mask,
            temperature=float(self.config["residue_classmass_temperature"]),
            max_per_class=int(self.config["residue_classmass_max_per_class"]),
            seed=int(self.config["residue_classmass_seed"]) + int(epoch) * 1000003 + int(step),
        )
        weighted = self.residue_classmass_weight * raw
        stats.update(raw_loss=raw.detach(), weighted_loss=weighted.detach())
        return weighted, stats

    def _pad_edge_context(
        self,
        ligand_edge_index: torch.Tensor,
        ligand_edge_attr: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate bond features per atom and scatter to [B, M, edge_dim+1]."""
        batch_size, max_atoms = ligand_mask.shape
        n_atoms = int(ligand_batch.numel())
        edge_attr_dim = int(self.config.get("edge_dim", 6))
        out = ligand_edge_attr.new_zeros((batch_size, max_atoms, edge_attr_dim + 1))
        if n_atoms == 0:
            return out
        if not bool((ligand_batch[1:] >= ligand_batch[:-1]).all()):
            raise ValueError("ligand_batch must be non-decreasing for V11m13 edge scatter")

        flat_sum = ligand_edge_attr.new_zeros((n_atoms, edge_attr_dim))
        flat_deg = ligand_edge_attr.new_zeros((n_atoms, 1))
        if ligand_edge_index.numel() > 0 and ligand_edge_attr.numel() > 0:
            src = ligand_edge_index[0].long()
            valid_src = (src >= 0) & (src < n_atoms)
            if valid_src.any():
                src = src[valid_src]
                edge_attr = ligand_edge_attr[valid_src, :edge_attr_dim]
                flat_sum.index_add_(0, src, edge_attr.to(dtype=flat_sum.dtype))
                flat_deg.index_add_(0, src, torch.ones((src.numel(), 1), device=src.device, dtype=flat_deg.dtype))
        mean_attr = flat_sum / flat_deg.clamp(min=1.0)
        degree = flat_deg.clamp(max=8.0) / 8.0
        flat_ctx = torch.cat([mean_attr, degree], dim=-1)

        counts = torch.bincount(ligand_batch, minlength=batch_size)
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=ligand_batch.device)
        cumsum[1:] = counts.cumsum(0)
        local_idx = torch.arange(n_atoms, device=ligand_batch.device) - cumsum[ligand_batch]
        valid = local_idx < max_atoms
        out[ligand_batch[valid], local_idx[valid]] = flat_ctx[valid].to(dtype=out.dtype)
        return out

    def _apply_interaction_kv_delta(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = super()._apply_interaction_kv_delta(k, v, local54)
        if self.local54_edge_context_k is None or self.local54_edge_context_v is None:
            return k, v
        edge_ctx = self._cached_v11m13_edge_context
        if edge_ctx is None:
            return k, v
        edge_ctx = edge_ctx.to(device=k.device, dtype=k.dtype)
        k_res = self.local54_edge_context_k(edge_ctx)
        v_res = self.local54_edge_context_v(edge_ctx).to(dtype=v.dtype)
        alpha = self._bounded_scale(
            self.local54_edge_context_raw_alpha.to(dtype=k.dtype),
            self.local54_edge_context_alpha_max,
        )
        valid = edge_ctx.abs().sum(dim=-1) > 0
        zero = k.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m13_stats["local54_edge_context_abs_mean"] = (
            edge_ctx.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        self._cached_v11m13_stats["local54_edge_context_kv_abs_mean"] = (
            v_res.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return k + alpha * k_res, v + alpha.to(dtype=v.dtype) * v_res

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ligand_edge_index = kwargs.get("ligand_edge_index")
        ligand_edge_attr = kwargs.get("ligand_edge_attr")
        ligand_batch = kwargs.get("ligand_batch")
        ligand_mask = kwargs.get("ligand_mask")
        if (
            self.use_local54_edge_context_kv
            and ligand_edge_index is not None
            and ligand_edge_attr is not None
            and ligand_batch is not None
            and ligand_mask is not None
        ):
            self._cached_v11m13_edge_context = self._pad_edge_context(
                ligand_edge_index=ligand_edge_index,
                ligand_edge_attr=ligand_edge_attr,
                ligand_batch=ligand_batch,
                ligand_mask=ligand_mask,
            )
        else:
            self._cached_v11m13_edge_context = None
        self._cached_v11m13_stats = {}

        self._cached_classmass_hidden = None
        try:
            outputs = super().forward(*args, **kwargs)
            if self._cached_classmass_hidden is not None:
                outputs["residue_classmass_hidden"] = self._cached_classmass_hidden
        finally:
            self._cached_classmass_hidden = None
        outputs["local54_edge_context_alpha"] = self._bounded_scale(
            self.local54_edge_context_raw_alpha.detach(),
            self.local54_edge_context_alpha_max,
        )
        for key, value in self._cached_v11m13_stats.items():
            outputs[key] = value
        return outputs
