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


class PocketGVPFusionV11m13(PocketGVPFusionV11m10):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m10.DEFAULT_CONFIG,
        "use_local54_edge_context_kv": False,
        "local54_edge_context_hidden": 96,
        "local54_edge_context_alpha_init": 0.03,
        "local54_edge_context_alpha_max": 0.20,
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

        outputs = super().forward(*args, **kwargs)
        outputs["local54_edge_context_alpha"] = self._bounded_scale(
            self.local54_edge_context_raw_alpha.detach(),
            self.local54_edge_context_alpha_max,
        )
        for key, value in self._cached_v11m13_stats.items():
            outputs[key] = value
        return outputs
