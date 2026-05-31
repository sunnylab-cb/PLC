"""
Pocket-GVP-Fusion V11m7: V11m6n mainline with group-wise local54 adapter.

V11m7 keeps the stable V11m6n recipe:
- late local ligand delta only;
- no global/UniMol gate on the delta by default;
- fractional protein gradient for protected no-detach learning.

The only architectural change is inside the local54 chemistry path.  Instead
of projecting raw 54D atom features directly into local delta keys/values, it
splits local54 into chemistry groups and gives the empirically important
SASA/H-bond, flexibility/rotatable, and electronic groups more capacity.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m6 import PocketGVPFusionV11m6


LOCAL54_GROUP_SPECS = (
    ("base33", ((0, 33),), "local54_group_dim_base33", 48),
    ("electronic3", ((33, 36),), "local54_group_dim_electronic3", 64),
    ("sasa_hbond4", ((38, 42),), "local54_group_dim_sasa_hbond4", 96),
    ("flex_rot5", ((36, 38), (44, 47)), "local54_group_dim_flex_rot5", 80),
    ("metal4", ((47, 51),), "local54_group_dim_metal4", 16),
    ("stereo5", ((42, 44), (51, 54)), "local54_group_dim_stereo5", 16),
)


class Local54GroupAdapter(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        dropout = float(cfg.get("local54_group_dropout", 0.05))
        self.specs = []
        adapters = {}
        total_dim = 0
        for name, ranges, cfg_key, default_dim in LOCAL54_GROUP_SPECS:
            in_dim = int(sum(end - start for start, end in ranges))
            out_dim = int(cfg.get(cfg_key, default_dim))
            self.specs.append((name, ranges))
            total_dim += out_dim
            adapters[name] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
        self.adapters = nn.ModuleDict(adapters)
        out_dim = int(cfg.get("local54_group_adapter_dim", 128))
        self.out_dim = out_dim
        self.mix = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, local54: torch.Tensor) -> torch.Tensor:
        parts = []
        for name, ranges in self.specs:
            x = torch.cat([local54[..., start:end] for start, end in ranges], dim=-1)
            parts.append(self.adapters[name](x))
        return self.mix(torch.cat(parts, dim=-1))


class PocketGVPFusionV11m7(PocketGVPFusionV11m6):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m6.DEFAULT_CONFIG,
        "use_local54_group_adapter": True,
        "local54_group_adapter_dim": 128,
        "local54_group_dropout": 0.05,
        "local54_group_dim_base33": 48,
        "local54_group_dim_electronic3": 64,
        "local54_group_dim_sasa_hbond4": 96,
        "local54_group_dim_flex_rot5": 80,
        "local54_group_dim_metal4": 16,
        "local54_group_dim_stereo5": 16,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_local54_group_adapter = bool(cfg.get("use_local54_group_adapter", True))
        if self.use_local_delta_only and self.use_local54_group_adapter:
            d_model = int(self.config["d_model"])
            self.local54_group_adapter = Local54GroupAdapter(cfg)
            adapter_dim = int(self.local54_group_adapter.out_dim)
            self.local_delta_k_local = nn.Linear(adapter_dim, self.local_delta_dim, bias=False)
            self.local_delta_v_local = nn.Linear(adapter_dim, d_model, bias=False)
        else:
            self.local54_group_adapter = None

        self.config["use_local54_group_adapter"] = self.use_local54_group_adapter
        self.config["local54_group_adapter_dim"] = int(cfg.get("local54_group_adapter_dim", 128))

    def _encode_local_delta_features(self, local54: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        local = local54.to(dtype=dtype)
        if self.local54_group_adapter is None:
            return super()._encode_local_delta_features(local54, dtype)
        return self.local54_group_adapter(local).to(dtype=dtype)
