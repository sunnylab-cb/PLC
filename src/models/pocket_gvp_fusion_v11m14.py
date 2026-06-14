"""
Pocket-GVP-Fusion V11m14: base33-conditioned local chemistry token.

V11m13/base4 showed that adding base33 as a direct evidence group does not open
the ceiling.  V11m14 keeps base33 as an identity anchor and forms one bounded
conditioned chemistry token from:
  base33 + electronic3 + sasa_hbond4 + flex_rot5
The token can add a small residual to local_delta K/V and to the residue-chem
compatibility readout without changing the V11m8x/co-attn trunk.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m8 import _atanh_ratio
from .pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13


class PocketGVPFusionV11m14(PocketGVPFusionV11m13):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m13.DEFAULT_CONFIG,
        "use_local54_basecond_token": False,
        "local54_basecond_hidden": 96,
        "local54_basecond_kv_alpha_init": 0.02,
        "local54_basecond_kv_alpha_max": 0.12,
        "local54_basecond_compat_alpha_init": 0.015,
        "local54_basecond_compat_alpha_max": 0.10,
        "local54_basecond_compat_gate_bias_init": 0.005,
        "local54_basecond_compat_read_scale": 0.30,
        "local54_basecond_compat_normalize_read": True,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_local54_basecond_token = bool(cfg.get("use_local54_basecond_token", False))
        self.local54_basecond_kv_alpha_max = float(
            cfg.get("local54_basecond_kv_alpha_max", 0.12)
        )
        self.local54_basecond_compat_alpha_max = float(
            cfg.get("local54_basecond_compat_alpha_max", 0.10)
        )
        self.local54_basecond_compat_read_scale = float(
            cfg.get("local54_basecond_compat_read_scale", 0.30)
        )
        self.local54_basecond_compat_normalize_read = bool(
            cfg.get("local54_basecond_compat_normalize_read", True)
        )

        d_model = int(self.config["d_model"])
        local_dim = int(self.local_delta_local_dim)
        hidden = int(cfg.get("local54_basecond_hidden", 96))
        in_dim = 33 + 3 + 4 + 5

        if self.use_local54_basecond_token:
            self.local54_basecond_token = nn.Sequential(
                nn.LayerNorm(in_dim, elementwise_affine=False),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.local54_basecond_k = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.local_delta_dim, bias=False),
            )
            self.local54_basecond_v = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, d_model, bias=False),
            )
            self.local54_basecond_compat_value = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, d_model, bias=False),
            )
            self.local54_basecond_compat_gate = nn.Sequential(
                nn.LayerNorm(d_model + d_model + local_dim),
                nn.Linear(d_model + d_model + local_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            if self.local54_basecond_compat_normalize_read:
                self.local54_basecond_compat_norm = nn.LayerNorm(
                    d_model,
                    elementwise_affine=False,
                )
            else:
                self.local54_basecond_compat_norm = None

            nn.init.zeros_(self.local54_basecond_k[-1].weight)
            nn.init.zeros_(self.local54_basecond_v[-1].weight)
            nn.init.zeros_(self.local54_basecond_compat_value[-1].weight)
            nn.init.zeros_(self.local54_basecond_compat_gate[-1].weight)
            nn.init.constant_(
                self.local54_basecond_compat_gate[-1].bias,
                float(cfg.get("local54_basecond_compat_gate_bias_init", 0.005)),
            )
            self.local54_basecond_kv_raw_alpha = nn.Parameter(
                torch.tensor(
                    _atanh_ratio(
                        float(cfg.get("local54_basecond_kv_alpha_init", 0.02)),
                        self.local54_basecond_kv_alpha_max,
                    )
                )
            )
            self.local54_basecond_compat_raw_alpha = nn.Parameter(
                torch.tensor(
                    _atanh_ratio(
                        float(cfg.get("local54_basecond_compat_alpha_init", 0.015)),
                        self.local54_basecond_compat_alpha_max,
                    )
                )
            )
        else:
            self.local54_basecond_token = None
            self.local54_basecond_k = None
            self.local54_basecond_v = None
            self.local54_basecond_compat_value = None
            self.local54_basecond_compat_gate = None
            self.local54_basecond_compat_norm = None
            self.register_buffer(
                "local54_basecond_kv_raw_alpha",
                torch.tensor(0.0),
                persistent=False,
            )
            self.register_buffer(
                "local54_basecond_compat_raw_alpha",
                torch.tensor(0.0),
                persistent=False,
            )

        self.config["use_local54_basecond_token"] = self.use_local54_basecond_token
        self._cached_v11m14_stats = {}

    def _basecond_input(self, local54: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pieces = [
            self._extract_local54_group(local54, "base33"),
            self._extract_local54_group(local54, "electronic3"),
            self._extract_local54_group(local54, "sasa_hbond4"),
            self._extract_local54_group(local54, "flex_rot5"),
        ]
        return torch.cat(pieces, dim=-1).to(dtype=dtype)

    def _basecond_token(self, local54: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.local54_basecond_token is None:
            raise RuntimeError("local54 base-conditioned token is not initialized")
        return self.local54_basecond_token(self._basecond_input(local54, dtype))

    def _apply_interaction_kv_delta(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = super()._apply_interaction_kv_delta(k, v, local54)
        if self.local54_basecond_k is None or self.local54_basecond_v is None:
            return k, v

        token = self._basecond_token(local54, k.dtype)
        k_res = self.local54_basecond_k(token)
        v_res = self.local54_basecond_v(token).to(dtype=v.dtype)
        alpha = self._bounded_scale(
            self.local54_basecond_kv_raw_alpha.to(dtype=k.dtype),
            self.local54_basecond_kv_alpha_max,
        )
        self._cached_v11m14_stats["local54_basecond_token_abs_mean"] = (
            token.detach().float().abs().mean()
        )
        self._cached_v11m14_stats["local54_basecond_kv_abs_mean"] = (
            v_res.detach().float().abs().mean()
        )
        return k + alpha * k_res, v + alpha.to(dtype=v.dtype) * v_res

    def _apply_residue_chem_compat(
        self,
        hp: torch.Tensor,
        lig_read: torch.Tensor,
        local54: torch.Tensor,
        ligand_context: torch.Tensor,
        attn: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        lig_read = super()._apply_residue_chem_compat(
            hp=hp,
            lig_read=lig_read,
            local54=local54,
            ligand_context=ligand_context,
            attn=attn,
            protein_mask=protein_mask,
        )
        if (
            self.local54_basecond_compat_value is None
            or self.local54_basecond_compat_gate is None
        ):
            return lig_read

        token = self._basecond_token(local54, lig_read.dtype)
        value = self.local54_basecond_compat_value(token).to(dtype=lig_read.dtype)
        compat_read = torch.matmul(attn, value)
        raw_read = compat_read
        if self.local54_basecond_compat_norm is not None:
            compat_read = self.local54_basecond_compat_norm(
                compat_read.to(dtype=hp.dtype)
            ).to(dtype=lig_read.dtype)
        if self.local54_basecond_compat_read_scale != 1.0:
            compat_read = compat_read * compat_read.new_tensor(
                self.local54_basecond_compat_read_scale
            )

        context = ligand_context.to(dtype=hp.dtype).unsqueeze(1).expand(-1, hp.size(1), -1)
        gate = torch.tanh(self.local54_basecond_compat_gate(torch.cat([hp, context], dim=-1)))
        gate = gate.to(dtype=lig_read.dtype)
        alpha = self._bounded_scale(
            self.local54_basecond_compat_raw_alpha.to(dtype=lig_read.dtype),
            self.local54_basecond_compat_alpha_max,
        )
        mask = protein_mask.unsqueeze(-1).to(dtype=lig_read.dtype)

        valid = protein_mask.bool()
        zero = hp.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m14_stats["local54_basecond_compat_gate_abs_mean"] = (
            gate.detach()[valid].abs().mean() if valid.any() else zero
        )
        self._cached_v11m14_stats["local54_basecond_compat_read_raw_abs_mean"] = (
            raw_read.detach()[valid].abs().mean() if valid.any() else zero
        )
        self._cached_v11m14_stats["local54_basecond_compat_read_abs_mean"] = (
            compat_read.detach()[valid].abs().mean() if valid.any() else zero
        )
        return lig_read + alpha * compat_read * gate * mask

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        self._cached_v11m14_stats = {}
        outputs = super().forward(*args, **kwargs)
        outputs["local54_basecond_kv_alpha"] = self._bounded_scale(
            self.local54_basecond_kv_raw_alpha.detach(),
            self.local54_basecond_kv_alpha_max,
        )
        outputs["local54_basecond_compat_alpha"] = self._bounded_scale(
            self.local54_basecond_compat_raw_alpha.detach(),
            self.local54_basecond_compat_alpha_max,
        )
        for key, value in self._cached_v11m14_stats.items():
            outputs[key] = value
        return outputs
