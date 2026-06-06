"""
Pocket-GVP-Fusion V11m8x: V11m8 all_light with V12-style cross attention.

This branch keeps the latest V11m8 self-attention/local-delta stack intact:
- protein self-attention remains the V11 RoPE/dist-bias context encoder;
- ligand self-attention remains the V11 ligand self-attn path;
- V11m8 all_light local_delta/FiLM/virtual-token/center probes are unchanged.

Only the bidirectional cross-attention loop is swapped to V12's padding-safe
gated cross-attention block, wrapped in a bounded residual scale so it cannot
overwrite the stable V11 trunk at initialization.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m8 import PocketGVPFusionV11m8, _atanh_ratio
from .pocket_gvp_fusion_v12m2 import V12CrossAttentionBlock


class _ScaledV12CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        drop_path_prob: float,
        attn_dropout: Optional[float],
        attn_logit_clip: Optional[float],
        alpha_init: float,
        alpha_max: float,
        use_flash_attn: bool,
        flash_attn_fallback: bool,
        zero_init_final: bool,
    ):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.block = V12CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            drop_path_prob=drop_path_prob,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        if zero_init_final:
            nn.init.zeros_(self.block.cross_attn.w_o.weight)
            if hasattr(self.block.ffn, "w_2"):
                nn.init.zeros_(self.block.ffn.w_2.weight)
        self.raw_alpha = nn.Parameter(torch.tensor(_atanh_ratio(alpha_init, self.alpha_max)))
        self._last_delta_norm = None

    def _alpha(self) -> torch.Tensor:
        if self.alpha_max <= 0:
            return self.raw_alpha
        return self.raw_alpha.tanh() * self.alpha_max

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.block(q=q, kv=kv, q_mask=q_mask, kv_mask=kv_mask)
        delta = out - q
        valid = q_mask.bool() if q_mask is not None else torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        self._last_delta_norm = (
            delta.detach().float().norm(dim=-1)[valid].mean()
            if valid.any()
            else q.new_tensor(0.0, dtype=torch.float32)
        )
        alpha = self._alpha().to(dtype=q.dtype)
        return q + alpha * delta


class _V12StyleBidirectionalCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        drop_path_prob: float,
        attn_dropout: Optional[float],
        attn_logit_clip: Optional[float],
        alpha_init: float,
        alpha_max: float,
        use_flash_attn: bool,
        flash_attn_fallback: bool,
        zero_init_final: bool,
    ):
        super().__init__()
        kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            drop_path_prob=drop_path_prob,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
            zero_init_final=zero_init_final,
        )
        self.prot_to_lig = _ScaledV12CrossAttentionBlock(**kwargs)
        self.lig_to_prot = _ScaledV12CrossAttentionBlock(**kwargs)


class PocketGVPFusionV11m8x(PocketGVPFusionV11m8):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m8.DEFAULT_CONFIG,
        "use_v12_cross_attention": True,
        "v12_cross_alpha_init": 0.25,
        "v12_cross_alpha_max": 0.60,
        "v12_cross_attn_logit_clip": 30.0,
        "v12_cross_zero_init_final": False,
        "v12_cross_use_flash_attn": False,
        "v12_cross_flash_attn_fallback": True,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_v12_cross_attention = bool(cfg.get("use_v12_cross_attention", True))
        self.v12_cross_alpha_init = float(cfg.get("v12_cross_alpha_init", 0.25))
        self.v12_cross_alpha_max = float(cfg.get("v12_cross_alpha_max", 0.60))
        self.v12_cross_attn_logit_clip = cfg.get("v12_cross_attn_logit_clip", 30.0)
        if self.v12_cross_attn_logit_clip is not None:
            self.v12_cross_attn_logit_clip = float(self.v12_cross_attn_logit_clip)
        self.v12_cross_zero_init_final = bool(cfg.get("v12_cross_zero_init_final", False))

        if self.use_v12_cross_attention:
            c = self.config
            n_cross_layers = int(c.get("n_cross_attention_layers", 4))
            stochastic_depth_prob = float(c.get("stochastic_depth_prob", 0.0))
            drop_path_probs = [
                stochastic_depth_prob * i / max(n_cross_layers - 1, 1)
                for i in range(n_cross_layers)
            ]
            self.cross_attention_layers = nn.ModuleList([
                _V12StyleBidirectionalCrossAttentionBlock(
                    d_model=int(c["d_model"]),
                    n_heads=int(c["n_heads"]),
                    dropout=float(c["dropout"]),
                    drop_path_prob=drop_path_probs[i],
                    attn_dropout=c.get("attn_dropout", None),
                    attn_logit_clip=self.v12_cross_attn_logit_clip,
                    alpha_init=self.v12_cross_alpha_init,
                    alpha_max=self.v12_cross_alpha_max,
                    use_flash_attn=bool(cfg.get("v12_cross_use_flash_attn", False)),
                    flash_attn_fallback=bool(cfg.get("v12_cross_flash_attn_fallback", True)),
                    zero_init_final=self.v12_cross_zero_init_final,
                )
                for i in range(n_cross_layers)
            ])

        self.config["use_v12_cross_attention"] = self.use_v12_cross_attention
        self.config["v12_cross_alpha_init"] = self.v12_cross_alpha_init
        self.config["v12_cross_alpha_max"] = self.v12_cross_alpha_max
        self.config["v12_cross_attn_logit_clip"] = self.v12_cross_attn_logit_clip
        self.config["v12_cross_zero_init_final"] = self.v12_cross_zero_init_final

    def _cross_alpha_values(self) -> list[torch.Tensor]:
        values = []
        for layer in self.cross_attention_layers:
            for block in (getattr(layer, "prot_to_lig", None), getattr(layer, "lig_to_prot", None)):
                if hasattr(block, "_alpha"):
                    values.append(block._alpha().detach())
        return values

    def _cross_delta_norms(self) -> list[torch.Tensor]:
        values = []
        for layer in self.cross_attention_layers:
            for block in (getattr(layer, "prot_to_lig", None), getattr(layer, "lig_to_prot", None)):
                value = getattr(block, "_last_delta_norm", None)
                if value is not None:
                    values.append(value.detach())
        return values

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = super().forward(*args, **kwargs)
        if self.use_v12_cross_attention:
            alpha_values = self._cross_alpha_values()
            if alpha_values:
                outputs["v12_cross_alpha_mean"] = torch.stack(alpha_values).float().mean()
            delta_norms = self._cross_delta_norms()
            if delta_norms:
                outputs["v12_cross_delta_norm_mean"] = torch.stack(delta_norms).float().mean()
        return outputs
