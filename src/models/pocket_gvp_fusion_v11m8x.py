"""
Pocket-GVP-Fusion V11m8x: V11m8 all_light with V12-style fusion probes.

This branch keeps the latest V11m8 self-attention/local-delta stack intact:
- protein self-attention remains the V11 RoPE/dist-bias context encoder;
- ligand self-attention remains the V11 ligand self-attn path;
- V11m8 all_light local_delta/FiLM/virtual-token/center probes are unchanged.

Two mutually exclusive fusion-loop replacements are supported:
- cross-only: replace each V11 bidirectional cross layer with V12 gated cross
  attention wrapped in a bounded residual scale.
- co-attn: replace each V11 fusion layer with a V12 CoAttentionBlock adapter
  after the original V11 protein/ligand self-attention has already run.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m8 import PocketGVPFusionV11m8, _atanh_ratio
from .pocket_gvp_fusion_v12m2 import (
    CoAttentionBlock,
    V12CrossAttentionBlock,
    zero_init_co_attention_residuals,
)


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


class _V12CoAttentionAsV11CrossBlock(nn.Module):
    """
    Adapter that lets the V11 fusion loop call a V12 CoAttentionBlock.

    V11 expects each layer to expose `prot_to_lig()` followed by `lig_to_prot()`.
    V12 CoAttentionBlock updates both streams in one call, so `prot_to_lig()`
    runs the full co-attention block and caches the ligand output for the
    immediately following `lig_to_prot()` call.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        drop_path_prob: float,
        attn_dropout: Optional[float],
        cross_attn_logit_clip: Optional[float],
        self_attn_logit_clip: Optional[float],
        d_unimol: int,
        use_film: bool,
        use_prot_dist_bias: bool,
        use_lig_dist_bias: bool,
        inner_residual_scale: float,
        cross_residual_scale: Optional[float],
        use_rezero_branch_gates: bool,
        rezero_gate_init: float,
        p2l_length_scale: bool,
        p2l_length_ref: float,
        p2l_rezero_gate_max: Optional[float],
        use_flash_attn: bool,
        flash_attn_fallback: bool,
        zero_init_residuals: bool,
    ):
        super().__init__()
        self.block = CoAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            drop_path_prob=drop_path_prob,
            attn_dropout=attn_dropout,
            d_unimol=d_unimol,
            use_film=use_film,
            use_prot_dist_bias=use_prot_dist_bias,
            use_lig_dist_bias=use_lig_dist_bias,
            cross_attn_logit_clip=cross_attn_logit_clip,
            self_attn_logit_clip=self_attn_logit_clip,
            inner_residual_scale=inner_residual_scale,
            cross_residual_scale=cross_residual_scale,
            use_rezero_branch_gates=use_rezero_branch_gates,
            rezero_gate_init=rezero_gate_init,
            p2l_length_scale=p2l_length_scale,
            p2l_length_ref=p2l_length_ref,
            p2l_rezero_gate_max=p2l_rezero_gate_max,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        if zero_init_residuals:
            zero_init_co_attention_residuals(self.block)
        self._pending_lig = None

    def forward_pair(
        self,
        h_prot: torch.Tensor,
        h_lig: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None,
    ):
        """Stateless joint update for activation-checkpoint replay."""
        h_prot, h_lig = self.block(
            h_prot=h_prot,
            h_lig=h_lig,
            unimol_embed=None,
            prot_mask=prot_mask,
            lig_mask=lig_mask,
            prot_coords=None,
            lig_coords=None,
            position_ids=None,
        )
        if lig_mask is not None:
            h_lig = h_lig.masked_fill(~lig_mask.unsqueeze(-1), 0.0)
        return h_prot, h_lig

    def prot_to_lig(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._pending_lig = None
        h_prot, h_lig = self.block(
            h_prot=q,
            h_lig=kv,
            unimol_embed=None,
            prot_mask=q_mask,
            lig_mask=kv_mask,
            prot_coords=None,
            lig_coords=None,
            position_ids=None,
        )
        self._pending_lig = h_lig
        return h_prot

    def lig_to_prot(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._pending_lig is None:
            raise RuntimeError("V12 co-attention adapter expected prot_to_lig() before lig_to_prot()")
        h_lig = self._pending_lig
        self._pending_lig = None
        if q_mask is not None:
            h_lig = h_lig.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return h_lig


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
        "use_v12_coattention_block": False,
        "v12_coattn_use_film": False,
        "v12_coattn_use_prot_dist_bias": False,
        "v12_coattn_use_lig_dist_bias": False,
        "v12_coattn_self_attn_logit_clip": 30.0,
        "v12_coattn_inner_residual_scale": 0.05,
        "v12_coattn_cross_residual_scale": 0.25,
        "v12_coattn_use_rezero_branch_gates": True,
        "v12_coattn_rezero_gate_init": 0.05,
        "v12_coattn_p2l_length_scale": False,
        "v12_coattn_p2l_length_ref": 256.0,
        "v12_coattn_p2l_rezero_gate_max": 0.20,
        "v12_coattn_zero_init_residuals": True,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_v12_coattention_block = bool(cfg.get("use_v12_coattention_block", False))
        self.use_v12_cross_attention = bool(cfg.get("use_v12_cross_attention", True)) and not self.use_v12_coattention_block
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
        elif self.use_v12_coattention_block:
            c = self.config
            n_cross_layers = int(c.get("n_cross_attention_layers", 4))
            stochastic_depth_prob = float(c.get("stochastic_depth_prob", 0.0))
            drop_path_probs = [
                stochastic_depth_prob * i / max(n_cross_layers - 1, 1)
                for i in range(n_cross_layers)
            ]
            self.cross_attention_layers = nn.ModuleList([
                _V12CoAttentionAsV11CrossBlock(
                    d_model=int(c["d_model"]),
                    n_heads=int(c["n_heads"]),
                    dropout=float(c["dropout"]),
                    drop_path_prob=drop_path_probs[i],
                    attn_dropout=c.get("attn_dropout", None),
                    cross_attn_logit_clip=self.v12_cross_attn_logit_clip,
                    self_attn_logit_clip=cfg.get("v12_coattn_self_attn_logit_clip", 30.0),
                    d_unimol=int(c.get("d_unimol", 512)),
                    use_film=bool(cfg.get("v12_coattn_use_film", False)),
                    use_prot_dist_bias=bool(cfg.get("v12_coattn_use_prot_dist_bias", False)),
                    use_lig_dist_bias=bool(cfg.get("v12_coattn_use_lig_dist_bias", False)),
                    inner_residual_scale=float(cfg.get("v12_coattn_inner_residual_scale", 0.05)),
                    cross_residual_scale=cfg.get("v12_coattn_cross_residual_scale", 0.25),
                    use_rezero_branch_gates=bool(cfg.get("v12_coattn_use_rezero_branch_gates", True)),
                    rezero_gate_init=float(cfg.get("v12_coattn_rezero_gate_init", 0.05)),
                    p2l_length_scale=bool(cfg.get("v12_coattn_p2l_length_scale", False)),
                    p2l_length_ref=float(cfg.get("v12_coattn_p2l_length_ref", 256.0)),
                    p2l_rezero_gate_max=cfg.get("v12_coattn_p2l_rezero_gate_max", 0.20),
                    use_flash_attn=bool(cfg.get("v12_cross_use_flash_attn", False)),
                    flash_attn_fallback=bool(cfg.get("v12_cross_flash_attn_fallback", True)),
                    zero_init_residuals=bool(cfg.get("v12_coattn_zero_init_residuals", True)),
                )
                for i in range(n_cross_layers)
            ])

        self.config["use_v12_coattention_block"] = self.use_v12_coattention_block
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

    def _coattn_stat_values(self, attr: str) -> list[torch.Tensor]:
        values = []
        for layer in self.cross_attention_layers:
            block = getattr(layer, "block", None)
            value = getattr(block, attr, None)
            if value is not None:
                values.append(value.detach().float().mean())
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
        if self.use_v12_coattention_block:
            for out_key, attr in (
                ("v12_coattn_p2l_gate_mean", "_cached_p2l_gate"),
                ("v12_coattn_l2p_gate_mean", "_cached_l2p_gate"),
                ("v12_coattn_p2l_delta_norm_mean", "_cached_p2l_delta_norm"),
                ("v12_coattn_l2p_delta_norm_mean", "_cached_l2p_delta_norm"),
            ):
                values = self._coattn_stat_values(attr)
                if values:
                    outputs[out_key] = torch.stack(values).float().mean()
        return outputs
