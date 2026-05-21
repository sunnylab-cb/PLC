"""
Pocket-GVP-Fusion V12m3: V12m2 + residue-atom pair-evidence head.

Architecture changes from V11:
  移除: 獨立 Protein Context Encoder (Module 4)
        獨立 Ligand Self-Attention (Module 5b)
        Cross-Attention Loop (Module 6)
  新增: N=6 個 CoAttentionBlock，每個 Block 包含:
          1. FiLM conditioning (Uni-Mol CLS → protein 全域化學調製)
          2. Protein Self-Attention (RoPE + 3D distance bias)
          3. Ligand Self-Attention (3D distance bias)
          4. Bidirectional Cross-Attention (P→L + L→P)

保留自 V11: Modules 1-3 (GVP + ESM Fusion), Module 5a (Ligand GNN), Module 7 (Classifier)
             DistanceBias, LigandDistanceBias, RoPE, GVP Output Dropout, Coord Noise

新增 Innovation:
  [1] FiLMLayer: 每個 Block 以 Uni-Mol 全域化學語義廣播調製蛋白質

重要限制:
  蛋白質坐標與配體坐標各自處於獨立坐標系（非 Holo 對齊態）。
  絕不能計算跨模態幾何距離（Cα ↔ 配體原子），此類特徵屬於偽特徵。
  合法幾何信息：蛋白質內 Cα-Cα 距離、配體原子間距離（各自坐標系內部）。

接受 unimol_embed [B, 512] 作為新輸入 (若不提供則退化為零向量，FiLM = identity)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple

from .gvp_layers import GVPEncoder, build_protein_graph, rbf_encode
from .ligand_encoder_3d import Ligand3DEncoder
from .chain_encoding import get_cumulative_gap_position_ids
# Re-use from V11 (avoid code duplication)
try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # Optional HPC-only dependency
    flash_attn_varlen_func = None

from .pocket_gvp_fusion_v11 import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    DistanceBias,
    LigandDistanceBias,
    SwiGLU,
    PairwiseBridge,
    drop_path,
)


def convert_layernorm_to_rmsnorm(module: nn.Module) -> None:
    """
    V12-local normalization swap.

    ROCm training probes showed repeated non-finite gradients in LayerNorm
    affine parameters after optimizer steps, even with co-attn disabled.
    RMSNorm removes the mean-subtraction/bias path and was stable in the same
    probes.  This helper intentionally mutates only the V12 model instance; it
    does not touch V11 or shared transformer classes.
    """
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
            convert_layernorm_to_rmsnorm(child)


def freeze_norm_parameters(module: nn.Module) -> int:
    """Freeze direct parameters of norm modules under a given module tree."""
    norm_types = (nn.LayerNorm, nn.RMSNorm)
    n_frozen = 0
    for child in module.modules():
        if isinstance(child, norm_types):
            for p in child.parameters(recurse=False):
                if p.requires_grad:
                    p.requires_grad_(False)
                    n_frozen += p.numel()
    return n_frozen


# ===========================================================================
# V12-local RoPE Self-Attention
# ===========================================================================

class V12RoPEGatedMultiHeadAttention(nn.Module):
    """
    V12-local RoPE gated attention.

    This intentionally does not reuse V11's RoPEGatedMultiHeadAttention because
    V12 needs stricter numerical guards for repeated co-attention blocks:
    fp32 softmax, optional finite-logit clipping, all-masked-row protection, and
    query-padding output zeroing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_gate = use_gate
        self.attn_logit_clip = attn_logit_clip

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        if use_gate:
            self.gate = nn.Linear(d_model, d_model, bias=False)

        _attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.attn_dropout = nn.Dropout(_attn_dropout)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(L, x.device, position_ids=position_ids)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias

        if self.attn_logit_clip is not None:
            clip = float(self.attn_logit_clip)
            attn = torch.nan_to_num(
                attn, nan=0.0, posinf=clip, neginf=-clip
            )
            attn = attn.clamp(-clip, clip)

        query_mask = None
        if mask is not None:
            if mask.dim() == 2:
                query_mask = mask
                key_mask = mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, L]
            elif mask.dim() == 3:
                key_mask = mask.unsqueeze(1)                # [B, 1, L, L]
            else:
                raise ValueError(f"Unsupported attention mask shape: {mask.shape}")
            attn = attn.masked_fill(~key_mask, float('-inf'))

        all_masked = torch.isneginf(attn).all(dim=-1, keepdim=True)
        if all_masked.any():
            attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.d_model)

        if query_mask is not None:
            out = out.masked_fill(~query_mask.unsqueeze(-1), 0.0)

        out = self.w_o(out)
        if self.use_gate:
            out = out * torch.sigmoid(self.gate(x))
        return out


class V12RoPEGatedTransformerBlock(nn.Module):
    """V12-local Transformer block with padding-safe self-attention and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gated_attn: bool = True,
        use_swiglu: bool = True,
        drop_path_prob: float = 0.0,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = V12RoPEGatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_gate=use_gated_attn,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop_path_prob = drop_path_prob

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_mask = mask if mask is not None and mask.dim() == 2 else None

        attn_out = self.attn(
            self.norm1(x),
            mask=mask,
            attn_bias=attn_bias,
            position_ids=position_ids,
        )
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        if query_mask is not None:
            x = x.masked_fill(~query_mask.unsqueeze(-1), 0.0)

        ffn_out = self.ffn(self.norm2(x))
        if query_mask is not None:
            ffn_out = ffn_out.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        x = x + drop_path(ffn_out, self.drop_path_prob, self.training)
        if query_mask is not None:
            x = x.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        return x


class V12GatedCrossAttention(nn.Module):
    """V12-local cross attention with all-masked-row protection."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.attn_logit_clip = attn_logit_clip
        self.use_flash_attn = bool(use_flash_attn)
        self.flash_attn_fallback = bool(flash_attn_fallback)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)

        _attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.attn_dropout = nn.Dropout(_attn_dropout)


    def _flash_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor],
        kv_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_flash_attn:
            return None
        if flash_attn_varlen_func is None:
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("use_flash_attn=True but flash-attn is not installed")
        if q.dtype not in (torch.float16, torch.bfloat16):
            if self.flash_attn_fallback:
                return None
            raise RuntimeError(f"flash-attn requires fp16/bf16, got {q.dtype}")

        B, H, L, D = q.shape
        S = k.size(2)
        if q_mask is None:
            q_mask = torch.ones(B, L, dtype=torch.bool, device=q.device)
        if kv_mask is None:
            kv_mask = torch.ones(B, S, dtype=torch.bool, device=q.device)
        if q_mask.dim() != 2 or kv_mask.dim() != 2:
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("flash-attn cross attention supports only 2D q/kv masks")

        q_mask = q_mask.bool()
        kv_mask = kv_mask.bool()
        q_lens = q_mask.sum(dim=1, dtype=torch.int32)
        kv_lens = kv_mask.sum(dim=1, dtype=torch.int32)
        if (q_lens == 0).any() or (kv_lens == 0).any():
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("flash-attn received an empty q/kv sequence")

        q_b = q.transpose(1, 2).contiguous()  # [B, L, H, D]
        k_b = k.transpose(1, 2).contiguous()  # [B, S, H, D]
        v_b = v.transpose(1, 2).contiguous()
        q_unpad = q_b[q_mask]
        k_unpad = k_b[kv_mask]
        v_unpad = v_b[kv_mask]
        cu_q = F.pad(q_lens.cumsum(0), (1, 0)).to(torch.int32)
        cu_k = F.pad(kv_lens.cumsum(0), (1, 0)).to(torch.int32)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, cu_q, cu_k,
            int(q_lens.max().item()), int(kv_lens.max().item()),
            dropout_p=dropout_p, softmax_scale=self.scale, causal=False,
        )
        out = q_b.new_zeros(B, L, H, D)
        out[q_mask] = out_unpad
        return out.transpose(1, 2).reshape(B, L, self.d_model)

    def forward(
        self,
        q_in: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = q_in.shape
        S = kv.size(1)

        q = self.w_q(q_in).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        flash_out = self._flash_forward(q, k, v, q_mask=q_mask, kv_mask=kv_mask)
        if flash_out is not None:
            out = self.w_o(flash_out)
            out = out * torch.sigmoid(self.gate(q_in))
            if q_mask is not None:
                out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
            return out

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.attn_logit_clip is not None:
            clip = float(self.attn_logit_clip)
            attn = torch.nan_to_num(
                attn, nan=0.0, posinf=clip, neginf=-clip
            )
            attn = attn.clamp(-clip, clip)

        if kv_mask is not None:
            if kv_mask.dim() == 2:
                key_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            elif kv_mask.dim() == 3:
                key_mask = kv_mask.unsqueeze(1)               # [B, 1, L, S]
            else:
                raise ValueError(f"Unsupported cross-attn mask shape: {kv_mask.shape}")
            attn = attn.masked_fill(~key_mask, float('-inf'))

        all_masked = torch.isneginf(attn).all(dim=-1, keepdim=True)
        if all_masked.any():
            attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        if all_masked.any():
            attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.d_model)
        out = self.w_o(out)
        out = out * torch.sigmoid(self.gate(q_in))
        if q_mask is not None:
            out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return out


class V12CrossAttentionBlock(nn.Module):
    """V12-local CrossAttentionBlock with padding-safe attention and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = V12GatedCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_prob = drop_path_prob

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = q
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        if q_mask is not None:
            q_normed = q_normed.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        if kv_mask is not None and kv_mask.dim() == 2:
            kv_normed = kv_normed.masked_fill(~kv_mask.unsqueeze(-1), 0.0)
        out = self.cross_attn(q_normed, kv_normed, q_mask=q_mask, kv_mask=kv_mask)
        out = drop_path(self.dropout(out), self.drop_path_prob, self.training) + residual
        if q_mask is not None:
            out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)

        residual = out
        ffn_out = self.ffn(self.norm2(out))
        if q_mask is not None:
            ffn_out = ffn_out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        out = drop_path(self.dropout(ffn_out), self.drop_path_prob, self.training) + residual
        if q_mask is not None:
            out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return out


# ===========================================================================
# FiLM Conditioning Layer  [V12 Innovation #1]
# ===========================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation from Uni-Mol global CLS embedding.

    Residual delta form:
        h_prot ← h_prot + γ * LN(h_prot) + β

    γ, β are predicted from unimol_embed [B, d_unimol].
    Zero-init → γ=0, β=0 → delta=0 → true identity (output = h_prot exactly).

    Using LN inside delta (not on main path) keeps the conditioning numerically
    stable while leaving h_prot unmodified when unimol_embed is absent.
    """

    def __init__(
        self,
        d_unimol: int = 512,
        d_model: int = 512,
        zero_init: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(d_unimol, 2 * d_model, bias=True)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_prot: torch.Tensor,
        unimol_embed: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_prot:       [B, L, d_model]
            unimol_embed: [B, d_unimol]
        Returns:
            [B, L, d_model]  — equals h_prot exactly when proj is zero-init
        """
        params = self.proj(unimol_embed)               # [B, 2*d_model]
        gamma, beta = params.chunk(2, dim=-1)          # [B, d_model] each
        delta = gamma.unsqueeze(1) * self.norm(h_prot) + beta.unsqueeze(1)
        if prot_mask is not None:
            delta = delta.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
        h_out = h_prot + delta
        if prot_mask is not None:
            h_out = h_out.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
        return h_out


# ===========================================================================
# SASA Residual Gate  [V12 Innovation #2]
# ===========================================================================

class SASAResidualGate(nn.Module):
    """
    Learnable SASA residual gate applied to h_prot before the classifier.

    Form:  h_out = h_prot * (1 + amplitude * tanh(linear(rsa) * rsa_mask))

    - Zero-init linear → tanh(0) = 0 → gate = 1 → identity at training start.
    - After training, the gate direction and magnitude are fully learnable;
      it is NOT a fixed rule that surface residues get amplified.
    - amplitude bounds the modulation to ±amplitude (default 0.5 → [0.5×, 1.5×]).
    - rsa_mask (0/1 float): where 0, forces delta=0 (gate=identity) regardless of
      rsa_raw value. Prevents spurious bias when spatial_scalar_raw_features is
      absent in older .pt files (missing RSA ≠ RSA=0 = "fully buried").

    Inputs:
        h_prot:    [B, L, d_model]
        rsa_raw:   [B, L, 1]  raw RSA values in [0, 1]
        rsa_mask:  [B, L, 1]  float mask: 1=real RSA, 0=absent (gate forced to identity)
        prot_mask: [B, L]     bool mask: True = valid residue (padding zeroed)
    """

    def __init__(self, amplitude: float = 0.5):
        super().__init__()
        self.amplitude = amplitude
        self.proj = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        h_prot:    torch.Tensor,
        rsa_raw:   torch.Tensor,
        rsa_mask:  Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.amplitude * torch.tanh(self.proj(rsa_raw))   # [B, L, 1]
        if rsa_mask is not None:
            delta = delta * rsa_mask.to(dtype=delta.dtype)        # zero out absent RSA
        h_out = h_prot * (1.0 + delta)
        if prot_mask is not None:
            h_out = h_out * prot_mask.unsqueeze(-1).float()
        return h_out


# ===========================================================================
# Inter-Block AttnRes V2  [V12: partial-biased init]
# ===========================================================================

class InterBlockAttnResV2(nn.Module):
    """
    Weighted merge of anchor representations + current partial.

    Like V11's InterBlockAttnRes but with partial-biased initialization so that
    at training start the output ≈ partial (identity), satisfying V12's
    "zero-disruption" principle. This is in contrast to the original which
    starts as a uniform average over all sources.

    Mechanism:
        sources = stack(anchors + [partial])          # [K+1, B, L, d]
        logits  = proj(RMSNorm(sources)) + partial_bias  # partial gets +bias
        alpha   = softmax(logits, dim=0)              # [K+1, B, L]
        output  = (alpha[..., None] * sources).sum(0) # [B, L, d]

    With proj zero-init and partial_logit_bias=4.0:
        K=1 anchor: softmax([0, 4]) ≈ [0.018, 0.982] → ~98% partial
        K=2 anchors: softmax([0, 0, 4]) ≈ [0.009, 0.009, 0.982]
    Training gradually learns to increase anchor weights as needed.

    partial_logit_bias is a *learnable scalar* — the model can widen or narrow
    the residual window over training.
    """

    def __init__(self, d_model: int, init_partial_bias: float = 4.0):
        super().__init__()
        self.proj      = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.key_norm  = nn.RMSNorm(d_model)
        # Learnable bias that gives partial an initial logit advantage
        self.partial_logit_bias = nn.Parameter(torch.tensor(init_partial_bias))

    def forward(
        self,
        blocks:  List[torch.Tensor],   # anchor tensors  [B, L, d] each
        partial: torch.Tensor,          # current repr    [B, L, d]
        mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sources = torch.stack(blocks + [partial], dim=0)   # [K+1, B, L, d]
        keys    = self.key_norm(sources)
        logits  = self.proj(keys).squeeze(-1)              # [K+1, B, L]
        # Fully functional bias: one_hot selects last index, no slice assignment.
        bias = F.one_hot(
            torch.tensor(logits.shape[0] - 1, device=logits.device),
            num_classes=logits.shape[0],
        ).to(dtype=logits.dtype) * self.partial_logit_bias  # [K+1]
        logits = logits + bias[:, None, None]               # boost partial logit
        alpha   = logits.softmax(dim=0)                    # [K+1, B, L]
        out = (alpha.unsqueeze(-1) * sources).sum(dim=0)    # [B, L, d]
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


# ===========================================================================
# Co-Attention Block  (V12 core building block)
# ===========================================================================

class CoAttentionBlock(nn.Module):
    """
    Interleaved Co-Attention Block (AlphaFold2 Evoformer style).

    Sub-operations per block (in order):
      1. FiLM:  Uni-Mol CLS embed → (γ, β) applied to h_prot
      2. Prot SA: protein residues attend to each other (RoPE + intra-protein 3D dist bias)
      3. Lig SA:  ligand atoms attend to each other (intra-ligand 3D dist bias)
      4. P→L CA: protein residues query ligand atoms (reads pre-update h_lig)
      5. L→P CA: ligand atoms query protein residues (reads pre-update h_prot, parallel with 4)

    Geometric distances used:
      - Protein self-attn: Cα-Cα distances (within protein coordinate system)
      - Ligand self-attn:  atom-atom distances (within ligand coordinate system)
      *** Cross-attention: NO distance bias — protein and ligand are in separate
          coordinate systems (NOT Holo state), so cross-modal distances are meaningless.

    After N=6 blocks, protein and ligand co-evolve in the same representation
    space; no separate alignment step is needed at classification.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,
        attn_dropout: Optional[float] = None,
        d_unimol: int = 512,
        use_film: bool = True,
        use_prot_dist_bias: bool = True,
        use_lig_dist_bias: bool = True,
        cross_attn_logit_clip: Optional[float] = None,
        self_attn_logit_clip: Optional[float] = None,
        inner_residual_scale: float = 1.0,
        cross_residual_scale: Optional[float] = None,
        use_rezero_branch_gates: bool = True,
        rezero_gate_init: float = 0.0,
        rezero_film_init: Optional[float] = None,
        rezero_prot_sa_init: Optional[float] = None,
        rezero_lig_sa_init: Optional[float] = None,
        rezero_p2l_init: Optional[float] = None,
        rezero_l2p_init: Optional[float] = None,
        p2l_length_scale: bool = False,
        p2l_length_ref: float = 256.0,
        p2l_rezero_gate_max: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        self.inner_residual_scale = inner_residual_scale
        self.cross_residual_scale = (
            inner_residual_scale if cross_residual_scale is None else float(cross_residual_scale)
        )
        self.use_rezero_branch_gates = use_rezero_branch_gates
        self.p2l_length_scale = bool(p2l_length_scale)
        self.p2l_length_ref = float(p2l_length_ref)
        self.p2l_rezero_gate_max = p2l_rezero_gate_max

        # 1. FiLM conditioning
        self.film = FiLMLayer(
            d_unimol,
            d_model,
            zero_init=not use_rezero_branch_gates,
        ) if use_film else None

        # 2. Protein Self-Attention (RoPE + intra-protein 3D dist bias)
        self.prot_self_attn = V12RoPEGatedTransformerBlock(
            d_model, n_heads, dropout,
            use_gated_attn=True, use_swiglu=True,
            drop_path_prob=drop_path_prob, attn_dropout=attn_dropout,
            attn_logit_clip=self_attn_logit_clip,
        )
        self.prot_dist_bias = DistanceBias(n_heads) if use_prot_dist_bias else None

        # 3. Ligand Self-Attention (intra-ligand 3D dist bias; no RoPE)
        self.lig_self_attn = V12RoPEGatedTransformerBlock(
            d_model, n_heads, dropout,
            use_gated_attn=True, use_swiglu=True,
            drop_path_prob=drop_path_prob, attn_dropout=attn_dropout,
            attn_logit_clip=self_attn_logit_clip,
        )
        self.lig_dist_bias = LigandDistanceBias(n_heads) if use_lig_dist_bias else None

        # 4 & 5. Bidirectional Cross-Attention (standard; no geometric bias)
        self.prot_to_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.lig_to_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )

        if use_rezero_branch_gates:
            init = float(rezero_gate_init)
            film_init = init if rezero_film_init is None else float(rezero_film_init)
            prot_sa_init = init if rezero_prot_sa_init is None else float(rezero_prot_sa_init)
            lig_sa_init = init if rezero_lig_sa_init is None else float(rezero_lig_sa_init)
            p2l_init = init if rezero_p2l_init is None else float(rezero_p2l_init)
            l2p_init = init if rezero_l2p_init is None else float(rezero_l2p_init)

            self.rezero_film = nn.Parameter(torch.tensor(film_init)) if use_film else None
            self.rezero_prot_sa = nn.Parameter(torch.tensor(prot_sa_init))
            self.rezero_lig_sa = nn.Parameter(torch.tensor(lig_sa_init))
            self.rezero_p2l = nn.Parameter(torch.tensor(
                self._raw_bounded_gate_init(p2l_init, p2l_rezero_gate_max)
            ))
            self.rezero_l2p = nn.Parameter(torch.tensor(l2p_init))
        else:
            self.rezero_film = None
            self.register_buffer('rezero_prot_sa', torch.tensor(1.0), persistent=False)
            self.register_buffer('rezero_lig_sa', torch.tensor(1.0), persistent=False)
            self.register_buffer('rezero_p2l', torch.tensor(1.0), persistent=False)
            self.register_buffer('rezero_l2p', torch.tensor(1.0), persistent=False)

    @staticmethod
    def _bounded_gate(gate: torch.Tensor, max_abs: Optional[float]) -> torch.Tensor:
        if max_abs is None:
            return gate
        return float(max_abs) * torch.tanh(gate)

    @staticmethod
    def _raw_bounded_gate_init(effective_init: float, max_abs: Optional[float]) -> float:
        """Convert desired effective gate init to raw tanh-param init."""
        if max_abs is None:
            return float(effective_init)
        max_abs = float(max_abs)
        if max_abs <= 0:
            raise ValueError(f"p2l_rezero_gate_max must be positive, got {max_abs}")
        if abs(effective_init) >= max_abs:
            raise ValueError(
                f"rezero_p2l_init={effective_init} must satisfy "
                f"abs(init) < p2l_rezero_gate_max={max_abs}"
            )
        ratio = float(effective_init) / max_abs
        return math.atanh(ratio)

    def _scale_p2l_delta(
        self,
        delta: torch.Tensor,
        prot_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.p2l_length_scale or prot_mask is None:
            self._cached_p2l_length_scale = None
            return delta
        n_res = prot_mask.sum(dim=1).clamp_min(1).to(dtype=delta.dtype)
        ref = torch.as_tensor(self.p2l_length_ref, device=delta.device, dtype=delta.dtype)
        scale = torch.sqrt(ref / n_res).clamp(max=1.0)
        self._cached_p2l_length_scale = scale.detach()
        return delta * scale.view(-1, 1, 1)

    def forward(
        self,
        h_prot:       torch.Tensor,
        h_lig:        torch.Tensor,
        unimol_embed: torch.Tensor,
        prot_mask:    Optional[torch.Tensor],
        lig_mask:     Optional[torch.Tensor],     # [B, M+1] includes global token
        prot_coords:  Optional[torch.Tensor],     # [B, L, 3]  Cα (protein coord sys)
        lig_coords:   Optional[torch.Tensor],     # [B, M+1, 3] atom pos (ligand coord sys)
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. FiLM: global ligand chemistry conditions protein repr
        if self.film is not None:
            h_prot_film = self.film(h_prot, unimol_embed, prot_mask=prot_mask)
            if self.rezero_film is not None:
                h_prot = h_prot + self.rezero_film * (h_prot_film - h_prot)
                if prot_mask is not None:
                    h_prot = h_prot.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
            else:
                h_prot = h_prot_film

        # 2. Protein Self-Attention (intra-protein geometry only)
        prot_attn_bias = None
        if self.prot_dist_bias is not None and prot_coords is not None:
            prot_attn_bias = self.prot_dist_bias(prot_coords, mask=prot_mask)
        h_prot_full = self.prot_self_attn(
            h_prot, mask=prot_mask, attn_bias=prot_attn_bias, position_ids=position_ids
        )
        h_prot = h_prot + (
            self.rezero_prot_sa * self.inner_residual_scale * (h_prot_full - h_prot)
        )
        if prot_mask is not None:
            h_prot = h_prot.masked_fill(~prot_mask.unsqueeze(-1), 0.0)

        # 3. Ligand Self-Attention (intra-ligand geometry only)
        # lig_coords [B, M+1, 3]: position 0 = global token (zeros), 1..M = atoms.
        # LigandDistanceBias expects atom positions only (strip global token at idx 0).
        # Returns [B, n_heads, M+1, M+1] with zero bias for global token row/col.
        #
        # RoPE neutralization: pass all-zero position_ids so every atom gets
        # identical RoPE phase → pairwise relative encoding = 0 for all pairs
        # → RoPE has zero effect. Ligand atom order is arbitrary (RDKit enumeration),
        # so injecting sequential RoPE would be a spurious positional bias.
        lig_attn_bias = None
        if self.lig_dist_bias is not None and lig_coords is not None:
            atom_coords = lig_coords[:, 1:, :]                                     # [B, M, 3]
            atom_mask   = lig_mask[:, 1:] if lig_mask is not None else None        # [B, M]
            lig_attn_bias = self.lig_dist_bias(atom_coords, atom_mask=atom_mask)
        B_lig, M1, _ = h_lig.shape
        lig_zero_pos = torch.zeros(B_lig, M1, dtype=torch.long, device=h_lig.device)
        h_lig_full = self.lig_self_attn(
            h_lig, mask=lig_mask, attn_bias=lig_attn_bias, position_ids=lig_zero_pos
        )
        h_lig = h_lig + (
            self.rezero_lig_sa * self.inner_residual_scale * (h_lig_full - h_lig)
        )
        if lig_mask is not None:
            h_lig = h_lig.masked_fill(~lig_mask.unsqueeze(-1), 0.0)

        # 4 & 5. Parallel Bidirectional Cross-Attention (true Evoformer-style)
        # Both branches read from pre-update h_prot / h_lig so neither side
        # influences the other within the same block — gradient flows are symmetric.
        h_prot_full = self.prot_to_lig(
            q=h_prot, kv=h_lig, q_mask=prot_mask, kv_mask=lig_mask
        )
        h_lig_full = self.lig_to_prot(
            q=h_lig, kv=h_prot, q_mask=lig_mask, kv_mask=prot_mask   # note: h_prot not h_prot_new
        )
        p2l_delta = self._scale_p2l_delta(h_prot_full - h_prot, prot_mask)
        p2l_gate = self._bounded_gate(self.rezero_p2l, self.p2l_rezero_gate_max)
        l2p_delta = h_lig_full - h_lig
        self._cached_p2l_gate = p2l_gate.detach()
        self._cached_l2p_gate = self.rezero_l2p.detach()
        self._cached_p2l_delta_norm = (
            p2l_delta.detach().float().norm(dim=-1).mean()
        )
        self._cached_l2p_delta_norm = (
            l2p_delta.detach().float().norm(dim=-1).mean()
        )
        h_prot_new = h_prot + (
            p2l_gate * self.cross_residual_scale * p2l_delta
        )
        h_lig_new = h_lig + (
            self.rezero_l2p * self.cross_residual_scale * l2p_delta
        )
        if prot_mask is not None:
            h_prot_new = h_prot_new.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
        if lig_mask is not None:
            h_lig_new = h_lig_new.masked_fill(~lig_mask.unsqueeze(-1), 0.0)

        return h_prot_new, h_lig_new


# ===========================================================================
# Interaction Token Block  (V12m3 bottleneck)
# ===========================================================================

class InteractionTokenBlock(nn.Module):
    """
    Learnable interaction-token bottleneck.

    Flow per block:
        protein ─┐
                 ├─> interaction tokens ─┬─> protein
        ligand  ─┘                       └─> ligand

    The tokens are a compact, ligand-aware interaction workspace.  The classifier
    later reads both residue features and the pooled interaction tokens, so the
    model gets a direct training signal to maintain useful cross-modal state.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        drop_path_prob: float = 0.0,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        residual_scale: float = 0.25,
        gate_init: float = 0.05,
        gate_max: Optional[float] = 0.2,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.gate_max = gate_max

        self.int_from_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.int_from_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.prot_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.lig_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )

        raw = CoAttentionBlock._raw_bounded_gate_init(gate_init, gate_max)
        self.rezero_i_from_p = nn.Parameter(torch.tensor(raw))
        self.rezero_i_from_l = nn.Parameter(torch.tensor(raw))
        self.rezero_p_from_i = nn.Parameter(torch.tensor(raw))
        self.rezero_l_from_i = nn.Parameter(torch.tensor(raw))

    def _gate(self, raw: torch.Tensor) -> torch.Tensor:
        return CoAttentionBlock._bounded_gate(raw, self.gate_max)

    def forward(
        self,
        h_prot: torch.Tensor,
        h_lig: torch.Tensor,
        h_int: torch.Tensor,
        prot_mask: Optional[torch.Tensor],
        lig_mask: Optional[torch.Tensor],
        int_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        int_from_p = self.int_from_prot(
            q=h_int, kv=h_prot, q_mask=int_mask, kv_mask=prot_mask
        )
        int_from_l = self.int_from_lig(
            q=h_int, kv=h_lig, q_mask=int_mask, kv_mask=lig_mask
        )
        h_int_new = h_int + self.residual_scale * (
            self._gate(self.rezero_i_from_p) * (int_from_p - h_int)
            + self._gate(self.rezero_i_from_l) * (int_from_l - h_int)
        )
        if int_mask is not None:
            h_int_new = h_int_new.masked_fill(~int_mask.unsqueeze(-1), 0.0)

        prot_from_i = self.prot_from_int(
            q=h_prot, kv=h_int_new, q_mask=prot_mask, kv_mask=int_mask
        )
        lig_from_i = self.lig_from_int(
            q=h_lig, kv=h_int_new, q_mask=lig_mask, kv_mask=int_mask
        )
        h_prot_new = h_prot + (
            self.residual_scale * self._gate(self.rezero_p_from_i) * (prot_from_i - h_prot)
        )
        h_lig_new = h_lig + (
            self.residual_scale * self._gate(self.rezero_l_from_i) * (lig_from_i - h_lig)
        )
        if prot_mask is not None:
            h_prot_new = h_prot_new.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
        if lig_mask is not None:
            h_lig_new = h_lig_new.masked_fill(~lig_mask.unsqueeze(-1), 0.0)

        self._cached_i_from_p_gate = self._gate(self.rezero_i_from_p).detach()
        self._cached_i_from_l_gate = self._gate(self.rezero_i_from_l).detach()
        self._cached_p_from_i_gate = self._gate(self.rezero_p_from_i).detach()
        self._cached_l_from_i_gate = self._gate(self.rezero_l_from_i).detach()
        self._cached_int_delta_norm = (h_int_new - h_int).detach().float().norm(dim=-1).mean()
        self._cached_prot_int_delta_norm = (h_prot_new - h_prot).detach().float().norm(dim=-1).mean()
        self._cached_lig_int_delta_norm = (h_lig_new - h_lig).detach().float().norm(dim=-1).mean()

        return h_prot_new, h_lig_new, h_int_new


# ===========================================================================
# Pair Evidence Head  (V12m3 residue-atom evidence)
# ===========================================================================

class PairEvidenceHead(nn.Module):
    """
    Coordinate-free residue-atom evidence head.

    It scores each residue against ligand atoms using learned compatibility of
    their encoded representations.  No protein-ligand coordinate distance is
    used: protein and ligand internal geometry have already been encoded inside
    h_prot / h_lig_atoms by their own encoders.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_pair: int = 128,
        d_context: int = 128,
        d_features: int = 32,
        topk: int = 8,
        logit_clip: float = 20.0,
        chunk_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_pair = int(d_pair)
        self.d_context = int(d_context)
        self.d_features = int(d_features)
        self.topk = int(topk)
        self.logit_clip = float(logit_clip)
        self.chunk_size = int(chunk_size)
        self.res_proj = nn.Linear(d_model, self.d_pair, bias=False)
        self.atom_proj = nn.Linear(d_model, self.d_pair, bias=False)
        self.atom_value = nn.Linear(d_model, self.d_context, bias=False)
        self.feature_mlp = nn.Sequential(
            nn.Linear(4, max(32, self.d_features)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, self.d_features), self.d_features),
        )
        self.context_norm = nn.LayerNorm(self.d_context)
        self.feature_norm = nn.LayerNorm(self.d_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_prot: torch.Tensor,
        h_lig_atoms: torch.Tensor,
        prot_mask: Optional[torch.Tensor],
        lig_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, _ = h_prot.shape
        M = h_lig_atoms.size(1)
        device = h_prot.device
        dtype = h_prot.dtype

        if M == 0:
            context = torch.zeros(B, L, self.d_context, device=device, dtype=dtype)
            features = torch.zeros(B, L, self.d_features, device=device, dtype=dtype)
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return context, features, {
                'pair_score_max_mean': zero,
                'pair_score_topk_mean': zero,
                'pair_entropy_mean': zero,
                'pair_context_norm': zero,
            }

        prot_valid = prot_mask.bool() if prot_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)
        lig_valid = lig_mask.bool() if lig_mask is not None else torch.ones(B, M, device=device, dtype=torch.bool)

        atom_key = self.atom_proj(h_lig_atoms)
        atom_value = self.atom_value(h_lig_atoms).masked_fill(~lig_valid.unsqueeze(-1), 0.0)
        k = max(1, min(self.topk, M))
        scale = 1.0 / math.sqrt(float(self.d_pair))

        contexts: List[torch.Tensor] = []
        features: List[torch.Tensor] = []
        stat_max: List[torch.Tensor] = []
        stat_topk: List[torch.Tensor] = []
        stat_entropy: List[torch.Tensor] = []

        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            h_chunk = h_prot[:, start:end, :]
            mask_chunk = prot_valid[:, start:end]
            res_key = self.res_proj(h_chunk)
            logits = torch.matmul(res_key, atom_key.transpose(-1, -2)) * scale  # [B, C, M]
            clip = self.logit_clip
            logits = torch.nan_to_num(logits, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
            logits = logits.masked_fill(~lig_valid.unsqueeze(1), float('-inf'))

            no_atom = ~lig_valid.any(dim=1)  # [B]
            if no_atom.any():
                logits = torch.where(no_atom[:, None, None], torch.zeros_like(logits), logits)

            topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
            weights = F.softmax(topk_vals.float(), dim=-1).to(dtype=dtype)
            if no_atom.any():
                weights = torch.where(no_atom[:, None, None], torch.zeros_like(weights), weights)

            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_context)
            value_exp = atom_value.unsqueeze(1).expand(-1, end - start, -1, -1)
            topk_values = torch.gather(value_exp, dim=2, index=gather_idx)
            context = (weights.unsqueeze(-1) * topk_values).sum(dim=2)
            context = self.context_norm(context)
            context = self.dropout(context)

            finite_topk = torch.where(torch.isfinite(topk_vals), topk_vals, torch.zeros_like(topk_vals))
            max_score = finite_topk[..., 0]
            mean_topk = finite_topk.mean(dim=-1)
            probs = weights.float().clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1) / math.log(float(k) + 1e-8)
            lse = torch.logsumexp(torch.where(torch.isfinite(logits), logits.float(), torch.full_like(logits.float(), -1e4)), dim=-1)
            raw_feat = torch.stack([max_score.float(), mean_topk.float(), entropy.float(), lse.float()], dim=-1)
            raw_feat = torch.where(no_atom[:, None, None], torch.zeros_like(raw_feat), raw_feat)
            pair_feat = self.feature_norm(self.feature_mlp(raw_feat).to(dtype=dtype))

            context = context.masked_fill(~mask_chunk.unsqueeze(-1), 0.0)
            pair_feat = pair_feat.masked_fill(~mask_chunk.unsqueeze(-1), 0.0)
            contexts.append(context)
            features.append(pair_feat)

            valid_scores = mask_chunk & (~no_atom[:, None])
            if valid_scores.any():
                stat_max.append(max_score.detach().float()[valid_scores])
                stat_topk.append(mean_topk.detach().float()[valid_scores])
                stat_entropy.append(entropy.detach().float()[valid_scores])

        pair_context = torch.cat(contexts, dim=1)
        pair_features = torch.cat(features, dim=1)
        valid_context = prot_valid.unsqueeze(-1).to(dtype=pair_context.dtype)
        context_norm = (pair_context.detach().float().norm(dim=-1) * prot_valid.float()).sum() / prot_valid.float().sum().clamp_min(1.0)

        def _mean_or_zero(items: List[torch.Tensor]) -> torch.Tensor:
            if not items:
                return torch.zeros((), device=device, dtype=torch.float32)
            return torch.cat(items).mean()

        stats = {
            'pair_score_max_mean': _mean_or_zero(stat_max),
            'pair_score_topk_mean': _mean_or_zero(stat_topk),
            'pair_entropy_mean': _mean_or_zero(stat_entropy),
            'pair_context_norm': context_norm,
        }
        return pair_context * valid_context, pair_features * valid_context, stats


def zero_init_co_attention_residuals(block: CoAttentionBlock) -> None:
    """
    Make a CoAttentionBlock identity-like at initialization.

    V12 stacks several strong residual branches per block:
      protein SA+FFN, ligand SA+FFN, P->L CA+FFN, L->P CA+FFN.
    Default random initialization lets these branches rewrite V11-style fused
    representations from step 0 and can produce huge LayerNorm gradients.

    Zero-initializing each residual branch's final projection is the standard
    ReZero/LayerScale-style safe start: branch output is exactly zero at init,
    but gradients still update the final projection so the branch can open
    gradually during training.
    """
    for self_block_name in ("prot_self_attn", "lig_self_attn"):
        self_block = getattr(block, self_block_name)
        nn.init.zeros_(self_block.attn.w_o.weight)
        if hasattr(self_block.ffn, "w_2"):
            nn.init.zeros_(self_block.ffn.w_2.weight)

    for cross_block_name in ("prot_to_lig", "lig_to_prot"):
        cross_block = getattr(block, cross_block_name)
        nn.init.zeros_(cross_block.cross_attn.w_o.weight)
        if hasattr(cross_block.ffn, "w_2"):
            nn.init.zeros_(cross_block.ffn.w_2.weight)


# ===========================================================================
# Main Model
# ===========================================================================

class PocketGVPFusionV12M3(nn.Module):
    """
    Pocket-GVP-Fusion V12m3: bottleneck-only interaction tokens plus coordinate-free residue-atom pair evidence.

    Config is backward-compatible with V11 (same key names for shared modules).
    V12-specific keys:
        n_co_attn_blocks:    int  = 6     # Co-Attention block count
        use_film:            bool = True  # FiLM from Uni-Mol
        use_prot_dist_bias:  bool = True  # intra-protein 3D dist bias in protein SA
        use_lig_dist_bias:   bool = True  # intra-ligand 3D dist bias in ligand SA
        d_unimol:            int  = 512   # Uni-Mol embedding dim
    """

    DEFAULT_CONFIG: Dict = {
        # ── Shared with V11 (same keys for checkpoint compatibility) ─────────
        'd_esm':              1280,
        'd_phys':             62,
        'd_spatial_scalar':   89,
        'd_phys_encoded':     128,
        'd_gvp_internal':     256,
        'd_model':            512,
        'd_ligand_atom':      109,
        'n_gvp_vectors':      12,
        'n_gvp_vector_hidden': 20,
        'n_gvp_layers':       3,
        'n_ligand_layers':    3,
        'n_heads':            8,
        'dropout':            0.1,
        'attn_dropout':       0.1,
        'classifier_dropout': 0.2,
        'stochastic_depth_prob': 0.1,
        'esm_dropout_prob':   0.1,
        'physchem_dropout_prob': 0.2,
        'gvp_dropout_prob':   0.1,
        'coord_noise_std':    0.5,
        'gvp_edge_rbf_bins':  16,
        'gvp_edge_rbf_dmax':  20.0,
        'ligand_edge_rbf_bins': 8,
        'gvp_radius':         10.0,
        'gvp_k_neighbors':    20,
        'edge_dim':           6,
        'use_gradient_checkpointing': False,
        'n_classes':          3,
        'use_chain_features': True,
        'use_chain_rope':     None,   # None → inherit from use_chain_features
        'use_chain_edge':     None,   # None → inherit from use_chain_features
        'chain_gap_size':     1000,
        'use_dist_aux':       False,
        # ── V12-specific ─────────────────────────────────────────────────────
        'd_unimol':             512,
        'use_unimol_ligand_input': False,
        'd_unimol_ligand_proj': 64,
        'unimol_ligand_alpha_init': 0.02,
        'n_co_attn_blocks':     6,
        'use_film':             True,
        'use_prot_dist_bias':   True,
        'use_lig_dist_bias':    True,
        'use_attn_res':         True,
        'co_attn_residual_scale': 0.25,
        'zero_init_co_attn_residuals': True,
        'cross_attn_logit_clip': 30.0,
        'self_attn_logit_clip': 30.0,
        'co_attn_inner_residual_scale': 0.05,
        'co_attn_cross_residual_scale': None,
        'use_rezero_branch_gates': True,
        'rezero_gate_init': 0.0,
        'p2l_length_scale': True,
        'p2l_length_ref': 256.0,
        'p2l_rezero_gate_max': None,
        'norm_type': 'rmsnorm',
        'freeze_all_norms': True,
        'freeze_co_attn_norms': True,
        'esm_proj_output_grad_clip': 0.0,
        'sasa_gate_amplitude':  0.5,   # tanh gate range: [1-amp, 1+amp] × h_prot
        'debug_asserts':        False,
        # Optional HPC-only FlashAttention backend for cross-attention.
        # Self-attention keeps the standard path because it uses distance bias.
        'use_flash_attn':      False,
        'flash_attn_fallback': True,
        # ── V12m3 bottleneck-only interaction-token fusion ─────────────────────────────
        'use_interaction_tokens': True,
        'n_interaction_tokens': 4,
        'interaction_token_init_std': 0.02,
        'interaction_residual_scale': 0.25,
        'interaction_gate_init': 0.05,
        'interaction_gate_max': 0.2,
        'interaction_classifier_concat': True,
        'fusion_mode': 'bottleneck_only',
        'classifier_readout': 'conditioned_plus_inter',
        # ── V12m3 pair-evidence head ───────────────────────────────────────
        'use_pair_evidence_head': False,
        'pair_dim': 128,
        'pair_context_dim': 128,
        'pair_feature_dim': 32,
        'pair_atom_topk': 8,
        'pair_logit_clip': 20.0,
        'pair_chunk_size': 256,
        'pair_context_dropout': 0.05,
        'pair_int_pool_dim': 128,
        'classifier_pair_concat': True,
        'pair_keep_full_interaction_readout': False,
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config
        self.raw_ligand_atom_dim = int(c['d_ligand_atom'])
        self.use_unimol_ligand_input = bool(c.get('use_unimol_ligand_input', False))
        self.d_unimol_ligand_proj = int(c.get('d_unimol_ligand_proj', 64))
        if self.use_unimol_ligand_input:
            c['raw_ligand_atom_dim'] = self.raw_ligand_atom_dim
            c['d_ligand_atom'] = self.raw_ligand_atom_dim + self.d_unimol_ligand_proj

        # Resolve chain flags (None → inherit master switch)
        _master = c.get('use_chain_features', True)
        self._use_chain_rope = c['use_chain_rope'] if c.get('use_chain_rope') is not None else _master
        self._use_chain_edge = c['use_chain_edge'] if c.get('use_chain_edge') is not None else _master

        # ── Module 1: PhysChem MLP Encoder (identical to V11) ───────────────
        self.physchem_encoder = nn.Sequential(
            nn.Linear(c['d_phys'], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c['d_phys_encoded']),
            nn.LayerNorm(c['d_phys_encoded']),
            nn.GELU(),
        )

        # ── Module 2: Protein Geometric Encoder (GVP) — identical to V11 ────
        _rbf_bins   = c.get('gvp_edge_rbf_bins', 16)
        _edge_s_dim = _rbf_bins + 4 + (1 if self._use_chain_edge else 0)
        _edge_v_dim = 2
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
            node_h_dim =(c['d_gvp_internal'],   c['n_gvp_vector_hidden']),
            edge_in_dim=(_edge_s_dim,            _edge_v_dim),
            n_layers   = c['n_gvp_layers'],
            drop_rate  = c['dropout'],
        )

        # ── Module 3: Bridge & Gating Fusion (identical to V11) ─────────────
        self.pairwise_bridge = PairwiseBridge(n_vec_channels=c['n_gvp_vector_hidden'])
        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        d_geo  = c['d_gvp_internal'] + n_pairwise + c['d_phys_encoded']
        d_half = c['d_model'] // 2

        self.gvp_proj = nn.Sequential(
            nn.Linear(d_geo, d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout']),
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(c['d_esm'], d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout']),
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(d_half * 2, d_half),
            nn.ReLU(),
            nn.Linear(d_half, d_half),
            nn.Sigmoid(),
        )
        self.input_norm = nn.LayerNorm(c['d_model'])

        # ── Optional Uni-Mol ligand atom augmentation (V11m3-style) ───────────
        # Unlike the global-token path, this makes ligand chemistry visible to
        # Ligand3DEncoder from its first layer and gives it direct gradients.
        if self.use_unimol_ligand_input:
            self.unimol_ligand_proj = nn.Sequential(
                nn.LayerNorm(c['d_unimol']),
                nn.Linear(c['d_unimol'], self.d_unimol_ligand_proj),
                nn.GELU(),
                nn.LayerNorm(self.d_unimol_ligand_proj),
            )
            self.unimol_ligand_alpha = nn.Parameter(
                torch.tensor(float(c.get('unimol_ligand_alpha_init', 0.02)))
            )
        else:
            self.unimol_ligand_proj = None
            self.register_buffer('unimol_ligand_alpha', torch.tensor(0.0), persistent=False)

        # ── Module 5a: Ligand 3D GNN Encoder (identical to V11) ─────────────
        _lig_edge_in = c['edge_dim'] + c.get('ligand_edge_rbf_bins', 8)
        self.ligand_encoder = Ligand3DEncoder(
            node_in_dim   = c['d_ligand_atom'],
            node_h_dim    = c['d_gvp_internal'] // 2,
            n_vec_channels = 4,
            n_vec_hidden  = 16,
            edge_in_dim   = _lig_edge_in,
            out_dim       = c['d_model'],
            n_layers      = c['n_ligand_layers'],
            dropout       = c['dropout'],
            radius        = 5.0,
            max_neighbors = 32,
        )

        # ── Module 6 (V12): N × CoAttentionBlock ────────────────────────────
        n_blocks  = c['n_co_attn_blocks']
        sdp_max   = c.get('stochastic_depth_prob', 0.0)
        dpp       = [sdp_max * i / max(n_blocks - 1, 1) for i in range(n_blocks)]
        attn_drop = c.get('attn_dropout', None)

        self.fusion_mode = str(c.get('fusion_mode', 'bottleneck_only'))
        if self.fusion_mode == 'bottleneck_only':
            # V12m3 keeps direct P<->L CoAttentionBlock outside the trainable path.
            # Keep a length-matched placeholder list so the block loop can share
            # indexing with interaction_blocks without allocating unused parameters.
            self.co_attn_blocks = nn.ModuleList([nn.Identity() for _ in range(n_blocks)])
        else:
            self.co_attn_blocks = nn.ModuleList([
                CoAttentionBlock(
                    d_model          = c['d_model'],
                    n_heads          = c['n_heads'],
                    dropout          = c['dropout'],
                    drop_path_prob   = dpp[i],
                    attn_dropout     = attn_drop,
                    d_unimol         = c['d_unimol'],
                    use_film         = c['use_film'],
                    use_prot_dist_bias = c['use_prot_dist_bias'],
                    use_lig_dist_bias  = c['use_lig_dist_bias'],
                    cross_attn_logit_clip = c.get('cross_attn_logit_clip', None),
                    self_attn_logit_clip = c.get('self_attn_logit_clip', None),
                    inner_residual_scale = c.get('co_attn_inner_residual_scale', 1.0),
                    cross_residual_scale = c.get('co_attn_cross_residual_scale', None),
                    use_rezero_branch_gates = c.get('use_rezero_branch_gates', True),
                    rezero_gate_init = c.get('rezero_gate_init', 0.0),
                    rezero_film_init = c.get('rezero_film_init', None),
                    rezero_prot_sa_init = c.get('rezero_prot_sa_init', None),
                    rezero_lig_sa_init = c.get('rezero_lig_sa_init', None),
                    rezero_p2l_init = c.get('rezero_p2l_init', None),
                    rezero_l2p_init = c.get('rezero_l2p_init', None),
                    p2l_length_scale = c.get('p2l_length_scale', True),
                    p2l_length_ref = c.get('p2l_length_ref', 256.0),
                    p2l_rezero_gate_max = c.get('p2l_rezero_gate_max', None),
                    use_flash_attn = c.get('use_flash_attn', False),
                    flash_attn_fallback = c.get('flash_attn_fallback', True),
                )
                for i in range(n_blocks)
            ])
            if (
                c.get('zero_init_co_attn_residuals', True)
                and not c.get('use_rezero_branch_gates', True)
            ):
                for block in self.co_attn_blocks:
                    zero_init_co_attention_residuals(block)

        # ── V12m3: bottleneck-only interaction-token workspace ─────────────────────
        self.use_interaction_tokens = bool(c.get('use_interaction_tokens', True))
        if self.fusion_mode == 'bottleneck_only' and not self.use_interaction_tokens:
            raise ValueError(
                "V12m3 bottleneck_only requires use_interaction_tokens=true. "
                "Use V12/V12m1 if you want direct P<->L fusion without tokens."
            )
        self.interaction_classifier_concat = bool(
            c.get('interaction_classifier_concat', True)
        )
        if self.use_interaction_tokens:
            n_int = int(c.get('n_interaction_tokens', 4))
            init_std = float(c.get('interaction_token_init_std', 0.02))
            self.interaction_tokens = nn.Parameter(
                torch.empty(n_int, c['d_model']).normal_(mean=0.0, std=init_std)
            )
            self.interaction_blocks = nn.ModuleList([
                InteractionTokenBlock(
                    d_model=c['d_model'],
                    n_heads=c['n_heads'],
                    dropout=c['dropout'],
                    drop_path_prob=dpp[i],
                    attn_dropout=attn_drop,
                    attn_logit_clip=c.get('cross_attn_logit_clip', None),
                    residual_scale=c.get('interaction_residual_scale', 0.25),
                    gate_init=c.get('interaction_gate_init', 0.05),
                    gate_max=c.get('interaction_gate_max', 0.2),
                    use_flash_attn=c.get('use_flash_attn', False),
                    flash_attn_fallback=c.get('flash_attn_fallback', True),
                )
                for i in range(n_blocks)
            ])
        else:
            self.interaction_tokens = None
            self.interaction_blocks = nn.ModuleList()

        # ── V12m3: residue-atom pair evidence head ─────────────────────────
        self.use_pair_evidence_head = bool(c.get('use_pair_evidence_head', False))
        self.classifier_pair_concat = bool(c.get('classifier_pair_concat', True))
        self.pair_keep_full_interaction_readout = bool(
            c.get('pair_keep_full_interaction_readout', False)
        )
        if self.use_pair_evidence_head:
            self.pair_evidence = PairEvidenceHead(
                d_model=c['d_model'],
                d_pair=int(c.get('pair_dim', 128)),
                d_context=int(c.get('pair_context_dim', 128)),
                d_features=int(c.get('pair_feature_dim', 32)),
                topk=int(c.get('pair_atom_topk', 8)),
                logit_clip=float(c.get('pair_logit_clip', 20.0)),
                chunk_size=int(c.get('pair_chunk_size', 256)),
                dropout=float(c.get('pair_context_dropout', 0.0)),
            )
            if self.pair_keep_full_interaction_readout:
                self.pair_int_pool_proj = nn.Identity()
            else:
                pair_int_dim = int(c.get('pair_int_pool_dim', 128))
                self.pair_int_pool_proj = nn.Sequential(
                    nn.Linear(c['d_model'], pair_int_dim),
                    nn.GELU(),
                    nn.LayerNorm(pair_int_dim),
                )
        else:
            self.pair_evidence = None
            self.pair_int_pool_proj = None

        # ── RSA Input Projection  [V12 Innovation #3] ───────────────────────
        # Inject surface signal into h_prot BEFORE co-attention so that protein
        # SA and P→L cross-attn both see the RSA prior from block 0 onward.
        # Zero-init → identity at training start.  rsa_mask guards against
        # missing-RSA samples being treated as RSA=0 ("fully buried").
        self.rsa_input_proj = nn.Linear(1, c['d_model'], bias=False)
        self.rsa_input_alpha = nn.Parameter(torch.tensor(c.get('rezero_gate_init', 0.0)))
        if not c.get('use_rezero_branch_gates', True):
            nn.init.zeros_(self.rsa_input_proj.weight)

        # ── Uni-Mol → ligand global token  [V12 Innovation #4] ─────────────
        # Augments the mean-pool global token (h_lig[:,0]) with Uni-Mol info
        # so protein residues can selectively query it via P→L cross-attention,
        # not just receive it as a global FiLM broadcast.
        # Zero-init → starts as pure mean-pool; learns to expose Uni-Mol signal.
        self.unimol_to_global = nn.Linear(c['d_unimol'], c['d_model'], bias=False)
        self.unimol_global_alpha = nn.Parameter(torch.tensor(c.get('rezero_gate_init', 0.0)))
        if not c.get('use_rezero_branch_gates', True):
            nn.init.zeros_(self.unimol_to_global.weight)

        # Final LN before classifier
        self.final_norm = nn.LayerNorm(c['d_model'])

        # ── Inter-Block AttnRes V2 (one per co-attention block) ─────────────
        # Applied BEFORE each CoAttentionBlock so anchors influence every layer.
        # Anchors: [h_prot_0] for protein, [h_lig_0] for ligand (pre-blocks repr).
        # partial_logit_bias=4.0 → output ≈ 98% partial at init → identity-like.
        n_blocks = c.get('n_co_attn_blocks', 6)
        self.prot_attn_res = nn.ModuleList([
            InterBlockAttnResV2(d_model=c['d_model']) for _ in range(n_blocks)
        ])
        self.lig_attn_res = nn.ModuleList([
            InterBlockAttnResV2(d_model=c['d_model']) for _ in range(n_blocks)
        ])

        # ── SASA Residual Gate (soft surface weighting, identity-init) ───────
        self.sasa_gate = SASAResidualGate(amplitude=c.get('sasa_gate_amplitude', 0.5))

        # ── Module 7: Classifier ────────────────────────────────────────────
        d_cls = c['d_model'] + c['d_phys_encoded']
        if self.use_pair_evidence_head and self.classifier_pair_concat:
            d_cls += int(c.get('pair_context_dim', 128))
            d_cls += int(c.get('pair_feature_dim', 32))
            if self.pair_keep_full_interaction_readout:
                d_cls += c['d_model']
            else:
                d_cls += int(c.get('pair_int_pool_dim', 128))
        elif self.use_interaction_tokens and self.interaction_classifier_concat:
            d_cls += c['d_model']
        self.classifier_input_norm = nn.LayerNorm(d_cls)
        self.classifier = nn.Sequential(
            nn.Linear(d_cls, 256),
            nn.ReLU(),
            nn.Dropout(c.get('classifier_dropout', 0.2)),
            nn.Linear(256, c['n_classes']),
        )

        # Optional distance regression aux head
        if c.get('use_dist_aux', False):
            self.dist_aux_head = nn.Sequential(
                nn.Linear(c['d_model'], 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Softplus(),
            )
        else:
            self.dist_aux_head = None

        # V12 stability controls.  Keep these instance-local so V11/shared
        # modules are unaffected.
        norm_type = str(c.get('norm_type', 'layernorm')).lower()
        if norm_type in ('rmsnorm', 'rms'):
            convert_layernorm_to_rmsnorm(self)
        elif norm_type not in ('layernorm', 'layer_norm', 'ln'):
            raise ValueError(f"Unsupported V12 norm_type: {c.get('norm_type')}")

        if c.get('freeze_all_norms', False):
            self._frozen_norm_params = freeze_norm_parameters(self)
            self._frozen_co_attn_norm_params = 0
        elif c.get('freeze_co_attn_norms', False):
            self._frozen_co_attn_norm_params = sum(
                freeze_norm_parameters(block) for block in self.co_attn_blocks
            )
            self._frozen_norm_params = self._frozen_co_attn_norm_params
        else:
            self._frozen_co_attn_norm_params = 0
            self._frozen_norm_params = 0

    # ─────────────────────────────────────────────────────────────────────────

    def _augment_ligand_x(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        unimol_embed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Optionally concatenate Uni-Mol chemistry to every ligand atom."""
        if not self.use_unimol_ligand_input:
            self._cached_unimol_ligand_delta = None
            return ligand_x

        expected_aug_dim = self.raw_ligand_atom_dim + self.d_unimol_ligand_proj
        if ligand_x.size(-1) == expected_aug_dim:
            self._cached_unimol_ligand_delta = None
            return ligand_x
        if ligand_x.size(-1) != self.raw_ligand_atom_dim:
            raise ValueError(
                f"Unexpected ligand_x dim={ligand_x.size(-1)}; expected "
                f"{self.raw_ligand_atom_dim} or {expected_aug_dim}"
            )

        if ligand_x.size(0) == 0:
            unimol_atom = ligand_x.new_zeros((0, self.d_unimol_ligand_proj))
        else:
            if ligand_batch.min().item() < 0 or ligand_batch.max().item() >= batch_size:
                raise ValueError(
                    f"ligand_batch out of range for batch_size={batch_size}: "
                    f"min={ligand_batch.min().item()} max={ligand_batch.max().item()}"
                )
            ue = unimol_embed.to(device=ligand_x.device, dtype=ligand_x.dtype)
            if ue.shape != (batch_size, self.config['d_unimol']):
                raise ValueError(
                    f"unimol_embed shape={tuple(ue.shape)}, "
                    f"expected ({batch_size}, {self.config['d_unimol']})"
                )
            unimol_atom = self.unimol_ligand_proj(ue)[ligand_batch]
            unimol_atom = self.unimol_ligand_alpha.to(dtype=ligand_x.dtype) * unimol_atom

        self._cached_unimol_ligand_delta = unimol_atom.detach()
        return torch.cat([ligand_x, unimol_atom], dim=-1)

    def forward(
        self,
        esm_embeddings:         torch.Tensor,           # [B, L, 1280]
        protein_physchem:       torch.Tensor,           # [B, L, d_phys]
        protein_spatial_scalar: torch.Tensor,           # [B, L, d_spatial_scalar]
        protein_spatial_vector: torch.Tensor,           # [B, L, n_gvp_vectors_spatial, 3]
        protein_gvp_vectors:    torch.Tensor,           # [B, L, n_gvp_vectors_backbone, 3]
        protein_coords:         torch.Tensor,           # [B, L, 3]  Cα (protein coord sys)
        protein_mask:           torch.Tensor,           # [B, L]     True=valid
        ligand_x:               torch.Tensor,           # [N_atoms, d_ligand_atom]
        ligand_pos:             torch.Tensor,           # [N_atoms, 3] ligand coord sys
        ligand_edge_index:      torch.Tensor,           # [2, N_edges]
        ligand_edge_attr:       torch.Tensor,           # [N_edges, edge_dim]
        ligand_batch:           torch.Tensor,           # [N_atoms]  sample index
        ligand_mask:            torch.Tensor,           # [B, M_max] True=valid
        unimol_embed:           Optional[torch.Tensor] = None,  # [B, 512]
        protein_rsa_raw:        Optional[torch.Tensor] = None,  # [B, L, 1] raw RSA in [0,1]
        protein_rsa_mask:       Optional[torch.Tensor] = None,  # [B, L, 1] 1=real RSA, 0=absent
        chain_ids:              Optional[torch.Tensor] = None,  # [B, L]; required when use_chain_edge=True
    ) -> Dict[str, torch.Tensor]:

        B, L, _ = esm_embeddings.shape
        device   = esm_embeddings.device
        c        = self.config

        # chain_ids is mandatory when use_chain_edge=True (GVP edge dim depends on it)
        if self._use_chain_edge and chain_ids is None:
            raise ValueError(
                "chain_ids must be provided when use_chain_edge=True "
                "(pass torch.zeros(B, L, dtype=torch.long) for single-chain proteins)"
            )

        # Chain-gap RoPE positions
        position_ids = None
        if self._use_chain_rope and chain_ids is not None:
            position_ids = get_cumulative_gap_position_ids(
                chain_ids, gap_size=c.get('chain_gap_size', 1000)
            )

        # ── Module 1: PhysChem ───────────────────────────────────────────────
        protein_physchem_encoded = self.physchem_encoder(protein_physchem)
        if self.training and c.get('physchem_dropout_prob', 0.0) > 0:
            pm = (torch.rand(B, 1, 1, device=device) > c['physchem_dropout_prob']).float()
            protein_physchem_encoded = protein_physchem_encoded * pm

        # ── Module 2: GVP Encoder ───────────────────────────────────────────
        protein_vectors_combined = torch.cat([protein_gvp_vectors, protein_spatial_vector], dim=2)
        protein_n_coords = protein_coords + protein_gvp_vectors[:, :, 1, :]
        protein_c_coords = protein_coords + protein_gvp_vectors[:, :, 3, :]

        mask_bool          = protein_mask.bool()
        n_valid_per_sample = mask_bool.sum(dim=1)
        total_valid        = n_valid_per_sample.sum().item()

        gvp_rbf_bins    = c.get('gvp_edge_rbf_bins', 16)
        gvp_rbf_dmax    = c.get('gvp_edge_rbf_dmax', 20.0)
        coord_noise_std = c.get('coord_noise_std', 0.0)

        if total_valid > 0:
            all_scalar  = protein_spatial_scalar[mask_bool]
            all_vectors = protein_vectors_combined[mask_bool]
            all_ca      = protein_coords[mask_bool]
            all_n       = protein_n_coords[mask_bool]
            all_cpos    = protein_c_coords[mask_bool]
            all_chain   = chain_ids[mask_bool] if chain_ids is not None else None

            if self.training and coord_noise_std > 0:
                noise    = torch.randn_like(all_ca) * coord_noise_std
                all_ca   = all_ca   + noise
                all_n    = all_n    + noise
                all_cpos = all_cpos + noise

            edge_index_list, edge_s_list, edge_v_list = [], [], []
            node_offset = 0
            for b in range(B):
                n_b = n_valid_per_sample[b].item()
                if n_b == 0:
                    continue
                chain_b = all_chain[node_offset:node_offset + n_b] if all_chain is not None else None
                ei_b, es_b, ev_b = build_protein_graph(
                    coords_ca = all_ca  [node_offset:node_offset + n_b],
                    coords_n  = all_n   [node_offset:node_offset + n_b],
                    coords_c  = all_cpos[node_offset:node_offset + n_b],
                    k         = c['gvp_k_neighbors'],
                    radius    = c['gvp_radius'],
                    num_rbf   = gvp_rbf_bins,
                    rbf_dmax  = gvp_rbf_dmax,
                    chain_ids = chain_b if self._use_chain_edge else None,
                )
                edge_index_list.append(ei_b + node_offset)
                edge_s_list.append(es_b)
                edge_v_list.append(ev_b)
                node_offset += n_b

            all_edge_index = torch.cat(edge_index_list, dim=1)
            all_edge_s     = torch.cat(edge_s_list, dim=0)
            all_edge_v     = torch.cat(edge_v_list, dim=0)

            h_s_flat, h_v_flat = self.gvp_encoder(
                node_s=all_scalar, node_v=all_vectors,
                edge_index=all_edge_index, edge_s=all_edge_s, edge_v=all_edge_v,
            )
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'],     device=device, dtype=h_s_flat.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v_flat.dtype)
            h_s_gvp[mask_bool] = h_s_flat
            h_v_gvp[mask_bool] = h_v_flat
        else:
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'],     device=device, dtype=protein_spatial_scalar.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype)

        # GVP Output Dropout (training only)
        if self.training and c.get('gvp_dropout_prob', 0.0) > 0:
            gm = torch.bernoulli(torch.full((B, 1, 1), 1.0 - c['gvp_dropout_prob'], device=device))
            h_s_gvp = h_s_gvp * gm
            h_v_gvp = h_v_gvp * gm.unsqueeze(-1)

        # ── Module 3: Bridge & Gating Fusion ────────────────────────────────
        v_inv  = self.pairwise_bridge(h_v_gvp)
        h_geo  = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)

        current_esm = esm_embeddings
        if self.training and c.get('esm_dropout_prob', 0.0) > 0:
            em = torch.bernoulli(torch.full((B, 1, 1), 1.0 - c['esm_dropout_prob'], device=device))
            current_esm = current_esm * em

        h_gvp_proj = self.gvp_proj(h_geo)
        h_gvp_proj = h_gvp_proj.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        # Keep the public module/state_dict layout as `esm_proj.0/1/2`, but run
        # the steps explicitly so the hook can sit directly on Linear output.
        #
        # Under ROCm fp16 autocast, LinearBackward may accumulate dW in a dtype
        # that can overflow on long proteins even when dL/dz is clipped. Keep
        # this projection path in fp32; downstream layers may still autocast.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            z_esm = self.esm_proj[0](current_esm.float())
            esm_grad_clip = float(c.get('esm_proj_output_grad_clip', 0.0) or 0.0)
            if self.training and esm_grad_clip > 0.0 and z_esm.requires_grad:
                # Clamp dL/d(z_esm) before LinearBackward computes
                # dW = grad_output^T @ ESM.  Hooking after Norm/Dropout is too late
                # if Norm backward itself produces the spike.
                def _clip_esm_linear_output_grad(grad: torch.Tensor) -> torch.Tensor:
                    return torch.nan_to_num(
                        grad,
                        nan=0.0,
                        posinf=esm_grad_clip,
                        neginf=-esm_grad_clip,
                    ).clamp_(min=-esm_grad_clip, max=esm_grad_clip)

                z_esm.register_hook(_clip_esm_linear_output_grad)
            h_esm_proj = self.esm_proj[1](z_esm)
            h_esm_proj = self.esm_proj[2](h_esm_proj)
        h_esm_proj = h_esm_proj.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        h_concat   = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate       = self.modality_gate(h_concat)
        h_prot     = self.input_norm(
            torch.cat([gate * h_gvp_proj, (1.0 - gate) * h_esm_proj], dim=-1)
        )  # [B, L, d_model]
        # Keep padded residues out of V12 anchors/AttnRes/FiLM gradients.
        h_prot = h_prot.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        # Cache modality features for training diagnostics (ESM/GVP contribution).
        self._cached_gvp_scalar = h_s_gvp.detach()
        self._cached_gvp_vector = h_v_gvp.detach()
        self._cached_h_geo = h_geo.detach()
        self._cached_current_esm = current_esm.detach()
        self._cached_h_gvp_proj = h_gvp_proj.detach()
        self._cached_h_esm_proj = h_esm_proj.detach()
        self._cached_esm_mask = em.detach() if 'em' in locals() else None

        # ── Uni-Mol embedding ────────────────────────────────────────────────
        if unimol_embed is None:
            unimol_embed = torch.zeros(B, c['d_unimol'], device=device, dtype=h_prot.dtype)
        else:
            unimol_embed = unimol_embed.to(device=device, dtype=h_prot.dtype)
        self._cached_unimol_embed = unimol_embed.detach()

        # ── Module 5a: Ligand GNN Encoder ───────────────────────────────────
        ligand_x = self._augment_ligand_x(
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            unimol_embed=unimol_embed,
            batch_size=B,
        )
        _lig_rbf_bins = c.get('ligand_edge_rbf_bins', 8)
        if ligand_edge_index.size(1) > 0:
            lig_row, lig_col = ligand_edge_index
            lig_dist = torch.norm(ligand_pos[lig_col] - ligand_pos[lig_row], dim=-1)
            lig_rbf  = rbf_encode(lig_dist, D_min=0.0, D_max=5.0, num_rbf=_lig_rbf_bins)
            lig_edge_enriched = torch.cat([ligand_edge_attr, lig_rbf], dim=-1)
        else:
            lig_edge_enriched = torch.cat([
                ligand_edge_attr,
                torch.zeros(0, _lig_rbf_bins, device=device, dtype=ligand_pos.dtype),
            ], dim=-1)

        h_ligand_nodes, _ = self.ligand_encoder(
            x=ligand_x, pos=ligand_pos,
            edge_index=ligand_edge_index, edge_attr=lig_edge_enriched,
            batch=ligand_batch,
        )  # [N_atoms, d_model]

        # Unpack PyG sparse → dense [B, M_max, d_model]
        B_dim, Max_M = ligand_mask.shape
        h_lig      = torch.zeros(B_dim, Max_M, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        lig_coords = torch.zeros(B_dim, Max_M, 3,            device=device, dtype=ligand_pos.dtype)
        N_atoms = h_ligand_nodes.size(0)
        if N_atoms > 0:
            if c.get('debug_asserts', False):
                if ligand_batch.numel() != N_atoms:
                    raise RuntimeError(
                        f"ligand_batch length mismatch: {ligand_batch.numel()} vs {N_atoms}"
                    )
                if not torch.isfinite(h_ligand_nodes).all():
                    raise RuntimeError("h_ligand_nodes contains NaN/Inf")
                if not torch.isfinite(ligand_pos).all():
                    raise RuntimeError("ligand_pos contains NaN/Inf")
                if ligand_batch.min().item() < 0 or ligand_batch.max().item() >= B_dim:
                    raise RuntimeError(
                        f"ligand_batch out of range: min={ligand_batch.min().item()} "
                        f"max={ligand_batch.max().item()} B={B_dim}"
                    )
            counts    = torch.bincount(ligand_batch, minlength=B_dim)
            if c.get('debug_asserts', False):
                mask_counts = ligand_mask.sum(dim=1).to(dtype=counts.dtype)
                if not torch.equal(counts[:B_dim], mask_counts):
                    raise RuntimeError(
                        "ligand_batch counts do not match ligand_mask counts: "
                        f"counts={counts[:B_dim].detach().cpu().tolist()} "
                        f"mask={mask_counts.detach().cpu().tolist()}"
                    )
                if (counts[:B_dim] > Max_M).any():
                    raise RuntimeError(
                        f"ligand atoms exceed dense Max_M={Max_M}: "
                        f"{counts[:B_dim].detach().cpu().tolist()}"
                    )
            cumsum    = torch.zeros(B_dim + 1, dtype=torch.long, device=device)
            cumsum[1:] = counts.cumsum(0)
            local_idx  = torch.arange(N_atoms, device=device) - cumsum[ligand_batch]
            valid_m    = local_idx < Max_M
            if c.get('debug_asserts', False):
                if (local_idx < 0).any():
                    raise RuntimeError(
                        f"negative ligand local_idx detected: min={local_idx.min().item()}"
                    )
                if not valid_m.all():
                    bad = (~valid_m).sum().item()
                    raise RuntimeError(
                        f"ligand dense scatter would drop {bad} atoms with Max_M={Max_M}"
                    )
            h_lig     [ligand_batch[valid_m], local_idx[valid_m]] = h_ligand_nodes[valid_m]
            lig_coords[ligand_batch[valid_m], local_idx[valid_m]] = ligand_pos[valid_m]

        # Dynamic ligand global token (masked mean pool)
        mask_f   = ligand_mask.float().unsqueeze(-1)                        # [B, M, 1]
        glob_tok = (h_lig * mask_f).sum(1, keepdim=True) \
                 / mask_f.sum(1, keepdim=True).clamp(min=1)                # [B, 1, d]
        h_lig    = torch.cat([glob_tok, h_lig], dim=1)                     # [B, M+1, d]
        g_mask   = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)
        lig_mask_g  = torch.cat([g_mask, ligand_mask], dim=1)              # [B, M+1]
        # Pad lig_coords with zero row for global token at position 0
        lig_coords = torch.cat([
            torch.zeros(B_dim, 1, 3, device=device, dtype=lig_coords.dtype),
            lig_coords,
        ], dim=1)                                                           # [B, M+1, 3]

        # [Innovation #4] Augment ligand global token with Uni-Mol embedding.
        # h_lig[:, 0, :] starts as masked mean-pool; adding Uni-Mol projection
        # makes the chemical fingerprint directly queryable by protein residues
        # via P→L cross-attention (selective), not only broadcast via FiLM.
        # zero-init proj → no change at start of training.
        # Non-in-place cat to avoid autograd / torch.compile instability.
        unimol_global_delta = (
            self.unimol_global_alpha * self.unimol_to_global(unimol_embed).unsqueeze(1)
        )
        self._cached_unimol_global_base = h_lig[:, 0:1, :].detach()
        self._cached_unimol_global_delta = unimol_global_delta.detach()
        h_lig_global = h_lig[:, 0:1, :] + unimol_global_delta
        h_lig = torch.cat([h_lig_global, h_lig[:, 1:, :]], dim=1)

        # [Innovation #3] RSA pre-injection: inject surface prior into h_prot
        # before co-attention so all 6 blocks see RSA.  rsa_mask zeroes out
        # the contribution for residues with missing raw RSA data.
        # Also multiply by protein_mask to ensure padding positions stay zero.
        if c.get('use_rsa_input', True) and protein_rsa_raw is not None:
            rsa_f = protein_rsa_raw.to(dtype=h_prot.dtype)           # [B, L, 1]
            rsa_m = protein_rsa_mask.to(dtype=h_prot.dtype) \
                    if protein_rsa_mask is not None \
                    else torch.ones_like(rsa_f)                       # [B, L, 1]
            rsa_m = rsa_m * protein_mask.unsqueeze(-1).to(dtype=h_prot.dtype)
            h_prot = h_prot + self.rsa_input_alpha * rsa_m * self.rsa_input_proj(rsa_f)
            h_prot = h_prot.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        # ── Module 6 (V12): N × CoAttentionBlock with per-block AttnRes ─────
        # AttnRes anchor: h_prot_0 / h_lig_0 are the representations after
        # encoder fusion + RSA/UniMol injection, before any co-attention.
        # At each block i, AttnRes blends [anchor, current_partial] with learned
        # per-position weights (~98% partial at init → does not disrupt baseline).
        h_prot_0 = h_prot.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        h_lig_0  = h_lig.masked_fill(~lig_mask_g.unsqueeze(-1), 0.0)
        if self.use_interaction_tokens:
            h_int = self.interaction_tokens.to(dtype=h_prot.dtype, device=device)
            h_int = h_int.unsqueeze(0).expand(B, -1, -1).contiguous()
            int_mask = torch.ones(
                B, h_int.size(1), device=device, dtype=protein_mask.dtype
            )
        else:
            h_int = None
            int_mask = None

        for i, block in enumerate(self.co_attn_blocks):
            # AttnRes before each block: weighted merge of anchor + current repr.
            # Equivalent to V11's attn_res([b0], partial=h_prot_fused) before CA.
            if c.get('use_attn_res', True):
                h_prot = self.prot_attn_res[i](
                    blocks=[h_prot_0], partial=h_prot, mask=protein_mask
                )
                h_lig = self.lig_attn_res[i](
                    blocks=[h_lig_0], partial=h_lig, mask=lig_mask_g
                )

            # V12m3 bottleneck-only fusion. Tokens are the main cross-modal
            # exchange path in the main loop: they read both streams and write
            # compact interaction context back to protein/ligand. Direct P↔L
            # CoAttentionBlock is intentionally skipped to reduce protein shortcut.
            if self.use_interaction_tokens:
                h_prot, h_lig, h_int = self.interaction_blocks[i](
                    h_prot=h_prot,
                    h_lig=h_lig,
                    h_int=h_int,
                    prot_mask=protein_mask,
                    lig_mask=lig_mask_g,
                    int_mask=int_mask,
                )

            # V12m3 intentionally does not run the direct P<->L CoAttentionBlock.
            # Protein/ligand self-attention inside CoAttentionBlock is also skipped
            # here; the controlled cross-modal bottleneck is the interaction-token
            # workspace above. This keeps the experiment clean: any ligand effect
            # must pass through h_int before reaching the classifier.
            if self.fusion_mode != 'bottleneck_only':
                raise ValueError(
                    "PocketGVPFusionV12M3 expects fusion_mode='bottleneck_only'. "
                    "Use train_v12m1.py for the hybrid direct+token path."
                )

        # ── Module 7: Classifier ─────────────────────────────────────────────
        h_prot = self.final_norm(h_prot)
        h_prot = h_prot.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        # SASA Residual Gate: learnable modulation conditioned on RSA.
        # Identity at init (zero-init linear → delta = 0).
        # rsa_mask zeros out delta for residues with absent raw RSA (old .pt files),
        # preventing spurious bias from treating missing data as RSA=0 ("fully buried").
        # Skipped entirely if protein_rsa_raw is None (backward compat).
        if c.get('use_sasa_gate', True) and protein_rsa_raw is not None:
            rsa = protein_rsa_raw.to(dtype=h_prot.dtype)
            rsa_m = protein_rsa_mask.to(dtype=h_prot.dtype) if protein_rsa_mask is not None else None
            h_prot = self.sasa_gate(h_prot, rsa, rsa_mask=rsa_m, prot_mask=protein_mask)

        protein_physchem_encoded = protein_physchem_encoded.masked_fill(
            ~protein_mask.unsqueeze(-1), 0.0
        )
        h_int_broadcast = None
        pair_context = None
        pair_features = None
        pair_stats: Dict[str, torch.Tensor] = {}
        if self.use_pair_evidence_head and self.classifier_pair_concat:
            pair_context, pair_features, pair_stats = self.pair_evidence(
                h_prot=h_prot,
                h_lig_atoms=h_lig[:, 1:, :],
                prot_mask=protein_mask,
                lig_mask=lig_mask_g[:, 1:],
            )
            h_int_pool = h_int.mean(dim=1)
            h_int_pool = self.pair_int_pool_proj(h_int_pool)
            h_int_broadcast = h_int_pool.unsqueeze(1).expand(-1, L, -1)
            h_int_broadcast = h_int_broadcast.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
            x_final = torch.cat([
                h_prot,
                pair_context,
                pair_features,
                h_int_broadcast,
                protein_physchem_encoded,
            ], dim=-1)
        elif self.use_interaction_tokens and self.interaction_classifier_concat:
            h_int_pool = h_int.mean(dim=1)
            h_int_broadcast = h_int_pool.unsqueeze(1).expand(-1, L, -1)
            h_int_broadcast = h_int_broadcast.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
            x_final = torch.cat([h_prot, h_int_broadcast, protein_physchem_encoded], dim=-1)
        else:
            x_final = torch.cat([h_prot, protein_physchem_encoded], dim=-1)
        x_final = x_final.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        logits  = self.classifier(self.classifier_input_norm(x_final))

        result: Dict[str, torch.Tensor] = {
            'logits':         logits,
            'h_fused':        h_prot,
            'h_ligand_fused': h_lig,
        }
        if self.use_interaction_tokens:
            result['h_interaction_tokens'] = h_int
            if h_int_broadcast is not None:
                result['h_interaction_broadcast'] = h_int_broadcast
        if pair_context is not None:
            result['h_pair_context'] = pair_context
            result['h_pair_features'] = pair_features
            for key, value in pair_stats.items():
                result[key] = value
        if self.dist_aux_head is not None:
            result['dist_pred'] = self.dist_aux_head(h_prot).squeeze(-1)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
