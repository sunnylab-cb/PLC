"""
Pocket-GVP-Fusion V11: Ligand Internal Self-Attention.

V11 updates over V10m8:
- Ligand-internal self-attention (n_ligand_self_attn_layers, default=2):
  After ligand GNN encoder + global token assembly, run N transformer blocks
  over [global_token, atom_1, ..., atom_M] to build intra-molecular context.
  Addresses the parameter imbalance: protein had self-attention (31%), ligand
  had only GNN (1%). Self-attention output feeds cross-attention as richer K/V.
  AttnRes b_lig0 stays anchored to raw encoder output (before self-attn).

繼承 V10m8:
  Per-layer distance bias, Distance regression aux head,
  Dynamic Ligand Global Token, GVP Output Dropout, Coordinate Noise,
  RoPE, Ligand AttnRes, GVP Block (b_gvp)

Block划分策略:
  b_gvp: h_s_gvp (纯GVP几何输出，融合前)
  b0:    h_seq (GVP+ESM融合后、进入Transformer前)
  b1:    h_prot_global (4层Transformer编码后)

  配体侧:
  b_lig0: h_ligand_with_global (配体编码器输出 + dynamic global token) [AttnRes锚点]
  h_lig_ctx: ligand_self_attn(b_lig0) [cross-attention起点]

  Cross-Attention循环(4层):
    蛋白侧: h_prot_fused = attn_res([b_gvp, b0, b1], partial=h_prot_fused)
    配体侧: h_lig_fused = lig_attn_res([b_lig0], partial=h_lig_fused)
    然后标准双向交叉注意力
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple

from .gvp_layers import GVPEncoder, build_knn_graph, build_protein_graph, rbf_encode
from .transformer_layers import BidirectionalCrossAttentionBlock
from .ligand_encoder_3d import Ligand3DEncoder
from .chain_encoding import get_cumulative_gap_position_ids


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to Q/K based on position index, enabling relative position encoding.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin: [seq_len, dim] if position_ids is None,
                      [B, seq_len, dim] if position_ids is [B, seq_len]
        """
        if position_ids is not None:
            t = position_ids.to(dtype=self.inv_freq.dtype)          # [B, L]
            freqs = torch.einsum('bi,j->bij', t, self.inv_freq)     # [B, L, dim//2]
        else:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)                   # [L, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to input tensor.

    Args:
        x:       [B, n_heads, L, head_dim]
        cos/sin: [L, head_dim]  OR  [B, L, head_dim]  (batched when chain_ids provided)
    """
    # Normalise x to [B, n_heads, L, head_dim]
    if x.dim() == 4 and x.shape[2] != cos.shape[-2]:
        x = x.transpose(1, 2)
        transposed = True
    else:
        transposed = False

    x1, x2 = x.chunk(2, dim=-1)
    half = x1.shape[-1]

    if cos.dim() == 2:                                          # [L, head_dim]
        cos = cos[:, :half].unsqueeze(0).unsqueeze(0)          # [1, 1, L, half]
        sin = sin[:, :half].unsqueeze(0).unsqueeze(0)
    else:                                                       # [B, L, head_dim]
        cos = cos[:, :, :half].unsqueeze(1)                    # [B, 1, L, half]
        sin = sin[:, :, :half].unsqueeze(1)

    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    if transposed:
        out = out.transpose(1, 2)
    return out


# ---------------------------------------------------------------------------
# RoPE-enabled Attention and Transformer (minimal, based on transformer_layers.py)
# ---------------------------------------------------------------------------

class RoPEGatedMultiHeadAttention(nn.Module):
    """GatedMultiHeadAttention with RoPE support."""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 use_gate: bool = True, attn_dropout: Optional[float] = None,
                 use_sdpa: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_gate = use_gate
        self.use_sdpa = use_sdpa

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        if use_gate:
            self.gate = nn.Linear(d_model, d_model, bias=False)

        _attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.attn_dropout = nn.Dropout(_attn_dropout)
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # [B, L] chain-gap positions
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if kv is None:
            kv = x
        S = kv.size(1)

        q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (position_ids enables chain-gap encoding)
        cos_q, sin_q = self.rope(L, x.device, position_ids=position_ids)
        cos_k, sin_k = self.rope(S, x.device, position_ids=position_ids)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

        if self.use_sdpa:
            # SDPA accepts an additive float mask/bias broadcastable to
            # [B, n_heads, L, S].  This path is intended for protein
            # self-attention, where long L makes fused attention worthwhile.
            sdpa_mask = attn_bias.to(dtype=q.dtype) if attn_bias is not None else None
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, S]
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)                # [B, 1, L, S]
                pad_bias = torch.zeros(
                    mask.shape, device=x.device, dtype=q.dtype
                ).masked_fill(~mask, float('-inf'))
                sdpa_mask = pad_bias if sdpa_mask is None else sdpa_mask + pad_bias

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).reshape(B, L, self.d_model)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # V10m4: add pre-softmax 3D distance bias (protein-internal, no leakage)
            if attn_bias is not None:
                attn = attn + attn_bias

            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(2)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                attn = attn.masked_fill(~mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.d_model)
        out = self.w_o(out)

        if self.use_gate:
            gate = torch.sigmoid(self.gate(x))
            out = out * gate

        return out


class DistanceBias(nn.Module):
    """
    V10m4: 3D Distance-Biased Attention for protein self-attention.

    Computes per-head attention bias from CA-CA pairwise distances.
    Uses RBF encoding → linear projection (one bias scalar per head per pair).

    Key properties:
    - Protein-ONLY coords → no protein-ligand joint information → NO LEAKAGE
    - Zero initialization → starts as standard attention, bias learned gradually
    - Pre-computed once per forward pass and shared across all Transformer layers

    The bias is ADDITIVE before softmax:
        attn_scores = QK^T/sqrt(d) + dist_bias
    Negative bias for distant pairs suppresses long-range attention;
    positive bias for close pairs boosts local attention.

    Args:
        n_heads:  number of attention heads
        n_rbf:    number of RBF bins for distance encoding
        max_dist: saturation distance (Å); beyond this distances are clamped
    """
    def __init__(self, n_heads: int, n_rbf: int = 16, max_dist: float = 20.0):
        super().__init__()
        self.n_heads  = n_heads
        self.n_rbf    = n_rbf
        self.max_dist = max_dist

        # RBF centres (fixed, not learned) stored as buffer
        self.register_buffer('rbf_mu', torch.linspace(0.0, max_dist, n_rbf))
        self.rbf_sigma = max_dist / n_rbf

        # Projection: n_rbf → n_heads  (one bias scalar per head per residue pair)
        # Zero init → model starts from standard attention, no training instability
        self.proj = nn.Linear(n_rbf, n_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coords: [B, L, 3]  protein CA coordinates (PROTEIN-ONLY)
            mask:   [B, L]     True = valid residue

        Returns:
            bias: [B, n_heads, L, L]  additive pre-softmax bias
        """
        B, L, _ = coords.shape
        device   = coords.device
        dtype    = coords.dtype

        # Pairwise CA-CA distances  [B, L, L]
        dist = torch.cdist(coords, coords)
        dist = dist.clamp(max=self.max_dist)

        # RBF encoding  [B, L, L, n_rbf]
        rbf_mu = self.rbf_mu.to(dtype=dtype)
        rbf    = torch.exp(
            -((dist.unsqueeze(-1) - rbf_mu) ** 2) / (2.0 * self.rbf_sigma ** 2)
        )

        # Project to per-head bias  [B, L, L, n_heads] → [B, n_heads, L, L]
        bias = self.proj(rbf).permute(0, 3, 1, 2)

        # Mask padding positions: columns (key positions) of padded residues get -inf
        if mask is not None:
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, L]
            bias = bias.masked_fill(pad_mask, float('-inf'))

        return bias


class LigandDistanceBias(nn.Module):
    """
    V11m2: 3D Distance-Biased Attention for ligand self-attention.

    Symmetric to DistanceBias (protein), adapted for ligand's
    [global_token, atom_1, ..., atom_M] layout:
    - Global token (index 0) has no 3D atom position → its bias rows/columns are zero.
    - Atom-atom pairs [1:, 1:] receive RBF(inter-atomic distance) → per-head bias.

    Zero-init → starts as standard attention, bias learned gradually (safe to enable).

    Args:
        n_heads:  number of attention heads
        n_rbf:    RBF bins (8 is sufficient; ligand atoms within ~8Å of each other)
        max_dist: saturation distance (Å)
    """
    def __init__(self, n_heads: int, n_rbf: int = 8, max_dist: float = 8.0):
        super().__init__()
        self.n_heads  = n_heads
        self.n_rbf    = n_rbf
        self.max_dist = max_dist
        self.register_buffer('rbf_mu', torch.linspace(0.0, max_dist, n_rbf))
        self.rbf_sigma = max_dist / n_rbf
        # Zero-init → no change at start, bias learned gradually
        self.proj = nn.Linear(n_rbf, n_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, atom_pos: torch.Tensor,
                atom_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_pos:  [B, M, 3]  atom positions (NO global token)
            atom_mask: [B, M]     True = valid atom
        Returns:
            bias: [B, n_heads, M+1, M+1]
                  row/col 0 (global token) = zero
                  rows/cols [1:, 1:] = RBF atom-atom bias
        """
        B, M, _ = atom_pos.shape
        device   = atom_pos.device
        dtype    = atom_pos.dtype

        # Pairwise atom-atom distances [B, M, M]
        dist   = torch.cdist(atom_pos, atom_pos).clamp(max=self.max_dist)

        # RBF encoding [B, M, M, n_rbf]
        rbf_mu = self.rbf_mu.to(dtype=dtype)
        rbf    = torch.exp(
            -((dist.unsqueeze(-1) - rbf_mu) ** 2) / (2.0 * self.rbf_sigma ** 2)
        )

        # Project to per-head bias [B, M, M, n_heads] → [B, n_heads, M, M]
        atom_bias = self.proj(rbf).permute(0, 3, 1, 2)

        # Mask padding atoms (key columns)
        if atom_mask is not None:
            pad_mask  = (~atom_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            atom_bias = atom_bias.masked_fill(pad_mask, float('-inf'))

        # Prepend zero row and column for global token → [B, n_heads, M+1, M+1]
        zeros_col = torch.zeros(B, self.n_heads, M, 1,     device=device, dtype=atom_bias.dtype)
        atom_bias = torch.cat([zeros_col, atom_bias], dim=-1)            # [B, n_heads, M, M+1]
        zeros_row = torch.zeros(B, self.n_heads, 1, M + 1, device=device, dtype=atom_bias.dtype)
        full_bias  = torch.cat([zeros_row, atom_bias], dim=-2)           # [B, n_heads, M+1, M+1]

        return full_bias


class SwiGLU(nn.Module):
    """SwiGLU FFN (copied from transformer_layers.py)."""
    def __init__(self, d_model: int, d_ffn: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(8 / 3 * d_model)
            d_ffn = ((d_ffn + 63) // 64) * 64
        self.w_1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_2 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        x = self.w_1(x) * gate
        x = self.dropout(x)
        x = self.w_2(x)
        return x


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic Depth."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    rand_tensor = torch.floor(rand_tensor + keep_prob)
    return x / keep_prob * rand_tensor


class RoPEGatedTransformerBlock(nn.Module):
    """Transformer block with RoPE, SwiGLU, and Stochastic Depth."""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 use_gated_attn: bool = True, use_swiglu: bool = True,
                 drop_path_prob: float = 0.0, attn_dropout: Optional[float] = None,
                 use_sdpa: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEGatedMultiHeadAttention(
            d_model, n_heads, dropout, use_gate=use_gated_attn,
            attn_dropout=attn_dropout, use_sdpa=use_sdpa
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.drop_path_prob = drop_path_prob

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # [B, L] chain-gap positions
        chain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        extra = {"chain_ids": chain_ids} if getattr(self.attn, "requires_chain_ids", False) else {}
        attn_out = self.attn(self.norm1(x), mask=mask, attn_bias=attn_bias,
                             position_ids=position_ids, **extra)
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        ffn_out = self.ffn(self.norm2(x))
        x = x + drop_path(ffn_out, self.drop_path_prob, self.training)
        return x


class RoPEProteinContextEncoder(nn.Module):
    """Protein Context Encoder with RoPE + optional 3D Distance-Biased Attention (V10m4)."""
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, stochastic_depth_prob: float = 0.0,
                 attn_dropout: Optional[float] = None,
                 use_dist_bias: bool = False,        # V10m4
                 dist_bias_n_rbf: int = 16,          # V10m4
                 dist_bias_max_dist: float = 20.0,    # V10m4
                 use_sdpa: bool = False, hybrid_pattern: Optional[str] = None,
                 gdn_backend: str = "fla"):
        super().__init__()
        drop_path_probs = [stochastic_depth_prob * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            RoPEGatedTransformerBlock(
                d_model, n_heads, dropout, use_gated_attn=True, use_swiglu=True,
                drop_path_prob=drop_path_probs[i], attn_dropout=attn_dropout,
                use_sdpa=use_sdpa
            ) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # V10m8: per-layer distance bias (AlphaFold2/ESMFold style)
        # Each layer has its own DistanceBias — shallow layers learn local geometry,
        # deep layers learn global context. Zero-init ensures safe start.
        if use_dist_bias:
            self.dist_bias_modules = nn.ModuleList([
                DistanceBias(n_heads, n_rbf=dist_bias_n_rbf, max_dist=dist_bias_max_dist)
                for _ in range(n_layers)
            ])
        else:
            self.dist_bias_modules = None

        self.hybrid_pattern = hybrid_pattern or "F" * n_layers
        if len(self.hybrid_pattern) != n_layers or set(self.hybrid_pattern) - {"F", "G"}:
            raise ValueError("protein_hybrid_pattern must contain one F/G per front layer")
        if "G" in self.hybrid_pattern:
            from .protein_bidirectional_gdn import BidirectionalProteinGDN
            # Preserve initialization of every unchanged module in matched runs.
            with torch.random.fork_rng(devices=[]):
                for i, kind in enumerate(self.hybrid_pattern):
                    if kind == "G":
                        self.layers[i].attn = BidirectionalProteinGDN(
                            d_model, n_heads, dropout=dropout, backend=gdn_backend)
                        if self.dist_bias_modules is not None:
                            self.dist_bias_modules[i] = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # [B, L] chain-gap positions
        chain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            attn_bias = None
            if self.hybrid_pattern[i] == "F" and self.dist_bias_modules is not None and coords is not None:
                attn_bias = self.dist_bias_modules[i](coords, mask=mask)
            x = layer(x, mask=mask, attn_bias=attn_bias, position_ids=position_ids, chain_ids=chain_ids)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------------

class PairwiseBridge(nn.Module):
    """Pairwise dot product bridge for translation-invariant vector features."""
    def __init__(self, n_vec_channels: int):
        super().__init__()
        self.n_pairs = (n_vec_channels * (n_vec_channels + 1)) // 2

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        gram = torch.einsum('bnci,bnji->bncj', h_v, h_v)
        batch, n_res, c, _ = gram.shape
        indices = torch.triu_indices(c, c, device=gram.device)
        gram_flat = gram.view(batch * n_res, c, c)
        flat_pairs = gram_flat[:, indices[0], indices[1]]
        return flat_pairs.view(batch, n_res, self.n_pairs)


class InterBlockAttnRes(nn.Module):
    """Inter-Block Attention Residuals (AttnRes, Kimi 2025)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.key_norm = nn.RMSNorm(d_model)

    def forward(self, blocks: List[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
        sources = torch.stack(blocks + [partial], dim=0)
        keys = self.key_norm(sources)
        logits = self.proj(keys).squeeze(-1)
        alpha = logits.softmax(dim=0)
        out = (alpha.unsqueeze(-1) * sources).sum(dim=0)
        return out


# ---------------------------------------------------------------------------
# V11: Ligand Internal Self-Attention
# ---------------------------------------------------------------------------

class LigandSelfAttentionEncoder(nn.Module):
    """
    V11: Intra-molecular self-attention for ligand atoms.

    Ligand atoms have only a GNN encoder (1% of total params), while the protein
    has a 4-layer Transformer self-attention (31%). This module closes that gap by
    running N transformer blocks over [global_token, atom_1, ..., atom_M] so the
    ligand side has comparable intra-molecular context before cross-attention.

    Design choices:
    - Pre-LN, Gated MHA + SwiGLU FFN (consistent with protein_context_encoder)
    - No RoPE: atoms have no meaningful linear order; sequential default positions
      are used and act as a mild soft positional prior (acceptable for ≤128 atoms)
    - Trailing LayerNorm for stable hand-off to cross-attention

    V11m2 addition: optional per-layer 3D distance bias (LigandDistanceBias).
    Symmetric to protein's DistanceBias. Global token (index 0) always has zero bias;
    atom-atom pairs use RBF(inter-atomic distance). Zero-init, safe to enable.
    """
    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 dropout: float, attn_dropout: Optional[float] = None,
                 use_dist_bias: bool = False,
                 dist_bias_n_rbf: int = 8,
                 dist_bias_max_dist: float = 8.0):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEGatedTransformerBlock(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                use_gated_attn=True, use_swiglu=True,
                drop_path_prob=0.0,            # no stochastic depth on ligand side
                attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # V11m2: optional per-layer 3D distance bias
        if use_dist_bias:
            self.dist_bias_modules = nn.ModuleList([
                LigandDistanceBias(n_heads, n_rbf=dist_bias_n_rbf, max_dist=dist_bias_max_dist)
                for _ in range(n_layers)
            ])
        else:
            self.dist_bias_modules = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                atom_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:        [B, M+1, d_model]  global token + atoms
            mask:     [B, M+1]           True = valid position
            atom_pos: [B, M, 3]          atom positions (no global token); for dist bias
        Returns:
            [B, M+1, d_model]
        """
        for i, layer in enumerate(self.layers):
            attn_bias = None
            if self.dist_bias_modules is not None and atom_pos is not None:
                atom_mask = mask[:, 1:] if mask is not None else None  # [B, M], skip global
                attn_bias = self.dist_bias_modules[i](atom_pos, atom_mask=atom_mask)
            x = layer(x, mask=mask, attn_bias=attn_bias)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PocketGVPFusionV11(nn.Module):
    """
    Pocket-GVP-Fusion V11.

    V11 addition over V10m8:
    1. GVP Output Dropout (gvp_dropout_prob): sample-level mask on GVP encoder output
       - Symmetric to esm_dropout_prob / physchem_dropout_prob
       - Forces model not to over-rely on geometric features
    2. Coordinate Noise (coord_noise_std): Gaussian noise on CA coords before build_knn_graph
       - Changes graph topology + edge distances -> stronger structural regularization
       - Applied only during training

    三層 GVP 正則化:
        L1: internal GVP Dropout  (layer-level, existing, drop_rate=dropout)
        L2: gvp_dropout_prob      (output-level sample mask, NEW)
        L3: coord_noise_std       (input coord noise, NEW)

    Block layout:
        b_gvp = h_s_gvp (纯GVP几何输出)
        b0 = h_seq (融合后)
        b1 = h_prot_global (Transformer后)
        b_lig0 = h_ligand_with_global (配体编码器输出)
    """

    DEFAULT_CONFIG = {
        'd_esm': 1280,
        'use_esm_mlc': False,
        'esm_mlc_num_layers': 0,
        'esm_mlc_fusion': 'weighted_sum',  # weighted_sum | concat_proj
        'esm_mlc_init': 'uniform',         # uniform | last
        'esm_mlc_last_logit_bias': 4.0,
        'esm_mlc_concat_zero_init': True,
        'esm_mlc_dropout': 0.0,
        'use_esm3': False,
        'd_esm3': 1536,
        'esm3_fusion': 'global_prior',     # global_prior | residue_gate | concat_proj
        'esm3_alpha_init': 0.02,
        'esm3_alpha_max': 0.20,
        'esm3_gate_bias_init': 1.5,
        'esm3_zero_init': True,
        'esm3_dropout_prob': 0.0,
        'd_phys': 62,
        'd_spatial_scalar': 89,
        'd_phys_encoded': 128,
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 109,
        'n_gvp_vectors': 12,
        'n_gvp_vector_hidden': 20,
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 4,
        'n_heads': 8,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'classifier_dropout': 0.2,
        'stochastic_depth_prob': 0.1,
        'esm_dropout_prob': 0.1,
        'physchem_dropout_prob': 0.2,
        'gvp_dropout_prob': 0.1,      # V10m4: GVP output sample-level dropout
        'coord_noise_std': 0.5,       # V10m4: CA coordinate Gaussian noise (Å)
        # V10m4: 3D Distance-Biased Attention (protein self-attention only, no leakage)
        'use_dist_bias': True,        # enable distance bias in protein context encoder
        'dist_bias_n_rbf': 16,        # RBF bins for distance encoding
        'dist_bias_max_dist': 20.0,   # saturation distance (Å)
        'use_sdpa_protein_self_attn': False,  # fused PyTorch SDPA for long protein self-attn
        # V10m4: enriched protein GVP edges
        'gvp_edge_rbf_bins': 16,      # RBF bins for CA-CA distance encoding
        'gvp_edge_rbf_dmax': 20.0,    # RBF max distance (Å)
        # gvp edge_in_dim is derived: (rbf_bins + 4, 2) = (20, 2) by default
        # V10m4: ligand edge enrichment (chemical bond + RBF distance)
        'ligand_edge_rbf_bins': 8,    # RBF bins added to ligand edge features
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'use_gradient_checkpointing': False,
        'n_classes': 3,
        'edge_dim': 6,
        # Multi-chain awareness
        # use_chain_features: master switch (backward compat)
        # use_chain_rope:     RoPE chain-gap position IDs (Phase 1); None = inherit master
        # use_chain_edge:     GVP is_same_chain edge bit  (Phase 2); None = inherit master
        'use_chain_features': True,
        'use_chain_rope': None,
        'use_chain_edge': None,
        'chain_gap_size': 1000,
        # V10m8: Distance regression auxiliary head
        'use_dist_aux': False,
        # V11: Ligand internal self-attention (0 = disabled, backward compat)
        'n_ligand_self_attn_layers': 0,
        # V11m2: Ligand self-attention distance bias + b_lig1 AttnRes anchor
        'use_lig_dist_bias': False,       # enable 3D distance bias in ligand self-attention
        'lig_dist_bias_n_rbf': 8,         # RBF bins (fewer than protein; ligand radius < 8Å)
        'lig_dist_bias_max_dist': 8.0,    # saturation distance (Å)
        'use_lig_attnres_b1': False,      # add b_lig1 (post-self-attn) as 2nd ligand AttnRes anchor
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

        # Resolve fine-grained chain flags (None → inherit master switch)
        _master = c.get('use_chain_features', True)
        self._use_chain_rope = c['use_chain_rope'] if c.get('use_chain_rope') is not None else _master
        self._use_chain_edge = c['use_chain_edge'] if c.get('use_chain_edge') is not None else _master

        # Module 1: PhysChem MLP Encoder
        self.physchem_encoder = nn.Sequential(
            nn.Linear(c['d_phys'], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c['d_phys_encoded']),
            nn.LayerNorm(c['d_phys_encoded']),
            nn.GELU()
        )

        # Module 2: Protein Geometric Encoder (GVP)
        # edge scalar: rbf_bins + 4 [+ 1 if use_chain_edge] — RBF + seq offset [+ is_same_chain]
        # edge vector: 2 — global dir + local backbone frame dir
        _rbf_bins    = c.get('gvp_edge_rbf_bins', 16)
        _edge_s_dim  = _rbf_bins + 4 + (1 if self._use_chain_edge else 0)
        _edge_v_dim  = 2
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
            node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
            edge_in_dim=(_edge_s_dim, _edge_v_dim),
            n_layers=c['n_gvp_layers'],
            drop_rate=c['dropout']
        )

        # Module 3: Bridge & Gating Fusion
        self.pairwise_bridge = PairwiseBridge(n_vec_channels=c['n_gvp_vector_hidden'])
        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        d_geo = c['d_gvp_internal'] + n_pairwise + c['d_phys_encoded']
        d_half = c['d_model'] // 2

        self.use_esm_mlc = bool(c.get('use_esm_mlc', False))
        self.esm_mlc_fusion = str(c.get('esm_mlc_fusion', 'weighted_sum')).lower()
        self.esm_mlc_num_layers = int(c.get('esm_mlc_num_layers', 0) or 0)
        if self.use_esm_mlc:
            if self.esm_mlc_num_layers <= 0:
                raise ValueError("use_esm_mlc=true requires model.esm_mlc_num_layers > 0")
            if self.esm_mlc_fusion not in {"weighted_sum", "concat_proj"}:
                raise ValueError(
                    "model.esm_mlc_fusion must be 'weighted_sum' or 'concat_proj', "
                    f"got {self.esm_mlc_fusion!r}"
                )
            self.esm_mlc_layer_norm = nn.LayerNorm(c['d_esm'])
            init_logits = torch.zeros(self.esm_mlc_num_layers)
            if str(c.get('esm_mlc_init', 'uniform')).lower() in {'last', 'final'}:
                init_logits[-1] = float(c.get('esm_mlc_last_logit_bias', 4.0))
            self.esm_mlc_logits = nn.Parameter(init_logits)
            if self.esm_mlc_fusion == "concat_proj":
                self.esm_mlc_concat_proj = nn.Sequential(
                    nn.LayerNorm(c['d_esm'] * self.esm_mlc_num_layers),
                    nn.Linear(c['d_esm'] * self.esm_mlc_num_layers, c['d_esm']),
                    nn.GELU(),
                    nn.Dropout(float(c.get('esm_mlc_dropout', 0.0))),
                    nn.LayerNorm(c['d_esm']),
                )
                if bool(c.get('esm_mlc_concat_zero_init', True)):
                    nn.init.zeros_(self.esm_mlc_concat_proj[1].weight)
                    nn.init.zeros_(self.esm_mlc_concat_proj[1].bias)
            else:
                self.esm_mlc_concat_proj = None
        else:
            self.esm_mlc_layer_norm = None
            self.register_buffer("esm_mlc_logits", torch.zeros(0), persistent=False)
            self.esm_mlc_concat_proj = None

        self.gvp_proj = nn.Sequential(
            nn.Linear(d_geo, d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout'])
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(c['d_esm'], d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout'])
        )
        self.use_esm3 = bool(c.get('use_esm3', False))
        self.esm3_fusion = str(c.get('esm3_fusion', 'global_prior')).lower()
        self.esm3_alpha_max = float(c.get('esm3_alpha_max', 0.20))
        if self.use_esm3:
            if self.esm3_fusion not in {"global_prior", "residue_gate", "concat_proj"}:
                raise ValueError(
                    "model.esm3_fusion must be one of global_prior/residue_gate/concat_proj, "
                    f"got {self.esm3_fusion!r}"
                )
            d_esm3 = int(c.get('d_esm3', 1536))
            self.esm3_input_norm = nn.LayerNorm(d_esm3, elementwise_affine=False)
            self.esm3_proj = nn.Sequential(
                nn.Linear(d_esm3, d_half),
                nn.LayerNorm(d_half),
                nn.Dropout(c['dropout']),
            )
            if self.esm3_fusion == "residue_gate":
                self.esm3_gate = nn.Linear(d_half * 2, d_half)
                nn.init.constant_(self.esm3_gate.bias, float(c.get('esm3_gate_bias_init', 1.5)))
                self.esm3_concat_proj = None
            elif self.esm3_fusion == "concat_proj":
                self.esm3_gate = None
                self.esm3_concat_proj = nn.Sequential(
                    nn.LayerNorm(d_half * 2),
                    nn.Linear(d_half * 2, d_half),
                    nn.GELU(),
                    nn.LayerNorm(d_half),
                )
                if bool(c.get('esm3_zero_init', True)):
                    nn.init.zeros_(self.esm3_concat_proj[1].weight)
                    nn.init.zeros_(self.esm3_concat_proj[1].bias)
            else:
                self.esm3_gate = None
                self.esm3_concat_proj = None
            ratio = float(c.get('esm3_alpha_init', 0.02)) / max(self.esm3_alpha_max, 1e-8)
            ratio = min(max(ratio, 1e-5), 1.0 - 1e-5)
            self.esm3_raw_alpha = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        else:
            self.esm3_input_norm = None
            self.esm3_proj = None
            self.esm3_gate = None
            self.esm3_concat_proj = None
            self.register_buffer("esm3_raw_alpha", torch.tensor(0.0), persistent=False)
        self.modality_gate = nn.Sequential(
            nn.Linear(d_half * 2, d_half),
            nn.ReLU(),
            nn.Linear(d_half, d_half),
            nn.Sigmoid()
        )

        # V10m3: project b_gvp (GVP scalar 256) → d_model (512) for AttnRes
        self.b_gvp_proj = nn.Linear(c['d_gvp_internal'], c['d_model'], bias=False)

        # Module 4: RoPE Protein Context Encoder (V10m4: optional 3D Distance Bias)
        self.protein_context_encoder = RoPEProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            stochastic_depth_prob=c.get('stochastic_depth_prob', 0.0),
            attn_dropout=c.get('attn_dropout', None),
            use_dist_bias=c.get('use_dist_bias', False),
            dist_bias_n_rbf=c.get('dist_bias_n_rbf', 16),
            dist_bias_max_dist=c.get('dist_bias_max_dist', 20.0),
            use_sdpa=c.get('use_sdpa_protein_self_attn', False),
            hybrid_pattern=c.get('protein_hybrid_pattern'),
            gdn_backend=c.get('protein_gdn_backend', 'fla'),
        )

        # Module 5: 3D Ligand Encoder
        # V10m4: ligand edge = chemical bond features (6D) + RBF(atom distance, 8 bins)
        #   This combines topological (bond type/ring/aromaticity) with 3D geometry
        _lig_rbf_bins    = c.get('ligand_edge_rbf_bins', 8)
        _lig_edge_in_dim = c['edge_dim'] + _lig_rbf_bins   # 6 + 8 = 14 default
        self.ligand_encoder = Ligand3DEncoder(
            node_in_dim=c['d_ligand_atom'],
            node_h_dim=c['d_gvp_internal'] // 2,
            n_vec_channels=4,
            n_vec_hidden=16,
            edge_in_dim=_lig_edge_in_dim,
            out_dim=c['d_model'],
            n_layers=c['n_ligand_layers'],
            dropout=c['dropout'],
            radius=5.0,
            max_neighbors=32
        )
        # V10m5: ligand global token is now computed dynamically (masked mean pool)
        # instead of a fixed nn.Parameter — no learnable parameter here

        # Module 5b: V11 Ligand Internal Self-Attention
        # Runs AFTER ligand GNN encoder + global token, BEFORE cross-attention.
        # Gives ligand atoms intra-molecular context comparable to protein's self-attention.
        _n_lig_sa = c.get('n_ligand_self_attn_layers', 0)
        if _n_lig_sa > 0:
            self.ligand_self_attn = LigandSelfAttentionEncoder(
                d_model=c['d_model'],
                n_layers=_n_lig_sa,
                n_heads=c['n_heads'],
                dropout=c['dropout'],
                attn_dropout=c.get('attn_dropout', None),
                use_dist_bias=c.get('use_lig_dist_bias', False),          # V11m2
                dist_bias_n_rbf=c.get('lig_dist_bias_n_rbf', 8),          # V11m2
                dist_bias_max_dist=c.get('lig_dist_bias_max_dist', 8.0),  # V11m2
            )
        else:
            self.ligand_self_attn = None

        # Module 6: Bidirectional Cross-Attention
        n_cross_layers = c.get('n_cross_attention_layers', 4)
        stochastic_depth_prob = c.get('stochastic_depth_prob', 0.0)
        cross_drop_path_probs = [
            stochastic_depth_prob * i / max(n_cross_layers - 1, 1)
            for i in range(n_cross_layers)
        ]
        self.cross_attention_layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout'],
                drop_path_prob=cross_drop_path_probs[i],
                attn_dropout=c.get('attn_dropout', None)
            )
            for i in range(n_cross_layers)
        ])

        # V10m3: Protein-side AttnRes (3 blocks: b_gvp, b0, b1)
        self.cross_attn_res = nn.ModuleList([
            InterBlockAttnRes(d_model=c['d_model'])
            for _ in range(n_cross_layers)
        ])

        # V10m3: Ligand-side AttnRes (1 block: b_lig0)
        self.lig_attn_res = nn.ModuleList([
            InterBlockAttnRes(d_model=c['d_model'])
            for _ in range(n_cross_layers)
        ])

        # Module 7: Classifier
        d_classifier_in = c['d_model'] + c['d_phys_encoded']
        classifier_dropout = c.get('classifier_dropout', 0.2)
        # LayerNorm before classifier: h_prot_fused (cross-attn residual, no trailing norm)
        # and protein_physchem_encoded (has LayerNorm at encoder output) may differ in scale.
        # LN is a low-risk stabilizer; whether it materially helps depends on observed norm gap.
        self.classifier_input_norm = nn.LayerNorm(d_classifier_in)
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, c['n_classes'])
        )

        # V10m8: Distance regression auxiliary head
        # Predicts continuous d_min (residue-ligand distance) alongside classification.
        # Provides smoother gradient for boundary residues (4.9Å vs 5.1Å get similar gradients).
        # Head is dropped at inference (zero cost).
        if c.get('use_dist_aux', False):
            self.dist_aux_head = nn.Sequential(
                nn.Linear(c['d_model'], 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()       # distance >= 0, smooth and differentiable
            )
        else:
            self.dist_aux_head = None

        self._cached_esm_mlc_weights = None
        self._cached_esm3_alpha = None
        self._cached_esm3_prior_abs_mean = None
        self._cached_esm3_delta_abs_mean = None
        self._cached_esm3_gate_mean = None

    def _apply_esm3_fusion(
        self,
        h_esm_proj: torch.Tensor,
        esm3_embeddings: Optional[torch.Tensor],
        protein_mask: torch.Tensor,
        esm3_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject optional ESM3 prior into the ESM-side half-width stream."""
        self._cached_esm3_alpha = None
        self._cached_esm3_prior_abs_mean = None
        self._cached_esm3_delta_abs_mean = None
        self._cached_esm3_gate_mean = None
        if not self.use_esm3:
            return h_esm_proj
        if esm3_embeddings is None:
            raise ValueError("model.use_esm3=true requires esm3_embeddings in forward inputs")
        if esm3_embeddings.ndim != 3:
            raise ValueError(f"esm3_embeddings must be [B,L,D], got {tuple(esm3_embeddings.shape)}")
        if esm3_embeddings.shape[:2] != h_esm_proj.shape[:2]:
            raise ValueError(
                "ESM3 shape mismatch: "
                f"esm3={tuple(esm3_embeddings.shape)}, h_esm={tuple(h_esm_proj.shape)}"
            )
        valid_esm3 = protein_mask.bool()
        if esm3_valid_mask is not None:
            if esm3_valid_mask.shape != protein_mask.shape:
                raise ValueError("ESM3 validity mask shape must match protein mask")
            valid_esm3 = valid_esm3 & esm3_valid_mask.bool()
        x3 = self.esm3_input_norm(esm3_embeddings.float()).to(dtype=h_esm_proj.dtype)
        if self.training and float(self.config.get('esm3_dropout_prob', 0.0)) > 0:
            keep = 1.0 - float(self.config.get('esm3_dropout_prob', 0.0))
            mask = torch.bernoulli(torch.full((x3.shape[0], 1, 1), keep, device=x3.device, dtype=x3.dtype))
            x3 = x3 * mask
        h3 = self.esm3_proj(x3)
        alpha = self.esm3_alpha_max * torch.sigmoid(self.esm3_raw_alpha.to(dtype=h_esm_proj.dtype))
        self._cached_esm3_alpha = alpha.detach()

        if self.esm3_fusion == "global_prior":
            valid = valid_esm3.to(dtype=h3.dtype).unsqueeze(-1)
            pooled = (h3 * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            delta = alpha * pooled
            self._cached_esm3_prior_abs_mean = pooled.detach().float().abs().mean()
            self._cached_esm3_delta_abs_mean = delta.detach().float().abs().mean()
            return h_esm_proj + delta
        if self.esm3_fusion == "residue_gate":
            gate = torch.sigmoid(self.esm3_gate(torch.cat([h_esm_proj, h3], dim=-1)))
            delta = alpha * gate * (h3 - h_esm_proj)
            if esm3_valid_mask is not None:
                delta = delta.masked_fill(~valid_esm3.unsqueeze(-1), 0.0)
            self._cached_esm3_prior_abs_mean = h3.detach().float().abs().mean()
            self._cached_esm3_delta_abs_mean = delta.detach().float().abs().mean()
            self._cached_esm3_gate_mean = gate.detach().float().mean()
            return h_esm_proj + delta
        delta = self.esm3_concat_proj(torch.cat([h_esm_proj, h3], dim=-1))
        self._cached_esm3_prior_abs_mean = delta.detach().float().abs().mean()
        delta = alpha * delta
        if esm3_valid_mask is not None:
            delta = delta.masked_fill(~valid_esm3.unsqueeze(-1), 0.0)
        self._cached_esm3_delta_abs_mean = delta.detach().float().abs().mean()
        return h_esm_proj + delta

    def _fuse_esm_mlc(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        """Fuse optional multi-layer ESM [B, L, K, D] to standard [B, L, D]."""
        if esm_embeddings.ndim == 3:
            self._cached_esm_mlc_weights = None
            return esm_embeddings
        if esm_embeddings.ndim != 4:
            raise ValueError(
                "esm_embeddings must be [B,L,D] or [B,L,K,D], "
                f"got shape={tuple(esm_embeddings.shape)}"
            )
        if not self.use_esm_mlc:
            raise ValueError(
                "Received multi-layer ESM embeddings [B,L,K,D], but "
                "model.use_esm_mlc is false."
            )

        n_layers = int(esm_embeddings.size(2))
        if n_layers != self.esm_mlc_num_layers:
            raise ValueError(
                f"ESM MLC layer count mismatch: input K={n_layers}, "
                f"config esm_mlc_num_layers={self.esm_mlc_num_layers}"
            )

        x = self.esm_mlc_layer_norm(esm_embeddings)
        weights = torch.softmax(self.esm_mlc_logits.to(dtype=x.dtype), dim=0)
        self._cached_esm_mlc_weights = weights.detach()
        if self.esm_mlc_fusion == "concat_proj":
            fused = self.esm_mlc_concat_proj(x.flatten(start_dim=2))
            # Residual weighted sum keeps concat-proj anchored to a stable ESM scale.
            return fused + torch.sum(weights.view(1, 1, -1, 1) * x, dim=2)
        return torch.sum(weights.view(1, 1, -1, 1) * x, dim=2)

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        protein_physchem: torch.Tensor,
        protein_spatial_scalar: torch.Tensor,
        protein_spatial_vector: torch.Tensor,
        protein_gvp_vectors: torch.Tensor,
        protein_coords: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_edge_index: torch.Tensor,
        ligand_edge_attr: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
        chain_ids: Optional[torch.Tensor] = None,   # [B, L] integer chain indices
        ligand_to_protein: Optional[torch.Tensor] = None,  # legacy placeholder, ignored
        esm3_embeddings: Optional[torch.Tensor] = None,
        esm_valid_mask: Optional[torch.Tensor] = None,
        esm3_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        raw_esm_embeddings = esm_embeddings
        esm_embeddings = self._fuse_esm_mlc(esm_embeddings)
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

        # Chain-gap RoPE position IDs
        position_ids = None
        if self._use_chain_rope and chain_ids is not None:
            position_ids = get_cumulative_gap_position_ids(
                chain_ids, gap_size=c.get('chain_gap_size', 1000)
            )  # [B, L]

        # Module 1: PhysChem Encoder
        protein_physchem_encoded = self.physchem_encoder(protein_physchem)
        physchem_dropout_prob = c.get('physchem_dropout_prob', 0.0)
        if self.training and physchem_dropout_prob > 0:
            physchem_dropout_mask = (torch.rand(B, 1, 1, device=device) > physchem_dropout_prob).float()
            protein_physchem_encoded = protein_physchem_encoded * physchem_dropout_mask

        # Module 2: GVP Encoder
        protein_vectors_combined = torch.cat([
            protein_gvp_vectors,
            protein_spatial_vector
        ], dim=2)

        # V10m4: Coordinate noise std (L3 regularization)
        coord_noise_std  = c.get('coord_noise_std', 0.0)
        gvp_rbf_bins     = c.get('gvp_edge_rbf_bins', 16)
        gvp_rbf_dmax     = c.get('gvp_edge_rbf_dmax', 20.0)

        # V10m4: Recover N and C backbone coordinates from protein_gvp_vectors.
        # _compute_gvp_vectors stores:
        #   gvp_vectors[:, :, 1, :] = N  - CA (N displacement from CA)
        #   gvp_vectors[:, :, 3, :] = C  - CA (C displacement from CA)
        # So absolute positions: N = CA + gvp_vec[:,1], C = CA + gvp_vec[:,3]
        protein_n_coords = protein_coords + protein_gvp_vectors[:, :, 1, :]   # [B, L, 3]
        protein_c_coords = protein_coords + protein_gvp_vectors[:, :, 3, :]   # [B, L, 3]

        # V10m6: Batched GVP graph — build one flat graph for all samples,
        # process in a single GPU call, then scatter back to [B, L, d].
        # Replaces the per-sample Python for-loop from V10m5.
        mask_bool = protein_mask.bool()                                   # [B, L]
        n_valid_per_sample = mask_bool.sum(dim=1)                         # [B]
        total_valid = n_valid_per_sample.sum().item()

        if total_valid > 0:
            # Flatten valid residues from all samples into [N_total, ...]
            all_scalar   = protein_spatial_scalar[mask_bool]               # [N_total, d_scalar]
            all_vectors  = protein_vectors_combined[mask_bool]             # [N_total, n_vec, 3]
            all_ca       = protein_coords[mask_bool]                       # [N_total, 3]
            all_n        = protein_n_coords[mask_bool]                     # [N_total, 3]
            all_c        = protein_c_coords[mask_bool]                     # [N_total, 3]
            all_chain    = chain_ids[mask_bool] if chain_ids is not None else None  # [N_total]

            # V10m4 L3: Gaussian noise on CA coords (training only)
            if self.training and coord_noise_std > 0:
                noise = torch.randn_like(all_ca) * coord_noise_std
                all_ca = all_ca + noise
                all_n  = all_n  + noise
                all_c  = all_c  + noise

            # Build batch indicator for per-sample graph isolation
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), n_valid_per_sample
            )  # [N_total]

            # Build per-sample graphs and merge with offset edge indices
            edge_index_list, edge_s_list, edge_v_list = [], [], []
            node_offset = 0
            for b in range(B):
                n_b = n_valid_per_sample[b].item()
                if n_b == 0:
                    continue
                ca_b    = all_ca[node_offset:node_offset + n_b]
                n_b_c   = all_n[node_offset:node_offset + n_b]
                c_b_c   = all_c[node_offset:node_offset + n_b]
                chain_b = all_chain[node_offset:node_offset + n_b] if all_chain is not None else None

                ei_b, es_b, ev_b = build_protein_graph(
                    coords_ca=ca_b, coords_n=n_b_c, coords_c=c_b_c,
                    k=c['gvp_k_neighbors'], radius=c['gvp_radius'],
                    num_rbf=gvp_rbf_bins, rbf_dmax=gvp_rbf_dmax,
                    chain_ids=chain_b if self._use_chain_edge else None,
                )
                edge_index_list.append(ei_b + node_offset)
                edge_s_list.append(es_b)
                edge_v_list.append(ev_b)
                node_offset += n_b

            # Concatenate all per-sample graphs into one batched graph
            all_edge_index = torch.cat(edge_index_list, dim=1)            # [2, E_total]
            all_edge_s     = torch.cat(edge_s_list, dim=0)                # [E_total, d_es]
            all_edge_v     = torch.cat(edge_v_list, dim=0)                # [E_total, 2, 3]

            # Single GVP forward pass on the entire batched graph
            h_s_flat, h_v_flat = self.gvp_encoder(
                node_s=all_scalar, node_v=all_vectors,
                edge_index=all_edge_index, edge_s=all_edge_s, edge_v=all_edge_v
            )  # h_s_flat: [N_total, d_gvp], h_v_flat: [N_total, n_vec_h, 3]

            # Scatter back to padded [B, L, d] tensors
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'], device=device, dtype=h_s_flat.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v_flat.dtype)
            h_s_gvp[mask_bool] = h_s_flat
            h_v_gvp[mask_bool] = h_v_flat
        else:
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'], device=device, dtype=protein_spatial_scalar.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype)

        # V10m4 L2: GVP Output Dropout — sample-level mask, symmetric to ESM/PhysChem dropout
        # Randomly zeros entire GVP output for some training samples.
        # Effect: forces model to reconstruct binding signal from ESM alone,
        #         preventing over-reliance on geometric features.
        # During eval: always uses full GVP output (no scaling needed, train-only mask).
        gvp_dropout_prob = c.get('gvp_dropout_prob', 0.0)
        if self.training and gvp_dropout_prob > 0:
            # [B, 1, 1]: one mask per sample, broadcast over L and feature dim
            gvp_drop_mask = torch.bernoulli(
                torch.full((B, 1, 1), 1.0 - gvp_dropout_prob, device=device)
            )
            h_s_gvp = h_s_gvp * gvp_drop_mask                        # [B, L, d_gvp_internal]
            h_v_gvp = h_v_gvp * gvp_drop_mask.unsqueeze(-1)          # [B, L, n_vec, 3]

        _do_cache = c.get('cache_intermediates', False)

        # Always cache health-monitoring intermediates (needed for esm_contribution_ratio
        # in TensorBoard regardless of cache_intermediates setting).
        # These are small tensors; the extra attribute assignment is negligible.
        self._cached_gvp_scalar = h_s_gvp
        self._cached_gvp_vector = h_v_gvp

        # V10m3: b_gvp = pure GVP scalar output projected to d_model (512)
        b_gvp = self.b_gvp_proj(h_s_gvp)

        # Module 3: Bridge & Gating Fusion
        v_inv = self.pairwise_bridge(h_v_gvp)
        h_geo = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)
        self._cached_h_geo = h_geo  # always cached for esm_contribution_ratio
        current_esm = esm_embeddings
        if self.training:
            p_drop_esm = c.get('esm_dropout_prob', 0.3)
            esm_dropout_mask = torch.bernoulli(torch.full((B, 1, 1), 1 - p_drop_esm, device=device))
            current_esm = current_esm * esm_dropout_mask
        self._cached_raw_esm_input = raw_esm_embeddings
        self._cached_current_esm = current_esm  # always cached (post-fusion/dropout ESM)
        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)
        if esm_valid_mask is not None:
            if esm_valid_mask.shape != protein_mask.shape:
                raise ValueError("ESM2 validity mask shape must match protein mask")
            h_esm_proj = h_esm_proj.masked_fill(~esm_valid_mask.bool().unsqueeze(-1), 0.0)
        h_esm_proj = self._apply_esm3_fusion(
            h_esm_proj, esm3_embeddings, protein_mask, esm3_valid_mask)
        h_concat = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate = self.modality_gate(h_concat)
        h_seq = torch.cat([gate * h_gvp_proj, (1 - gate) * h_esm_proj], dim=-1)
        self._cached_h_gvp_proj = h_gvp_proj   # always cached
        self._cached_h_esm_proj = h_esm_proj   # always cached
        if _do_cache:
            self._cached_gate = gate            # debug only: full gate tensor
        # b0: block before Transformer
        b0 = h_seq

        # Optional training-only context augmentation, outside checkpoint replay.
        context_noise = getattr(self, '_protein_context_noise', None)
        if self.training and context_noise is not None:
            h_seq = context_noise(h_seq, protein_spatial_scalar, protein_coords, protein_mask)

        # Module 4: RoPE Protein Context Encoder
        # V10m4: pass protein_coords for 3D Distance-Biased Attention
        # protein_coords = CA positions (PROTEIN-ONLY) → no leakage.
        # The bias is pre-softmax additive: close residues get higher attention.
        # V10m5: fix gradient checkpointing — pass coords & position_ids via lambda
        if self.config['use_gradient_checkpointing'] and self.training:
            def _run_encoder(_h_seq, _mask):
                return self.protein_context_encoder(
                    _h_seq, mask=_mask, coords=protein_coords,
                    position_ids=position_ids, chain_ids=chain_ids
                )
            h_prot_global = checkpoint(
                _run_encoder, h_seq, protein_mask, use_reentrant=False
            )
        else:
            h_prot_global = self.protein_context_encoder(
                h_seq, mask=protein_mask, coords=protein_coords,
                position_ids=position_ids, chain_ids=chain_ids
            )

        # b1: block after Transformer
        b1 = h_prot_global

        # Module 5: Ligand Encoder
        # V10m4: Enrich ligand edge features with RBF(inter-atom distance).
        # Chemical bond features (6D: bond type / ring / aromaticity) tell the
        # graph about topology; RBF distance adds 3D geometric context, giving
        # the GVP richer information for cross-attention with the protein.
        _lig_rbf_bins = c.get('ligand_edge_rbf_bins', 8)
        if ligand_edge_index.size(1) > 0:
            lig_row, lig_col = ligand_edge_index
            lig_dist = torch.norm(ligand_pos[lig_col] - ligand_pos[lig_row], dim=-1)  # [E_lig]
            lig_rbf  = rbf_encode(lig_dist, D_min=0.0, D_max=5.0, num_rbf=_lig_rbf_bins)  # [E_lig, 8]
            lig_edge_enriched = torch.cat([ligand_edge_attr, lig_rbf], dim=-1)          # [E_lig, 6+8]
        else:
            lig_edge_enriched = torch.cat([
                ligand_edge_attr,
                torch.zeros(0, _lig_rbf_bins, device=device, dtype=ligand_pos.dtype)
            ], dim=-1)

        h_ligand_nodes, ligand_pos_out = self.ligand_encoder(
            x=ligand_x, pos=ligand_pos,
            edge_index=ligand_edge_index, edge_attr=lig_edge_enriched,
            batch=ligand_batch
        )

        B_dim, Max_M_dim = ligand_mask.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        # V11m2: flag for atom_pos scatter (needed only when ligand dist bias is active)
        _need_atom_pos = (self.ligand_self_attn is not None and
                          getattr(self.ligand_self_attn, 'dist_bias_modules', None) is not None)
        atom_pos_padded = None
        N_atoms = h_ligand_nodes.size(0)
        if N_atoms > 0:
            # Vectorized scatter: requires ligand_batch to be sorted/contiguous by
            # sample index (i.e. all atoms of sample 0 come before sample 1, etc.),
            # which is guaranteed by pocket_collate_fn_v10 / PyG batching.
            # If atom order is ever shuffled, revert to the per-sample loop.
            assert bool((ligand_batch[1:] >= ligand_batch[:-1]).all()), \
                "ligand_batch must be non-decreasing (contiguous by sample)"
            counts = torch.bincount(ligand_batch, minlength=B_dim)          # [B]
            cumsum = torch.zeros(B_dim + 1, dtype=torch.long, device=device)
            cumsum[1:] = counts.cumsum(0)
            local_idx = torch.arange(N_atoms, device=device) - cumsum[ligand_batch]  # atom pos within sample
            valid = local_idx < Max_M_dim
            h_ligand[ligand_batch[valid], local_idx[valid]] = h_ligand_nodes[valid]
            # V11m2: scatter atom positions for ligand self-attn distance bias
            if _need_atom_pos:
                atom_pos_padded = torch.zeros(B_dim, Max_M_dim, 3, device=device, dtype=ligand_pos.dtype)
                atom_pos_padded[ligand_batch[valid], local_idx[valid]] = ligand_pos[valid]

        # V10m5: Dynamic ligand global token via masked mean pool
        mask_f = ligand_mask.float().unsqueeze(-1)                        # [B, M, 1]
        global_token = (h_ligand * mask_f).sum(1, keepdim=True) \
                       / mask_f.sum(1, keepdim=True).clamp(min=1)         # [B, 1, d]
        h_ligand_with_global = torch.cat([global_token, h_ligand], dim=1)
        global_token_mask = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)
        ligand_mask_with_global = torch.cat([global_token_mask, ligand_mask], dim=1)

        # b_lig0: ligand AttnRes anchor (raw encoder output, before self-attn)
        b_lig0 = h_ligand_with_global

        # Module 5b: V11 Ligand Internal Self-Attention
        # b_lig0 is kept as the AttnRes residual anchor so cross-attn can always
        # "skip back" to the raw encoder output. h_lig_ctx is the enriched starting
        # point fed into the cross-attention loop.
        if self.ligand_self_attn is not None:
            h_lig_ctx = self.ligand_self_attn(
                h_ligand_with_global, mask=ligand_mask_with_global,
                atom_pos=atom_pos_padded                              # V11m2: for ligand dist bias
            )
        else:
            h_lig_ctx = h_ligand_with_global

        # Module 6: Cross-Attention with AttnRes
        attn_res_blocks = [b_gvp, b0, b1]  # V10m3: 3 protein blocks
        # V11m2: optionally add b_lig1 (post-self-attn) as 2nd ligand AttnRes anchor.
        # InterBlockAttnRes.proj is Linear(d,1) — operates per-position, source-count agnostic.
        lig_attn_res_blocks = [b_lig0]     # base: 1 ligand block (raw encoder anchor)
        if c.get('use_lig_attnres_b1', False) and self.ligand_self_attn is not None:
            lig_attn_res_blocks = [b_lig0, h_lig_ctx]   # V11m2: add post-self-attn anchor
        h_prot_fused = h_prot_global
        h_lig_fused = h_lig_ctx            # V11: starts from self-attn output

        for i, cross_attn_layer in enumerate(self.cross_attention_layers):
            # V10m3: Protein-side AttnRes (3 blocks)
            h_prot_fused = self.cross_attn_res[i](
                blocks=attn_res_blocks, partial=h_prot_fused
            )

            # V10m3: Ligand-side AttnRes (1 block)
            h_lig_fused = self.lig_attn_res[i](
                blocks=lig_attn_res_blocks, partial=h_lig_fused
            )

            if self.config['use_gradient_checkpointing'] and self.training:
                h_prot_updated = checkpoint(
                    cross_attn_layer.prot_to_lig,
                    h_prot_fused, h_lig_fused,
                    protein_mask, ligand_mask_with_global,
                    use_reentrant=False
                )
                h_lig_updated = checkpoint(
                    cross_attn_layer.lig_to_prot,
                    h_lig_fused, h_prot_updated,
                    ligand_mask_with_global, protein_mask,
                    use_reentrant=False
                )
            else:
                h_prot_updated = cross_attn_layer.prot_to_lig(
                    q=h_prot_fused, kv=h_lig_fused,
                    q_mask=protein_mask, kv_mask=ligand_mask_with_global
                )
                h_lig_updated = cross_attn_layer.lig_to_prot(
                    q=h_lig_fused, kv=h_prot_updated,
                    q_mask=ligand_mask_with_global, kv_mask=protein_mask
                )

            h_prot_fused = h_prot_updated
            h_lig_fused = h_lig_updated

        # Module 7: Classifier
        x_final = torch.cat([h_prot_fused, protein_physchem_encoded], dim=-1)
        x_final = self.classifier_input_norm(x_final)
        logits = self.classifier(x_final)

        result = {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_prot_fused,
            # Ligand-only state before bidirectional cross-attention.  Downstream
            # assignment heads must use this rather than a protein-conditioned
            # ligand state to avoid shortcut matching.
            'h_ligand_context': h_lig_ctx,
            'h_ligand_fused': h_lig_fused,
            'protein_physchem_encoded': protein_physchem_encoded,  # exposed for subclass reuse
        }
        if self._cached_esm_mlc_weights is not None:
            for idx, weight in enumerate(self._cached_esm_mlc_weights):
                result[f'esm_mlc_weight_{idx}'] = weight.detach()
        if self._cached_esm3_alpha is not None:
            result['esm3_alpha'] = self._cached_esm3_alpha.detach()
        if self._cached_esm3_prior_abs_mean is not None:
            result['esm3_prior_abs_mean'] = self._cached_esm3_prior_abs_mean.detach()
        if self._cached_esm3_delta_abs_mean is not None:
            result['esm3_delta_abs_mean'] = self._cached_esm3_delta_abs_mean.detach()
        if self._cached_esm3_gate_mean is not None:
            result['esm3_gate_mean'] = self._cached_esm3_gate_mean.detach()

        # V10m8: Distance regression auxiliary head
        if self.dist_aux_head is not None:
            result['pred_dist'] = self.dist_aux_head(h_prot_fused).squeeze(-1)  # [B, L]

        return result
