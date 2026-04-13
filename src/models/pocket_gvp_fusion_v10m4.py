"""
Pocket-GVP-Fusion V10m4: GVP Output Dropout + Coordinate Noise.

V10m4 updates over V10m3:
- GVP Output Dropout (sample-level, symmetric to ESM dropout):
    - gvp_dropout_prob: randomly zero entire GVP output for some training samples
    - Forces model to not over-rely on geometric features; improves generalization
- Coordinate Noise (structural augmentation):
    - coord_noise_std: add Gaussian noise to CA coordinates before build_knn_graph
    - Changes both graph topology and edge distances -> stronger structural regularization

原有內部 GVP Dropout（layer/channel-level）保持不變。

三層 GVP 正則化:
  L1: internal GVP Dropout     - layer-level scalar+vector channel dropout (現有, drop_rate=0.1)
  L2: gvp_dropout_prob         - output-level sample mask (新增, default 0.1)
  L3: coord_noise_std          - input coordinate noise (新增, default 0.5Å)

繼承 V10m3:
  RoPE + Ligand AttnRes + GVP Block (b_gvp)

Block划分策略:
  b_gvp: h_s_gvp (纯GVP几何输出，融合前)
  b0:    h_seq (GVP+ESM融合后、进入Transformer前)
  b1:    h_prot_global (4层Transformer编码后)

  配体侧:
  b_lig0: h_ligand_with_global (配体编码器输出 + global token)

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
                 use_gate: bool = True, attn_dropout: Optional[float] = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_gate = use_gate

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
                 drop_path_prob: float = 0.0, attn_dropout: Optional[float] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEGatedMultiHeadAttention(
            d_model, n_heads, dropout, use_gate=use_gated_attn, attn_dropout=attn_dropout
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
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask=mask, attn_bias=attn_bias,
                             position_ids=position_ids)
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
                 dist_bias_max_dist: float = 20.0):  # V10m4
        super().__init__()
        drop_path_probs = [stochastic_depth_prob * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            RoPEGatedTransformerBlock(
                d_model, n_heads, dropout, use_gated_attn=True, use_swiglu=True,
                drop_path_prob=drop_path_probs[i], attn_dropout=attn_dropout
            ) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # V10m4: optional distance bias (computed once, shared across all layers)
        self.dist_bias_module = (
            DistanceBias(n_heads, n_rbf=dist_bias_n_rbf, max_dist=dist_bias_max_dist)
            if use_dist_bias else None
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # [B, L] chain-gap positions
    ) -> torch.Tensor:
        attn_bias = None
        if self.dist_bias_module is not None and coords is not None:
            attn_bias = self.dist_bias_module(coords, mask=mask)

        for layer in self.layers:
            x = layer(x, mask=mask, attn_bias=attn_bias, position_ids=position_ids)
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
# Main model
# ---------------------------------------------------------------------------

class PocketGVPFusionV10m4(nn.Module):
    """
    Pocket-GVP-Fusion V10m4.

    V10m4 additions over V10m3:
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
        self.ligand_global_token = nn.Parameter(torch.randn(1, 1, c['d_model']))

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
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, c['n_classes'])
        )

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
        ligand_to_protein: torch.Tensor,
        ligand_mask: torch.Tensor,
        chain_ids: Optional[torch.Tensor] = None,   # [B, L] integer chain indices
    ) -> Dict[str, torch.Tensor]:

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

        h_s_gvp_list, h_v_gvp_list = [], []
        for b in range(B):
            mask_b = protein_mask[b]
            n_valid = mask_b.sum()
            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_spatial_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype))
                continue

            scalar_b   = protein_spatial_scalar[b, mask_b]
            vectors_b  = protein_vectors_combined[b, mask_b]
            coords_ca_b = protein_coords[b, mask_b]
            chain_ids_b = chain_ids[b, mask_b] if chain_ids is not None else None
            coords_n_b  = protein_n_coords[b, mask_b]
            coords_c_b  = protein_c_coords[b, mask_b]

            # V10m4 L3: Gaussian noise on CA coords before graph construction.
            # Perturbs both KNN topology AND the backbone frame (via CA), giving
            # stronger structural regularization than post-GVP noise.
            # Only during training; eval uses clean coords.
            if self.training and coord_noise_std > 0:
                noise = torch.randn_like(coords_ca_b) * coord_noise_std
                coords_ca_b = coords_ca_b + noise
                # Apply same noise to N and C to preserve local frame geometry
                coords_n_b  = coords_n_b  + noise
                coords_c_b  = coords_c_b  + noise

            # Enriched protein graph — RBF dist + seq offset + local frame + is_same_chain
            edge_index, edge_s, edge_v = build_protein_graph(
                coords_ca=coords_ca_b,
                coords_n=coords_n_b,
                coords_c=coords_c_b,
                k=c['gvp_k_neighbors'],
                radius=c['gvp_radius'],
                num_rbf=gvp_rbf_bins,
                rbf_dmax=gvp_rbf_dmax,
                chain_ids=chain_ids_b if self._use_chain_edge else None,
            )
            h_s, h_v = self.gvp_encoder(
                node_s=scalar_b, node_v=vectors_b,
                edge_index=edge_index, edge_s=edge_s, edge_v=edge_v
            )

            h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device, dtype=h_s.dtype)
            h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v.dtype)
            h_s_full[mask_b] = h_s
            h_v_full[mask_b] = h_v
            h_s_gvp_list.append(h_s_full)
            h_v_gvp_list.append(h_v_full)

        h_s_gvp = torch.stack(h_s_gvp_list, dim=0)
        h_v_gvp = torch.stack(h_v_gvp_list, dim=0)

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

        self._cached_gvp_scalar = h_s_gvp
        self._cached_gvp_vector = h_v_gvp

        # V10m3: b_gvp = pure GVP scalar output projected to d_model (512)
        b_gvp = self.b_gvp_proj(h_s_gvp)

        # Module 3: Bridge & Gating Fusion
        v_inv = self.pairwise_bridge(h_v_gvp)
        h_geo = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)
        self._cached_h_geo = h_geo

        current_esm = esm_embeddings
        if self.training:
            p_drop_esm = c.get('esm_dropout_prob', 0.3)
            esm_dropout_mask = torch.bernoulli(torch.full((B, 1, 1), 1 - p_drop_esm, device=device))
            current_esm = current_esm * esm_dropout_mask
        self._cached_current_esm = current_esm

        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)
        h_concat = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate = self.modality_gate(h_concat)
        h_seq = torch.cat([gate * h_gvp_proj, (1 - gate) * h_esm_proj], dim=-1)
        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj
        self._cached_gate = gate

        # b0: block before Transformer
        b0 = h_seq

        # Module 4: RoPE Protein Context Encoder
        # V10m4: pass protein_coords for 3D Distance-Biased Attention
        # protein_coords = CA positions (PROTEIN-ONLY) → no leakage.
        # The bias is pre-softmax additive: close residues get higher attention.
        # gradient_checkpointing does not support extra kwargs, so bypass when active.
        if self.config['use_gradient_checkpointing'] and self.training:
            h_prot_global = checkpoint(
                self.protein_context_encoder, h_seq, protein_mask, use_reentrant=False
            )
        else:
            h_prot_global = self.protein_context_encoder(
                h_seq, mask=protein_mask, coords=protein_coords,
                position_ids=position_ids
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

        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        for b in range(B_dim):
            mask_b = ligand_batch == b
            if mask_b.any():
                nodes_b = h_ligand_nodes[mask_b]
                n_atoms = min(nodes_b.shape[0], Max_M_dim)
                h_ligand[b, :n_atoms] = nodes_b[:n_atoms]

        global_token = self.ligand_global_token.expand(B_dim, -1, -1)
        h_ligand_with_global = torch.cat([global_token, h_ligand], dim=1)
        global_token_mask = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)
        ligand_mask_with_global = torch.cat([global_token_mask, ligand_mask], dim=1)

        # b_lig0: ligand block
        b_lig0 = h_ligand_with_global

        # Module 6: Cross-Attention with AttnRes
        attn_res_blocks = [b_gvp, b0, b1]  # V10m3: 3 protein blocks
        lig_attn_res_blocks = [b_lig0]     # V10m3: 1 ligand block
        h_prot_fused = h_prot_global
        h_lig_fused = h_ligand_with_global

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
        logits = self.classifier(x_final)

        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_prot_fused,
            'h_ligand_fused': h_lig_fused,
        }
