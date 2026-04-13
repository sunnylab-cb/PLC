"""
Pocket-GVP-Fusion V10m2: RoPE + Ligand-side AttnRes + GVP Block.

V10m2 updates over V10m1:
- RoPE (Rotary Position Embedding) in ProteinContextEncoder Transformer
- Ligand-side Inter-Block AttnRes (配体侧也回溯早期表征)
- GVP output as additional block (b_gvp) for AttnRes (纯几何信号直达后期)

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

from .gvp_layers import GVPEncoder, build_knn_graph
from .transformer_layers import BidirectionalCrossAttentionBlock
from .ligand_encoder_3d import Ligand3DEncoder


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

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin: [seq_len, dim] rotation matrices
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to input tensor.

    Args:
        x: [B, L, n_heads, head_dim] or [B, n_heads, L, head_dim]
        cos, sin: [L, head_dim]
    Returns:
        Rotated x with same shape
    """
    # Assume x is [B, n_heads, L, head_dim]
    if x.dim() == 4 and x.shape[2] != cos.shape[0]:
        # x is [B, L, n_heads, head_dim], transpose to [B, n_heads, L, head_dim]
        x = x.transpose(1, 2)
        transposed = True
    else:
        transposed = False

    # Split into two halves
    x1, x2 = x.chunk(2, dim=-1)  # each [B, n_heads, L, head_dim//2]

    # cos/sin are [L, head_dim], take only first half to match x1/x2
    half = x1.shape[-1]
    cos = cos[:, :half].unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim//2]
    sin = sin[:, :half].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    out = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        if kv is None:
            kv = x
        S = kv.size(1)

        q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D]
        k = self.w_k(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        cos_q, sin_q = self.rope(L, x.device)
        cos_k, sin_k = self.rope(S, x.device)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

        # Standard attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        ffn_out = self.ffn(self.norm2(x))
        x = x + drop_path(ffn_out, self.drop_path_prob, self.training)
        return x


class RoPEProteinContextEncoder(nn.Module):
    """Protein Context Encoder with RoPE."""
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, stochastic_depth_prob: float = 0.0,
                 attn_dropout: Optional[float] = None):
        super().__init__()
        drop_path_probs = [stochastic_depth_prob * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            RoPEGatedTransformerBlock(
                d_model, n_heads, dropout, use_gated_attn=True, use_swiglu=True,
                drop_path_prob=drop_path_probs[i], attn_dropout=attn_dropout
            ) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
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

class PocketGVPFusionV10m2(nn.Module):
    """
    Pocket-GVP-Fusion V10m2.

    V10m2 improvements over V10m1:
    1. RoPE in Transformer (相对位置编码)
    2. Ligand-side AttnRes (配体侧回溯)
    3. GVP output as additional block (b_gvp, 纯几何信号)

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
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'use_gradient_checkpointing': False,
        'n_classes': 3,
        'edge_dim': 6,
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

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
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
            node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
            edge_in_dim=(1, 1),
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

        # V10m2: project b_gvp (GVP scalar 256) → d_model (512) for AttnRes
        self.b_gvp_proj = nn.Linear(c['d_gvp_internal'], c['d_model'], bias=False)

        # Module 4: RoPE Protein Context Encoder
        self.protein_context_encoder = RoPEProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            stochastic_depth_prob=c.get('stochastic_depth_prob', 0.0),
            attn_dropout=c.get('attn_dropout', None)
        )

        # Module 5: 3D Ligand Encoder
        self.ligand_encoder = Ligand3DEncoder(
            node_in_dim=c['d_ligand_atom'],
            node_h_dim=c['d_gvp_internal'] // 2,
            n_vec_channels=4,
            n_vec_hidden=16,
            edge_in_dim=c['edge_dim'],
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

        # V10m2: Protein-side AttnRes (3 blocks: b_gvp, b0, b1)
        self.cross_attn_res = nn.ModuleList([
            InterBlockAttnRes(d_model=c['d_model'])
            for _ in range(n_cross_layers)
        ])

        # V10m2: Ligand-side AttnRes (1 block: b_lig0)
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
    ) -> Dict[str, torch.Tensor]:

        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

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

        h_s_gvp_list, h_v_gvp_list = [], []
        for b in range(B):
            mask_b = protein_mask[b]
            n_valid = mask_b.sum()
            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_spatial_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype))
                continue

            scalar_b = protein_spatial_scalar[b, mask_b]
            vectors_b = protein_vectors_combined[b, mask_b]
            coords_b = protein_coords[b, mask_b]

            edge_index, edge_dist, edge_dir = build_knn_graph(
                coords_b, k=c['gvp_k_neighbors'], radius=c['gvp_radius']
            )
            h_s, h_v = self.gvp_encoder(
                node_s=scalar_b, node_v=vectors_b,
                edge_index=edge_index, edge_s=edge_dist, edge_v=edge_dir
            )

            h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device, dtype=h_s.dtype)
            h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v.dtype)
            h_s_full[mask_b] = h_s
            h_v_full[mask_b] = h_v
            h_s_gvp_list.append(h_s_full)
            h_v_gvp_list.append(h_v_full)

        h_s_gvp = torch.stack(h_s_gvp_list, dim=0)
        h_v_gvp = torch.stack(h_v_gvp_list, dim=0)
        self._cached_gvp_scalar = h_s_gvp
        self._cached_gvp_vector = h_v_gvp

        # V10m2: b_gvp = pure GVP scalar output projected to d_model (512)
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
        if self.config['use_gradient_checkpointing'] and self.training:
            h_prot_global = checkpoint(
                self.protein_context_encoder, h_seq, protein_mask, use_reentrant=False
            )
        else:
            h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)

        # b1: block after Transformer
        b1 = h_prot_global

        # Module 5: Ligand Encoder
        h_ligand_nodes, ligand_pos_out = self.ligand_encoder(
            x=ligand_x, pos=ligand_pos,
            edge_index=ligand_edge_index, edge_attr=ligand_edge_attr,
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
        attn_res_blocks = [b_gvp, b0, b1]  # V10m2: 3 protein blocks
        lig_attn_res_blocks = [b_lig0]     # V10m2: 1 ligand block
        h_prot_fused = h_prot_global
        h_lig_fused = h_ligand_with_global

        for i, cross_attn_layer in enumerate(self.cross_attention_layers):
            # V10m2: Protein-side AttnRes (3 blocks)
            h_prot_fused = self.cross_attn_res[i](
                blocks=attn_res_blocks, partial=h_prot_fused
            )

            # V10m2: Ligand-side AttnRes (1 block)
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
