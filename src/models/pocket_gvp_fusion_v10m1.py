"""
Pocket-GVP-Fusion V10m1: AttnRes over Cross-Attention layers.

V10m1 updates over V10 (which itself is V9m2 with new feature dimensions):
- Inter-Block Attention Residuals (AttnRes, Kimi 2025) applied to the
  Cross-Attention block stack.

Block划分策略:
  Block 0 (b0): h_seq — GVP+ESM融合后、进入Transformer前的表征
  Block 1 (b1): h_prot_global — 4层Transformer编码后的输出

  Cross-Attention循环(4层)整体作为第2个Block。
  每一层交叉注意力执行前，蛋白侧先做一次Inter-Block AttnRes:
      h_prot_fused = attn_res([b0, b1], partial=h_prot_fused)
  使得每层都能以可学习的、内容依赖的权重回溯早期结构信息。

Input dimensions (same as V10 / complete_features_v4 + ligand_features_v5):
  d_phys         = 62
  d_spatial_scalar = 89
  d_ligand_atom  = 109
  n_gvp_vectors  = 12  (4 runtime + 8 protein v4 spatial vectors)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional

from .gvp_layers import GVPEncoder, build_knn_graph
from .transformer_layers import ProteinContextEncoder, BidirectionalCrossAttentionBlock
from .ligand_encoder_3d import Ligand3DEncoder


# ---------------------------------------------------------------------------
# Shared sub-modules (identical to V9m2)
# ---------------------------------------------------------------------------

class PairwiseBridge(nn.Module):
    """
    Pairwise dot product bridge for translation-invariant vector features.
    Computes Gram matrix of vector features and flattens the upper triangle.
    """
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


# ---------------------------------------------------------------------------
# Inter-Block Attention Residual module (AttnRes, Kimi 2025)
# ---------------------------------------------------------------------------

class InterBlockAttnRes(nn.Module):
    """
    Inter-Block Attention Residuals.

    Replaces the fixed residual addition between blocks with a learnable,
    input-dependent softmax attention over all completed block representations
    plus the current partial block sum.

    For N completed blocks and a partial sum h:
        keys_i  = RMSNorm(b_i)          for i in 0..N-1, and h
        logit_i = w · keys_i            (scalar per token position)
        alpha   = softmax([logit_0, ..., logit_{N-1}, logit_h])
        output  = sum_i alpha_i * b_i   +  alpha_h * h

    The pseudo-query vector w is zero-initialised so that at the start of
    training the softmax is uniform (equivalent to mean pooling), preventing
    early instability.

    Args:
        d_model: Feature dimension D.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Pseudo-query: one scalar per token dimension. init=0 is critical.
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        # RMSNorm on keys prevents large-magnitude blocks from dominating softmax
        self.key_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        blocks: List[torch.Tensor],   # N tensors of shape [B, L, D]
        partial: torch.Tensor,        # [B, L, D] current partial block sum
    ) -> torch.Tensor:
        """
        Returns:
            Tensor [B, L, D] — softmax-weighted combination of all sources.
        """
        # Stack all sources: [N+1, B, L, D]
        sources = torch.stack(blocks + [partial], dim=0)

        # Normalize keys along D
        keys = self.key_norm(sources)          # [N+1, B, L, D]

        # Scalar logit per source per token: proj maps D->1
        logits = self.proj(keys).squeeze(-1)   # [N+1, B, L]

        # Softmax over source dimension
        alpha = logits.softmax(dim=0)          # [N+1, B, L]

        # Weighted sum
        out = (alpha.unsqueeze(-1) * sources).sum(dim=0)  # [B, L, D]
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PocketGVPFusionV10m1(nn.Module):
    """
    Pocket-GVP-Fusion V10m1.

    Architecture identical to V10 (= V9m2 with updated input dims) except
    that the Cross-Attention loop uses Inter-Block AttnRes (Kimi 2025) on the
    protein side before each cross-attention layer.

    Block layout:
        b0 = h_seq           (GVP+ESM fusion output, before Transformer)
        b1 = h_prot_global   (4-layer Transformer output)
        For each cross-attention layer i:
            h_prot_fused = InterBlockAttnRes([b0, b1], partial=h_prot_fused)
            h_prot_fused, h_lig_fused = cross_attn_layer_i(...)

    Args:
        config: Configuration dictionary. Missing keys fall back to DEFAULT_CONFIG.
    """

    DEFAULT_CONFIG = {
        # Dimensions (V10: updated for complete_features_v4 + ligand_features_v5)
        'd_esm': 1280,
        'd_phys': 62,
        'd_spatial_scalar': 89,       # V10: 89D (was 72D in V9m2)
        'd_phys_encoded': 128,
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 109,         # V10: 109D (was 96D in V9m2)
        'n_gvp_vectors': 12,          # V10: 12 (4 runtime + 8 protein v4 spatial)
        'n_gvp_vector_hidden': 20,

        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 4,
        'n_heads': 8,

        # Dropout
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'classifier_dropout': 0.2,

        # Stochastic Depth
        'stochastic_depth_prob': 0.1,

        # Flash Attention
        'use_flash_attn': False,

        # Modality dropout
        'esm_dropout_prob': 0.1,
        'physchem_dropout_prob': 0.2,

        # Settings
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'use_gradient_checkpointing': False,

        # Output
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

        # Module 3: Bridge & Learnable Gating Fusion
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

        # Module 4: Protein Context Encoder (4-layer Transformer = Block 1)
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            pe_type='sinusoidal',
            stochastic_depth_prob=c.get('stochastic_depth_prob', 0.0),
            attn_dropout=c.get('attn_dropout', None),
            use_flash_attn=c.get('use_flash_attn', False)
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

        # Module 6: Sequential Bidirectional Cross-Attention
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

        # V10m1: One InterBlockAttnRes per cross-attention layer (protein side).
        # Each has its own pseudo-query w_l (zero-init) and shared-structure RMSNorm.
        # blocks = [b0, b1] → N+1 = 3 sources (b0, b1, partial).
        self.cross_attn_res = nn.ModuleList([
            InterBlockAttnRes(d_model=c['d_model'])
            for _ in range(n_cross_layers)
        ])

        # Module 7: Late Injection & Classifier
        d_classifier_in = c['d_model'] + c['d_phys_encoded']
        classifier_dropout = c.get('classifier_dropout', 0.2)
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, c['n_classes'])
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        esm_embeddings: torch.Tensor,          # [B, L, 1280]
        protein_physchem: torch.Tensor,        # [B, L, 62]
        protein_spatial_scalar: torch.Tensor,  # [B, L, 89]
        protein_spatial_vector: torch.Tensor,  # [B, L, 6, 3]
        protein_gvp_vectors: torch.Tensor,     # [B, L, 4, 3]  (runtime)
        protein_coords: torch.Tensor,          # [B, L, 3]
        protein_mask: torch.Tensor,            # [B, L]
        ligand_x: torch.Tensor,                # [N_atoms, 109]
        ligand_pos: torch.Tensor,              # [N_atoms, 3]
        ligand_edge_index: torch.Tensor,       # [2, E]
        ligand_edge_attr: torch.Tensor,        # [E, edge_dim]
        ligand_batch: torch.Tensor,            # [N_atoms]
        ligand_to_protein: torch.Tensor,       # [B, max_atoms, d_model]
        ligand_mask: torch.Tensor,             # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:

        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

        # ----------------------------------------------------------------
        # Module 1: PhysChem MLP Encoder
        # ----------------------------------------------------------------
        protein_physchem_encoded = self.physchem_encoder(protein_physchem)  # [B, L, 128]

        physchem_dropout_prob = c.get('physchem_dropout_prob', 0.0)
        if self.training and physchem_dropout_prob > 0:
            physchem_dropout_mask = (torch.rand(B, 1, 1, device=device) > physchem_dropout_prob).float()
            protein_physchem_encoded = protein_physchem_encoded * physchem_dropout_mask
        else:
            physchem_dropout_mask = torch.ones(B, 1, 1, device=device)
        self._cached_physchem_mask = physchem_dropout_mask

        # ----------------------------------------------------------------
        # Module 2: GVP Encoder
        # ----------------------------------------------------------------
        # n_gvp_vectors = 12: cat 4 runtime vectors + 8 protein-v4 spatial vectors
        protein_vectors_combined = torch.cat([
            protein_gvp_vectors,    # [B, L, 4, 3]
            protein_spatial_vector  # [B, L, 8, 3]  (v4 has 8 spatial vectors)
        ], dim=2)                   # [B, L, 12, 3]

        h_s_gvp_list, h_v_gvp_list = [], []
        for b in range(B):
            mask_b = protein_mask[b]
            n_valid = mask_b.sum()
            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_spatial_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype))
                continue

            scalar_b  = protein_spatial_scalar[b, mask_b]       # [n_valid, 89]
            vectors_b = protein_vectors_combined[b, mask_b]      # [n_valid, 12, 3]
            coords_b  = protein_coords[b, mask_b]

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

        # ----------------------------------------------------------------
        # Module 3: Bridge & Gating Fusion → h_seq  (= b0)
        # ----------------------------------------------------------------
        v_inv = self.pairwise_bridge(h_v_gvp)                                      # [B, L, 210]
        h_geo = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)      # [B, L, 594]
        self._cached_h_geo = h_geo

        current_esm = esm_embeddings
        if self.training:
            p_drop_esm = c.get('esm_dropout_prob', 0.3)
            esm_dropout_mask = torch.bernoulli(torch.full((B, 1, 1), 1 - p_drop_esm, device=device))
            current_esm = current_esm * esm_dropout_mask
        else:
            esm_dropout_mask = torch.ones((B, 1, 1), device=device)
        self._cached_esm_mask = esm_dropout_mask
        self._cached_current_esm = current_esm

        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)
        h_concat   = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate       = self.modality_gate(h_concat)
        h_seq      = torch.cat([gate * h_gvp_proj, (1 - gate) * h_esm_proj], dim=-1)  # [B, L, 512]

        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj
        self._cached_gate = gate

        # b0: block representation before Transformer
        b0 = h_seq  # [B, L, D]

        # ----------------------------------------------------------------
        # Module 4: Protein Context Encoder (Block 1)
        # ----------------------------------------------------------------
        if self.config['use_gradient_checkpointing'] and self.training:
            h_prot_global = checkpoint(
                self.protein_context_encoder, h_seq, protein_mask, use_reentrant=False
            )
        else:
            h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)

        # b1: block representation after Transformer
        b1 = h_prot_global  # [B, L, D]

        # ----------------------------------------------------------------
        # Module 5: 3D Ligand Encoder
        # ----------------------------------------------------------------
        h_ligand_nodes, ligand_pos_out = self.ligand_encoder(
            x=ligand_x, pos=ligand_pos,
            edge_index=ligand_edge_index, edge_attr=ligand_edge_attr,
            batch=ligand_batch
        )

        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand         = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        ligand_pos_batched = torch.zeros(B_dim, Max_M_dim, 3, device=device, dtype=ligand_pos.dtype)
        for b in range(B_dim):
            mask_b = ligand_batch == b
            if mask_b.any():
                nodes_b = h_ligand_nodes[mask_b]
                pos_b   = ligand_pos_out[mask_b]
                n_atoms = min(nodes_b.shape[0], Max_M_dim)
                h_ligand[b, :n_atoms]          = nodes_b[:n_atoms]
                ligand_pos_batched[b, :n_atoms] = pos_b[:n_atoms]

        global_token           = self.ligand_global_token.expand(B_dim, -1, -1)
        h_ligand_with_global   = torch.cat([global_token, h_ligand], dim=1)
        global_token_mask      = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)
        ligand_mask_with_global = torch.cat([global_token_mask, ligand_mask], dim=1)

        # ----------------------------------------------------------------
        # Module 6: Sequential Bidirectional Cross-Attention  (Block 2)
        #   V10m1: protein side uses InterBlockAttnRes before each layer.
        #   Blocks passed to AttnRes: [b0, b1]
        #   partial = h_prot_fused (current cross-attn running state)
        # ----------------------------------------------------------------
        attn_res_blocks = [b0, b1]  # completed block representations
        h_prot_fused = h_prot_global
        h_lig_fused  = h_ligand_with_global

        for i, cross_attn_layer in enumerate(self.cross_attention_layers):
            # Inter-Block AttnRes: protein side attends back to b0 and b1
            h_prot_fused = self.cross_attn_res[i](
                blocks=attn_res_blocks,
                partial=h_prot_fused
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
            h_lig_fused  = h_lig_updated

        h_lig_atoms = h_lig_fused[:, 1:, :]  # [B, Max_M, D]

        # ----------------------------------------------------------------
        # Module 7: Late Injection & Classifier
        # ----------------------------------------------------------------
        x_final = torch.cat([h_prot_fused, protein_physchem_encoded], dim=-1)
        logits  = self.classifier(x_final)

        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_prot_fused,
            'h_ligand_fused': h_lig_fused,
        }
