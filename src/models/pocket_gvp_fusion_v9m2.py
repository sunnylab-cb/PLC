"""
Pocket-GVP-Fusion V9m2: Enhanced Regularization and Efficiency over V9.

V9m2 improvements over V9:
- V9m2: Flash Attention support (triton version for ROCm) for faster training
- V9m2: Enhanced Label Smoothing (0.15 vs 0.1)
- V9m2: Boundary Loss for improved binding site edge prediction

Architecture identical to V9 — efficiency and loss improvements only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple

from .gvp_layers import GVPEncoder, build_knn_graph
from .transformer_layers import ProteinContextEncoder, BidirectionalCrossAttentionBlock
from .ligand_encoder_3d import Ligand3DEncoder


class PairwiseBridge(nn.Module):
    """
    Pairwise dot product bridge for translation-invariant vector features.

    Computes Gram matrix of vector features and flattens the upper triangle.
    """
    def __init__(self, n_vec_channels: int):
        super().__init__()
        self.n_pairs = (n_vec_channels * (n_vec_channels + 1)) // 2

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_v: Vector features [Batch, N, C, 3]
        Returns:
            Flattened pairwise inner products [Batch, N, n_pairs]
        """
        # Compute Gram Matrix per residue (Batch, N, C, C)
        gram = torch.einsum('bnci,bnji->bncj', h_v, h_v)

        # Extract Upper Triangle (unique pairs only)
        batch, n_res, c, _ = gram.shape
        indices = torch.triu_indices(c, c, device=gram.device)

        # Flatten and gather
        gram_flat = gram.view(batch * n_res, c, c)
        flat_pairs = gram_flat[:, indices[0], indices[1]]
        flat_pairs = flat_pairs.view(batch, n_res, self.n_pairs)

        return flat_pairs


class PocketGVPFusionV9m2(nn.Module):
    """
    Pocket-GVP-Fusion V9m2: V9 architecture with efficiency and loss improvements.

    V9m2 Additions:
    1. Flash Attention: faster attention computation (triton version for ROCm)
    2. Enhanced Label Smoothing: 0.15 (vs 0.1 in V9)
    3. Boundary Loss: additional loss for binding site edge residues

    Args:
        config: Configuration dictionary with model hyperparameters
    """

    # Default configuration
    DEFAULT_CONFIG = {
        # Dimensions
        'd_esm': 1280,
        'd_phys': 62,
        'd_spatial_scalar': 72,
        'd_phys_encoded': 128,
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 96,
        'n_gvp_vectors': 10,
        'n_gvp_vector_hidden': 20,

        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 1,
        'n_heads': 8,

        # Dropout
        'dropout': 0.1,
        'attn_dropout': 0.1,          # V9: separate attn weight dropout
        'classifier_dropout': 0.2,    # V9: classifier head dropout (up from 0.1)

        # V9: Stochastic Depth
        'stochastic_depth_prob': 0.1,     # V9: max drop_path probability

        # V9m2: Flash Attention
        'use_flash_attn': False,      # V9m2: enable Flash Attention

        # Modality dropout
        'esm_dropout_prob': 0.3,
        'physchem_dropout_prob': 0.3,

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

        # Merge with defaults
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

        # Module 4: Protein Context Encoder
        # V9: pass stochastic_depth_prob and attn_dropout
        # V9m2: pass use_flash_attn
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            pe_type='sinusoidal',
            stochastic_depth_prob=c.get('stochastic_depth_prob', 0.0),   # V9
            attn_dropout=c.get('attn_dropout', None),                     # V9
            use_flash_attn=c.get('use_flash_attn', False)                 # V9m2
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
        # V9: pass stochastic_depth_prob and attn_dropout with linear scaling
        n_cross_layers = c.get('n_cross_attention_layers', 1)
        stochastic_depth_prob = c.get('stochastic_depth_prob', 0.0)

        # V9: linearly scale drop_path_prob from 0 to stochastic_depth_prob across layers
        cross_drop_path_probs = [
            stochastic_depth_prob * i / max(n_cross_layers - 1, 1)
            for i in range(n_cross_layers)
        ]

        self.cross_attention_layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout'],
                drop_path_prob=cross_drop_path_probs[i],   # V9: linearly scaled
                attn_dropout=c.get('attn_dropout', None)    # V9
            )
            for i in range(n_cross_layers)
        ])

        # Module 7: Late Injection & Classifier
        # V9: classifier_dropout (default 0.2, up from 0.1)
        d_classifier_in = c['d_model'] + c['d_phys_encoded']
        classifier_dropout = c.get('classifier_dropout', 0.2)

        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),   # V9: configurable classifier dropout
            nn.Linear(256, c['n_classes'])
        )

    def forward(
        self,
        # Protein features
        esm_embeddings: torch.Tensor,       # [B, L, 1280]
        protein_physchem: torch.Tensor,     # [B, L, 62] - V8: Physical-chemical features
        protein_spatial_scalar: torch.Tensor,  # [B, L, 72] - V8: Spatial scalar features
        protein_spatial_vector: torch.Tensor,  # [B, L, 6, 3] - V8: Spatial vector features
        protein_gvp_vectors: torch.Tensor,  # [B, L, 4, 3] - GVP vectors (computed at runtime)
        protein_coords: torch.Tensor,       # [B, L, 3]
        protein_mask: torch.Tensor,         # [B, L]

        # Ligand features (batched graph)
        ligand_x: torch.Tensor,             # [N_atoms_total, 96]
        ligand_pos: torch.Tensor,           # [N_atoms_total, 3]
        ligand_edge_index: torch.Tensor,    # [2, E_total]
        ligand_edge_attr: torch.Tensor,     # [E_total, edge_dim]
        ligand_batch: torch.Tensor,         # [N_atoms_total]
        ligand_to_protein: torch.Tensor,    # [B, max_atoms, d_model]
        ligand_mask: torch.Tensor,          # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with V3 complete features integration.

        V8 Changes:
        - Uses 62D physical-chemical features (pure physicochemical properties)
        - Uses 72D spatial scalar features for GVP node_s
        - Concatenates 6 V3 spatial vectors with 4 GVP vectors for GVP node_v (total 10 vectors)

        Returns:
            Dict with logits, h_prot_global, h_fused, h_ligand_fused
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

        # ============================================================
        # Module 1: PhysChem MLP Encoder with Modality Dropout
        # ============================================================
        # V8: Encode 62D physical-chemical features
        protein_physchem_encoded = self.physchem_encoder(protein_physchem)  # [B, L, 128]

        # PhysChem modality dropout
        physchem_dropout_prob = c.get('physchem_dropout_prob', 0.0)
        if self.training and physchem_dropout_prob > 0:
            physchem_dropout_mask = torch.rand(B, 1, 1, device=device) > physchem_dropout_prob
            physchem_dropout_mask = physchem_dropout_mask.float()
            protein_physchem_encoded = protein_physchem_encoded * physchem_dropout_mask
        else:
            physchem_dropout_mask = torch.ones(B, 1, 1, device=device)

        # Cache for diagnostics
        self._cached_physchem_mask = physchem_dropout_mask

        # ============================================================
        # Module 2: GVP Encoder with V3 Features
        # ============================================================
        # V8: Concatenate 6 V3 spatial vectors with 4 GVP vectors
        protein_vectors_combined = torch.cat([
            protein_gvp_vectors,      # [B, L, 4, 3]
            protein_spatial_vector    # [B, L, 6, 3]
        ], dim=2)  # [B, L, 10, 3]

        h_s_gvp_list = []
        h_v_gvp_list = []

        for b in range(B):
            mask_b = protein_mask[b]
            n_valid = mask_b.sum()

            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_spatial_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype))
                continue

            # V8: Use spatial scalar features for GVP node_s
            scalar_b = protein_spatial_scalar[b, mask_b]  # [n_valid, 72]
            vectors_b = protein_vectors_combined[b, mask_b]  # [n_valid, 10, 3]
            coords_b = protein_coords[b, mask_b]

            edge_index, edge_dist, edge_dir = build_knn_graph(
                coords_b,
                k=c['gvp_k_neighbors'],
                radius=c['gvp_radius']
            )

            h_s, h_v = self.gvp_encoder(
                node_s=scalar_b,
                node_v=vectors_b,
                edge_index=edge_index,
                edge_s=edge_dist,
                edge_v=edge_dir
            )

            h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device, dtype=h_s.dtype)
            h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v.dtype)

            h_s_full[mask_b] = h_s
            h_v_full[mask_b] = h_v

            h_s_gvp_list.append(h_s_full)
            h_v_gvp_list.append(h_v_full)

        h_s_gvp = torch.stack(h_s_gvp_list, dim=0)
        h_v_gvp = torch.stack(h_v_gvp_list, dim=0)

        # Cache GVP features for diagnostics
        self._cached_gvp_scalar = h_s_gvp
        self._cached_gvp_vector = h_v_gvp

        # ============================================================
        # Module 3: Bridge & Early Injection
        # ============================================================
        v_inv = self.pairwise_bridge(h_v_gvp)  # [B, L, 210]
        h_geo = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)  # [B, L, 594]

        # Cache h_geo for diagnostics
        self._cached_h_geo = h_geo

        # Modality Dropout for ESM
        current_esm = esm_embeddings
        esm_dropout_mask = None
        if self.training:
            p_drop_esm = c.get('esm_dropout_prob', 0.3)
            esm_dropout_mask = torch.bernoulli(
                torch.full((B, 1, 1), 1 - p_drop_esm, device=device)
            )
            current_esm = current_esm * esm_dropout_mask
        else:
            esm_dropout_mask = torch.ones((B, 1, 1), device=device)

        # Cache ESM features for diagnostics
        self._cached_esm_mask = esm_dropout_mask
        self._cached_current_esm = current_esm

        # Separate projections
        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)

        # Channel-wise gating fusion
        h_concat = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate = self.modality_gate(h_concat)

        # Apply channel-wise gating
        h_gvp_gated = gate * h_gvp_proj
        h_esm_gated = (1 - gate) * h_esm_proj
        h_seq = torch.cat([h_gvp_gated, h_esm_gated], dim=-1)

        # Cache projected features and gate for diagnostics
        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj
        self._cached_gate = gate

        # ============================================================
        # Module 4: Protein Context Encoder
        # ============================================================
        if self.config['use_gradient_checkpointing'] and self.training:
            h_prot_global = checkpoint(
                self.protein_context_encoder,
                h_seq,
                protein_mask,
                use_reentrant=False
            )
        else:
            h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)

        # ============================================================
        # Module 5: 3D Ligand Encoder
        # ============================================================
        h_ligand_nodes, ligand_pos_out = self.ligand_encoder(
            x=ligand_x,
            pos=ligand_pos,
            edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr,
            batch=ligand_batch
        )

        # Scatter to [B, Max_M, D] and [B, Max_M, 3]
        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        ligand_pos_batched = torch.zeros(B_dim, Max_M_dim, 3, device=device, dtype=ligand_pos.dtype)

        for b in range(B_dim):
            mask_b = ligand_batch == b
            if mask_b.any():
                nodes_b = h_ligand_nodes[mask_b]
                pos_b = ligand_pos_out[mask_b]
                n_atoms = min(nodes_b.shape[0], Max_M_dim)
                h_ligand[b, :n_atoms] = nodes_b[:n_atoms]
                ligand_pos_batched[b, :n_atoms] = pos_b[:n_atoms]

        # Add global ligand token
        global_token = self.ligand_global_token.expand(B_dim, -1, -1)  # [B, 1, D]
        h_ligand_with_global = torch.cat([global_token, h_ligand], dim=1)  # [B, 1+Max_M, D]

        # Update ligand mask to include global token
        global_token_mask = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)  # [B, 1]
        ligand_mask_with_global = torch.cat([global_token_mask, ligand_mask], dim=1)  # [B, 1+Max_M]

        # ============================================================
        # Module 6: Sequential Bidirectional Cross-Attention
        # ============================================================
        h_prot_fused = h_prot_global
        h_lig_fused = h_ligand_with_global

        for cross_attn_layer in self.cross_attention_layers:
            if self.config['use_gradient_checkpointing'] and self.training:
                # Step 1: Update protein with ligand info
                h_prot_updated = checkpoint(
                    cross_attn_layer.prot_to_lig,
                    h_prot_fused,
                    h_lig_fused,
                    protein_mask,
                    ligand_mask_with_global,
                    use_reentrant=False
                )

                # Step 2: Update ligand with UPDATED protein info
                h_lig_updated = checkpoint(
                    cross_attn_layer.lig_to_prot,
                    h_lig_fused,
                    h_prot_updated,
                    ligand_mask_with_global,
                    protein_mask,
                    use_reentrant=False
                )
            else:
                # Step 1: Update protein with ligand info
                h_prot_updated = cross_attn_layer.prot_to_lig(
                    q=h_prot_fused,
                    kv=h_lig_fused,
                    q_mask=protein_mask,
                    kv_mask=ligand_mask_with_global
                )

                # Step 2: Update ligand with UPDATED protein info
                h_lig_updated = cross_attn_layer.lig_to_prot(
                    q=h_lig_fused,
                    kv=h_prot_updated,
                    q_mask=ligand_mask_with_global,
                    kv_mask=protein_mask
                )

            h_prot_fused = h_prot_updated
            h_lig_fused = h_lig_updated

        # Extract atom-level features (remove global token)
        h_lig_atoms = h_lig_fused[:, 1:, :]  # [B, Max_M, D]

        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        x_final = torch.cat([h_prot_fused, protein_physchem_encoded], dim=-1)
        logits = self.classifier(x_final)

        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_prot_fused,
            'h_ligand_fused': h_lig_fused
        }
