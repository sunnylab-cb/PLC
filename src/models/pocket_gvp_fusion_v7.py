"""
Pocket-GVP-Fusion V7: Enhanced Features (Ligand V4 + Complete Features V2).

Key improvements:
- V7: Ligand Features V4 (96D atom features, up from 80D)
- V7: Complete Features V2 (94D protein features, up from 82D)
- V6m9: Channel-wise gating (no information bottleneck)
- V6m9: q_mask application in cross-attention output (reduce padding noise)
- V6m9: Global ligand token for cross-attention (global chemical semantics)
- V6m9: Optional gradient checkpointing (memory efficiency)
- V6m8: Sequential bidirectional cross-attention (stronger information flow)
- V6m7: Label Smoothing (0.1) for better generalization
- V6m6: 3D Ligand GVP Encoder (uses real 3D coordinates)
- V6m5: GELU activation instead of ReLU to reduce information loss

Architecture:
    Module 1: Input Featurizer (handled in forward)
    Module 1.5: PhysChem MLP Encoder with LayerNorm + GELU
    Module 2: Protein Geometric Encoder (GVP)
    Module 3: Bridge & Channel-wise Gating Fusion
    Module 4: Protein Context Encoder (Gated Transformer)
    Module 5: 3D Ligand Encoder (GVP) + Global Token
    Module 6: Sequential Bidirectional Cross-Attention with q_mask
    Module 7: Late Injection & Classifier
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


class PocketGVPFusionV7(nn.Module):
    """
    Pocket-GVP-Fusion V7 with enhanced features (Ligand V4 + Complete Features V2).

    Key improvements over V6m9:
    1. Ligand Features V4: 96D atom features (up from 80D)
       - More comprehensive atom-level descriptors
    2. Complete Features V2: 94D protein features (up from 82D)
       - Enhanced physicochemical, surface, geometry, pocketness, flexibility, environmental features

    Args:
        config: Configuration dictionary with model hyperparameters
    """

    # Default configuration
    DEFAULT_CONFIG = {
        # Dimensions
        'd_esm': 1280,
        'd_phys': 94,  # V7: Complete Features V2 (94D, up from 82D)
        'd_phys_encoded': 128,  # V7: Increased to handle richer features
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 96,  # V7: Ligand Features V4 (96D, up from 80D)
        'n_gvp_vectors': 4,
        'n_gvp_vector_hidden': 16,

        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 1,  # Number of bidirectional blocks
        'n_heads': 8,

        # Settings
        'dropout': 0.1,
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'esm_dropout_prob': 0.3,  # ESM modality dropout
        'physchem_dropout_prob': 0.3,  # V6m6: PhysChem modality dropout
        'use_gradient_checkpointing': False,  # V7: Optional gradient checkpointing

        # Output
        'n_classes': 3,

        # Edge features
        'edge_dim': 6,
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Merge with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

        # Module 1: PhysChem MLP Encoder with LayerNorm and GELU
        # V7: Enhanced to handle Complete Features V2 (94D)
        # Maps raw physicochemical features to higher-dimensional latent space
        # to capture non-linear interactions
        # LayerNorm added to fix feature scale mismatch with GVP features
        # GELU replaces ReLU to avoid information loss from negative values
        self.physchem_encoder = nn.Sequential(
            nn.Linear(c['d_phys'], 128),  # V7: Increased capacity for richer features
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c['d_phys_encoded']),
            nn.LayerNorm(c['d_phys_encoded']),
            nn.GELU()
        )

        # Module 2: Protein Geometric Encoder (GVP)
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_phys'], c['n_gvp_vectors']),
            node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
            edge_in_dim=(1, 1),
            n_layers=c['n_gvp_layers'],
            drop_rate=c['dropout']
        )

        # Module 3: Bridge & Learnable Gating Fusion (V6m8)
        self.pairwise_bridge = PairwiseBridge(
            n_vec_channels=c['n_gvp_vector_hidden']
        )

        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        # Updated: Use d_phys_encoded instead of d_phys
        d_geo = c['d_gvp_internal'] + n_pairwise + c['d_phys_encoded']
        d_half = c['d_model'] // 2

        # Separate projection paths
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

        # V7: Channel-wise gating for ESM/GVP fusion
        # Computes per-residue, per-channel adaptive weights
        # Output: concat(g*h_gvp, (1-g)*h_esm) - preserves full expressiveness
        self.modality_gate = nn.Sequential(
            nn.Linear(d_half * 2, d_half),  # Input: concatenated features
            nn.ReLU(),
            nn.Linear(d_half, d_half),  # Output: d_half logits (per-channel gate)
            nn.Sigmoid()  # Normalize to [0, 1] per-channel weights
        )

        # Module 4: Protein Context Encoder
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            pe_type='sinusoidal'
        )

        # Module 5: 3D Ligand Encoder (V6m6: GVP-based)
        self.ligand_encoder = Ligand3DEncoder(
            node_in_dim=c['d_ligand_atom'],
            node_h_dim=c['d_gvp_internal'] // 2,  # 128
            n_vec_channels=4,
            n_vec_hidden=16,
            edge_in_dim=c['edge_dim'],
            out_dim=c['d_model'],
            n_layers=c['n_ligand_layers'],
            dropout=c['dropout'],
            radius=5.0,  # 5 Angstrom radius for ligand atoms
            max_neighbors=32
        )

        # V7: Global ligand token (learnable)
        # Captures global chemical semantics (e.g., molecular properties, drug-likeness)
        self.ligand_global_token = nn.Parameter(torch.randn(1, 1, c['d_model']))

        # Module 6: Sequential Bidirectional Cross-Attention (V6m8)
        # Sequential update: first protein, then ligand (with updated protein)
        n_cross_layers = c.get('n_cross_attention_layers', 1)
        self.cross_attention_layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout']
            )
            for _ in range(n_cross_layers)
        ])

        # Module 7: Late Injection & Classifier
        # Updated: Use d_phys_encoded instead of d_phys
        d_classifier_in = c['d_model'] + c['d_phys_encoded']

        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(c['dropout']),
            nn.Linear(256, c['n_classes'])
        )

    def forward(
        self,
        # Protein features
        esm_embeddings: torch.Tensor,       # [B, L, 1280]
        protein_scalar: torch.Tensor,       # [B, L, 35]
        protein_vectors: torch.Tensor,      # [B, L, 4, 3]
        protein_coords: torch.Tensor,       # [B, L, 3]
        protein_mask: torch.Tensor,         # [B, L]

        # Ligand features (batched graph)
        ligand_x: torch.Tensor,             # [N_atoms_total, 64]
        ligand_pos: torch.Tensor,           # [N_atoms_total, 3] - V6m6: 3D coordinates
        ligand_edge_index: torch.Tensor,    # [2, E_total]
        ligand_edge_attr: torch.Tensor,     # [E_total, edge_dim]
        ligand_batch: torch.Tensor,         # [N_atoms_total]
        ligand_to_protein: torch.Tensor,    # [B, max_atoms, d_model]
        ligand_mask: torch.Tensor,          # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with 3D ligand encoding and distance-aware attention.

        V6m6 Changes:
        - Uses 3D ligand coordinates for GVP encoding
        - Computes protein-ligand distance matrices for attention bias

        Returns:
            Dict with logits, h_prot_global, h_fused, h_ligand_fused
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

        # ============================================================
        # Module 1: PhysChem MLP Encoder with Modality Dropout (V6m6)
        # ============================================================
        # Encode raw physicochemical features to capture non-linear interactions
        protein_scalar_encoded = self.physchem_encoder(protein_scalar)  # [B, L, 64]

        # V6m6: PhysChem modality dropout (similar to ESM dropout)
        physchem_dropout_prob = c.get('physchem_dropout_prob', 0.0)
        if self.training and physchem_dropout_prob > 0:
            # Random dropout mask for each sample in batch
            physchem_dropout_mask = torch.rand(B, 1, 1, device=device) > physchem_dropout_prob
            physchem_dropout_mask = physchem_dropout_mask.float()
            protein_scalar_encoded = protein_scalar_encoded * physchem_dropout_mask
        else:
            physchem_dropout_mask = torch.ones(B, 1, 1, device=device)

        # Cache for diagnostics
        self._cached_physchem_mask = physchem_dropout_mask

        # ============================================================
        # Module 2: GVP Encoder
        # ============================================================
        h_s_gvp_list = []
        h_v_gvp_list = []

        for b in range(B):
            mask_b = protein_mask[b]
            n_valid = mask_b.sum()

            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors.dtype))
                continue

            scalar_b = protein_scalar[b, mask_b]
            vectors_b = protein_vectors[b, mask_b]
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
        v_inv = self.pairwise_bridge(h_v_gvp)
        # Use encoded physicochemical features instead of raw
        h_geo = torch.cat([h_s_gvp, v_inv, protein_scalar_encoded], dim=-1)

        # Cache h_geo for diagnostics
        self._cached_h_geo = h_geo

        # Modality Dropout
        current_esm = esm_embeddings
        esm_dropout_mask = None
        if self.training:
            p_drop_esm = c.get('esm_dropout_prob', 0.3)
            esm_dropout_mask = torch.bernoulli(
                torch.full((B, 1, 1), 1 - p_drop_esm, device=device)
            )
            current_esm = current_esm * esm_dropout_mask
        else:
            # During eval, no dropout - mask is all ones
            esm_dropout_mask = torch.ones((B, 1, 1), device=device)

        # Cache ESM features for diagnostics
        self._cached_esm_mask = esm_dropout_mask
        self._cached_current_esm = current_esm

        # Separate projections
        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)

        # V7: Channel-wise gating fusion
        # Compute per-residue, per-channel adaptive weights for GVP and ESM modalities
        h_concat = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)  # [B, L, d_model]
        gate = self.modality_gate(h_concat)  # [B, L, d_half], values in [0, 1]

        # Apply channel-wise gating: concat(g*h_gvp, (1-g)*h_esm)
        # This preserves full d_model expressiveness (no information bottleneck)
        h_gvp_gated = gate * h_gvp_proj  # [B, L, d_half]
        h_esm_gated = (1 - gate) * h_esm_proj  # [B, L, d_half]
        h_seq = torch.cat([h_gvp_gated, h_esm_gated], dim=-1)  # [B, L, d_model]

        # Cache projected features and gate for diagnostics
        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj
        self._cached_gate = gate  # V7: Cache per-channel gate

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
        # Module 5: 3D Ligand Encoder (V6m6: GVP-based)
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

        # V7: Add global ligand token
        # Prepend global token to ligand representation: [B, 1+Max_M, D]
        global_token = self.ligand_global_token.expand(B_dim, -1, -1)  # [B, 1, D]
        h_ligand_with_global = torch.cat([global_token, h_ligand], dim=1)  # [B, 1+Max_M, D]

        # Update ligand mask to include global token (always valid)
        global_token_mask = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)  # [B, 1]
        ligand_mask_with_global = torch.cat([global_token_mask, ligand_mask], dim=1)  # [B, 1+Max_M]

        # ============================================================
        # Module 6: Sequential Bidirectional Cross-Attention (V6m8)
        # V7: Use ligand representation with global token
        # ============================================================
        h_prot_fused = h_prot_global
        h_lig_fused = h_ligand_with_global  # V7: Include global token

        for cross_attn_layer in self.cross_attention_layers:
            # V6m8: Sequential update for stronger information flow
            # V7: Optional gradient checkpointing for memory efficiency
            if self.config['use_gradient_checkpointing'] and self.training:
                # Step 1: Update protein with ligand info (including global token)
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
                # Step 1: Update protein with ligand info (including global token)
                h_prot_updated = cross_attn_layer.prot_to_lig(
                    q=h_prot_fused,
                    kv=h_lig_fused,
                    q_mask=protein_mask,
                    kv_mask=ligand_mask_with_global  # V7: Use mask with global token
                )

                # Step 2: Update ligand with UPDATED protein info
                h_lig_updated = cross_attn_layer.lig_to_prot(
                    q=h_lig_fused,
                    kv=h_prot_updated,  # V6m8: Use updated protein features
                    q_mask=ligand_mask_with_global,  # V7: Use mask with global token
                    kv_mask=protein_mask
                )

            h_prot_fused = h_prot_updated
            h_lig_fused = h_lig_updated

        # V7: Extract atom-level features (remove global token for final prediction)
        h_lig_atoms = h_lig_fused[:, 1:, :]  # [B, Max_M, D] - remove first token

        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        # Use encoded physicochemical features instead of raw
        x_final = torch.cat([h_prot_fused, protein_scalar_encoded], dim=-1)
        logits = self.classifier(x_final)

        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_prot_fused,
            'h_ligand_fused': h_lig_fused  # NEW: Also return updated ligand features
        }

    def predict(self, **kwargs) -> torch.Tensor:
        """
        Convenience method for inference.

        Returns:
            predictions: [B, L] - predicted class per residue
        """
        outputs = self.forward(**kwargs)
        return outputs['logits'].argmax(dim=-1)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss with class weights and padding masking.

        Args:
            logits: [B, L, n_classes]
            labels: [B, L]
            mask: [B, L] (True = valid, False = padding)
            class_weights: [n_classes]
            gamma: Focal loss gamma parameter

        Returns:
            Scalar loss value
        """
        B, L, C = logits.shape
        device = logits.device

        # Flatten
        logits_flat = logits.view(-1, C)
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)

        # Compute softmax probabilities
        probs = F.softmax(logits_flat, dim=-1)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')

        # Focal weight
        p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** gamma

        # Apply class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
            sample_weights = class_weights[labels_flat]
            focal_loss = focal_weight * sample_weights * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply padding mask
        focal_loss = focal_loss * mask_flat.float()

        # Mean over valid tokens
        n_valid = mask_flat.sum()
        if n_valid > 0:
            loss = focal_loss.sum() / n_valid
        else:
            loss = focal_loss.sum() * 0

        return loss

    def load_gvp_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, int]:
        """
        Load GVP-related weights from an existing checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, raise error on missing/unexpected keys

        Returns:
            Dict with 'loaded', 'skipped', 'missing' counts
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        source_state_dict = checkpoint.get('model_state_dict', checkpoint)

        gvp_prefixes = ('gvp_encoder.', 'pairwise_bridge.', 'gvp_proj.')
        gvp_keys = {k: v for k, v in source_state_dict.items()
                    if any(k.startswith(prefix) for prefix in gvp_prefixes)}

        current_state = self.state_dict()

        loaded_count = 0
        skipped_count = 0
        missing_keys = []

        for key, value in gvp_keys.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    current_state[key] = value
                    loaded_count += 1
                else:
                    print(f"  [GVP Load] Shape mismatch for {key}")
                    skipped_count += 1
            else:
                missing_keys.append(key)
                skipped_count += 1

        self.load_state_dict(current_state)

        print(f"[GVP Checkpoint] Loaded {loaded_count} parameters, "
              f"skipped {skipped_count}")

        if missing_keys and strict:
            raise ValueError(f"Missing keys in model: {missing_keys}")

        return {
            'loaded': loaded_count,
            'skipped': skipped_count,
            'missing': len(missing_keys)
        }

    def freeze_gvp(self, freeze: bool = True) -> int:
        """
        Freeze or unfreeze GVP-related parameters.

        Args:
            freeze: If True, freeze GVP params

        Returns:
            Number of parameters affected
        """
        gvp_modules = [self.gvp_encoder, self.pairwise_bridge, self.gvp_proj]

        param_count = 0
        for module in gvp_modules:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = not freeze
                    param_count += param.numel()

        status = "frozen" if freeze else "unfrozen"
        print(f"[GVP] {status.capitalize()} {param_count:,} parameters")

        return param_count

