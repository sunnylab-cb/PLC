"""
Pocket-GVP-Fusion v6: Bidirectional Cross-Attention Architecture.

Key improvement over v5:
- V5: One-way cross-attention (Protein queries Ligand)
- V6: Bidirectional co-attention (Protein ↔ Ligand)

This allows both protein and ligand features to be mutually enhanced,
better modeling the "induced fit" phenomenon in protein-ligand binding.

Architecture:
    Module 1: Input Featurizer (handled in forward)
    Module 2: Protein Geometric Encoder (GVP)
    Module 3: Bridge & Early Injection
    Module 4: Protein Context Encoder (Gated Transformer)
    Module 5: Ligand Encoder (GINEConv)
    Module 6: Bidirectional Cross-Attention Fusion  ← NEW in V6
    Module 7: Late Injection & Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .gvp_layers import GVPEncoder, build_knn_graph
from .transformer_layers import ProteinContextEncoder, BidirectionalCrossAttentionBlock
from .ligand_encoder import LigandEncoder


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


class PocketGVPFusionV6m2(nn.Module):
    """
    Pocket-GVP-Fusion V6m2 with bidirectional cross-attention and PhysChem encoder.

    Key improvements:
    1. Bidirectional co-attention allows both protein and ligand features
       to be mutually enhanced, modeling induced fit.
    2. PhysChem MLP Encoder processes physicochemical features (35D -> 64D)
       to capture non-linear interactions before fusion.

    Args:
        config: Configuration dictionary with model hyperparameters
    """

    # Default configuration (same as V6, with d_phys_encoded added)
    DEFAULT_CONFIG = {
        # Dimensions
        'd_esm': 1280,
        'd_phys': 35,
        'd_phys_encoded': 64,  # NEW: Encoded physicochemical feature dimension
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 64,
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
        'esm_dropout_prob': 0.3,

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

        # Module 1: PhysChem MLP Encoder (NEW in V6m2)
        # Maps raw physicochemical features to higher-dimensional latent space
        # to capture non-linear interactions (e.g., "large + hydrophobic" vs "small + hydrophobic")
        self.physchem_encoder = nn.Sequential(
            nn.Linear(c['d_phys'], 64),
            nn.ReLU(),
            nn.Linear(64, c['d_phys_encoded']),
            nn.ReLU()
        )

        # Module 2: Protein Geometric Encoder (GVP)
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_phys'], c['n_gvp_vectors']),
            node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
            edge_in_dim=(1, 1),
            n_layers=c['n_gvp_layers'],
            drop_rate=c['dropout']
        )

        # Module 3: Bridge & Early Injection
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

        # Module 4: Protein Context Encoder
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            pe_type='sinusoidal'
        )

        # Module 5: Ligand Encoder
        self.ligand_encoder = LigandEncoder(
            node_in_dim=c['d_ligand_atom'],
            edge_in_dim=c['edge_dim'],
            hidden_dim=c['d_gvp_internal'],
            out_dim=c['d_model'],
            n_layers=c['n_ligand_layers'],
            dropout=c['dropout']
        )

        # Module 6: Bidirectional Cross-Attention Fusion (NEW in V6)
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
        ligand_edge_index: torch.Tensor,    # [2, E_total]
        ligand_edge_attr: torch.Tensor,     # [E_total, edge_dim]
        ligand_batch: torch.Tensor,         # [N_atoms_total]
        ligand_to_protein: torch.Tensor,    # [B, max_atoms, d_model]
        ligand_mask: torch.Tensor,          # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bidirectional cross-attention and PhysChem encoding.

        Returns:
            Dict with logits, h_prot_global, h_fused, h_ligand_fused
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config

        # ============================================================
        # Module 1: PhysChem MLP Encoder (NEW in V6m2)
        # ============================================================
        # Encode raw physicochemical features to capture non-linear interactions
        protein_scalar_encoded = self.physchem_encoder(protein_scalar)  # [B, L, 64]

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
        h_seq = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)

        # Cache projected features for diagnostics
        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj

        # ============================================================
        # Module 4: Protein Context Encoder
        # ============================================================
        h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)

        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        h_ligand_nodes = self.ligand_encoder(
            x=ligand_x,
            edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr,
            batch=ligand_batch
        )

        # Scatter to [B, Max_M, D]
        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)

        for b in range(B_dim):
            mask_b = ligand_batch == b
            if mask_b.any():
                nodes_b = h_ligand_nodes[mask_b]
                n_atoms = min(nodes_b.shape[0], Max_M_dim)
                h_ligand[b, :n_atoms] = nodes_b[:n_atoms]

        # ============================================================
        # Module 6: Bidirectional Cross-Attention Fusion (NEW in V6)
        # ============================================================
        h_prot_fused = h_prot_global
        h_lig_fused = h_ligand

        for cross_attn_layer in self.cross_attention_layers:
            h_prot_fused, h_lig_fused = cross_attn_layer(
                h_prot=h_prot_fused,
                h_lig=h_lig_fused,
                prot_mask=protein_mask,
                lig_mask=ligand_mask
            )

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

