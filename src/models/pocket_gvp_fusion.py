"""
Pocket-GVP-Fusion v2: Main model architecture.

A hybrid architecture combining:
- GVP encoder for local 3D geometry
- Gated Transformer for global protein context
- GINEConv for ligand topology
- Cross-attention for protein-ligand fusion
- Multi-class classification head (Contact/Near/Far)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .gvp_layers import GVPEncoder, build_knn_graph
from .transformer_layers import ProteinContextEncoder, CrossAttentionBlock
from .ligand_encoder import LigandEncoder


class PairwiseBridge(nn.Module):
    """
    Pairwise dot product bridge for translation-invariant vector features.
    
    Computes Gram matrix of vector features and flattens the upper triangle.
    No projection layer - raw pairwise inner products are returned to be
    concatenated with scalar features before a unified projection.
    """
    def __init__(self, n_vec_channels: int):
        super().__init__()
        # Number of unique pairs in symmetric matrix: C * (C+1) / 2
        self.n_pairs = (n_vec_channels * (n_vec_channels + 1)) // 2

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h_v: Vector features [Batch, N, C, 3]
            
        Returns:
            Flattened pairwise inner products [Batch, N, n_pairs]
        """
        # 1. Compute Gram Matrix per residue (Batch, N, C, C)
        # dot product over the spatial dimension (last dim)
        gram = torch.einsum('bnci,bnji->bncj', h_v, h_v)
        
        # 2. Extract Upper Triangle (unique pairs only)
        batch, n_res, c, _ = gram.shape
        
        # Create indices for upper triangle
        indices = torch.triu_indices(c, c, device=gram.device)
        
        # Flatten batch and residue dims for efficient gathering
        gram_flat = gram.view(batch * n_res, c, c)
        
        # Gather: [Batch*N, N_Pairs]
        flat_pairs = gram_flat[:, indices[0], indices[1]]
        
        # Reshape back: [Batch, N, N_Pairs]
        flat_pairs = flat_pairs.view(batch, n_res, self.n_pairs)
        
        # Return raw pairwise inner products (no projection)
        return flat_pairs


class PocketGVPFusion(nn.Module):
    """
    Pocket-GVP-Fusion v2 model for protein-ligand binding site prediction.
    
    Architecture:
        Module 1: Input Featurizer (handled in forward)
        Module 2: Protein Geometric Encoder (GVP)
        Module 3: Bridge & Early Injection
        Module 4: Protein Context Encoder (Gated Transformer)
        Module 5: Ligand Encoder (GINEConv)
        Module 6: Cross-Attention Fusion
        Module 7: Late Injection & Classifier
    
    Args:
        config: Configuration dictionary with model hyperparameters
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Dimensions
        'd_esm': 1280,           # ESM-2 embedding size
        'd_phys': 35,            # Physicochemical features
        'd_gvp_internal': 256,   # GVP hidden dimension
        'd_model': 512,          # Transformer dimension (d_model/2 per projection path)
        'd_ligand_atom': 64,     # Ligand atom features
        'n_gvp_vectors': 4,      # Number of GVP vector features
        'n_gvp_vector_hidden': 16,  # GVP vector hidden dimension
        # Note: d_pairwise_bridge removed - pairwise bridge returns raw inner products (136 dims)
        
        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 1,  # Multi-layer cross-attention (default 1 for backward compat)
        'n_heads': 8,
        
        # Settings
        'dropout': 0.1,
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'esm_dropout_prob': 0.3,  # Modality dropout: probability of dropping ESM during training
        
        # Output
        'n_classes': 3,
        
        # Edge features
        'edge_dim': 6,  # Bond type (4) + conjugated + in_ring
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Merge with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config
        
        # ============================================================
        # Module 2: Protein Geometric Encoder (GVP)
        # ============================================================
        # Input: scalar (d_phys=35) + vector (4×3 or depends on feature extraction)
        # Note: GVP input features exclude coordinates, coords only used for graph
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_phys'], c['n_gvp_vectors']),
            node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
            edge_in_dim=(1, 1),  # distance + direction
            n_layers=c['n_gvp_layers'],
            drop_rate=c['dropout']
        )
        
        # ============================================================
        # Module 3: Bridge & Early Injection (Separate Projections)
        # ============================================================
        # Step A: Pairwise Bridge for vectors (no projection - raw inner products)
        self.pairwise_bridge = PairwiseBridge(
            n_vec_channels=c['n_gvp_vector_hidden']
        )
        
        # Number of pairwise inner products: C * (C+1) / 2 = 16 * 17 / 2 = 136
        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        
        # Construct d_geo = [h_s_gvp (256) + pairwise_out (136) + P_raw (35)] = 427
        # Note: Physchem is included in GVP projection since it's geometric/structural
        d_geo = c['d_gvp_internal'] + n_pairwise + c['d_phys']
        
        # Step B: SEPARATE projection paths for GVP and ESM
        # This prevents ESM from dominating early training when GVP weights are random
        d_half = c['d_model'] // 2  # 256 each
        
        # GVP path: (GVP scalar + pairwise + physchem) -> d_half
        self.gvp_proj = nn.Sequential(
            nn.Linear(d_geo, d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout'])
        )
        
        # ESM path: ESM embeddings -> d_half
        self.esm_proj = nn.Sequential(
            nn.Linear(c['d_esm'], d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout'])
        )
        
        # ============================================================
        # Module 4: Protein Context Encoder (Gated Transformer)
        # ============================================================
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout'],
            pe_type='sinusoidal'
        )
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        self.ligand_encoder = LigandEncoder(
            node_in_dim=c['d_ligand_atom'],
            edge_in_dim=c['edge_dim'],
            hidden_dim=c['d_gvp_internal'],
            out_dim=c['d_model'],
            n_layers=c['n_ligand_layers'],
            dropout=c['dropout']
        )
        
        # ============================================================
        # Module 6: Cross-Attention Fusion (Multi-Layer)
        # ============================================================
        n_cross_layers = c.get('n_cross_attention_layers', 1)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout']
            )
            for _ in range(n_cross_layers)
        ])
        
        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        # Input: Concat(H_fused, P_raw)
        # Size: d_model (512) + d_phys (35) = 547
        # NO ResNet skip from Protein Encoder H_prot_global
        
        d_classifier_in = c['d_model'] + c['d_phys']
        
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(c['dropout']),
            nn.Linear(256, c['n_classes'])
        )
    
    @staticmethod
    def convert_legacy_state_dict(state_dict: Dict) -> Dict:
        """
        Convert old checkpoint keys to new multi-layer format.
        
        Old format: cross_attention.norm_q.weight, etc.
        New format: cross_attention_layers.0.norm_q.weight, etc.
        
        This ensures backward compatibility with old checkpoints.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('cross_attention.'):
                # Convert to cross_attention_layers.0.*
                new_key = key.replace('cross_attention.', 'cross_attention_layers.0.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
        
    def forward(
        self,
        # Protein features
        esm_embeddings: torch.Tensor,       # [B, L, 1280]
        protein_scalar: torch.Tensor,       # [B, L, 35]
        protein_vectors: torch.Tensor,      # [B, L, 4, 3]
        protein_coords: torch.Tensor,       # [B, L, 3]
        protein_mask: torch.Tensor,         # [B, L] (True = valid)
        
        # Ligand features (batched graph)
        ligand_x: torch.Tensor,             # [N_atoms_total, 64]
        ligand_edge_index: torch.Tensor,    # [2, E_total]
        ligand_edge_attr: torch.Tensor,     # [E_total, edge_dim]
        ligand_batch: torch.Tensor,         # [N_atoms_total]
        ligand_to_protein: torch.Tensor,    # [B, max_atoms, d_model] padded ligand features
        ligand_mask: torch.Tensor,          # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            esm_embeddings: ESM-2 embeddings
            protein_scalar: Physicochemical features
            protein_vectors: GVP vector features
            protein_coords: Cα coordinates
            protein_mask: Padding mask
            ligand_x: Ligand atom features
            ligand_edge_index: Ligand edge indices
            ligand_edge_attr: Ligand edge features
            ligand_batch: Ligand batch assignment
            ligand_to_protein: Padded ligand features for cross-attn
            ligand_mask: Ligand padding mask
            
        Returns:
            Dict with logits, h_prot_global, h_fused
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config
        
        # ============================================================
        # Module 2: GVP Encoder [Updated Flow]
        # ============================================================
        # Note: GVP input = phys + vectors. Coordinates ONLY used for graph building.
        
        h_s_gvp_list = []
        h_v_gvp_list = []
        
        for b in range(B):
            # Get valid residues for this protein
            mask_b = protein_mask[b]  # [L]
            n_valid = mask_b.sum()  # Keep as tensor to avoid graph break

            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_scalar.dtype))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors.dtype))
                continue
            
            # Extract valid residues
            scalar_b = protein_scalar[b, mask_b]  # [n_valid, 35]
            vectors_b = protein_vectors[b, mask_b]  # [n_valid, 4, 3]
            coords_b = protein_coords[b, mask_b]  # [n_valid, 3]
            
            # Build k-NN graph
            edge_index, edge_dist, edge_dir = build_knn_graph(
                coords_b,
                k=c['gvp_k_neighbors'],
                radius=c['gvp_radius']
            )
            
            # Apply GVP encoder
            h_s, h_v = self.gvp_encoder(
                node_s=scalar_b,
                node_v=vectors_b,
                edge_index=edge_index,
                edge_s=edge_dist,
                edge_v=edge_dir
            )
            
            # Pad back to full length
            h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device, dtype=h_s.dtype)
            h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v.dtype)

            h_s_full[mask_b] = h_s
            h_v_full[mask_b] = h_v
            
            h_s_gvp_list.append(h_s_full)
            h_v_gvp_list.append(h_v_full)
        
        h_s_gvp = torch.stack(h_s_gvp_list, dim=0)  # [B, L, 256]
        h_v_gvp = torch.stack(h_v_gvp_list, dim=0)  # [B, L, 16, 3]
        
        # Cache for diagnostics
        self._cached_gvp_scalar = h_s_gvp
        self._cached_gvp_vector = h_v_gvp
        
        # ============================================================
        # Module 3: Bridge & Early Injection [Separate Projections]
        # ============================================================
        
        # Step A: Pairwise Bridge (no projection - raw inner products)
        # [B, L, 16, 3] -> [B, L, 136]
        v_inv = self.pairwise_bridge(h_v_gvp)
        
        # h_geo = [h_s_gvp, v_inv, P_raw] -> [B, L, 256 + 136 + 35 = 427]
        h_geo = torch.cat([h_s_gvp, v_inv, protein_scalar], dim=-1)
        
        # Cache final geometric embedding for diagnostics
        self._cached_h_geo = h_geo
        
        # --- Modality Dropout: Applied BEFORE projection ---
        # This forces the model to leverage GVP features during training
        current_esm = esm_embeddings
        esm_dropout_mask = None
        
        if self.training:
            # Dropout probability from config (default: 0.5)
            p_drop_esm = c.get('esm_dropout_prob', 0.5)
            
            # Create Bernoulli mask: [B, 1, 1]
            # Same decision for all residues in a protein, independent per protein
            # 1 = Keep, 0 = Drop
            esm_dropout_mask = torch.bernoulli(
                torch.full((B, 1, 1), 1 - p_drop_esm, device=device)
            )
            
            # Apply mask (no scaling - LayerNorm will handle variance)
            current_esm = current_esm * esm_dropout_mask
        
        # Cache for diagnostics
        self._cached_esm_mask = esm_dropout_mask
        self._cached_current_esm = current_esm
        
        # Step B: SEPARATE projection paths
        # GVP path: [B, L, 427] -> [B, L, 256]
        h_gvp_proj = self.gvp_proj(h_geo)
        
        # ESM path: [B, L, 1280] -> [B, L, 256]
        h_esm_proj = self.esm_proj(current_esm)
        
        # Cache projected features for diagnostics
        self._cached_h_gvp_proj = h_gvp_proj
        self._cached_h_esm_proj = h_esm_proj
        
        # Step C: Concatenate projections for self-attention
        # [B, L, 256] + [B, L, 256] -> [B, L, 512]
        h_seq = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        
        # ============================================================
        # Module 4: Protein Context Encoder
        # ============================================================
        h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)  # [B, L, 512]
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        h_ligand_nodes = self.ligand_encoder(
            x=ligand_x,
            edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr,
            batch=ligand_batch
        )
        # Use padded ligand features from dataloader (assumes correct collation/batching)
        # Ideally, we should scatter h_ligand_nodes back to [B, M, D]
        # For now, leveraging the passed 'ligand_to_protein' if it contains encoded features
        # BUT 'ligand_to_protein' arg currently holds raw features from collate. 
        # We need to reshape h_ligand_nodes. 
        # Since standard scattering is complex with variable sizes, and we are using GINEConv which is global,
        # let's assume ligand_to_protein is a placeholder or we need to implementation proper scattering.
        #
        # FIX: The simplified current implementation in `collate.py` passes raw features in `ligand_to_protein`.
        # We need to scatter the GNN output `h_ligand_nodes` into a [B, Max_M, D] tensor for CrossAttn.
        
        # Create zero tensor [B, Max_M, D]
        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        # ligand_to_protein passed in is currently raw features [B, M, 64] from inputs

        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        
        # Scatter node embeddings
        # ligand_batch: [N_total] -> tells us which batch index
        # We need per-batch atom indices.
        
        # Naive loop for scatter (can be optimized)
        for b in range(B_dim):
             mask_b = ligand_batch == b
             if mask_b.any():
                 nodes_b = h_ligand_nodes[mask_b]
                 n_atoms = nodes_b.shape[0]
                 # Safety clip
                 n_atoms = min(n_atoms, Max_M_dim)
                 h_ligand[b, :n_atoms] = nodes_b[:n_atoms]
        
        # ============================================================
        # Module 6: Cross-Attention Fusion (Multi-Layer)
        # ============================================================
        h_fused = h_prot_global
        for cross_attn_layer in self.cross_attention_layers:
            h_fused = cross_attn_layer(
                q=h_fused,
                kv=h_ligand,
                q_mask=protein_mask,
                kv_mask=ligand_mask
            )  # [B, L, 512]
        
        # ============================================================
        # Module 7: Late Injection & Classifier [Updated Flow]
        # ============================================================
        # No ResNet skip from h_prot_global
        # Input: Concat(H_fused, P_raw)
        
        x_final = torch.cat([h_fused, protein_scalar], dim=-1)  # [B, L, 512+35]
        
        logits = self.classifier(x_final)  # [B, L, 3]
        
        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_fused
        }
    
    def predict(
        self,
        **kwargs
    ) -> torch.Tensor:
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
            class_weights: [n_classes] - per-class weights
            gamma: Focal loss gamma parameter
            
        Returns:
            Scalar loss value
        """
        B, L, C = logits.shape
        device = logits.device
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, C)  # [B*L, C]
        labels_flat = labels.view(-1)      # [B*L]
        mask_flat = mask.view(-1)          # [B*L]
        
        # Compute softmax probabilities
        probs = F.softmax(logits_flat, dim=-1)  # [B*L, C]
        
        # Get probability of true class
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # [B*L]
        
        # Focal weight: (1 - p_t)^gamma
        p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*L]
        focal_weight = (1 - p_t) ** gamma
        
        # Apply class weights if provided
        if class_weights is not None:
            class_weights = class_weights.to(device)
            sample_weights = class_weights[labels_flat]  # [B*L]
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
            loss = focal_loss.sum() * 0  # Avoids NaN, returns 0
        
        return loss
    
    def load_gvp_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, int]:
        """
        Load GVP-related weights from an existing checkpoint.
        
        This is useful for transferring pre-trained GVP weights from an
        ESM-independent ablation run (use_esm=False) to initialize the GVP pathway.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            strict: If True, raise error on missing/unexpected keys. Default: False
            
        Returns:
            Dict with 'loaded', 'skipped', 'missing' counts
        """
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        source_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # GVP-related prefixes to transfer
        gvp_prefixes = ('gvp_encoder.', 'pairwise_bridge.', 'gvp_proj.')
        
        # Filter source keys to only GVP-related ones
        gvp_keys = {k: v for k, v in source_state_dict.items() 
                    if any(k.startswith(prefix) for prefix in gvp_prefixes)}
        
        # Get current model state
        current_state = self.state_dict()
        
        # Match and load
        loaded_count = 0
        skipped_count = 0
        missing_keys = []
        
        for key, value in gvp_keys.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    current_state[key] = value
                    loaded_count += 1
                else:
                    print(f"  [GVP Load] Shape mismatch for {key}: "
                          f"checkpoint={value.shape}, model={current_state[key].shape}")
                    skipped_count += 1
            else:
                missing_keys.append(key)
                skipped_count += 1
        
        # Load the updated state
        self.load_state_dict(current_state)
        
        print(f"[GVP Checkpoint] Loaded {loaded_count} parameters, "
              f"skipped {skipped_count} (missing/shape mismatch)")
        
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
        
        When frozen, GVP encoder and projection layers will not receive gradients,
        allowing other components (ESM pathway, transformers) to adapt while
        GVP features remain stable.
        
        Args:
            freeze: If True, freeze GVP params. If False, unfreeze.
            
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


class PocketGVPFusionAblation(nn.Module):
    """
    Unified ablation model for component contribution analysis.
    
    Supports selective disabling of model components via config flags:
    - use_gvp: Enable/disable GVP encoder (default: True)
    - use_esm: Enable/disable ESM embeddings (default: True)
    - use_physchem_early: Enable/disable physchem in early injection (default: True)
    - use_physchem_late: Enable/disable physchem in late injection/classifier (default: True)
    - use_self_attention: Enable/disable protein self-attention before cross-attention (default: True)
    
    Maintains the SAME forward interface as PocketGVPFusion for drop-in replacement.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Merge with defaults
        self.config = {**PocketGVPFusion.DEFAULT_CONFIG, **(config or {})}
        c = self.config
        
        # Ablation flags (default all True = full model)
        self.use_gvp = c.get('use_gvp', True)
        self.use_esm = c.get('use_esm', True)
        self.use_physchem_early = c.get('use_physchem_early', True)
        self.use_physchem_late = c.get('use_physchem_late', True)
        self.use_self_attention = c.get('use_self_attention', True)
        
        # Log ablation configuration
        print(f"[Ablation Config] use_gvp={self.use_gvp}, use_esm={self.use_esm}, "
              f"use_physchem_early={self.use_physchem_early}, use_physchem_late={self.use_physchem_late}, "
              f"use_self_attention={self.use_self_attention}")
        
        # ============================================================
        # Module 2: GVP Encoder (conditional)
        # ============================================================
        # Compute pairwise dimension: C * (C+1) / 2
        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        
        if self.use_gvp:
            self.gvp_encoder = GVPEncoder(
                node_in_dim=(c['d_phys'], c['n_gvp_vectors']),
                node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
                edge_in_dim=(1, 1),
                n_layers=c['n_gvp_layers'],
                drop_rate=c['dropout']
            )
            self.pairwise_bridge = PairwiseBridge(
                n_vec_channels=c['n_gvp_vector_hidden']
            )
            # d_geo = GVP scalar + pairwise + (optionally physchem)
            d_geo = c['d_gvp_internal'] + n_pairwise
            if self.use_physchem_early:
                d_geo += c['d_phys']  # Include physchem in GVP projection
        else:
            self.gvp_encoder = None
            self.pairwise_bridge = None
            d_geo = 0
            # If no GVP but physchem_early is enabled, it goes to ESM path or standalone
            if self.use_physchem_early and not self.use_esm:
                d_geo = c['d_phys']
        
        # ============================================================
        # Module 3: Bridge & Early Injection (Separate Projections)
        # ============================================================
        d_half = c['d_model'] // 2  # 256 each
        
        # GVP path (if enabled)
        if self.use_gvp or (self.use_physchem_early and not self.use_esm):
            self.gvp_proj = nn.Sequential(
                nn.Linear(d_geo, d_half),
                nn.LayerNorm(d_half),
                nn.Dropout(c['dropout'])
            )
        else:
            self.gvp_proj = None
        
        # ESM path (if enabled)
        # When self-attention is disabled and only ESM is active, project to full d_model
        if self.use_esm:
            d_esm_in = c['d_esm']
            # If no GVP but physchem_early, add physchem to ESM path
            if not self.use_gvp and self.use_physchem_early:
                d_esm_in += c['d_phys']
            
            # Determine output dimension for ESM projection
            # If no GVP and no self-attention, project directly to d_model
            if not self.use_gvp and not self.use_self_attention:
                d_esm_out = c['d_model']  # Full model dim for direct cross-attn
            else:
                d_esm_out = d_half
            
            self.esm_proj = nn.Sequential(
                nn.Linear(d_esm_in, d_esm_out),
                nn.LayerNorm(d_esm_out),
                nn.Dropout(c['dropout'])
            )
        else:
            self.esm_proj = None
        
        # Validate at least one projection exists
        if self.gvp_proj is None and self.esm_proj is None:
            raise ValueError("At least one of GVP or ESM must be enabled")
        
        # Compute final model dimension based on active projections
        if self.gvp_proj is not None and self.esm_proj is not None:
            d_proj_out = c['d_model']  # d_half + d_half = 512
        elif not self.use_gvp and not self.use_self_attention:
            d_proj_out = c['d_model']  # ESM already projected to full d_model
        else:
            d_proj_out = d_half  # Only one active = 256
        
        # If only one projection, need to scale up to d_model
        if d_proj_out != c['d_model']:
            self.proj_scale = nn.Linear(d_proj_out, c['d_model'])
        else:
            self.proj_scale = None
        
        # ============================================================
        # Module 4: Protein Context Encoder (Gated Transformer) - Optional
        # ============================================================
        if self.use_self_attention:
            self.protein_context_encoder = ProteinContextEncoder(
                d_model=c['d_model'],
                n_layers=c['n_transformer_layers'],
                n_heads=c['n_heads'],
                dropout=c['dropout'],
                pe_type='sinusoidal'
            )
        else:
            self.protein_context_encoder = None
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        self.ligand_encoder = LigandEncoder(
            node_in_dim=c['d_ligand_atom'],
            edge_in_dim=c['edge_dim'],
            hidden_dim=c['d_gvp_internal'],
            out_dim=c['d_model'],
            n_layers=c['n_ligand_layers'],
            dropout=c['dropout']
        )
        
        # ============================================================
        # Module 6: Cross-Attention Fusion (Multi-Layer)
        # ============================================================
        n_cross_layers = c.get('n_cross_attention_layers', 1)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout']
            )
            for _ in range(n_cross_layers)
        ])
        
        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        d_classifier_in = c['d_model']
        if self.use_physchem_late:
            d_classifier_in += c['d_phys']  # +35
        
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(c['dropout']),
            nn.Linear(256, c['n_classes'])
        )
    
    def forward(
        self,
        # Protein features - SAME interface as PocketGVPFusion
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
        Forward pass with configurable ablation.
        
        Components are conditionally used based on ablation flags set in __init__.
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config
        
        # ============================================================
        # Module 2: GVP Encoder (conditional)
        # ============================================================
        if self.use_gvp:
            h_s_gvp_list = []
            h_v_gvp_list = []
            
            for b in range(B):
                mask_b = protein_mask[b]
                n_valid = mask_b.sum()  # Keep as tensor to avoid graph break

                if n_valid == 0:
                    h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device, dtype=protein_scalar.dtype))
                    h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors.dtype))
                    continue
                
                scalar_b = protein_scalar[b, mask_b]
                vectors_b = protein_vectors[b, mask_b]
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
            
            v_inv = self.pairwise_bridge(h_v_gvp)
            # Concatenate GVP scalar + pairwise + (optionally physchem)
            if self.use_physchem_early:
                h_geo = torch.cat([h_s_gvp, v_inv, protein_scalar], dim=-1)
            else:
                h_geo = torch.cat([h_s_gvp, v_inv], dim=-1)
        else:
            h_geo = None
        
        # ============================================================
        # Module 3: Bridge & Early Injection (Separate Projections)
        # ============================================================
        projections = []
        
        # GVP path
        if self.gvp_proj is not None:
            if h_geo is not None:
                h_gvp_proj = self.gvp_proj(h_geo)
            else:
                # Fallback: physchem only (no GVP)
                h_gvp_proj = self.gvp_proj(protein_scalar)
            projections.append(h_gvp_proj)
        
        # ESM path
        if self.esm_proj is not None:
            if not self.use_gvp and self.use_physchem_early:
                # Physchem goes with ESM when no GVP
                esm_input = torch.cat([esm_embeddings, protein_scalar], dim=-1)
            else:
                esm_input = esm_embeddings
            h_esm_proj = self.esm_proj(esm_input)
            projections.append(h_esm_proj)
        
        # Concatenate projections
        h_seq = torch.cat(projections, dim=-1)
        
        # Scale to d_model if needed
        if self.proj_scale is not None:
            h_seq = self.proj_scale(h_seq)
        
        # ============================================================
        # Module 4: Protein Context Encoder (conditional)
        # ============================================================
        if self.use_self_attention and self.protein_context_encoder is not None:
            h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)
        else:
            # Skip self-attention: use projected features directly
            h_prot_global = h_seq
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        h_ligand_nodes = self.ligand_encoder(
            x=ligand_x, edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr, batch=ligand_batch
        )
        
        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        
        for b in range(B_dim):
            mask_b = ligand_batch == b
            if mask_b.any():
                nodes_b = h_ligand_nodes[mask_b]
                n_atoms = min(nodes_b.shape[0], Max_M_dim)
                h_ligand[b, :n_atoms] = nodes_b[:n_atoms]
        
        # ============================================================
        # Module 6: Cross-Attention Fusion (Multi-Layer)
        # ============================================================
        h_fused = h_prot_global
        for cross_attn_layer in self.cross_attention_layers:
            h_fused = cross_attn_layer(
                q=h_fused, kv=h_ligand,
                q_mask=protein_mask, kv_mask=ligand_mask
            )
        
        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        if self.use_physchem_late:
            x_final = torch.cat([h_fused, protein_scalar], dim=-1)
        else:
            x_final = h_fused
        
        logits = self.classifier(x_final)
        
        return {
            'logits': logits,
            'h_prot_global': h_prot_global,
            'h_fused': h_fused
        }
    
    def predict(self, **kwargs) -> torch.Tensor:
        """Convenience method for inference."""
        outputs = self.forward(**kwargs)
        return outputs['logits'].argmax(dim=-1)
    
    def load_gvp_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, int]:
        """
        Load GVP-related weights from an existing checkpoint.
        
        This is useful for transferring pre-trained GVP weights from an
        ESM-independent ablation run (use_esm=False) to initialize the GVP pathway.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            strict: If True, raise error on missing/unexpected keys. Default: False
            
        Returns:
            Dict with 'loaded', 'skipped', 'missing' counts
        """
        if not self.use_gvp:
            print("[GVP Checkpoint] Warning: GVP is disabled in this model. Skipping load.")
            return {'loaded': 0, 'skipped': 0, 'missing': 0}
        
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        source_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # GVP-related prefixes to transfer
        gvp_prefixes = ('gvp_encoder.', 'pairwise_bridge.', 'gvp_proj.')
        
        # Filter source keys to only GVP-related ones
        gvp_keys = {k: v for k, v in source_state_dict.items() 
                    if any(k.startswith(prefix) for prefix in gvp_prefixes)}
        
        # Get current model state
        current_state = self.state_dict()
        
        # Match and load
        loaded_count = 0
        skipped_count = 0
        missing_keys = []
        
        for key, value in gvp_keys.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    current_state[key] = value
                    loaded_count += 1
                else:
                    print(f"  [GVP Load] Shape mismatch for {key}: "
                          f"checkpoint={value.shape}, model={current_state[key].shape}")
                    skipped_count += 1
            else:
                missing_keys.append(key)
                skipped_count += 1
        
        # Load the updated state
        self.load_state_dict(current_state)
        
        print(f"[GVP Checkpoint] Loaded {loaded_count} parameters, "
              f"skipped {skipped_count} (missing/shape mismatch)")
        
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
        
        When frozen, GVP encoder and projection layers will not receive gradients,
        allowing other components (ESM pathway, transformers) to adapt while
        GVP features remain stable.
        
        Args:
            freeze: If True, freeze GVP params. If False, unfreeze.
            
        Returns:
            Number of parameters affected
        """
        if not self.use_gvp:
            print("[GVP] Warning: GVP is disabled in this model. Nothing to freeze.")
            return 0
        
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


# Backward compatibility alias
PocketGVPFusionNoGVP = PocketGVPFusionAblation


class PocketGVPFusionSimplified(nn.Module):
    """
    Simplified version of Pocket-GVP-Fusion for testing/debugging.
    
    Uses pre-batched tensors instead of per-protein GVP processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = {**PocketGVPFusion.DEFAULT_CONFIG, **(config or {})}
        c = self.config
        
        # Simplified: Skip GVP, directly project features
        d_input = c['d_esm'] + c['d_phys']
        
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, c['d_model']),
            nn.LayerNorm(c['d_model']),
            nn.Dropout(c['dropout'])
        )
        
        self.protein_context_encoder = ProteinContextEncoder(
            d_model=c['d_model'],
            n_layers=c['n_transformer_layers'],
            n_heads=c['n_heads'],
            dropout=c['dropout']
        )
        
        self.ligand_proj = nn.Sequential(
            nn.Linear(c['d_ligand_atom'], c['d_model']),
            nn.LayerNorm(c['d_model'])
        )
        
        n_cross_layers = c.get('n_cross_attention_layers', 1)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=c['d_model'],
                n_heads=c['n_heads'],
                dropout=c['dropout']
            )
            for _ in range(n_cross_layers)
        ])
        
        d_classifier_in = c['d_model'] + c['d_phys']
        self.classifier = nn.Sequential(
            nn.Linear(d_classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(c['dropout']),
            nn.Linear(256, c['n_classes'])
        )
        
    def forward(
        self,
        esm_embeddings: torch.Tensor,
        protein_scalar: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_features: torch.Tensor,
        ligand_mask: torch.Tensor,
        **kwargs  # Accept extra args for compatibility
    ) -> Dict[str, torch.Tensor]:
        """Simplified forward pass without GVP."""
        
        # Combine ESM + scalar
        h_combined = torch.cat([esm_embeddings, protein_scalar], dim=-1)
        h_seq = self.input_proj(h_combined)
        
        # Protein context
        h_prot = self.protein_context_encoder(h_seq, mask=protein_mask)
        
        # Ligand projection
        h_lig = self.ligand_proj(ligand_features)
        
        # Cross-attention (multi-layer)
        h_fused = h_prot
        for cross_attn_layer in self.cross_attention_layers:
            h_fused = cross_attn_layer(
                q=h_fused, kv=h_lig,
                q_mask=protein_mask, kv_mask=ligand_mask
            )
        
        # Classifier with late injection
        x = h_fused + h_prot
        x_final = torch.cat([x, protein_scalar], dim=-1)
        logits = self.classifier(x_final)
        
        return {'logits': logits, 'h_prot_global': h_prot, 'h_fused': h_fused}
