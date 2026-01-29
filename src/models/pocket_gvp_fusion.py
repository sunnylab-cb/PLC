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
    """
    def __init__(self, n_vec_channels: int, d_out: int):
        super().__init__()
        # Number of unique pairs in symmetric matrix: C * (C+1) / 2
        self.n_pairs = (n_vec_channels * (n_vec_channels + 1)) // 2
        self.proj = nn.Linear(self.n_pairs, d_out)

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h_v: Vector features [Batch, N, C, 3]
            
        Returns:
            Projected scalar features [Batch, N, d_out]
        """
        # 1. Compute Gram Matrix per residue (Batch, N, C, C)
        # dot product over the spatial dimension (last dim)
        gram = torch.einsum('bnci,bnji->bncj', h_v, h_v)
        
        # 2. Extract Upper Triangle (unique pairs only)
        batch, n_res, c, _ = gram.shape
        
        # Create indices for upper triangle
        indices = torch.triu_indices(c, c, device=gram.device)
        
        # Gather unique pairs: [Batch, N, N_Pairs]
        # We need to expand indices to match batch and n_res dimensions
        # Or simply permute and gather
        
        # Flatten batch and residue dims for efficient gathering
        gram_flat = gram.view(batch * n_res, c, c)
        
        # Gather: [Batch*N, N_Pairs]
        flat_pairs = gram_flat[:, indices[0], indices[1]]
        
        # Reshape back: [Batch, N, N_Pairs]
        flat_pairs = flat_pairs.view(batch, n_res, self.n_pairs)
        
        # 3. Project back to reasonable dimension
        return self.proj(flat_pairs)


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
        'd_model': 512,          # Transformer dimension
        'd_ligand_atom': 64,     # Ligand atom features
        'n_gvp_vectors': 4,      # Number of GVP vector features
        'n_gvp_vector_hidden': 16,  # GVP vector hidden dimension
        'd_pairwise_bridge': 64,    # Output dim of pairwise bridge
        
        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_cross_attention_layers': 1,  # NEW: Multi-layer cross-attention (default 1 for backward compat)
        'n_heads': 8,
        
        # Settings
        'dropout': 0.1,
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'esm_dropout_prob': 0.5,  # Modality dropout: probability of dropping ESM during training
        
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
        # Module 3: Bridge & Early Injection
        # ============================================================
        # Step A: Pairwise Bridge for vectors
        self.pairwise_bridge = PairwiseBridge(
            n_vec_channels=c['n_gvp_vector_hidden'], 
            d_out=c['d_pairwise_bridge']
        )
        
        # Construct h_geo = [h_s_gvp (256) + pairwise_out (64)] = 320
        d_geo = c['d_gvp_internal'] + c['d_pairwise_bridge']
        
        # Step B: Early injection
        # h_combined = [h_geo (320) + P_raw (35) + ESM (1280)] = 1635
        d_bridge_in = d_geo + c['d_phys'] + c['d_esm']
        
        # Step C: Projection
        self.bridge_proj = nn.Sequential(
            nn.Linear(d_bridge_in, c['d_model']),
            nn.LayerNorm(c['d_model']),
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
            n_valid = mask_b.sum().item()
            
            if n_valid == 0:
                h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device))
                h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device))
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
            h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device)
            h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device)
            
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
        # Module 3: Bridge & Early Injection [Updated Flow]
        # ============================================================
        
        # Step A: Pairwise Bridge
        # [B, L, 16, 3] -> [B, L, 64]
        v_inv = self.pairwise_bridge(h_v_gvp)
        
        # h_geo = [h_s_gvp, v_inv] -> [B, L, 320]
        h_geo = torch.cat([h_s_gvp, v_inv], dim=-1)
        
        # Cache final geometric embedding for diagnostics
        self._cached_h_geo = h_geo
        
        # --- Modality Dropout: Applied BEFORE concatenation ---
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
        
        # Step B: Early Injection
        # Concat GVP out + Phys + ESM -> [B, L, 1635]
        # If masked, the ESM channels are zeroed out
        h_combined = torch.cat([h_geo, protein_scalar, current_esm], dim=-1)
        
        # Step C: Projection
        h_seq = self.bridge_proj(h_combined)  # [B, L, 512]
        
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
        
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device)
        
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


class PocketGVPFusionAblation(nn.Module):
    """
    Unified ablation model for component contribution analysis.
    
    Supports selective disabling of model components via config flags:
    - use_gvp: Enable/disable GVP encoder (default: True)
    - use_esm: Enable/disable ESM embeddings (default: True)
    - use_physchem_early: Enable/disable physchem in early injection (default: True)
    - use_physchem_late: Enable/disable physchem in late injection/classifier (default: True)
    
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
        
        # Log ablation configuration
        print(f"[Ablation Config] use_gvp={self.use_gvp}, use_esm={self.use_esm}, "
              f"use_physchem_early={self.use_physchem_early}, use_physchem_late={self.use_physchem_late}")
        
        # ============================================================
        # Module 2: GVP Encoder (conditional)
        # ============================================================
        if self.use_gvp:
            self.gvp_encoder = GVPEncoder(
                node_in_dim=(c['d_phys'], c['n_gvp_vectors']),
                node_h_dim=(c['d_gvp_internal'], c['n_gvp_vector_hidden']),
                edge_in_dim=(1, 1),
                n_layers=c['n_gvp_layers'],
                drop_rate=c['dropout']
            )
            self.pairwise_bridge = PairwiseBridge(
                n_vec_channels=c['n_gvp_vector_hidden'],
                d_out=c['d_pairwise_bridge']
            )
            d_geo = c['d_gvp_internal'] + c['d_pairwise_bridge']  # 320
        else:
            self.gvp_encoder = None
            self.pairwise_bridge = None
            d_geo = 0
        
        # ============================================================
        # Module 3: Bridge & Early Injection
        # ============================================================
        # Compute bridge input dimension based on enabled components
        d_bridge_in = d_geo
        if self.use_esm:
            d_bridge_in += c['d_esm']  # +1280
        if self.use_physchem_early:
            d_bridge_in += c['d_phys']  # +35
        
        # If nothing is enabled, use a minimum dimension (shouldn't happen in practice)
        if d_bridge_in == 0:
            raise ValueError("At least one of GVP, ESM, or early physchem must be enabled")
        
        self.bridge_proj = nn.Sequential(
            nn.Linear(d_bridge_in, c['d_model']),
            nn.LayerNorm(c['d_model']),
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
                n_valid = mask_b.sum().item()
                
                if n_valid == 0:
                    h_s_gvp_list.append(torch.zeros(L, c['d_gvp_internal'], device=device))
                    h_v_gvp_list.append(torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device))
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
                
                h_s_full = torch.zeros(L, c['d_gvp_internal'], device=device)
                h_v_full = torch.zeros(L, c['n_gvp_vector_hidden'], 3, device=device)
                h_s_full[mask_b] = h_s
                h_v_full[mask_b] = h_v
                
                h_s_gvp_list.append(h_s_full)
                h_v_gvp_list.append(h_v_full)
            
            h_s_gvp = torch.stack(h_s_gvp_list, dim=0)
            h_v_gvp = torch.stack(h_v_gvp_list, dim=0)
            
            v_inv = self.pairwise_bridge(h_v_gvp)
            h_geo = torch.cat([h_s_gvp, v_inv], dim=-1)
        else:
            h_geo = None
        
        # ============================================================
        # Module 3: Bridge & Early Injection
        # ============================================================
        components = []
        if h_geo is not None:
            components.append(h_geo)
        if self.use_esm:
            components.append(esm_embeddings)
        if self.use_physchem_early:
            components.append(protein_scalar)
        
        h_combined = torch.cat(components, dim=-1)
        h_seq = self.bridge_proj(h_combined)
        
        # ============================================================
        # Module 4: Protein Context Encoder
        # ============================================================
        h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        h_ligand_nodes = self.ligand_encoder(
            x=ligand_x, edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr, batch=ligand_batch
        )
        
        B_dim, Max_M_dim, _ = ligand_to_protein.shape
        h_ligand = torch.zeros(B_dim, Max_M_dim, c['d_model'], device=device)
        
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
