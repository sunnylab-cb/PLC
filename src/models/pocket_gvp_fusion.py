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
        
        # Layers
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_heads': 8,
        
        # Settings
        'dropout': 0.1,
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        
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
        # Input: scalar (d_phys=35) + vector (4×3)
        # Output: scalar (d_gvp_internal=256) + vector (16×3)
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
        # h_geo = [h_s_gvp (256) + ||h_v_gvp|| (16)] = 272
        # h_combined = [h_geo (272) + P_raw (35)] = 307
        # Also need space for ESM: 1280
        # Total input to bridge: 272 + 35 + 1280 = 1587
        
        d_bridge_in = c['d_gvp_internal'] + c['n_gvp_vector_hidden'] + c['d_phys'] + c['d_esm']
        
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
        # Module 6: Cross-Attention Fusion
        # ============================================================
        self.cross_attention = CrossAttentionBlock(
            d_model=c['d_model'],
            n_heads=c['n_heads'],
            dropout=c['dropout']
        )
        
        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        # ResNet skip: H_fused + H_prot_global (handled in forward)
        # DenseNet skip: Concat with P_raw (35)
        # Final dim: d_model + d_phys = 512 + 35 = 547
        
        d_classifier_in = c['d_model'] + c['d_phys']
        
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
        protein_mask: torch.Tensor,         # [B, L] (True = valid)
        
        # Ligand features (batched graph)
        ligand_x: torch.Tensor,             # [N_atoms_total, 64]
        ligand_edge_index: torch.Tensor,    # [2, E_total]
        ligand_edge_attr: torch.Tensor,     # [E_total, edge_dim]
        ligand_batch: torch.Tensor,         # [N_atoms_total]
        ligand_to_protein: torch.Tensor,    # [B, max_atoms] padded ligand features
        ligand_mask: torch.Tensor,          # [B, max_atoms]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            esm_embeddings: ESM-2 embeddings per residue
            protein_scalar: Physicochemical features per residue  
            protein_vectors: GVP vector features per residue
            protein_coords: Cα coordinates per residue
            protein_mask: Padding mask for protein (True = valid token)
            ligand_x: Atom features for all ligands in batch
            ligand_edge_index: Edge indices for ligand graph
            ligand_edge_attr: Edge features for ligand graph
            ligand_batch: Maps each atom to its batch index
            ligand_to_protein: Padded ligand atom features [B, max_atoms, d_model]
            ligand_mask: Padding mask for ligand atoms
            
        Returns:
            Dict with:
                'logits': [B, L, n_classes] - class logits per residue
                'h_prot_global': [B, L, d_model] - protein context embeddings
                'h_fused': [B, L, d_model] - fused embeddings
        """
        B, L, _ = esm_embeddings.shape
        device = esm_embeddings.device
        c = self.config
        
        # ============================================================
        # Module 2: GVP Encoder
        # ============================================================
        # Process each protein in batch through GVP
        # For efficiency, we flatten and rebuild
        
        h_s_gvp_list = []
        h_v_gvp_list = []
        
        for b in range(B):
            # Get valid residues for this protein
            mask_b = protein_mask[b]  # [L]
            n_valid = mask_b.sum().item()
            
            if n_valid == 0:
                # Handle empty case
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
        
        # ============================================================
        # Module 3: Bridge & Early Injection
        # ============================================================
        # Flatten vectors: ||h_v_gvp|| -> [B, L, 16]
        v_norm = torch.norm(h_v_gvp, dim=-1)  # [B, L, 16]
        
        # h_geo = [h_s_gvp, v_norm] -> [B, L, 272]
        h_geo = torch.cat([h_s_gvp, v_norm], dim=-1)
        
        # Early injection: concat with physics features + ESM
        # h_combined = [h_geo (272), P_raw (35), ESM (1280)] -> [B, L, 1587]
        h_combined = torch.cat([h_geo, protein_scalar, esm_embeddings], dim=-1)
        
        # Project to d_model
        h_seq = self.bridge_proj(h_combined)  # [B, L, 512]
        
        # ============================================================
        # Module 4: Protein Context Encoder
        # ============================================================
        h_prot_global = self.protein_context_encoder(h_seq, mask=protein_mask)  # [B, L, 512]
        
        # ============================================================
        # Module 5: Ligand Encoder
        # ============================================================
        # Encode all ligands in batch
        h_ligand_nodes = self.ligand_encoder(
            x=ligand_x,
            edge_index=ligand_edge_index,
            edge_attr=ligand_edge_attr,
            batch=ligand_batch
        )  # [N_atoms_total, 512]
        
        # Reshape to padded batch format for cross-attention
        # ligand_to_protein: [B, max_atoms] contains indices into h_ligand_nodes
        # or we use pre-padded ligand features
        h_ligand = ligand_to_protein  # [B, M, 512] (pre-padded in dataloader)
        
        # ============================================================
        # Module 6: Cross-Attention Fusion
        # ============================================================
        h_fused = self.cross_attention(
            q=h_prot_global,
            kv=h_ligand,
            q_mask=protein_mask,
            kv_mask=ligand_mask
        )  # [B, L, 512]
        
        # ============================================================
        # Module 7: Late Injection & Classifier
        # ============================================================
        # ResNet skip
        x = h_fused + h_prot_global  # [B, L, 512]
        
        # DenseNet skip (Late injection)
        x_final = torch.cat([x, protein_scalar], dim=-1)  # [B, L, 547]
        
        # Classification head
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
        
        self.cross_attention = CrossAttentionBlock(
            d_model=c['d_model'],
            n_heads=c['n_heads'],
            dropout=c['dropout']
        )
        
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
        
        # Cross-attention
        h_fused = self.cross_attention(
            q=h_prot, kv=h_lig,
            q_mask=protein_mask, kv_mask=ligand_mask
        )
        
        # Classifier with late injection
        x = h_fused + h_prot
        x_final = torch.cat([x, protein_scalar], dim=-1)
        logits = self.classifier(x_final)
        
        return {'logits': logits, 'h_prot_global': h_prot, 'h_fused': h_fused}
