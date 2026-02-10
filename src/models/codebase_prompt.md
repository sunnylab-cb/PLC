# 项目结构

项目`/home/husrcf/Code/ProtBind/protein-ligand-PoC/src/models/`的目录结构（已排除如`node_modules`等目录）：
```
/home/husrcf/Code/ProtBind/protein-ligand-PoC/src/models//
├── __init__.py
├── codebase_prompt.md
├── gvp_layers.py
├── ligand_encoder.py
├── package.py
├── pocket_gvp_fusion.py
└── transformer_layers.py
```

---

# 代码内容

## 文件: `gvp_layers.py`

```python
"""
GVP (Geometric Vector Perceptron) layers for Pocket-GVP-Fusion v2.

Implements GVPConv with vector gating (v2 style) for encoding local 3D geometry.
Refactored to avoid PyG MessagePassing tuple issues and support manual scatter.

Reference:
    Jing et al. "Learning from Protein Structure with Geometric Vector Perceptrons"
    https://arxiv.org/abs/2009.01411
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Tuple, Optional


class GVP(nn.Module):
    """
    Geometric Vector Perceptron with Safe Norm and Vector Gating.
    """
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations: tuple = (F.relu, None),
        vector_gate: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        self.vector_gate = vector_gate
        self.eps = eps
        
        # Scalar pathway
        # Input: scalar features + vector norms
        self.scalar_linear = nn.Linear(self.si + self.vi, self.so)
        
        # Vector pathway
        if self.vo > 0 and self.vi > 0:
            self.vector_linear = nn.Linear(self.vi, self.vo, bias=False)
            if self.vector_gate:
                self.gate_linear = nn.Linear(self.si + self.vi, self.vo)
        
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Compute Vector Norms (Safe Norm)
        # v: [*, vi, 3]
        v_sq = torch.sum(v ** 2, dim=-1)
        v_norm = torch.sqrt(v_sq + self.eps) # [*, vi]
        
        # 2. Scalar Pathway
        # Concatenate scalar input + vector norms
        s_combined = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_linear(s_combined)
        
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
            
        # 3. Vector Pathway
        v_out = v # Default to input if conditions not met
        if self.vo > 0 and self.vi > 0:
            # Linear transform on vector channels
            # [*, vi, 3] -> [*, 3, vi] -> Linear -> [*, 3, vo] -> [*, vo, 3]
            v_t = v.transpose(-2, -1)
            v_t = self.vector_linear(v_t)
            v_out = v_t.transpose(-2, -1)
            
            # Vector Gating
            if self.vector_gate:
                # Gate based on input scalars + input vector norms
                gate = self.gate_linear(s_combined)
                gate = torch.sigmoid(gate) # [*, vo]
                v_out = v_out * gate.unsqueeze(-1)
            
            # Vector Activation (usually None or normalization)
            if self.vector_act is not None:
                v_out = self.vector_act(v_out)
        elif self.vo > 0 and self.vi == 0:
            # If we need vector output but have no vector input, initialize zeros
            v_out = torch.zeros((*s.shape[:-1], self.vo, 3), device=s.device)
            
        return s_out, v_out


class Dropout(nn.Module):
    """
    GVP-specific Dropout. 
    Drops scalars normally.
    Drops vectors channel-wise (entire 3D vector is dropped).
    """
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, s, v):
        if not self.training:
            return s, v
            
        # Scalar dropout
        s = F.dropout(s, p=self.drop_rate, training=True)
        
        # Vector channel dropout
        # mask shape: [*, vi, 1]
        mask_shape = v.shape[:-1] + (1,)
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(mask_shape, device=v.device))
        v = v * mask * (1 / (1 - self.drop_rate))
        
        return s, v


class GVPConv(nn.Module):
    """
    GVP Graph Convolution Layer.
    Implemented explicitly (without MessagePassing class) for robust tuple handling.
    """
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        edge_dims: Tuple[int, int] = (1, 1),
        n_layers: int = 3,
        vector_gate: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        # Message Layers
        message_in = (self.si + self.se, self.vi + self.ve)
        message_hidden = (self.so, self.vo)
        
        self.message_func = nn.ModuleList()
        for i in range(n_layers):
            cur_in = message_in if i == 0 else message_hidden
            self.message_func.append(GVP(cur_in, message_hidden, vector_gate=vector_gate))
        
        # Update Layers
        update_in = (self.si + self.so, self.vi + self.vo)
        self.update_func = nn.ModuleList()
        for i in range(n_layers):
            cur_in = update_in if i == 0 else message_hidden
            cur_out = out_dims if i == n_layers - 1 else message_hidden
            self.update_func.append(GVP(cur_in, cur_out, vector_gate=vector_gate))
            
        self.dropout = Dropout(dropout)
        self.use_residual = (self.si == self.so and self.vi == self.vo)

    def forward(self, x_s, x_v, edge_index, edge_s, edge_v):
        """
        x_s: [N, si]
        x_v: [N, vi, 3]
        edge_index: [2, E]
        edge_s: [E, se]
        edge_v: [E, ve, 3]
        """
        src, dst = edge_index
        
        # 1. Create Messages
        # Gather source features
        s_j = x_s[src] # [E, si]
        v_j = x_v[src] # [E, vi, 3]
        
        # Concat with edge features
        msg_s = torch.cat([s_j, edge_s], dim=-1) # [E, si + se]
        msg_v = torch.cat([v_j, edge_v], dim=-2) # [E, vi + ve, 3]
        
        # Process messages
        for layer in self.message_func:
            msg_s, msg_v = layer(msg_s, msg_v)
            msg_s, msg_v = self.dropout(msg_s, msg_v)
            
        # 2. Aggregate (Sum/Mean)
        # Use scatter from torch_geometric.utils
        # Note: dim_size ensures we cover nodes with no incoming edges
        aggr_s = scatter(msg_s, dst, dim=0, dim_size=x_s.size(0), reduce='mean')
        aggr_v = scatter(msg_v, dst, dim=0, dim_size=x_v.size(0), reduce='mean')
        
        # 3. Update
        # Concat node features + aggregated messages
        upd_s = torch.cat([x_s, aggr_s], dim=-1)
        upd_v = torch.cat([x_v, aggr_v], dim=-2)
        
        for layer in self.update_func:
            upd_s, upd_v = layer(upd_s, upd_v)
            upd_s, upd_v = self.dropout(upd_s, upd_v)
            
        # Residual connection
        if self.use_residual:
            upd_s = upd_s + x_s
            upd_v = upd_v + x_v
            
        return upd_s, upd_v


class GVPEncoder(nn.Module):
    """
    GVP-based encoder for protein structure.
    
    Stacks multiple GVPConv layers to encode local 3D geometry.
    
    Args:
        node_in_dim: Input node dimensions (scalar, vector)
        node_h_dim: Hidden node dimensions (scalar, vector)
        edge_in_dim: Input edge dimensions (scalar, vector)
        n_layers: Number of GVPConv layers
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        node_in_dim: Tuple[int, int] = (35, 4),
        node_h_dim: Tuple[int, int] = (256, 16),
        edge_in_dim: Tuple[int, int] = (1, 1),
        n_layers: int = 3,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        
        # Initial projection
        self.W_v_in = nn.Linear(node_in_dim[1], node_h_dim[1], bias=False)
        self.W_s_in = nn.Linear(node_in_dim[0], node_h_dim[0])
        
        # GVP layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GVPConv(
                    in_dims=node_h_dim,
                    out_dims=node_h_dim,
                    edge_dims=edge_in_dim,
                    vector_gate=True,
                    dropout=drop_rate
                )
            )
        
        # Layer norm
        self.norm_s = nn.LayerNorm(node_h_dim[0])
        
    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_s: Node scalar features [N, si]
            node_v: Node vector features [N, vi, 3]
            edge_index: Edge indices [2, E]
            edge_s: Edge scalar features [E, se] (typically distance)
            edge_v: Edge vector features [E, ve, 3] (typically direction)
            
        Returns:
            Tuple of (node_s [N, so], node_v [N, vo, 3])
        """
        # Initial projection
        node_s = self.W_s_in(node_s)
        
        # Project vectors: [N, vi, 3] -> [N, vo, 3]
        # Transpose, apply linear, transpose back
        node_v = node_v.transpose(-2, -1)  # [N, 3, vi]
        node_v = self.W_v_in(node_v)       # [N, 3, vo]
        node_v = node_v.transpose(-2, -1)  # [N, vo, 3]
        
        # Apply GVP layers
        for layer in self.layers:
            # GVPConv (new) expects: x_s, x_v, edge_index, edge_s, edge_v
            # And it returns positions match s, v
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)
        
        # Final normalization
        node_s = self.norm_s(node_s)
        
        # Diagnostic tracking: cache norms for training monitoring
        # These can be read by the diagnostics module after forward pass
        with torch.no_grad():
            self._last_scalar_norm = node_s.norm(dim=-1).mean()
            self._last_vector_norm = node_v.norm(dim=-1).mean()
        
        return node_s, node_v


def build_knn_graph(
    coords: torch.Tensor,
    k: int = 20,
    radius: float = 10.0,
    batch: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build k-NN graph from coordinates.
    
    Args:
        coords: Node coordinates [N, 3]
        k: Number of nearest neighbors
        radius: Maximum edge distance (Angstroms)
        batch: Batch indices [N] for batched graphs
        
    Returns:
        edge_index: [2, E]
        edge_dist: [E, 1] - edge distances
        edge_dir: [E, 1, 3] - edge direction unit vectors
    """
    from torch_geometric.nn import knn_graph
    
    # Build k-NN graph
    try:
        if batch is not None:
            edge_index = knn_graph(coords, k=k, batch=batch, loop=False)
        else:
            edge_index = knn_graph(coords, k=k, loop=False)
    except (ImportError, OSError):
        # Fallback for when torch-cluster is not installed
        # Use simple cdist (slower but works without C++ extensions)
        if batch is None:
            batch_indices = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
            unique_batches = torch.tensor([0], device=coords.device)
        else:
            batch_indices = batch
            unique_batches = batch_indices.unique()
            
        edge_index_list = []
        for b in unique_batches:
            mask = batch_indices == b
            if not mask.any():
                continue
            
            # Local indices
            batch_coords = coords[mask]
            n_nodes = batch_coords.shape[0]
            curr_k = min(k, n_nodes - 1)
            
            if curr_k <= 0:
                continue
                
            # Compute distances
            dist = torch.cdist(batch_coords, batch_coords)
            
            # Get k-NN (excluding self which is usually index 0 in sort)
            # topk returns largest, so we negate or use sort
            # using topk with largest=False is faster
            _, indices = dist.topk(curr_k + 1, dim=1, largest=False)
            
            # Remove self-loops (first index is self)
            indices = indices[:, 1:]
            
            # Create edge index
            # source: indices[i, j], target: i
            # GVP convention: source -> target
            targets = torch.arange(n_nodes, device=coords.device).unsqueeze(1).expand(-1, curr_k)
            
            # Global indices
            global_indices = torch.where(mask)[0]
            src_global = global_indices[indices.flatten()]
            tgt_global = global_indices[targets.flatten()]
            
            edge_index_list.append(torch.stack([src_global, tgt_global], dim=0))
            
        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    
    # Filter by radius
    src, tgt = edge_index
    edge_vec = coords[tgt] - coords[src]  # [E, 3]
    edge_dist = torch.norm(edge_vec, dim=-1)  # [E]
    
    # Keep edges within radius
    mask = edge_dist <= radius
    edge_index = edge_index[:, mask]
    edge_vec = edge_vec[mask]
    edge_dist = edge_dist[mask]
    
    # Compute unit direction vectors
    edge_dir = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
    
    # Reshape for GVP
    edge_dist = edge_dist.unsqueeze(-1)  # [E, 1]
    edge_dir = edge_dir.unsqueeze(-2)    # [E, 1, 3]
    
    return edge_index, edge_dist, edge_dir

```

## 文件: `__init__.py`

```python
"""
Pocket-GVP-Fusion v2 model components.
"""

from .pocket_gvp_fusion import PocketGVPFusion
from .gvp_layers import GVPConv, GVPEncoder
from .transformer_layers import GatedTransformerBlock, SwiGLU
from .ligand_encoder import LigandEncoder

__all__ = [
    'PocketGVPFusion',
    'GVPConv',
    'GVPEncoder', 
    'GatedTransformerBlock',
    'SwiGLU',
    'LigandEncoder',
]

```

## 文件: `pocket_gvp_fusion.py`

```python
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

```

## 文件: `ligand_encoder.py`

```python
"""
Ligand encoder for Pocket-GVP-Fusion v2.

Implements GINEConv-based encoder for ligand topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from typing import Optional, Tuple


class LigandEncoder(nn.Module):
    """
    GINEConv-based encoder for ligand molecules.
    
    Encodes ligand topology using graph isomorphism network with edge features.
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        n_layers: Number of GINEConv layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_in_dim: int = 64,
        edge_in_dim: int = 4,
        hidden_dim: int = 256,
        out_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.out_dim = out_dim
        
        # Initial node projection
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        
        # Edge projection
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        
        # GINEConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            # MLP for GINEConv
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            batch: Batch assignment [N] (for pooling)
            
        Returns:
            Node embeddings [N, out_dim]
        """
        # Initial projection
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply GINEConv layers
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        # Output projection
        x = self.out_proj(x)
        x = self.out_norm(x)
        
        return x
    
    def pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mode: str = 'mean'
    ) -> torch.Tensor:
        """
        Pool node embeddings to graph-level representation.
        
        Args:
            x: Node embeddings [N, out_dim]
            batch: Batch assignment [N]
            mode: Pooling mode ('mean' or 'add')
            
        Returns:
            Graph embeddings [B, out_dim]
        """
        if mode == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_add_pool(x, batch)


def build_ligand_edge_features(mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge index and edge features for a ligand molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        edge_index: [2, E]
        edge_attr: [E, 4] - [bond_type_one_hot (3), is_conjugated]
    """
    from rdkit import Chem
    
    if mol is None:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)
    
    # Bond type mapping
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    
    edges_src = []
    edges_dst = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bidirectional edges
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        
        # Bond features
        bond_type = bond_type_map.get(bond.GetBondType(), 0)
        bond_onehot = [0.0] * 4
        bond_onehot[bond_type] = 1.0
        
        is_conjugated = float(bond.GetIsConjugated())
        is_in_ring = float(bond.IsInRing())
        
        feat = bond_onehot + [is_conjugated, is_in_ring]
        
        # Same features for both directions
        edge_features.extend([feat, feat])
    
    if len(edges_src) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 6), dtype=torch.float32)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    return edge_index, edge_attr


class LigandEncoderWithPooling(nn.Module):
    """
    Ligand encoder with optional global pooling for compatibility.
    
    Can output either node-level or graph-level representations.
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension  
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        n_layers: Number of GINEConv layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_in_dim: int = 64,
        edge_in_dim: int = 6,
        hidden_dim: int = 256,
        out_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = LigandEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            batch: Batch assignment [N]
            return_node_embeddings: If True, return node-level embeddings.
                                   If False, return graph-level pooled embeddings.
            
        Returns:
            If return_node_embeddings: [N, out_dim]
            Else: [B, out_dim]
        """
        node_emb = self.encoder(x, edge_index, edge_attr, batch)
        
        if return_node_embeddings:
            return node_emb
        else:
            if batch is None:
                # Single molecule
                return node_emb.mean(dim=0, keepdim=True)
            return self.encoder.pool(node_emb, batch, mode='mean')

```

## 文件: `transformer_layers.py`

```python
"""
Transformer layers for Pocket-GVP-Fusion v2.

Implements:
- SwiGLU FFN (LLaMA-style)
- Gated Attention (Qwen-style)
- GatedTransformerBlock combining both
- Positional Encoding (sinusoidal and learnable)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation with feedforward network (LLaMA-style).
    
    Computes: SwiGLU(x) = (x @ W_1) * SiLU(x @ W_gate) @ W_2
    
    Args:
        d_model: Input/output dimension
        d_ffn: Hidden dimension (typically 8/3 * d_model for similar param count as vanilla FFN)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Default hidden dim similar to 4x expansion with SwiGLU efficiency
        if d_ffn is None:
            d_ffn = int(8 / 3 * d_model)
            # Round to nearest multiple of 64 for efficiency
            d_ffn = ((d_ffn + 63) // 64) * 64
        
        self.w_1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_2 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [*, d_model]
            
        Returns:
            Output tensor [*, d_model]
        """
        # SwiGLU: element-wise product of gated and linear projections
        gate = F.silu(self.w_gate(x))
        x = self.w_1(x) * gate
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with output gating (Qwen-style).
    
    Adds a learnable gate to control the attention output contribution.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_gate: Whether to apply output gating
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_gate = use_gate
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Gating mechanism (Qwen-style)
        if use_gate:
            self.gate = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor [B, L, d_model]
            mask: Attention mask [B, L] or [B, L, L] (True = attend, False = mask)
            kv: Optional key/value tensor for cross-attention [B, S, d_model]
            
        Returns:
            Output tensor [B, L, d_model]
        """
        B, L, _ = x.shape
        
        # Use x for both K, V if not cross-attention
        if kv is None:
            kv = x
        S = kv.size(1)
        
        # Project Q, K, V
        q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D]
        k = self.w_k(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, D]
        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, D]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, S]
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:  # [B, S] -> [B, 1, 1, S]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # [B, L, S] -> [B, 1, L, S]
                mask = mask.unsqueeze(1)
            
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        
        # Diagnostic tracking: cache attention weights for entropy computation
        # Store before dropout for accurate entropy calculation
        with torch.no_grad():
            self._last_attn_weights = attn.detach()
            # Compute attention entropy: -sum(p * log(p))
            eps = 1e-8
            attn_clamped = attn.clamp(min=eps)
            self._last_attn_entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1).mean()
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]
        
        # Output projection
        out = self.w_o(out)
        
        # Apply gating
        if self.use_gate:
            gate = torch.sigmoid(self.gate(x))
            # Diagnostic tracking: cache gate openness
            with torch.no_grad():
                self._last_gate_openness = gate.mean()
            out = out * gate
        
        return out


class GatedTransformerBlock(nn.Module):
    """
    Transformer block with SwiGLU FFN and Gated Attention.
    
    Architecture:
        x -> LayerNorm -> GatedMHA -> Residual
          -> LayerNorm -> SwiGLU   -> Residual
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_gated_attn: Whether to use gated attention
        use_swiglu: Whether to use SwiGLU (vs standard FFN)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gated_attn: bool = True,
        use_swiglu: bool = True
    ):
        super().__init__()
        
        # Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_gate=use_gated_attn
        )
        
        # Feedforward
        self.norm2 = nn.LayerNorm(d_model)
        if use_swiglu:
            self.ffn = SwiGLU(d_model, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, d_model]
            mask: Attention mask [B, L] (True = valid, False = padding)
            
        Returns:
            Output tensor [B, L, d_model]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout(x) + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for protein-ligand fusion.
    
    Architecture:
        q -> LayerNorm -> CrossAttn(q, kv) -> Residual
          -> LayerNorm -> SwiGLU          -> Residual
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Cross-attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = GatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_gate=True
        )
        
        # Feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            q: Query tensor [B, L_q, d_model] (protein)
            kv: Key/Value tensor [B, L_kv, d_model] (ligand)
            q_mask: Query mask [B, L_q]
            kv_mask: Key/Value mask [B, L_kv]
            
        Returns:
            Output tensor [B, L_q, d_model]
        """
        # Cross-attention with residual
        residual = q
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        
        # Build cross-attention mask
        cross_mask = None
        if kv_mask is not None:
            cross_mask = kv_mask  # [B, L_kv]
        
        out = self.cross_attn(q_normed, mask=cross_mask, kv=kv_normed)
        out = self.dropout(out) + residual
        
        # FFN with residual
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = self.dropout(out) + residual
        
        return out


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al.).
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, L, d_model]
            
        Returns:
            Output tensor [B, L, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, L, d_model]
            
        Returns:
            Output tensor [B, L, d_model]
        """
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


class ProteinContextEncoder(nn.Module):
    """
    Protein Context Encoder using stacked GatedTransformerBlocks.
    
    Learns global context over the protein sequence.
    
    Args:
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
        max_len: Maximum sequence length for PE
        pe_type: Type of positional encoding ('sinusoidal' or 'learnable')
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2000,
        pe_type: str = 'sinusoidal'
    ):
        super().__init__()
        
        # Positional encoding
        if pe_type == 'sinusoidal':
            self.pe = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pe = LearnablePositionalEncoding(d_model, max_len, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_gated_attn=True,
                use_swiglu=True
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, d_model]
            mask: Padding mask [B, L] (True = valid, False = padding)
            
        Returns:
            Output tensor [B, L, d_model]
        """
        # Add positional encoding
        x = self.pe(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x

```

## 文件: `package.py`

```python
#!/usr/bin/env python3
import os
import re
import sys
import fnmatch

# 移除 from pathlib import Path，后续用 os 替代
# from pathlib import Path

# --- 配置区 ---
# 目标目录，使用 os 获取当前脚本所在目录
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

# 输出的Markdown文件
OUTPUT_FILE = "codebase_prompt.md"

# ==============================================================================
# 新增：仅包含（白名单）模式 - 用于筛选【文件内容】
# ==============================================================================
INCLUDE_ONLY_PATTERNS = [
    "*.py", "*.yaml"
]

# ==============================================================================
# 排除（黑名单）模式
# ==============================================================================
# 要排除的目录。此规则对【项目结构图】和【文件内容】两部分都生效。
EXCLUDE_DIRS = [
    "*/node_modules/*", "*/.git/*", "*/dist/*", "*/build/*", "*/.vscode/*", "*/.idea/*",
    "*/__pycache__/*", "*/venv/*", "*/.nuxt/*", "*/diaries/*", "*/runs/*", "*/output*/*"
]

# 要排除的文件类型或文件。
EXCLUDE_FILES = [
    "*.log", "*.tmp", "*.lock", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.DS_Store", "*.sqlite3", "*.db", "*.png", "*.ico", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.pth", "*.npy", "tokenizer.json", "alphagenome_pytorch"
]

# --- END 配置区 ---

def main():
    # 检查 SOURCE_DIR 是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源目录 '{SOURCE_DIR}' 不存在。")
        sys.exit(1)

    # 清空或创建输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('')

    # --- 1. 生成项目结构图 ---
    generate_project_structure()

    # --- 2. 拼接代码内容 ---
    generate_code_content()

    print(f"Done! 代码已拼接至 '{OUTPUT_FILE}'")

def generate_project_structure():
    """生成项目结构的Markdown文档"""
    # 预处理要排除的目录名
    exclude_names_pattern = []
    for dir_pattern in EXCLUDE_DIRS:
        dir_name_part = re.sub(r'/\*$', '', dir_pattern)
        dir_name_part = os.path.basename(dir_name_part)
        if dir_name_part and dir_name_part != '*' and dir_name_part != '.':
            exclude_names_pattern.append(dir_name_part)
    
    exclude_names_regex = '|'.join(exclude_names_pattern) if exclude_names_pattern else ''

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 项目结构\n\n")
        f.write(f"项目`{SOURCE_DIR}`的目录结构（已排除如`node_modules`等目录）：\n")
        f.write("```\n")
        f.write(f"{SOURCE_DIR}/\n")
        
        # 生成目录树
        generate_tree(SOURCE_DIR, '', exclude_names_regex, f)
        
        f.write("```\n\n")
        f.write("---\n\n")

def generate_tree(dir_path, prefix, exclude_regex, file_obj):
    """递归生成目录树"""
    # 获取当前目录下一级的所有文件和目录，并排序
    items = sorted(os.listdir(dir_path))
    
    for i, item in enumerate(items):
        # 检查是否需要排除这个文件或目录
        if exclude_regex and re.search(exclude_regex, item):
            continue
            
        item_path = os.path.join(dir_path, item)
        
        # 判断连接符
        connector = "├── "
        new_prefix = "│   "
        if i == len(items) - 1:
            connector = "└── "
            new_prefix = "    "
        
        # 判断是目录还是文件，并输出
        if os.path.isdir(item_path):
            file_obj.write(f"{prefix}{connector}{item}/\n")
            # 递归调用
            generate_tree(item_path, prefix + new_prefix, exclude_regex, file_obj)
        else:
            file_obj.write(f"{prefix}{connector}{item}\n")

def generate_code_content():
    """生成代码内容的Markdown文档"""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 代码内容\n\n")
        
        # 构建文件搜索条件
        if INCLUDE_ONLY_PATTERNS:
            print(f"模式：仅拼接匹配 {' '.join(INCLUDE_ONLY_PATTERNS)} 的文件内容")
            include_patterns = INCLUDE_ONLY_PATTERNS
        else:
            print("模式：拼接所有文件，除了黑名单中的文件和目录")
            include_patterns = None
        
        # 遍历所有文件
        for root, dirs, files in os.walk(SOURCE_DIR):
            # 排除目录
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), EXCLUDE_DIRS)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # 排除文件
                if should_exclude(file_path, EXCLUDE_DIRS + [f"*/{pattern}" for pattern in EXCLUDE_FILES]):
                    continue
                
                # 检查是否在包含列表中
                if include_patterns and not any(fnmatch.fnmatch(file_path, pattern) for pattern in include_patterns):
                    continue
                
                # 写入文件内容
                relative_path = os.path.relpath(file_path, SOURCE_DIR)
                f.write(f"## 文件: `{relative_path}`\n\n")
                
                # 获取文件扩展名作为代码块语言
                file_ext = os.path.splitext(file_path)[1].lstrip('.')
                f.write(f"```{'python' if file_ext == 'py' else file_ext}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"无法读取文件: {file_path} ({str(e)})\n")
                
                f.write("\n```\n\n")

def should_exclude(path, patterns):
    """检查路径是否应被排除"""
    path_str = str(path)
    return any(re.search(pattern.replace('*', '.*'), path_str) for pattern in patterns)

if __name__ == "__main__":
    main()
```

