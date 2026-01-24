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
