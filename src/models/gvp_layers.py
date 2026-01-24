"""
GVP (Geometric Vector Perceptron) layers for Pocket-GVP-Fusion v2.

Implements GVPConv with vector gating (v2 style) for encoding local 3D geometry.

Reference:
    Jing et al. "Learning from Protein Structure with Geometric Vector Perceptrons"
    https://arxiv.org/abs/2009.01411
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from typing import Tuple, Optional


class GVP(nn.Module):
    """
    Geometric Vector Perceptron - processes scalar and vector features.
    
    Args:
        in_dims: Tuple of (scalar_in, vector_in) dimensions
        out_dims: Tuple of (scalar_out, vector_out) dimensions
        activations: Tuple of (scalar_activation, vector_activation)
        vector_gate: Whether to use vector gating (v2 style)
    """
    
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations: Tuple[Optional[nn.Module], Optional[nn.Module]] = (F.relu, None),
        vector_gate: bool = True
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        self.vector_gate = vector_gate
        
        # Scalar pathway
        if self.vi > 0:
            # Include vector norms in scalar processing
            self.scalar_linear = nn.Linear(self.si + self.vi, self.so)
        else:
            self.scalar_linear = nn.Linear(self.si, self.so)
        
        # Vector pathway
        if self.vi > 0 and self.vo > 0:
            self.vector_linear = nn.Linear(self.vi, self.vo, bias=False)
            if self.vector_gate:
                # Gate for vector output (v2 style)
                self.gate_linear = nn.Linear(self.si + self.vi, self.vo)
        
    def forward(
        self, 
        s: torch.Tensor, 
        v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            s: Scalar features [*, si]
            v: Vector features [*, vi, 3]
            
        Returns:
            Tuple of (scalar_out [*, so], vector_out [*, vo, 3])
        """
        if v is not None and self.vi > 0:
            # Compute vector norms
            v_norm = torch.norm(v, dim=-1)  # [*, vi]
            
            # Scalar pathway: concat scalar + vector norms
            s_combined = torch.cat([s, v_norm], dim=-1)
            s_out = self.scalar_linear(s_combined)
        else:
            s_out = self.scalar_linear(s)
        
        # Apply scalar activation
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        
        # Vector pathway
        v_out = None
        if self.vo > 0 and v is not None and self.vi > 0:
            # Linear transform on vectors
            # v: [*, vi, 3] -> transpose to [*, 3, vi] for linear
            v_t = v.transpose(-2, -1)  # [*, 3, vi]
            v_t = self.vector_linear(v_t)  # [*, 3, vo]
            v_out = v_t.transpose(-2, -1)  # [*, vo, 3]
            
            # Apply vector gate if enabled
            if self.vector_gate:
                v_norm = torch.norm(v, dim=-1)  # [*, vi]
                s_combined = torch.cat([s, v_norm], dim=-1)
                gate = torch.sigmoid(self.gate_linear(s_combined))  # [*, vo]
                v_out = v_out * gate.unsqueeze(-1)  # [*, vo, 3]
            
            # Apply vector activation (typically just normalization)
            if self.vector_act is not None:
                v_out = self.vector_act(v_out)
        
        return s_out, v_out


class GVPConv(MessagePassing):
    """
    GVP-based message passing layer for graph neural networks.
    
    Operates on graphs with node scalar/vector features and edge features.
    
    Args:
        in_dims: Tuple of (scalar_in, vector_in) dimensions
        out_dims: Tuple of (scalar_out, vector_out) dimensions
        edge_dims: Tuple of (edge_scalar, edge_vector) dimensions
        n_layers: Number of GVP layers in message/update functions
        vector_gate: Whether to use vector gating
        dropout: Dropout rate
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
        super().__init__(aggr='mean')
        
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        # Message function: source node + edge -> message
        message_in = (self.si + self.se, self.vi + self.ve)
        message_hidden = (self.so, self.vo)
        
        self.message_layers = nn.ModuleList()
        curr_dims = message_in
        for i in range(n_layers):
            next_dims = message_hidden
            self.message_layers.append(
                GVP(curr_dims, next_dims, vector_gate=vector_gate)
            )
            curr_dims = next_dims
        
        # Update function: node + aggregated message -> output
        update_in = (self.si + self.so, self.vi + self.vo)
        
        self.update_layers = nn.ModuleList()
        curr_dims = update_in
        for i in range(n_layers):
            next_dims = out_dims if i == n_layers - 1 else message_hidden
            self.update_layers.append(
                GVP(curr_dims, next_dims, vector_gate=vector_gate)
            )
            curr_dims = next_dims
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if dimensions match)
        self.residual = (self.si == self.so and self.vi == self.vo)
        
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
            edge_s: Edge scalar features [E, se]
            edge_v: Edge vector features [E, ve, 3]
            
        Returns:
            Tuple of (node_s_out [N, so], node_v_out [N, vo, 3])
        """
        # Store input for residual
        if self.residual:
            s_res, v_res = node_s, node_v
        
        # Propagate messages
        s_out, v_out = self.propagate(
            edge_index,
            s=node_s,
            v=node_v,
            edge_s=edge_s,
            edge_v=edge_v
        )
        
        # Apply residual connection
        if self.residual:
            s_out = s_out + s_res
            v_out = v_out + v_res
        
        return s_out, v_out
    
    def message(
        self,
        s_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute messages from source nodes."""
        # Concatenate source node features with edge features
        s = torch.cat([s_j, edge_s], dim=-1)
        v = torch.cat([v_j, edge_v], dim=-2)
        
        # Apply message layers
        for layer in self.message_layers:
            s, v = layer(s, v)
            s = self.dropout(s)
        
        return s, v
    
    def aggregate(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate messages."""
        s, v = inputs
        
        # Mean aggregation for scalars
        s_agg = torch.zeros(dim_size, s.size(-1), device=s.device, dtype=s.dtype)
        s_agg.scatter_reduce_(0, index.unsqueeze(-1).expand_as(s), s, reduce='mean')
        
        # Mean aggregation for vectors
        v_agg = torch.zeros(dim_size, v.size(-2), 3, device=v.device, dtype=v.dtype)
        index_v = index.unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v_agg.scatter_reduce_(0, index_v, v, reduce='mean')
        
        return s_agg, v_agg
    
    def update(
        self,
        aggr_out: Tuple[torch.Tensor, torch.Tensor],
        s: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update node features with aggregated messages."""
        s_agg, v_agg = aggr_out
        
        # Concatenate node features with aggregated messages
        s = torch.cat([s, s_agg], dim=-1)
        v = torch.cat([v, v_agg], dim=-2)
        
        # Apply update layers
        for layer in self.update_layers:
            s, v = layer(s, v)
            s = self.dropout(s)
        
        return s, v


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
    from torch_geometric.nn import knn_graph, radius_graph
    
    # Build k-NN graph
    if batch is not None:
        edge_index = knn_graph(coords, k=k, batch=batch, loop=False)
    else:
        edge_index = knn_graph(coords, k=k, loop=False)
    
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
