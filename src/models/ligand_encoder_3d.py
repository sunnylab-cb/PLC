"""
3D Ligand Encoder for Pocket-GVP-Fusion V6m6.

Implements GVP-based encoder for 3D ligand structures with spatial coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from typing import Optional, Tuple

from .gvp_layers import GVP, GVPConv


class Ligand3DEncoder(nn.Module):
    """
    GVP-based encoder for 3D ligand molecules.

    Encodes ligand with 3D coordinates using Geometric Vector Perceptrons,
    capturing both scalar and vector (directional) features.

    Args:
        node_in_dim: Input node scalar feature dimension
        node_h_dim: Hidden node scalar dimension
        n_vec_channels: Number of vector channels
        n_vec_hidden: Hidden vector dimension
        edge_in_dim: Input edge feature dimension
        out_dim: Output dimension
        n_layers: Number of GVP layers
        dropout: Dropout rate
        radius: Radius for edge construction (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
    """

    def __init__(
        self,
        node_in_dim: int = 80,  # V3 ligand features
        node_h_dim: int = 128,
        n_vec_channels: int = 4,
        n_vec_hidden: int = 16,
        edge_in_dim: int = 6,
        out_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        radius: float = 5.0,
        max_neighbors: int = 32
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.out_dim = out_dim
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.n_vec_channels = n_vec_channels

        # Initial node projection (scalar only)
        self.node_proj = nn.Linear(node_in_dim, node_h_dim)

        # Edge projection
        self.edge_proj = nn.Linear(edge_in_dim, 32)

        # GVP layers
        self.gvp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dims = (node_h_dim, n_vec_channels)
            out_dims = (node_h_dim, n_vec_channels)
            edge_dims = (32, 1)  # Edge scalar + 1 vector (direction)

            self.gvp_layers.append(
                GVPConv(
                    in_dims=in_dims,
                    out_dims=out_dims,
                    edge_dims=edge_dims,
                    n_layers=3,
                    dropout=dropout
                )
            )

        # Output projection (scalar only)
        self.out_proj = nn.Sequential(
            nn.Linear(node_h_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def _build_edges(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on 3D coordinates.

        Args:
            pos: Atom coordinates [N, 3]
            batch: Batch assignment [N]

        Returns:
            edge_index: [2, E]
            edge_vectors: [E, 3] normalized direction vectors
        """
        # Use radius graph for spatial connectivity
        edge_index = radius_graph(
            pos,
            r=self.radius,
            batch=batch,
            max_num_neighbors=self.max_neighbors,
            loop=False
        )

        # Compute edge vectors
        row, col = edge_index
        edge_vectors = pos[col] - pos[row]  # [E, 3]

        # Normalize edge vectors
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)  # [E, 1]
        edge_vectors_normalized = edge_vectors / (edge_lengths + 1e-8)  # [E, 3]

        return edge_index, edge_vectors_normalized

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node scalar features [N, node_in_dim]
            pos: 3D coordinates [N, 3]
            edge_index: Optional pre-computed edge indices [2, E]
            edge_attr: Optional edge features [E, edge_in_dim]
            batch: Batch assignment [N]

        Returns:
            h_s: Node scalar embeddings [N, out_dim]
            pos: Original positions [N, 3] (for distance computation)
        """
        N = x.size(0)
        device = x.device

        # Build spatial edges if not provided
        if edge_index is None:
            edge_index, edge_vectors = self._build_edges(pos, batch)
        else:
            # Compute edge vectors from provided edge_index
            row, col = edge_index
            edge_vectors = pos[col] - pos[row]
            edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)
            edge_vectors = edge_vectors / (edge_lengths + 1e-8)

        # Initial node projection (scalar)
        h_s = self.node_proj(x)  # [N, node_h_dim]

        # Initialize vector features (zero vectors)
        h_v = torch.zeros(N, self.n_vec_channels, 3, device=device)  # [N, n_vec, 3]

        # Edge features
        if edge_attr is not None:
            edge_s = self.edge_proj(edge_attr)  # [E, 32]
        else:
            # Use distance-based edge features
            row, col = edge_index
            distances = torch.norm(pos[col] - pos[row], dim=-1, keepdim=True)  # [E, 1]
            edge_s = torch.cat([
                distances,
                torch.sin(distances),
                torch.cos(distances),
                1.0 / (distances + 1.0)
            ], dim=-1)  # [E, 4]
            edge_s = F.pad(edge_s, (0, 28))  # Pad to 32 dims

        # Edge vectors as 1-channel vector feature
        edge_v = edge_vectors.unsqueeze(1)  # [E, 1, 3]

        # Apply GVP layers
        for gvp_layer in self.gvp_layers:
            h_s, h_v = gvp_layer(
                x_s=h_s,
                x_v=h_v,
                edge_index=edge_index,
                edge_s=edge_s,
                edge_v=edge_v
            )

        # Output projection (scalar only)
        h_out = self.out_proj(h_s)  # [N, out_dim]

        return h_out, pos
