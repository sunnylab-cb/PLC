"""
Distance-Aware Cross-Attention for Pocket-GVP-Fusion V6m6.

Implements cross-attention with distance bias to focus on spatially relevant regions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistanceBiasModule(nn.Module):
    """
    Computes distance-based attention bias.

    Converts pairwise distances into attention biases using an MLP.

    Args:
        n_heads: Number of attention heads
        hidden_dim: Hidden dimension for MLP
    """

    def __init__(self, n_heads: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads

        # MLP to convert distance to bias
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads)
        )

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute distance bias.

        Args:
            dist_matrix: [B, L_q, L_kv] pairwise distances

        Returns:
            dist_bias: [B, n_heads, L_q, L_kv] attention bias
        """
        B, L_q, L_kv = dist_matrix.shape

        # Reshape for MLP: [B, L_q, L_kv, 1]
        dist_input = dist_matrix.unsqueeze(-1)

        # Apply MLP: [B, L_q, L_kv, n_heads]
        dist_bias = self.distance_mlp(dist_input)

        # Rearrange to [B, n_heads, L_q, L_kv]
        dist_bias = dist_bias.permute(0, 3, 1, 2)

        return dist_bias


class DistanceAwareCrossAttention(nn.Module):
    """
    Cross-attention with distance-aware bias.

    Incorporates spatial distance information into attention scores,
    allowing the model to focus on spatially relevant regions.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_distance_bias: Whether to use distance bias
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_distance_bias: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_distance_bias = use_distance_bias

        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Distance bias module
        if use_distance_bias:
            self.distance_bias = DistanceBiasModule(n_heads=n_heads)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with distance-aware attention.

        Args:
            q: Query features [B, L_q, d_model]
            kv: Key-Value features [B, L_kv, d_model]
            dist_matrix: Pairwise distances [B, L_q, L_kv] (optional)
            q_mask: Query mask [B, L_q]
            kv_mask: Key-Value mask [B, L_kv]

        Returns:
            Updated query features [B, L_q, d_model]
        """
        B, L_q, _ = q.shape
        L_kv = kv.size(1)

        # Project Q, K, V
        Q = self.w_q(q).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, L_q, d_head]
        K = self.w_k(kv).view(B, L_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, L_kv, d_head]
        V = self.w_v(kv).view(B, L_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, L_kv, d_head]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, n_heads, L_q, L_kv]

        # Add distance bias
        if self.use_distance_bias and dist_matrix is not None:
            dist_bias = self.distance_bias(dist_matrix)  # [B, n_heads, L_q, L_kv]
            attn_scores = attn_scores + dist_bias

        # Apply masks
        if kv_mask is not None:
            # kv_mask: [B, L_kv] -> [B, 1, 1, L_kv]
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~kv_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, L_q, L_kv]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, n_heads, L_q, d_head]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # [B, L_q, d_model]

        # Output projection
        out = self.w_o(out)
        out = self.dropout(out)

        # Residual connection and layer norm
        out = self.norm(q + out)

        return out


class DistanceAwareBidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention with distance awareness.

    Implements mutual information exchange with spatial distance bias:
    - Protein queries Ligand with protein-ligand distances
    - Ligand queries Protein with ligand-protein distances

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_distance_bias: Whether to use distance bias
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_distance_bias: bool = True
    ):
        super().__init__()

        # Prot -> Lig: Protein queries Ligand
        self.prot_to_lig = DistanceAwareCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_distance_bias=use_distance_bias
        )

        # Lig -> Prot: Ligand queries Protein
        self.lig_to_prot = DistanceAwareCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_distance_bias=use_distance_bias
        )

    def forward(
        self,
        h_prot: torch.Tensor,
        h_lig: torch.Tensor,
        prot_pos: torch.Tensor,
        lig_pos: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with distance-aware bidirectional cross-attention.

        Args:
            h_prot: Protein features [B, L_prot, d_model]
            h_lig: Ligand features [B, L_lig, d_model]
            prot_pos: Protein positions [B, L_prot, 3]
            lig_pos: Ligand positions [B, L_lig, 3]
            prot_mask: Protein mask [B, L_prot]
            lig_mask: Ligand mask [B, L_lig]

        Returns:
            Tuple of (updated_protein_features, updated_ligand_features)
        """
        # Compute pairwise distance matrices
        # prot_lig_dist: [B, L_prot, L_lig]
        prot_lig_dist = torch.cdist(prot_pos, lig_pos, p=2)

        # lig_prot_dist: [B, L_lig, L_prot]
        lig_prot_dist = prot_lig_dist.transpose(1, 2)

        # Prot -> Lig: Protein queries Ligand with distance bias
        h_prot_updated = self.prot_to_lig(
            q=h_prot,
            kv=h_lig,
            dist_matrix=prot_lig_dist,
            q_mask=prot_mask,
            kv_mask=lig_mask
        )

        # Lig -> Prot: Ligand queries Protein with distance bias
        h_lig_updated = self.lig_to_prot(
            q=h_lig,
            kv=h_prot,
            dist_matrix=lig_prot_dist,
            q_mask=lig_mask,
            kv_mask=prot_mask
        )

        return h_prot_updated, h_lig_updated
