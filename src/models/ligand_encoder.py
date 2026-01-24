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
