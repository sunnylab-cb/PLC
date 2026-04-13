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


def rbf_encode(
    dist: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    num_rbf: int = 16,
) -> torch.Tensor:
    """
    Radial Basis Function (RBF) encoding of distances.

    Each distance is mapped to a soft-assignment over num_rbf Gaussian
    centres uniformly spaced in [D_min, D_max].  Much richer than raw
    distance and lets the model learn smooth distance-dependent functions.

    Args:
        dist:    [E] distances (or any shape, last dim is NOT feature)
        D_min:   left edge of RBF range (Å)
        D_max:   right edge of RBF range (Å)
        num_rbf: number of RBF centres

    Returns:
        [E, num_rbf] encoded distances
    """
    D_mu    = torch.linspace(D_min, D_max, num_rbf, device=dist.device, dtype=dist.dtype)
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-((dist.unsqueeze(-1) - D_mu) ** 2) / (2.0 * D_sigma ** 2))


def build_protein_graph(
    coords_ca: torch.Tensor,
    coords_n:  torch.Tensor,
    coords_c:  torch.Tensor,
    k: int = 20,
    radius: float = 10.0,
    num_rbf: int = 16,
    rbf_dmax: float = 20.0,
    chain_ids: Optional[torch.Tensor] = None,   # [N] integer chain indices
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build protein residue graph with chemically-informed edge features.

    Improvements over build_knn_graph:
    1. Explicit backbone edges (i±1) are ALWAYS included (not subject to
       KNN filtering), ensuring peptide-bond connectivity survives coord noise.
    2. RBF distance encoding (num_rbf bins) instead of raw scalar distance.
    3. Sequential offset 4-class encoding — distinguishes backbone vs
       short-range vs medium-range vs long-range contacts.
    4. Local backbone reference frame (N-CA-C) as a second edge vector,
       letting GVP learn frame-relative directions (e.g. hydrogen-bond geometry).
    5. is_same_chain binary flag (1 = same chain, 0 = cross-chain).
       Only appended when chain_ids is provided; edge_s dim = num_rbf+4+1.

    Edge scalar features:
        chain_ids=None : [E, num_rbf + 4]
        chain_ids given: [E, num_rbf + 5]  (last dim = is_same_chain)

    Edge vector features [E, 2, 3]:
        [:, 0, :] = global CA_src→CA_dst unit direction
        [:, 1, :] = same direction in the N-CA-C local frame of source residue

    Args:
        coords_ca:  [N, 3]  C-alpha coordinates
        coords_n:   [N, 3]  backbone N coordinates
        coords_c:   [N, 3]  backbone C (carbonyl) coordinates
        k:          KNN neighbours (spatial proximity graph)
        radius:     maximum edge distance for KNN edges (Å)
        num_rbf:    number of RBF bins
        rbf_dmax:   max distance for RBF range (Å)
        chain_ids:  [N] integer chain indices (optional)

    Returns:
        edge_index: [2, E]
        edge_s:     [E, num_rbf + 4] or [E, num_rbf + 5]
        edge_v:     [E, 2, 3]
    """
    N      = coords_ca.size(0)
    device = coords_ca.device
    dtype  = coords_ca.dtype

    edge_s_dim = num_rbf + 4 + (1 if chain_ids is not None else 0)

    if N == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0, edge_s_dim), dtype=dtype, device=device),
            torch.zeros((0, 2, 3), dtype=dtype, device=device),
        )

    # ── 1. KNN spatial graph ─────────────────────────────────────────────────
    from torch_geometric.nn import knn_graph
    curr_k = min(k, N - 1)
    try:
        knn_ei = knn_graph(coords_ca, k=curr_k, loop=False)
    except Exception:
        dist_mat = torch.cdist(coords_ca, coords_ca)
        dist_mat.fill_diagonal_(float('inf'))
        _, nn_idx = dist_mat.topk(curr_k, dim=1, largest=False)  # [N, k]
        src_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, curr_k).reshape(-1)
        knn_ei  = torch.stack([src_idx, nn_idx.reshape(-1)], dim=0)

    # ── 2. Explicit backbone edges (always i→i±1) ────────────────────────────
    if N > 1:
        seq = torch.arange(N, device=device)
        bb_src = torch.cat([seq[:-1], seq[1:]])   # forward + backward
        bb_dst = torch.cat([seq[1:],  seq[:-1]])
        bb_ei  = torch.stack([bb_src, bb_dst], dim=0)
    else:
        bb_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

    # ── 3. Merge & deduplicate via unique edge keys ──────────────────────────
    combined = torch.cat([knn_ei, bb_ei], dim=1)          # [2, E_knn + E_bb]
    edge_key = combined[0] * N + combined[1]              # unique int per (src,dst)
    unique_keys, _ = torch.unique(edge_key, sorted=False, return_inverse=True)
    src = unique_keys // N
    dst = unique_keys %  N

    # ── 4. Radius filter (backbone edges bypass radius) ──────────────────────
    edge_vec_raw = coords_ca[dst] - coords_ca[src]         # [E, 3]
    dist_raw     = torch.norm(edge_vec_raw, dim=-1)        # [E]
    seq_offset   = (dst - src).abs()                       # [E]
    keep         = (dist_raw <= radius) | (seq_offset == 1)
    src, dst     = src[keep], dst[keep]
    edge_vec_raw = edge_vec_raw[keep]
    dist_raw     = dist_raw[keep]
    seq_offset   = seq_offset[keep]

    # ── 5. Edge scalar features ──────────────────────────────────────────────
    rbf          = rbf_encode(dist_raw, D_min=0.0, D_max=rbf_dmax, num_rbf=num_rbf)
    is_backbone  = (seq_offset == 1).float().unsqueeze(-1)
    is_short     = ((seq_offset >= 2) & (seq_offset <= 5)).float().unsqueeze(-1)
    is_medium    = ((seq_offset >= 6) & (seq_offset <= 12)).float().unsqueeze(-1)
    is_long      = (seq_offset > 12).float().unsqueeze(-1)
    edge_s_parts = [rbf, is_backbone, is_short, is_medium, is_long]

    # is_same_chain: 1.0 if src and dst belong to the same chain, else 0.0
    if chain_ids is not None:
        same_chain = (chain_ids[src] == chain_ids[dst]).float().unsqueeze(-1)  # [E, 1]
        edge_s_parts.append(same_chain)

    edge_s = torch.cat(edge_s_parts, dim=-1)

    # ── 6. Backbone local reference frame (N-CA-C) ───────────────────────────
    # forward direction: CA → C  (peptide extension)
    fwd   = F.normalize(coords_c  - coords_ca, dim=-1)    # [N, 3]
    # N → CA
    n2ca  = F.normalize(coords_ca - coords_n,  dim=-1)    # [N, 3]
    # up: normal of N-CA-C plane, perpendicular to fwd
    up    = F.normalize(torch.cross(fwd, n2ca, dim=-1), dim=-1)  # [N, 3]
    # right: completes right-hand frame
    right = torch.cross(fwd, up, dim=-1)                  # [N, 3]
    # frame columns = [fwd, up, right] → 3×3 rotation for each residue
    frame = torch.stack([fwd, up, right], dim=-1)         # [N, 3, 3]

    # ── 7. Edge vector features ──────────────────────────────────────────────
    global_dir = F.normalize(edge_vec_raw, dim=-1)        # [E, 3] global direction
    frame_src  = frame[src]                               # [E, 3, 3]
    # Project global direction into local frame of source residue
    local_dir  = torch.bmm(
        frame_src.transpose(1, 2),                        # [E, 3, 3]
        global_dir.unsqueeze(-1)                          # [E, 3, 1]
    ).squeeze(-1)                                         # [E, 3]

    edge_index = torch.stack([src, dst], dim=0)           # [2, E]
    edge_v     = torch.stack([global_dir, local_dir], dim=1)  # [E, 2, 3]

    return edge_index, edge_s, edge_v


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
