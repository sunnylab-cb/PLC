"""
Transformer layers for Pocket-GVP-Fusion v2.

Implements:
- SwiGLU FFN (LLaMA-style)
- Gated Attention (Qwen-style)
- GatedTransformerBlock combining both
- Positional Encoding (sinusoidal and learnable)
- Flash Attention support (V9m2)
- Inter-Block Attention Residuals (AttnRes, Kimi 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# V9m2: Flash Attention support (triton version for ROCm)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Stochastic Depth: randomly drop residual paths during training.
    Each sample in the batch is independently dropped with probability drop_prob.
    Dropped paths are scaled by 1/(1-drop_prob) to maintain expected value.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Shape: [B, 1, 1, ...] — broadcast over all dimensions except batch
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    rand_tensor = torch.floor(rand_tensor + keep_prob)  # 0 or 1
    return x / keep_prob * rand_tensor


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
        use_gate: bool = True,
        attn_dropout: Optional[float] = None,   # V9: separate attn weight dropout (default=dropout)
        use_flash_attn: bool = False  # V9m2: enable Flash Attention
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_gate = use_gate
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE  # V9m2

        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Gating mechanism (Qwen-style)
        if use_gate:
            self.gate = nn.Linear(d_model, d_model, bias=False)

        # V9: separate dropout for attention weights vs output
        _attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.attn_dropout_p = _attn_dropout  # V9m2: store dropout prob for flash_attn
        self.attn_dropout = nn.Dropout(_attn_dropout)
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
        q = self.w_q(x)  # [B, L, d_model]
        k = self.w_k(kv)  # [B, S, d_model]
        v = self.w_v(kv)  # [B, S, d_model]

        # V9m2: Use Flash Attention if enabled and available
        if self.use_flash_attn and self.training:
            # Flash Attention requires [B, L, H, D] format
            q = q.view(B, L, self.n_heads, self.d_head)
            k = k.view(B, S, self.n_heads, self.d_head)
            v = v.view(B, S, self.n_heads, self.d_head)

            # Flash Attention doesn't support arbitrary masks well, only causal
            # For now, use flash_attn only for self-attention without mask
            if mask is None and kv is None:
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attn_dropout_p if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=False
                )  # [B, L, H, D]
                out = out.reshape(B, L, self.d_model)
            else:
                # Fallback to standard attention if mask is present
                q = q.transpose(1, 2)  # [B, H, L, D]
                k = k.transpose(1, 2)  # [B, H, S, D]
                v = v.transpose(1, 2)  # [B, H, S, D]
                out = self._standard_attention(q, k, v, mask, B, L)
        else:
            # Standard attention
            q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D]
            k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, D]
            v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, D]
            out = self._standard_attention(q, k, v, mask, B, L)

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

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        B: int,
        L: int
    ) -> torch.Tensor:
        """Standard scaled dot-product attention"""
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

        # V9: use separate attn_dropout for attention weights
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]

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
        use_swiglu: bool = True,
        drop_path_prob: float = 0.0,    # V9: stochastic depth probability
        attn_dropout: Optional[float] = None,  # V9: separate attn weight dropout
        use_flash_attn: bool = False  # V9m2: enable Flash Attention
    ):
        super().__init__()

        self.drop_path_prob = drop_path_prob

        # Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_gate=use_gated_attn,
            attn_dropout=attn_dropout,  # V9
            use_flash_attn=use_flash_attn  # V9m2
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
        # Self-attention with residual + stochastic depth
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = drop_path(self.dropout(x), self.drop_path_prob, self.training) + residual

        # FFN with residual + stochastic depth
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = drop_path(self.dropout(x), self.drop_path_prob, self.training) + residual
        
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
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,    # V9: stochastic depth
        attn_dropout: Optional[float] = None  # V9: separate attn weight dropout
    ):
        super().__init__()

        self.drop_path_prob = drop_path_prob

        # Cross-attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = GatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_gate=True,
            attn_dropout=attn_dropout  # V9
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
        # Cross-attention with residual + stochastic depth
        residual = q
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)

        # Build cross-attention mask
        cross_mask = None
        if kv_mask is not None:
            cross_mask = kv_mask  # [B, L_kv]

        out = self.cross_attn(q_normed, mask=cross_mask, kv=kv_normed)
        out = drop_path(self.dropout(out), self.drop_path_prob, self.training) + residual

        # V6m9: Apply q_mask to output to reduce padding noise propagation
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).float()  # [B, L_q, d_model]

        # FFN with residual + stochastic depth
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = drop_path(self.dropout(out), self.drop_path_prob, self.training) + residual

        # V6m9: Apply q_mask again after FFN
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).float()

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


class BidirectionalCrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block for protein-ligand co-attention.

    Implements mutual information exchange:
    - Protein queries Ligand (Prot -> Lig): Updates protein features
    - Ligand queries Protein (Lig -> Prot): Updates ligand features

    This allows both modalities to be enhanced by each other, modeling
    the "induced fit" phenomenon where both protein and ligand adapt
    to each other's presence.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,    # V9: stochastic depth
        attn_dropout: Optional[float] = None  # V9: separate attn weight dropout
    ):
        super().__init__()

        # Prot -> Lig: Protein queries Ligand
        self.prot_to_lig = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            drop_path_prob=drop_path_prob,
            attn_dropout=attn_dropout
        )

        # Lig -> Prot: Ligand queries Protein
        self.lig_to_prot = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            drop_path_prob=drop_path_prob,
            attn_dropout=attn_dropout
        )

    def forward(
        self,
        h_prot: torch.Tensor,
        h_lig: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with bidirectional cross-attention.

        Args:
            h_prot: Protein features [B, L_prot, d_model]
            h_lig: Ligand features [B, L_lig, d_model]
            prot_mask: Protein mask [B, L_prot]
            lig_mask: Ligand mask [B, L_lig]

        Returns:
            Tuple of (updated_protein_features, updated_ligand_features)
        """
        # Prot -> Lig: Protein queries Ligand, updates protein
        h_prot_updated = self.prot_to_lig(
            q=h_prot,
            kv=h_lig,
            q_mask=prot_mask,
            kv_mask=lig_mask
        )

        # Lig -> Prot: Ligand queries Protein, updates ligand
        h_lig_updated = self.lig_to_prot(
            q=h_lig,
            kv=h_prot,
            q_mask=lig_mask,
            kv_mask=prot_mask
        )

        return h_prot_updated, h_lig_updated


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
        pe_type: str = 'sinusoidal',
        stochastic_depth_prob: float = 0.0,  # V9: max drop path prob (linearly scaled per layer)
        attn_dropout: Optional[float] = None,  # V9: separate attn weight dropout
        use_flash_attn: bool = False  # V9m2: enable Flash Attention
    ):
        super().__init__()

        # Positional encoding
        if pe_type == 'sinusoidal':
            self.pe = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pe = LearnablePositionalEncoding(d_model, max_len, dropout)

        # V9: linearly increase drop_path_prob from 0 to stochastic_depth_prob across layers
        # (DeiT/Swin convention: first layer gets 0, last gets max)
        drop_path_probs = [
            stochastic_depth_prob * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]

        # Transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_gated_attn=True,
                use_swiglu=True,
                drop_path_prob=drop_path_probs[i],
                attn_dropout=attn_dropout,
                use_flash_attn=use_flash_attn  # V9m2
            )
            for i in range(n_layers)
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


class InterBlockAttnRes(nn.Module):
    """
    Inter-Block Attention Residuals (AttnRes), Kimi 2025.

    Replaces the fixed-weight residual sum between blocks with a learnable,
    input-dependent softmax attention over completed block representations.

    Given completed block representations b_0, b_1, ..., b_{N-1} and the
    current partial sum h, computes:

        logits_i = w · RMSNorm(b_i)          (scalar per position)
        alpha    = softmax([logits_0, ..., logits_{N-1}, logit_h])
        output   = sum_i alpha_i * b_i  +  alpha_h * h

    The pseudo-query vector w is initialised to zero so that at the start of
    training all block representations (including the partial sum) are weighted
    equally (uniform softmax), matching the behaviour of a regular mean pool
    and avoiding early instability.

    Args:
        d_model: Feature dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Pseudo-query: scalar projection w ∈ R^d, init=0 (critical)
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        # RMSNorm on keys to prevent magnitude-dominant blocks hijacking softmax
        self.key_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        blocks: list,           # list of [B, L, D] completed block tensors
        partial: torch.Tensor,  # [B, L, D] current partial block sum
    ) -> torch.Tensor:
        """
        Args:
            blocks:  List of N tensors [B, L, D] — completed block representations.
            partial: Tensor [B, L, D] — intra-block partial sum up to current layer.

        Returns:
            Tensor [B, L, D] — attended combination of all sources.
        """
        # Stack all sources: shape [N+1, B, L, D]
        sources = torch.stack(blocks + [partial], dim=0)

        # Apply RMSNorm to keys (along D) — prevents large-magnitude blocks from
        # monopolising the softmax probability mass
        keys = self.key_norm(sources)  # [N+1, B, L, D]

        # Compute scalar logit per source per position:
        #   logit_i[b, l] = w · RMSNorm(sources[i, b, l])
        # proj maps D -> 1, so logits shape: [N+1, B, L, 1] -> squeeze -> [N+1, B, L]
        logits = self.proj(keys).squeeze(-1)  # [N+1, B, L]

        # Softmax over the source dimension
        alpha = logits.softmax(dim=0)  # [N+1, B, L]

        # Weighted sum: alpha[n, b, l] * sources[n, b, l, d] -> [B, L, D]
        out = (alpha.unsqueeze(-1) * sources).sum(dim=0)

        return out

