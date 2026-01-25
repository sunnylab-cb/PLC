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
