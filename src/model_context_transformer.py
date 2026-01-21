"""
Context-aware Transformer for binding site classification.

Takes a sequence of amino acid features (context window) and predicts
binding probability for the center residue using self-attention.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""
    
    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for short sequences."""
    
    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class ContextTransformer(nn.Module):
    """
    Transformer that takes a context window of amino acids and predicts
    binding for the center residue.
    
    Architecture:
        Input (B, seq_len, feature_dim)
            ↓
        Linear Projection → (B, seq_len, d_model)
            ↓
        Positional Encoding
            ↓
        Transformer Encoder Layers
            ↓
        Extract Center Token → (B, d_model)
            ↓
        Classification Head → (B, 1)
            ↓
        Sigmoid → Binding Probability
    """
    
    def __init__(
        self,
        feature_dim: int = 466,
        context_size: int = 5,  # Will compute window_size = 2*context_size + 1
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
        use_learnable_pos: bool = True,
    ):
        """
        Args:
            feature_dim: Dimension of input features per residue
            context_size: Number of residues on each side of center
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            use_learnable_pos: Use learnable positional encoding (vs sinusoidal)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.context_size = context_size
        self.window_size = 2 * context_size + 1
        self.d_model = d_model
        
        # Input projection: project each residue's features to d_model
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        if use_learnable_pos:
            self.pos_encoder = LearnablePositionalEncoding(
                d_model, max_len=self.window_size, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(
                d_model, max_len=self.window_size, dropout=dropout
            )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False,  # For compatibility
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, window_size, feature_dim)
            padding_mask: Optional mask for padded positions (True = ignore)
        
        Returns:
            Binding probability of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features: (B, seq_len, feature_dim) -> (B, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for padding if needed
        # The mask should be (B, seq_len) where True means ignore
        if padding_mask is None:
            # Detect zero-padded positions (all zeros in feature dim)
            # Sum along feature dim, if sum is 0 then it's padded
            # This is computed on the original input, but we need to track it
            pass  # We'll handle this in a simpler way below
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Extract center token (position = context_size)
        center_idx = self.context_size
        center_output = x[:, center_idx]  # (B, d_model)
        
        # Classification
        output = self.classifier(center_output)
        
        return output


class ContextTransformerWithCLS(nn.Module):
    """
    Alternative architecture using a CLS token for classification.
    
    Adds a learnable CLS token at the beginning, and uses its output
    for classification instead of the center token.
    """
    
    def __init__(
        self,
        feature_dim: int = 466,
        context_size: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.context_size = context_size
        self.window_size = 2 * context_size + 1
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding (window_size + 1 for CLS)
        self.pos_encoder = LearnablePositionalEncoding(
            d_model, max_len=self.window_size + 1, dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, window_size, feature_dim)
        
        Returns:
            Binding probability of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Project input features
        x = self.input_projection(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use CLS token output
        cls_output = x[:, 0]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


if __name__ == "__main__":
    # Quick test
    print("Testing ContextTransformer...")
    
    context_size = 5
    window_size = 2 * context_size + 1
    feature_dim = 466
    batch_size = 32
    
    model = ContextTransformer(
        feature_dim=feature_dim,
        context_size=context_size,
        d_model=256,
        nhead=8,
        num_layers=4,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, window_size, feature_dim)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    print("\nTesting ContextTransformerWithCLS...")
    
    model_cls = ContextTransformerWithCLS(
        feature_dim=feature_dim,
        context_size=context_size,
        d_model=256,
        nhead=8,
        num_layers=4,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
    
    y_cls = model_cls(x)
    print(f"Output shape: {y_cls.shape}")
    print(f"Output range: [{y_cls.min().item():.4f}, {y_cls.max().item():.4f}]")
    
    print("\nAll tests passed!")
