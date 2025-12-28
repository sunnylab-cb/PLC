"""
Transformer-based binding site classifier.

Treats the 466-dimensional feature vector as a sequence of feature groups
and applies self-attention to learn which feature interactions are important
for binding site prediction.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for feature position awareness."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
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


class BindingTransformer(nn.Module):
    """
    Transformer encoder for binding site classification.
    
    Architecture:
        Input (466 dims) → Reshape to patches → Linear projection → 
        Positional Encoding → Transformer Encoder → 
        Global Average Pooling → Classification Head → Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int = 466,
        patch_size: int = 20,  # Each patch covers ~20 features
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate number of patches (round up to cover all features)
        self.num_patches = math.ceil(input_dim / patch_size)
        # Pad input to be divisible by patch_size
        self.padded_dim = self.num_patches * patch_size
        
        # Input embedding: project each patch to d_model
        self.input_projection = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches, dropout=dropout)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Binding probability of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Pad input if necessary
        if self.padded_dim > self.input_dim:
            padding = torch.zeros(batch_size, self.padded_dim - self.input_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape to patches: (batch, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Project patches to d_model: (batch, num_patches, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use CLS token output for classification
        cls_output = x[:, 0]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


class BindingTransformerLite(nn.Module):
    """
    Lighter version of transformer for faster training.
    Uses fewer layers and smaller dimensions.
    """
    
    def __init__(
        self,
        input_dim: int = 466,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Direct projection from input dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Binding probability of shape (batch_size, 1)
        """
        # Project input: (batch, input_dim) -> (batch, d_model)
        x = self.input_projection(x)
        
        # Add sequence dimension for transformer: (batch, 1, d_model)
        x = x.unsqueeze(1)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension: (batch, d_model)
        x = x.squeeze(1)
        
        # Classification
        output = self.classifier(x)
        
        return output


if __name__ == "__main__":
    # Quick test
    print("Testing BindingTransformer...")
    model = BindingTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(32, 466)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    print("\nTesting BindingTransformerLite...")
    model_lite = BindingTransformerLite()
    print(f"Model parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
    
    y_lite = model_lite(x)
    print(f"Output shape: {y_lite.shape}")
    print(f"Output range: [{y_lite.min().item():.4f}, {y_lite.max().item():.4f}]")
    
    print("\nAll tests passed!")
