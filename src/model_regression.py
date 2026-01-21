"""
Regression model for predicting ligand-residue distance.
Based on BindingTransformerLite but modified for regression output.
"""

import torch
import torch.nn as nn


class DistanceRegressionModel(nn.Module):
    """
    Transformer-based model for distance regression.
    Predicts minimum distance (in Angstroms) between residue and ligand.
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
        
        # Regression head (no sigmoid, use ReLU for non-negative output)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # Ensure non-negative distance
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted distance of shape (batch_size, 1)
        """
        # Project input: (batch, input_dim) -> (batch, d_model)
        x = self.input_projection(x)
        
        # Add sequence dimension for transformer: (batch, 1, d_model)
        x = x.unsqueeze(1)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension: (batch, d_model)
        x = x.squeeze(1)
        
        # Regression output
        output = self.regressor(x)
        
        return output


class DistanceRegressionFCNN(nn.Module):
    """
    Simple FCNN for distance regression baseline.
    """
    
    def __init__(self, input_dim: int = 466):
        super().__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure non-negative distance
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # Quick test
    print("Testing DistanceRegressionModel...")
    model = DistanceRegressionModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(32, 466)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    print("\nTesting DistanceRegressionFCNN...")
    model_fcnn = DistanceRegressionFCNN()
    print(f"Model parameters: {sum(p.numel() for p in model_fcnn.parameters()):,}")
    
    y_fcnn = model_fcnn(x)
    print(f"Output shape: {y_fcnn.shape}")
    print(f"Output range: [{y_fcnn.min().item():.4f}, {y_fcnn.max().item():.4f}]")
    
    print("\nAll tests passed!")
