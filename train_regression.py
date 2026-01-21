"""
Training script for distance regression model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset_regression import get_regression_dataloaders
from src.model_regression import DistanceRegressionModel, DistanceRegressionFCNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import argparse
import numpy as np


def train():
    parser = argparse.ArgumentParser(description="Train Distance Regression Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"], help="Loss function")
    parser.add_argument("--fcnn", action="store_true", help="Use FCNN model instead of transformer")
    parser.add_argument("--log", action="store_true", help="Apply log(distance+1) transform to targets")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_regression"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, loss={args.loss}, log_transform={args.log}")

    # Load Data
    print("Loading data...")
    limit_files = 1 if args.debug else None
    train_loader, val_loader, test_loader = get_regression_dataloaders(
        DATA_DIR, 
        batch_size=args.batch_size, 
        limit_files=limit_files,
        log_transform=args.log
    )
    
    # Model
    if args.fcnn:
        print("Using DistanceRegressionFCNN")
        model = DistanceRegressionFCNN().to(DEVICE)
        model_save_path = "best_regression_fcnn.pth"
    else:
        print("Using DistanceRegressionModel (Transformer)")
        model = DistanceRegressionModel().to(DEVICE)
        model_save_path = "best_regression_transformer.pth"
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:  # huber
        criterion = nn.SmoothL1Loss()
    print(f"Using {args.loss.upper()} Loss")
    
    model_save_path = model_save_path.replace(".pth", f"_{args.loss}.pth")
    if args.log:
        model_save_path = model_save_path.replace(".pth", "_log.pth")
        
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training Loop
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training... Saving to {model_save_path}")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for features, distances in train_loader:
            features = features.to(DEVICE)
            distances = distances.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, distances)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(distances.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, distances in val_loader:
                features = features.to(DEVICE)
                distances = distances.to(DEVICE).unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, distances)
                
                val_loss += loss.item() * features.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(distances.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train MAE: {train_mae:.2f}Å | Train RMSE: {train_rmse:.2f}Å | "
              f"Val MAE: {val_mae:.2f}Å | Val RMSE: {val_rmse:.2f}Å | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save Best Model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'log_transform': args.log,
            }, model_save_path)
            print(f"  --> Saved new best model (Val MAE: {val_mae:.2f}Å)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
            
    print("=" * 70)
    print(f"Training complete. Best Val MAE: {best_val_mae:.2f}Å")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    train()
