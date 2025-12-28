"""
Training script for Transformer-based binding site classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset import get_dataloaders
from src.model_transformer import BindingTransformer, BindingTransformerLite
from src.loss import FocalLoss
from sklearn.metrics import roc_auc_score
import time
import argparse
import numpy as np


def train():
    parser = argparse.ArgumentParser(description="Train Transformer for binding prediction")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    parser.add_argument("--lite", action="store_true", help="Use lighter transformer model")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Load Data
    print("Loading data...")
    limit_files = 1 if args.debug else None
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, 
        batch_size=args.batch_size, 
        limit_files=limit_files
    )
    
    # Model
    if args.lite:
        print("Using BindingTransformerLite")
        model = BindingTransformerLite().to(DEVICE)
        model_save_path = "best_transformer_lite.pth"
    else:
        print("Using BindingTransformer")
        model = BindingTransformer().to(DEVICE)
        model_save_path = "best_transformer.pth"
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and Optimizer
    if args.focal:
        print("Using Focal Loss (alpha=0.9, gamma=2.0)")
        criterion = FocalLoss(alpha=0.9, gamma=2.0)
        model_save_path = model_save_path.replace(".pth", "_focal.pth")
    else:
        print("Using BCE Loss")
        criterion = nn.BCELoss()
        
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training Loop
    best_val_auc = 0.0
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
        
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save Best Model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
            }, model_save_path)
            print(f"  --> Saved new best model (Val AUC: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
            
    print("=" * 70)
    print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    train()
