"""
Training script for Context-aware Transformer binding site classifier.

This model takes a window of amino acids (context) and predicts
binding for the center residue.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset_context import get_context_dataloaders
from src.model_context_transformer import ContextTransformer, ContextTransformerWithCLS
from src.loss import FocalLoss
from sklearn.metrics import roc_auc_score
import time
import argparse
import numpy as np


def train():
    parser = argparse.ArgumentParser(description="Train Context-aware Transformer for binding prediction")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--context_size", type=int, default=5, help="Context size (residues on each side)")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit number of data files (for debugging)")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of center token")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Configuration:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  context_size={args.context_size} (window={2*args.context_size+1})")
    print(f"  d_model={args.d_model}, nhead={args.nhead}, num_layers={args.num_layers}")

    # Load Data
    print("\nLoading data with context windows...")
    train_loader, val_loader, test_loader = get_context_dataloaders(
        DATA_DIR, 
        context_size=args.context_size,
        batch_size=args.batch_size, 
        limit_files=args.limit_files,
    )
    
    # Model
    if args.use_cls:
        print("\nUsing ContextTransformerWithCLS (CLS token for classification)")
        model = ContextTransformerWithCLS(
            feature_dim=466,
            context_size=args.context_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(DEVICE)
        model_name = "context_transformer_cls"
    else:
        print("\nUsing ContextTransformer (center token for classification)")
        model = ContextTransformer(
            feature_dim=466,
            context_size=args.context_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_model * 2,
            dropout=args.dropout,
        ).to(DEVICE)
        model_name = "context_transformer"
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and Optimizer
    if args.focal:
        print("Using Focal Loss (alpha=0.9, gamma=2.0)")
        criterion = FocalLoss(alpha=0.9, gamma=2.0)
        model_save_path = f"best_{model_name}_focal.pth"
    else:
        print("Using BCE Loss")
        criterion = nn.BCELoss()
        model_save_path = f"best_{model_name}.pth"
        
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
    print("=" * 80)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (windows, labels) in enumerate(train_loader):
            windows = windows.to(DEVICE)  # (B, window_size, feature_dim)
            labels = labels.to(DEVICE).unsqueeze(1)  # (B, 1)
            
            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * windows.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')
            
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
            for windows, labels in val_loader:
                windows = windows.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                outputs = model(windows)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * windows.size(0)
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
                'config': {
                    'context_size': args.context_size,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'use_cls': args.use_cls,
                }
            }, model_save_path)
            print(f"  --> Saved new best model (Val AUC: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
            
    print("=" * 80)
    print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    train()
