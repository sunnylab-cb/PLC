import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import BindingFCNN
from src.loss import FocalLoss
import time
import os

import argparse

def train():
    parser = argparse.ArgumentParser(description="Train FCNN for binding prediction")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = 0.001
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load Data
    print("Loading data...")
    limit_files = 1 if args.debug else None
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, limit_files=limit_files)
    
    # Model
    model = BindingFCNN().to(DEVICE)
    
    # Loss and Optimizer
    if args.focal:
        print("Using Focal Loss (alpha=0.9, gamma=2.0)")
        criterion = FocalLoss(alpha=0.9, gamma=2.0)
        model_save_path = "best_fcnn_focal.pth"
    else:
        print("Using BCE Loss")
        criterion = nn.BCELoss()
        model_save_path = "best_fcnn_model.pth"
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_loss = float('inf')
    
    print(f"Starting training... Saving to {model_save_path}")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved better model (Val Loss: {val_loss:.4f})")
            
    print("Training complete.")

if __name__ == "__main__":
    train()
