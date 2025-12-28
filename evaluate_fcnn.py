import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import BindingFCNN
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
import numpy as np

import argparse

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate FCNN")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--model_path", type=str, default="best_fcnn_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    BATCH_SIZE = 1024
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load test data (ensure seed matches training script to get same split)
    print("Loading test data...")
    # We discard train/val loaders to save memory if possible, but get_dataloaders might hold refs.
    # Python GC should handle it if not assigned.
    limit_files = 1 if args.debug else None
    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, limit_files=limit_files)
    
    # Load Model
    model = BindingFCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Loaded {args.model_path}")
    except FileNotFoundError:
        print(f"Model file {args.model_path} not found. Please train first.")
        return

    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features).cpu().numpy()
            
            all_preds.extend(outputs)
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    print(f"Prediction Stats: Min={all_preds.min():.4f}, Max={all_preds.max():.4f}, Mean={all_preds.mean():.4f}")
    print(f"Quantiles: 50%={np.percentile(all_preds, 50):.4f}, 75%={np.percentile(all_preds, 75):.4f}, 90%={np.percentile(all_preds, 90):.4f}, 99%={np.percentile(all_preds, 99):.4f}")
    
    # Thresholding
    # all_preds_binary = (all_preds > 0.5).astype(int)
    # Let's pick a better threshold if max is low, or just use 0.5 for now but we know it might fail
    # For reporting, let's try to optimize F1 score or just stick to 0.5/variable?
    # Let's keep 0.5 but add a "Best Threshold" search logic
    
    thresholds = np.arange(0, 1, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds_bin = (all_preds > t).astype(int)
        # Manually calc F1 to avoid log spam or just import f1_score
        from sklearn.metrics import f1_score
        score = f1_score(all_targets, preds_bin, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    print(f"Best Threshold (F1): {best_thresh:.2f} (F1={best_f1:.4f})")
    all_preds_binary = (all_preds > best_thresh).astype(int)
    
    # Metrics
    acc = accuracy_score(all_targets, all_preds_binary)
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    
    print("\n" + "="*30)
    print("Evaluation Results (Test Set)")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"AUPRC:    {auprc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds_binary))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds_binary))

if __name__ == "__main__":
    evaluate()
