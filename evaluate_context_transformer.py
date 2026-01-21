"""
Evaluation script for Context-aware Transformer binding site classifier.

Loads a trained model and evaluates on the test set.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from src.dataset_context import get_context_dataloaders
from src.model_context_transformer import ContextTransformer, ContextTransformerWithCLS


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Context-aware Transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit data files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--save_plots", action="store_true", help="Save evaluation plots")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load checkpoint
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Extract config
    context_size = config.get('context_size', 5)
    d_model = config.get('d_model', 256)
    nhead = config.get('nhead', 8)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.2)
    use_cls = config.get('use_cls', False)
    
    print(f"Model config: context_size={context_size}, d_model={d_model}, "
          f"nhead={nhead}, num_layers={num_layers}, use_cls={use_cls}")

    # Load Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_context_dataloaders(
        DATA_DIR, 
        context_size=context_size,
        batch_size=args.batch_size, 
        limit_files=args.limit_files,
    )
    
    # Build Model
    if use_cls:
        model = ContextTransformerWithCLS(
            feature_dim=466,
            context_size=context_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)
    else:
        model = ContextTransformer(
            feature_dim=466,
            context_size=context_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_model * 2,
            dropout=dropout,
        ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loaded from epoch {checkpoint.get('epoch', 'N/A')} with val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for windows, labels in test_loader:
            windows = windows.to(DEVICE)
            outputs = model(windows)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    roc_auc = roc_auc_score(all_targets, all_preds)
    avg_precision = average_precision_score(all_targets, all_preds)
    
    # Binary predictions at threshold
    binary_preds = (all_preds >= args.threshold).astype(int)
    accuracy = accuracy_score(all_targets, binary_preds)
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nResults at threshold = {args.threshold}:")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Avg Precision:     {avg_precision:.4f}")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Precision (Bind):  {precision:.4f}")
    print(f"  Recall (Bind):     {recall:.4f}")
    print(f"  F1 (Bind):         {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    
    # Class distribution
    n_pos = int(all_targets.sum())
    n_neg = len(all_targets) - n_pos
    print(f"\nClass distribution: {n_neg:,} non-binding, {n_pos:,} binding ({100*n_pos/len(all_targets):.1f}%)")
    
    # Find optimal threshold based on F1
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    print(f"\nThreshold analysis:")
    for thresh in thresholds:
        preds_at_thresh = (all_preds >= thresh).astype(int)
        f1_at_thresh = f1_score(all_targets, preds_at_thresh, zero_division=0)
        recall_at_thresh = recall_score(all_targets, preds_at_thresh, zero_division=0)
        prec_at_thresh = precision_score(all_targets, preds_at_thresh, zero_division=0)
        
        print(f"  t={thresh:.2f}: Prec={prec_at_thresh:.3f}, Recall={recall_at_thresh:.3f}, F1={f1_at_thresh:.3f}")
        
        if f1_at_thresh > best_f1:
            best_f1 = f1_at_thresh
            best_thresh = thresh
    
    print(f"\nOptimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    
    # Save plots if requested
    if args.save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_targets, all_preds)
        axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_preds)
        axes[1].plot(recall_curve, precision_curve, 'b-', lw=2, label=f'AP = {avg_precision:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "context_transformer_eval_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved plots to {plot_path}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    evaluate()
