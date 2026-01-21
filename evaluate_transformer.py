"""
Comprehensive evaluation script for Transformer-based binding site classifier.
Computes accuracy, ROC AUC, PR AUC, precision, recall, F1, and confusion matrix.
"""

import torch
import numpy as np
from src.dataset import get_dataloaders
from src.model_transformer import BindingTransformer, BindingTransformerLite
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    average_precision_score, 
    precision_score,
    recall_score,
    f1_score,
    classification_report, 
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def find_optimal_threshold(y_true, y_pred, metric='f1'):
    """Find optimal classification threshold based on specified metric."""
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_score = 0
    best_thresh = 0.5
    
    for t in thresholds:
        preds_bin = (y_pred > t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds_bin, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, preds_bin, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, preds_bin, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh, best_score


def plot_curves(y_true, y_pred, save_path="transformer_eval_curves.png"):
    """Plot ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    axes[1].plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    # Baseline (random classifier)
    baseline = y_true.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved evaluation curves to {save_path}")


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--model_path", type=str, default="best_transformer_focal.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--lite", action="store_true", help="Use lighter transformer model")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--no_position", action="store_true", help="Disable relative position feature")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_enhanced"
    BATCH_SIZE = 1024
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load test data
    print("Loading test data...")
    limit_files = 1 if args.debug else None
    add_position = not args.no_position
    input_dim = 467 if add_position else 466
    print(f"Relative position: {'enabled' if add_position else 'disabled'}, input_dim={input_dim}")
    
    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, limit_files=limit_files, add_relative_position=add_position)
    
    # Load Model
    if args.lite:
        model = BindingTransformerLite(input_dim=input_dim).to(DEVICE)
    else:
        model = BindingTransformer(input_dim=input_dim).to(DEVICE)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {args.model_path} (epoch {checkpoint.get('epoch', '?')}, val_auc={checkpoint.get('val_auc', '?'):.4f})")
        else:
            model.load_state_dict(checkpoint)
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
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Min: {all_preds.min():.4f}, Max: {all_preds.max():.4f}, Mean: {all_preds.mean():.4f}")
    print(f"  Quantiles: 25%={np.percentile(all_preds, 25):.4f}, 50%={np.percentile(all_preds, 50):.4f}, "
          f"75%={np.percentile(all_preds, 75):.4f}, 95%={np.percentile(all_preds, 95):.4f}")
    
    # Class distribution
    n_positive = all_targets.sum()
    n_total = len(all_targets)
    print(f"\nClass Distribution:")
    print(f"  Binding (1): {n_positive:.0f} ({100*n_positive/n_total:.1f}%)")
    print(f"  Non-binding (0): {n_total - n_positive:.0f} ({100*(1 - n_positive/n_total):.1f}%)")
    
    # Find optimal threshold
    best_thresh, best_f1 = find_optimal_threshold(all_targets, all_preds, metric='f1')
    print(f"\nOptimal Threshold (F1): {best_thresh:.2f} (F1={best_f1:.4f})")
    
    # Binary predictions at optimal threshold
    all_preds_binary = (all_preds > best_thresh).astype(int)
    
    # Calculate all metrics
    acc = accuracy_score(all_targets, all_preds_binary)
    roc_auc = roc_auc_score(all_targets, all_preds)
    pr_auc = average_precision_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds_binary, zero_division=0)
    recall = recall_score(all_targets, all_preds_binary, zero_division=0)
    f1 = f1_score(all_targets, all_preds_binary, zero_division=0)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (Test Set)")
    print("=" * 50)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Accuracy':<25} {acc:>10.4f}")
    print(f"{'ROC AUC':<25} {roc_auc:>10.4f}")
    print(f"{'PR AUC (AUPRC)':<25} {pr_auc:>10.4f}")
    print(f"{'Precision (Binding)':<25} {precision:>10.4f}")
    print(f"{'Recall (Binding)':<25} {recall:>10.4f}")
    print(f"{'F1 Score (Binding)':<25} {f1:>10.4f}")
    print("-" * 37)
    print(f"{'Threshold used':<25} {best_thresh:>10.2f}")
    print("=" * 50)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds_binary, 
                                target_names=['Non-binding', 'Binding']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds_binary)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-bind  Binding")
    print(f"Actual Non-bind  {cm[0,0]:>8}  {cm[0,1]:>7}")
    print(f"Actual Binding   {cm[1,0]:>8}  {cm[1,1]:>7}")
    
    # Generate plots if requested
    if args.plot:
        plot_curves(all_targets, all_preds)
    
    # Summary comparison format (for notes)
    print("\n" + "=" * 50)
    print("SUMMARY (for comparison with other models):")
    print("=" * 50)
    print(f"| Transformer | {n_total:,} samples | {roc_auc:.4f} | {acc:.4f} | {recall:.2f} | {precision:.2f} |")


if __name__ == "__main__":
    evaluate()
