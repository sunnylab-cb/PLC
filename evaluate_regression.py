"""
Evaluation script for distance regression model.
"""

import torch
import numpy as np
from src.dataset_regression import get_regression_dataloaders
from src.model_regression import DistanceRegressionModel, DistanceRegressionFCNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Distance Regression Model")
    parser.add_argument("--debug", action="store_true", help="Debug mode (load fewer files)")
    parser.add_argument("--model_path", type=str, default="best_regression_transformer_mse.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--fcnn", action="store_true", help="Use FCNN model instead of transformer")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = "data/processed_regression"
    BATCH_SIZE = 1024
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load test data
    print("Loading test data...")
    limit_files = 1 if args.debug else None
    _, _, test_loader = get_regression_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, limit_files=limit_files)
    
    # Load Model
    if args.fcnn:
        model = DistanceRegressionFCNN().to(DEVICE)
    else:
        model = DistanceRegressionModel().to(DEVICE)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {args.model_path} (epoch {checkpoint.get('epoch', '?')}, val_mae={checkpoint.get('val_mae', '?'):.2f}Å)")
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
        for features, distances in test_loader:
            features = features.to(DEVICE)
            outputs = model(features).cpu().numpy()
            
            all_preds.extend(outputs)
            all_targets.extend(distances.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    # Correlation
    correlation = np.corrcoef(all_targets, all_preds)[0, 1]
    
    # Print results
    print("\n" + "=" * 50)
    print("REGRESSION EVALUATION RESULTS (Test Set)")
    print("=" * 50)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'MAE':<25} {mae:>12.2f} Å")
    print(f"{'RMSE':<25} {rmse:>12.2f} Å")
    print(f"{'MSE':<25} {mse:>12.2f} Å²")
    print(f"{'R² Score':<25} {r2:>15.4f}")
    print(f"{'Correlation':<25} {correlation:>15.4f}")
    print("=" * 50)
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Min: {all_preds.min():.2f}Å, Max: {all_preds.max():.2f}Å, Mean: {all_preds.mean():.2f}Å")
    print(f"  Ground Truth - Min: {all_targets.min():.2f}Å, Max: {all_targets.max():.2f}Å, Mean: {all_targets.mean():.2f}Å")
    
    # Threshold-based classification analysis
    print("\n" + "=" * 50)
    print("THRESHOLD ANALYSIS (for binding site classification)")
    print("=" * 50)
    
    for threshold in [5.0, 6.0, 7.0, 8.0]:
        pred_binding = (all_preds < threshold).astype(int)
        true_binding = (all_targets < threshold).astype(int)
        
        tp = ((pred_binding == 1) & (true_binding == 1)).sum()
        fp = ((pred_binding == 1) & (true_binding == 0)).sum()
        fn = ((pred_binding == 0) & (true_binding == 1)).sum()
        tn = ((pred_binding == 0) & (true_binding == 0)).sum()
        
        accuracy = (tp + tn) / len(true_binding)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold {threshold}Å: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
    
    # Generate plots if requested
    if args.plot:
        plt.figure(figsize=(10, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(all_targets, all_preds, alpha=0.1, s=1)
        plt.plot([0, 30], [0, 30], 'r--', linewidth=2, label='Perfect prediction')
        plt.xlabel('True Distance (Å)')
        plt.ylabel('Predicted Distance (Å)')
        plt.title(f'Distance Regression\nMAE={mae:.2f}Å, RMSE={rmse:.2f}Å, R²={r2:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = all_preds - all_targets
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error (Å)')
        plt.ylabel('Count')
        plt.title(f'Error Distribution\nMean={errors.mean():.2f}Å, Std={errors.std():.2f}Å')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_eval.png', dpi=150)
        plt.close()
        print(f"\nSaved evaluation plot to regression_eval.png")


if __name__ == "__main__":
    evaluate()
