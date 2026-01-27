#!/usr/bin/env python3
"""
Evaluation script for Pocket-GVP-Fusion v2.

Evaluates trained model on test set with comprehensive metrics:
- Per-class precision, recall, F1
- Confusion matrix
- Binding site detection analysis
- Per-protein breakdown

Usage:
    python evaluate.py --checkpoint outputs/experiment/checkpoints/best.pt --config outputs/experiment/config.yaml
    python evaluate.py --checkpoint outputs/experiment/checkpoints/best.pt --config outputs/experiment/config.yaml --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset import PocketDataset, load_splits, set_seed, create_dataloader
from src.data.collate import prepare_model_inputs
from src.models.pocket_gvp_fusion import PocketGVPFusion, PocketGVPFusionSimplified, PocketGVPFusionNoGVP


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: Dict, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model_config = config['model']
    
    # Check if GVP should be used (default True for backward compatibility)
    use_gvp = model_config.get('use_gvp', True)
    
    if use_gvp:
        print("Loading PocketGVPFusion (with GVP encoder)")
        model = PocketGVPFusion(model_config)
    else:
        print("Loading PocketGVPFusionNoGVP (ablation: no GVP encoder)")
        model = PocketGVPFusionNoGVP(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Convert legacy state dict for backward compatibility
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'convert_legacy_state_dict'):
        state_dict = model.convert_legacy_state_dict(state_dict)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return model


def compute_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    n_classes: int = 3
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for pred, label in zip(all_preds, all_labels):
        cm[label, pred] += 1
    
    return cm


def compute_detailed_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    n_classes: int = 3,
    class_names: List[str] = ['Contact', 'Near', 'Far']
) -> Dict:
    """Compute detailed classification metrics."""
    metrics = {}
    
    # Overall accuracy
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    metrics['accuracy'] = correct / total
    
    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, n_classes)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    per_class = {}
    macro_precision, macro_recall, macro_f1 = 0, 0, 0
    
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        support = (all_labels == c).sum()
        
        per_class[class_names[c]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
    
    metrics['per_class'] = per_class
    
    # Macro averages
    metrics['macro_precision'] = macro_precision / n_classes
    metrics['macro_recall'] = macro_recall / n_classes
    metrics['macro_f1'] = macro_f1 / n_classes
    
    # Weighted averages
    total_samples = len(all_labels)
    weighted_f1 = sum(
        per_class[name]['f1'] * per_class[name]['support'] / total_samples
        for name in class_names
    )
    metrics['weighted_f1'] = weighted_f1
    
    return metrics


def evaluate_binding_sites(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_probs: np.ndarray
) -> Dict:
    """
    Evaluate binding site detection performance.
    
    Binding sites = Class 0 (Contact) or Class 1 (Near)
    Non-binding = Class 2 (Far)
    """
    # Binary task: is this a binding site?
    pred_binding = (all_preds != 2).astype(np.int64)
    true_binding = (all_labels != 2).astype(np.int64)
    
    # Metrics
    tp = ((pred_binding == 1) & (true_binding == 1)).sum()
    fp = ((pred_binding == 1) & (true_binding == 0)).sum()
    fn = ((pred_binding == 0) & (true_binding == 1)).sum()
    tn = ((pred_binding == 0) & (true_binding == 0)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    # Probability-based: use max prob of binding classes
    prob_binding = 1 - all_probs[:, 2]  # P(not far)
    
    # AUROC approximation (simple)
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        auroc = roc_auc_score(true_binding, prob_binding)
        prec_curve, rec_curve, _ = precision_recall_curve(true_binding, prob_binding)
        auprc = auc(rec_curve, prec_curve)
    except ImportError:
        auroc = None
        auprc = None
    
    return {
        'binary_accuracy': float(accuracy),
        'binary_precision': float(precision),
        'binary_recall': float(recall),
        'binary_f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'auroc': float(auroc) if auroc is not None else None,
        'auprc': float(auprc) if auprc is not None else None
    }


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    n_classes: int = 3
) -> Dict:
    """Run full evaluation on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    per_protein = {}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)
        pdb_ids = batch['pdb_ids']
        
        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_scalar=inputs['protein_scalar'],
            protein_vectors=inputs['protein_vectors'],
            protein_coords=inputs['protein_coords'],
            protein_mask=inputs['protein_mask'],
            ligand_x=inputs['ligand_x'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_to_protein=inputs['ligand_to_protein'],
            ligand_mask=inputs['ligand_mask']
        )
        
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        # Collect predictions
        for b in range(len(pdb_ids)):
            mask_b = mask[b]
            preds_b = preds[b, mask_b].cpu().numpy()
            labels_b = labels[b, mask_b].cpu().numpy()
            probs_b = probs[b, mask_b].cpu().numpy()
            
            all_preds.extend(preds_b.tolist())
            all_labels.extend(labels_b.tolist())
            all_probs.extend(probs_b.tolist())
            
            # Per-protein tracking
            pdb_id = pdb_ids[b]
            correct = (preds_b == labels_b).sum()
            total = len(labels_b)
            acc = correct / total if total > 0 else 0
            
            per_protein[pdb_id] = {
                'accuracy': float(acc),
                'n_residues': int(total),
                'n_contact': int((labels_b == 0).sum()),
                'n_near': int((labels_b == 1).sum()),
                'n_far': int((labels_b == 2).sum())
            }
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    class_names = ['Contact', 'Near', 'Far']
    classification_metrics = compute_detailed_metrics(
        all_preds, all_labels, n_classes, class_names
    )
    
    binding_metrics = evaluate_binding_sites(
        all_preds, all_labels, all_probs
    )
    
    # Summary
    results = {
        'classification': classification_metrics,
        'binding_site': binding_metrics,
        'per_protein': per_protein,
        'n_samples': len(all_labels),
        'n_proteins': len(per_protein)
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    cls = results['classification']
    
    print(f"\n{'='*30} CLASSIFICATION {'='*30}")
    print(f"Overall Accuracy: {cls['accuracy']:.4f}")
    print(f"Macro F1: {cls['macro_f1']:.4f}")
    print(f"Weighted F1: {cls['weighted_f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for name, metrics in cls['per_class'].items():
        print(f"{name:<10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(cls['confusion_matrix'])
    print(f"{'':>10} {'Pred Contact':>12} {'Pred Near':>12} {'Pred Far':>12}")
    for i, name in enumerate(['Contact', 'Near', 'Far']):
        print(f"{name:>10} {cm[i, 0]:>12} {cm[i, 1]:>12} {cm[i, 2]:>12}")
    
    bind = results['binding_site']
    print(f"\n{'='*30} BINDING SITE DETECTION {'='*30}")
    print(f"Binary Accuracy: {bind['binary_accuracy']:.4f}")
    print(f"Binary Precision: {bind['binary_precision']:.4f}")
    print(f"Binary Recall: {bind['binary_recall']:.4f}")
    print(f"Binary F1: {bind['binary_f1']:.4f}")
    if bind.get('auroc'):
        print(f"AUROC: {bind['auroc']:.4f}")
    if bind.get('auprc'):
        print(f"AUPRC: {bind['auprc']:.4f}")
    
    print(f"\n{'='*30} DATASET INFO {'='*30}")
    print(f"Total Residues: {results['n_samples']}")
    print(f"Total Proteins: {results['n_proteins']}")
    
    # Best and worst proteins
    per_prot = results['per_protein']
    if per_prot:
        sorted_prots = sorted(per_prot.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\nTop 5 Proteins (by accuracy):")
        for pdb_id, stats in sorted_prots[:5]:
            print(f"  {pdb_id}: {stats['accuracy']:.4f} ({stats['n_residues']} residues)")
        
        print(f"\nBottom 5 Proteins (by accuracy):")
        for pdb_id, stats in sorted_prots[-5:]:
            print(f"  {pdb_id}: {stats['accuracy']:.4f} ({stats['n_residues']} residues)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pocket-GVP-Fusion v2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Load splits
    exp_dir = Path(args.checkpoint).parent.parent
    split_file = exp_dir / "splits.txt"
    
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        print("Please provide the correct experiment directory or split file.")
        return
    
    train_ids, val_ids, test_ids = load_splits(split_file)
    
    if args.split == 'train':
        pdb_ids = train_ids
    elif args.split == 'val':
        pdb_ids = val_ids
    else:
        pdb_ids = test_ids
    
    print(f"Evaluating on {args.split} split: {len(pdb_ids)} proteins")
    
    # Create dataloader
    data_cfg = config['data']
    dataloader = create_dataloader(
        feature_dir=data_cfg['feature_dir'],
        pdb_ids=pdb_ids,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        seed=config['experiment']['seed']
    )
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        n_classes=config['model']['n_classes']
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_json = json.loads(json.dumps(results, default=convert))
        
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
