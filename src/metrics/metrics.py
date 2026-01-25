"""
Publishable metrics for binding site prediction.

Implements:
- DICE coefficient / IoU for segmentation quality
- DCC Success Rate (Distance to Closest Cluster) for pocket localization
- Enhanced per-class metrics with macro/micro averages
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_dice_coefficient(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute DICE coefficient (F1 for segmentation).
    
    DICE = 2 * |pred ∩ true| / (|pred| + |true|)
    
    Args:
        pred_mask: Boolean prediction mask [N]
        true_mask: Boolean ground truth mask [N]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        DICE coefficient (0.0 to 1.0)
    """
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()
    
    intersection = (pred_mask * true_mask).sum()
    union_sum = pred_mask.sum() + true_mask.sum()
    
    dice = (2.0 * intersection + smooth) / (union_sum + smooth)
    return dice.item()


def compute_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    
    IoU = |pred ∩ true| / |pred ∪ true|
    
    Args:
        pred_mask: Boolean prediction mask [N]
        true_mask: Boolean ground truth mask [N]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()
    
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_segmentation_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    contact_class: int = 0
) -> Dict[str, float]:
    """
    Compute DICE and IoU for the contact class (binding site segmentation).
    
    Args:
        preds: Predicted classes [B, L] or flattened [N]
        labels: Ground truth classes [B, L] or flattened [N]
        mask: Valid residue mask [B, L] or flattened [N]
        contact_class: Class index for contact/binding (default 0)
        
    Returns:
        Dict with 'dice' and 'iou' scores
    """
    # Flatten if needed
    if preds.dim() > 1:
        preds = preds[mask]
        labels = labels[mask]
    
    pred_contact = (preds == contact_class)
    true_contact = (labels == contact_class)
    
    return {
        'dice': compute_dice_coefficient(pred_contact, true_contact),
        'iou': compute_iou(pred_contact, true_contact)
    }


def compute_dcc_success_rate(
    pred_probs: torch.Tensor,
    coords: torch.Tensor,
    ligand_centers: torch.Tensor,
    threshold: float = 4.0,
    top_k: int = 1,
    contact_class: int = 0
) -> Dict[str, float]:
    """
    Compute DCC (Distance to Closest Cluster) success rate.
    
    The DCC metric measures whether the center of the predicted binding pocket
    falls within a threshold distance of the actual ligand center.
    
    Args:
        pred_probs: Softmax probabilities [B, L, C] or list per protein
        coords: Cα coordinates [B, L, 3] or list per protein
        ligand_centers: Ligand center coordinates [B, 3]
        threshold: Success distance threshold in Angstroms (default 4.0 Å)
        top_k: Number of top predictions to consider
        contact_class: Class index for contact (default 0)
        
    Returns:
        Dict with success rates for different top-k values
    """
    if isinstance(pred_probs, torch.Tensor):
        B = pred_probs.shape[0]
        results = {'top1': 0, 'top3': 0, 'count': 0}
        
        for b in range(B):
            probs_b = pred_probs[b]  # [L, C]
            coords_b = coords[b]     # [L, 3]
            lig_center = ligand_centers[b]  # [3]
            
            # Get contact probabilities
            contact_probs = probs_b[:, contact_class]  # [L]
            
            # Find top-k predictions
            valid_mask = ~torch.isnan(contact_probs)
            if valid_mask.sum() == 0:
                continue
                
            contact_probs_valid = contact_probs[valid_mask]
            coords_valid = coords_b[valid_mask]
            
            # Sort by probability
            sorted_indices = torch.argsort(contact_probs_valid, descending=True)
            
            # Compute predicted cluster center (weighted by probability)
            for k in [1, 3]:
                top_indices = sorted_indices[:min(k, len(sorted_indices))]
                top_coords = coords_valid[top_indices]
                top_probs = contact_probs_valid[top_indices]
                
                # Weighted center
                weights = F.softmax(top_probs, dim=0)
                pred_center = (top_coords * weights.unsqueeze(-1)).sum(dim=0)
                
                # Distance to ligand center
                dist = torch.norm(pred_center - lig_center)
                
                if dist <= threshold:
                    results[f'top{k}'] += 1
            
            results['count'] += 1
        
        # Compute success rates
        if results['count'] > 0:
            return {
                'dcc_top1': results['top1'] / results['count'],
                'dcc_top3': results['top3'] / results['count'],
            }
    
    return {'dcc_top1': 0.0, 'dcc_top3': 0.0}


def compute_enhanced_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int = 3
) -> Dict[str, float]:
    """
    Compute enhanced classification metrics including macro/micro averages.
    
    Args:
        logits: Predicted logits [B, L, C]
        labels: Ground truth labels [B, L]
        mask: Padding mask [B, L] (True = valid)
        n_classes: Number of classes
        
    Returns:
        Dict with per-class and aggregated metrics
    """
    preds = logits.argmax(dim=-1)  # [B, L]
    
    # Flatten and apply mask
    preds_flat = preds[mask]
    labels_flat = labels[mask]
    
    n_samples = labels_flat.numel()
    if n_samples == 0:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
    
    # Basic accuracy
    correct = (preds_flat == labels_flat).sum().item()
    accuracy = correct / n_samples
    
    metrics = {'accuracy': accuracy}
    
    # Per-class metrics
    all_tp, all_fp, all_fn = 0, 0, 0
    f1_scores = []
    
    for c in range(n_classes):
        tp = ((preds_flat == c) & (labels_flat == c)).sum().item()
        fp = ((preds_flat == c) & (labels_flat != c)).sum().item()
        fn = ((preds_flat != c) & (labels_flat == c)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[f'precision_c{c}'] = precision
        metrics[f'recall_c{c}'] = recall
        metrics[f'f1_c{c}'] = f1
        
        f1_scores.append(f1)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    # Macro F1 (average across classes)
    metrics['macro_f1'] = sum(f1_scores) / len(f1_scores)
    
    # Micro F1 (aggregate confusion matrix)
    micro_precision = all_tp / (all_tp + all_fp + 1e-8)
    micro_recall = all_tp / (all_tp + all_fn + 1e-8)
    metrics['micro_f1'] = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    return metrics


def compute_all_publishable_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    ligand_centers: Optional[torch.Tensor] = None,
    n_classes: int = 3,
    contact_class: int = 0
) -> Dict[str, float]:
    """
    Compute all publishable metrics in one call.
    
    Args:
        logits: Predicted logits [B, L, C]
        labels: Ground truth labels [B, L]
        mask: Padding mask [B, L]
        coords: Optional Cα coordinates for DCC [B, L, 3]
        ligand_centers: Optional ligand centers for DCC [B, 3]
        n_classes: Number of classes
        contact_class: Class index for contact
        
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Enhanced F1/precision/recall
    clf_metrics = compute_enhanced_metrics(logits, labels, mask, n_classes)
    metrics.update(clf_metrics)
    
    # Segmentation metrics (DICE, IoU)
    preds = logits.argmax(dim=-1)
    seg_metrics = compute_segmentation_metrics(preds, labels, mask, contact_class)
    metrics['dice_c0'] = seg_metrics['dice']
    metrics['iou_c0'] = seg_metrics['iou']
    
    # DCC if coordinates available
    if coords is not None and ligand_centers is not None:
        probs = F.softmax(logits, dim=-1)
        dcc_metrics = compute_dcc_success_rate(
            probs, coords, ligand_centers, 
            threshold=4.0, contact_class=contact_class
        )
        metrics.update(dcc_metrics)
    
    return metrics
