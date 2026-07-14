"""
Publishable metrics for binding site prediction.

Implements:
- DICE coefficient / IoU for segmentation quality
- DCC Success Rate (Distance to Closest Cluster) for pocket localization
- Enhanced per-class metrics with macro/micro averages
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def _prepare_binary_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return finite CPU float64 scores and boolean targets."""
    scores = torch.as_tensor(scores).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    targets = torch.as_tensor(targets).detach().reshape(-1).to(device="cpu", dtype=torch.bool)
    if scores.numel() != targets.numel():
        raise ValueError(
            f"scores/targets size mismatch: {scores.numel()} vs {targets.numel()}"
        )
    finite = torch.isfinite(scores)
    return scores[finite], targets[finite]


def compute_binary_metrics_at_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute global binary metrics for ``scores >= threshold``."""
    scores, targets = _prepare_binary_scores(scores, targets)
    if scores.numel() == 0:
        return {
            "threshold": float(threshold),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
        }

    pred = scores >= float(threshold)
    tp = int((pred & targets).sum())
    fp = int((pred & ~targets).sum())
    fn = int((~pred & targets).sum())
    tn = int((~pred & ~targets).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def contact_gate_predictions(
    probabilities: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Apply a binary contact gate while preserving the non-contact classes.

    Class 0 is contact. Positions below the contact threshold are assigned to
    the most likely class among 1..C-1 instead of being collapsed to one class.
    """
    if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
        raise ValueError(
            "contact gate expects probabilities with at least two classes, "
            f"got shape={tuple(probabilities.shape)}"
        )
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"contact threshold must be in [0, 1], got {threshold}")

    non_contact = probabilities[..., 1:].argmax(dim=-1) + 1
    return torch.where(
        probabilities[..., 0] >= threshold,
        torch.zeros_like(non_contact),
        non_contact,
    )


def find_best_f1_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    default_threshold: float = 0.5,
) -> Dict[str, float]:
    """Find the exact score threshold maximizing global binary F1.

    Tied scores enter the predicted-positive set together. If multiple
    thresholds attain the same F1, the highest threshold is retained.
    """
    scores, targets = _prepare_binary_scores(scores, targets)
    if scores.numel() == 0:
        return compute_binary_metrics_at_threshold(scores, targets, default_threshold)

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].to(torch.float64)
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.arange(1, scores.numel() + 1, dtype=torch.float64) - tp
    total_positive = sorted_targets.sum()
    fn = total_positive - tp

    group_end = torch.ones(scores.numel(), dtype=torch.bool)
    if scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    candidate_indices = torch.nonzero(group_end, as_tuple=False).flatten()
    candidate_f1 = 2.0 * tp[candidate_indices] / (
        2.0 * tp[candidate_indices] + fp[candidate_indices] + fn[candidate_indices]
    ).clamp_min(1.0)
    best_idx = candidate_indices[int(torch.argmax(candidate_f1))]
    threshold = float(sorted_scores[best_idx])
    return compute_binary_metrics_at_threshold(scores, targets, threshold)


def compute_binary_ranking_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute exact global average precision (AUPRC) and AUROC."""
    scores, targets = _prepare_binary_scores(scores, targets)
    n_positive = int(targets.sum())
    n_negative = int((~targets).sum())
    if scores.numel() == 0:
        return {"auprc": float("nan"), "auroc": float("nan")}

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].to(torch.float64)
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)

    group_end = torch.ones(scores.numel(), dtype=torch.bool)
    if scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    tp = tp[group_end]
    fp = fp[group_end]

    if n_positive:
        recall = tp / n_positive
        precision = tp / (tp + fp).clamp_min(1.0)
        previous_recall = torch.cat((torch.zeros(1, dtype=torch.float64), recall[:-1]))
        auprc = float(((recall - previous_recall) * precision).sum())
    else:
        auprc = float("nan")

    if n_positive and n_negative:
        tpr = torch.cat((torch.zeros(1, dtype=torch.float64), tp / n_positive))
        fpr = torch.cat((torch.zeros(1, dtype=torch.float64), fp / n_negative))
        auroc = float(torch.trapz(tpr, fpr))
    else:
        auroc = float("nan")
    return {"auprc": auprc, "auroc": auroc}


def compute_contact_probability_metrics(
    sample_scores: Sequence[torch.Tensor],
    sample_targets: Sequence[torch.Tensor],
    top_l_divisors: Sequence[int] = (20, 10, 5),
) -> Dict[str, float]:
    """Compute val-only contact ranking, calibration, and Top-L metrics.

    ``sample_scores`` and ``sample_targets`` contain one valid-residue vector
    per protein-ligand row. Ranking metrics and threshold metrics are global
    over residues. Top-L precision/recall are macro-averaged over rows.
    """
    if len(sample_scores) != len(sample_targets):
        raise ValueError(
            f"sample list size mismatch: {len(sample_scores)} vs {len(sample_targets)}"
        )
    prepared = [
        _prepare_binary_scores(scores, targets)
        for scores, targets in zip(sample_scores, sample_targets)
    ]
    prepared = [(scores, targets) for scores, targets in prepared if scores.numel()]
    if not prepared:
        return {}

    all_scores = torch.cat([item[0] for item in prepared])
    all_targets = torch.cat([item[1] for item in prepared])
    ranking = compute_binary_ranking_metrics(all_scores, all_targets)
    at_half = compute_binary_metrics_at_threshold(all_scores, all_targets, 0.5)
    at_best = find_best_f1_threshold(all_scores, all_targets, 0.5)

    metrics = {
        "auprc_c0_global": ranking["auprc"],
        "auroc_c0_global": ranking["auroc"],
        "f1_c0_at_0_5_global": at_half["f1"],
        "mcc_c0_at_0_5_global": at_half["mcc"],
        "precision_c0_at_0_5_global": at_half["precision"],
        "recall_c0_at_0_5_global": at_half["recall"],
        "best_gate_c0": at_best["threshold"],
        "f1_c0_at_best_gate_global": at_best["f1"],
        "mcc_c0_at_best_gate_global": at_best["mcc"],
        "precision_c0_at_best_gate_global": at_best["precision"],
        "recall_c0_at_best_gate_global": at_best["recall"],
        "n_residues_c0_metrics": float(all_scores.numel()),
        "n_positive_c0_metrics": float(all_targets.sum()),
    }

    for divisor in top_l_divisors:
        if divisor <= 0:
            raise ValueError(f"Top-L divisor must be positive, got {divisor}")
        precisions = []
        recalls = []
        hit_rates = []
        for scores, targets in prepared:
            k = max(1, int(np.ceil(scores.numel() / divisor)))
            top_targets = targets[torch.topk(scores, k=k, largest=True).indices]
            n_top_positive = int(top_targets.sum())
            n_positive = int(targets.sum())
            precisions.append(n_top_positive / k)
            if n_positive:
                recalls.append(n_top_positive / n_positive)
            hit_rates.append(float(n_top_positive > 0))
        prefix = f"top_l{divisor}"
        metrics[f"{prefix}_precision_c0_macro"] = float(np.mean(precisions))
        metrics[f"{prefix}_recall_c0_macro"] = (
            float(np.mean(recalls)) if recalls else float("nan")
        )
        metrics[f"{prefix}_hit_rate_c0_macro"] = float(np.mean(hit_rates))

    return metrics


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
