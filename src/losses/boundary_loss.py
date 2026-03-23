#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Loss for Binding Site Prediction

对结合位点边界的残基施加额外的权重，提高边界预测精度。
边界定义：与不同类别残基相邻的残基（在3D空间中）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_boundary_mask(
    labels: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    distance_threshold: float = 8.0
) -> torch.Tensor:
    """
    计算边界残基的mask

    边界残基定义：在distance_threshold范围内存在不同类别的残基

    Args:
        labels: [B, N] 真实标签 (0=Contact, 1=Near, 2=Far)
        coords: [B, N, 3] Cα 坐标
        mask: [B, N] 有效残基 mask
        distance_threshold: 距离阈值（Å），用于判断相邻

    Returns:
        boundary_mask: [B, N] 边界残基mask (1=边界, 0=非边界)
    """
    B, N = labels.shape
    device = labels.device

    boundary_mask = torch.zeros_like(labels, dtype=torch.float32)

    for b in range(B):
        # 获取有效残基
        valid_indices = mask[b].nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            continue

        # 获取有效残基的坐标和标签
        valid_coords = coords[b, valid_indices]  # [n_valid, 3]
        valid_labels = labels[b, valid_indices]  # [n_valid]

        # 计算距离矩阵
        dist_matrix = torch.cdist(valid_coords, valid_coords, p=2)  # [n_valid, n_valid]

        # 找到距离阈值内的邻居
        neighbors = dist_matrix < distance_threshold  # [n_valid, n_valid]

        # 对每个残基，检查是否有不同类别的邻居
        for i, idx in enumerate(valid_indices):
            # 获取邻居的标签
            neighbor_labels = valid_labels[neighbors[i]]

            # 如果存在不同类别的邻居，则为边界残基
            if len(neighbor_labels) > 0 and (neighbor_labels != valid_labels[i]).any():
                boundary_mask[b, idx] = 1.0

    return boundary_mask


def compute_boundary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    boundary_weight: float = 2.0,
    distance_threshold: float = 8.0,
    label_smoothing: float = 0.0
) -> tuple:
    """
    计算带边界权重的交叉熵损失

    对边界残基施加更高的权重，提高模型对边界的预测精度。

    Args:
        logits: [B, N, C] 模型输出 logits
        labels: [B, N] 真实标签
        coords: [B, N, 3] Cα 坐标
        mask: [B, N] 有效残基 mask
        boundary_weight: 边界残基的权重倍数（默认2.0）
        distance_threshold: 距离阈值（Å）
        label_smoothing: Label smoothing参数

    Returns:
        loss: 总损失
        boundary_loss: 边界残基的损失
        non_boundary_loss: 非边界残基的损失
        boundary_ratio: 边界残基占比
    """
    B, N, C = logits.shape
    device = logits.device

    # 计算边界mask
    boundary_mask = compute_boundary_mask(labels, coords, mask, distance_threshold)

    # Flatten
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)
    boundary_flat = boundary_mask.view(-1)

    # 计算交叉熵损失（不reduction）
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
        label_smoothing=label_smoothing
    )

    # 应用mask
    ce_loss = ce_loss * mask_flat.float()

    # 分离边界和非边界损失
    boundary_indices = (boundary_flat == 1) & (mask_flat == 1)
    non_boundary_indices = (boundary_flat == 0) & (mask_flat == 1)

    # 计算边界损失
    if boundary_indices.sum() > 0:
        boundary_loss = ce_loss[boundary_indices].mean()
    else:
        boundary_loss = torch.tensor(0.0, device=device)

    # 计算非边界损失
    if non_boundary_indices.sum() > 0:
        non_boundary_loss = ce_loss[non_boundary_indices].mean()
    else:
        non_boundary_loss = torch.tensor(0.0, device=device)

    # 组合损失：边界残基权重更高
    n_boundary = boundary_indices.sum().float()
    n_non_boundary = non_boundary_indices.sum().float()
    n_total = n_boundary + n_non_boundary

    if n_total > 0:
        # 加权平均
        weighted_loss = (
            boundary_weight * boundary_loss * n_boundary +
            non_boundary_loss * n_non_boundary
        ) / (boundary_weight * n_boundary + n_non_boundary)
    else:
        weighted_loss = torch.tensor(0.0, device=device)

    # 计算边界残基占比
    boundary_ratio = n_boundary / n_total if n_total > 0 else 0.0

    return weighted_loss, boundary_loss, non_boundary_loss, boundary_ratio


def compute_combined_loss_with_boundary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float,
    dice_weight: float = 0.4,
    focal_weight: float = 0.4,
    boundary_weight: float = 0.2,
    boundary_multiplier: float = 2.0,
    distance_threshold: float = 8.0,
    label_smoothing: float = 0.0
) -> tuple:
    """
    计算组合损失: Dice + Focal + Boundary

    Args:
        logits: [B, N, C] 模型输出
        labels: [B, N] 真实标签
        coords: [B, N, 3] Cα 坐标
        mask: [B, N] 有效残基 mask
        class_weights: [C] 类别权重
        focal_gamma: Focal loss gamma参数
        dice_weight: Dice loss权重
        focal_weight: Focal loss权重
        boundary_weight: Boundary loss权重
        boundary_multiplier: 边界残基权重倍数
        distance_threshold: 边界判定距离阈值
        label_smoothing: Label smoothing参数

    Returns:
        total_loss: 总损失
        dice_loss: Dice损失
        focal_loss: Focal损失
        boundary_loss_val: 边界损失
        boundary_ratio: 边界残基占比
    """
    from .iou_loss import compute_iou_loss

    # 导入其他损失函数（避免循环导入）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.train.train_v9 import compute_dice_loss, compute_focal_loss

    # 1. Dice Loss
    dice_loss = compute_dice_loss(logits, labels, mask)

    # 2. Focal Loss
    focal_loss = compute_focal_loss(
        logits, labels, mask, class_weights, focal_gamma,
        label_smoothing=label_smoothing
    )

    # 3. Boundary Loss
    boundary_loss_val, boundary_loss_only, non_boundary_loss, boundary_ratio = compute_boundary_loss(
        logits, labels, coords, mask,
        boundary_weight=boundary_multiplier,
        distance_threshold=distance_threshold,
        label_smoothing=label_smoothing
    )

    # 组合损失
    total_loss = (
        dice_weight * dice_loss +
        focal_weight * focal_loss +
        boundary_weight * boundary_loss_val
    )

    return total_loss, dice_loss, focal_loss, boundary_loss_val, boundary_ratio
