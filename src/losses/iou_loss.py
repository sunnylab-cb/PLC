#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance-Weighted IoU Loss for Binding Site Prediction

结合区域重叠（IoU）和空间距离惩罚，对远距离的 False Positive 施加非线性增长的惩罚。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    power: float = 1.5,
    d_max: float = 10.0,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    计算 Distance-Weighted IoU Loss

    Args:
        logits: [B, N, 3] 模型输出 logits
        labels: [B, N] 真实标签 (0=Contact, 1=Near, 2=Far)
        coords: [B, N, 3] Cα 坐标
        mask: [B, N] 有效残基 mask
        power: 距离惩罚的指数 (默认 1.5)
        d_max: 距离归一化因子 (Å)
        smooth: 平滑项，避免除零

    Returns:
        loss: 标量损失值
    """
    # 获取预测类别
    preds = logits.argmax(dim=-1)  # [B, N]

    # 只将 Contact(0) 视为结合位点（正类），Near(1) 和 Far(2) 为负类
    pred_binding = (preds == 0).float()  # [B, N]
    true_binding = (labels == 0).float()  # [B, N]

    # 应用 mask
    pred_binding = pred_binding * mask
    true_binding = true_binding * mask

    # 1. 计算传统 IoU
    intersection = (pred_binding * true_binding).sum(dim=1)  # [B]
    union = (pred_binding + true_binding - pred_binding * true_binding).sum(dim=1)  # [B]
    iou = (intersection + smooth) / (union + smooth)  # [B]
    iou_loss = 1.0 - iou.mean()  # 标量

    # 2. 计算距离惩罚（针对 False Positives）
    # False Positive: 预测为结合位点，但实际不是
    fp_mask = (pred_binding == 1) & (true_binding == 0) & (mask == 1)  # [B, N]

    # 初始化为tensor标量
    distance_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n_fp_total = fp_mask.sum()

    if n_fp_total > 0:
        batch_size = logits.size(0)
        penalties = []

        for b in range(batch_size):
            # 当前样本的 FP 残基
            fp_indices = fp_mask[b].nonzero(as_tuple=True)[0]  # [n_fp]
            if len(fp_indices) == 0:
                continue

            # 当前样本的真实结合位点残基
            true_binding_indices = (true_binding[b] == 1).nonzero(as_tuple=True)[0]  # [n_true]
            if len(true_binding_indices) == 0:
                # 如果没有真实结合位点，施加最大惩罚
                penalties.append(torch.ones(len(fp_indices), device=logits.device))
                continue

            # 计算 FP 残基到最近真实结合位点的距离
            fp_coords = coords[b, fp_indices]  # [n_fp, 3]
            true_coords = coords[b, true_binding_indices]  # [n_true, 3]

            # 计算距离矩阵 [n_fp, n_true]
            dist_matrix = torch.cdist(fp_coords, true_coords, p=2)  # 欧氏距离

            # 找到每个 FP 到最近真实结合位点的距离
            min_distances, _ = dist_matrix.min(dim=1)  # [n_fp]

            # 应用非线性惩罚: (d / d_max)^power
            normalized_dist = torch.clamp(min_distances / d_max, max=1.0)  # 限制在 [0, 1]
            penalty = normalized_dist ** power  # [n_fp]

            penalties.append(penalty)

        if penalties:
            all_penalties = torch.cat(penalties)  # [total_fp]
            distance_penalty = all_penalties.mean()

    return iou_loss, distance_penalty


def compute_combined_iou_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    iou_weight: float = 1.0,
    dist_weight: float = 0.2,
    power: float = 1.5,
    d_max: float = 10.0
) -> tuple:
    """
    计算组合的 IoU Loss

    Args:
        logits: [B, N, 3] 模型输出
        labels: [B, N] 真实标签
        coords: [B, N, 3] Cα 坐标
        mask: [B, N] 有效残基 mask
        iou_weight: IoU loss 权重
        dist_weight: 距离惩罚权重
        power: 距离惩罚指数
        d_max: 距离归一化因子

    Returns:
        total_loss: 总损失
        iou_loss: IoU 损失
        dist_penalty: 距离惩罚
    """
    iou_loss, dist_penalty = compute_iou_loss(
        logits, labels, coords, mask, power=power, d_max=d_max
    )

    total_loss = iou_weight * iou_loss + dist_weight * dist_penalty

    return total_loss, iou_loss, dist_penalty


# ============================================================================
# 辅助函数：计算 IoU 指标（用于评估）
# ============================================================================

def compute_iou_metric(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor
) -> float:
    """
    计算 IoU 指标（用于评估，不计算梯度）

    Args:
        preds: [B, N] 预测类别
        labels: [B, N] 真实标签
        mask: [B, N] 有效残基 mask

    Returns:
        iou: IoU 值
    """
    # 只将 Contact(0) 视为结合位点
    pred_binding = (preds == 0).float() * mask
    true_binding = (labels == 0).float() * mask

    intersection = (pred_binding * true_binding).sum()
    union = (pred_binding + true_binding - pred_binding * true_binding).sum()

    if union == 0:
        return 0.0

    iou = (intersection / union).item()
    return iou
