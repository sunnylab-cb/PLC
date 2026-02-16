#!/usr/bin/env python3
"""
训练脚本 for Pocket-GVP-Fusion v6m7 (Label Smoothing)

V6m7 改进:
- ✅ Label Smoothing (0.1) 防止过拟合，提高泛化能力
- ✅ 双向交叉注意力 (Protein ↔ Ligand)
- ✅ PhysChem MLP Encoder (35D -> 64D) 捕捉非线性交互
- ✅ PhysChem Encoder 中添加 LayerNorm 修复特征尺度不匹配
- ✅ GELU 激活函数替代 ReLU，减少信息丢失
- ✅ 3D Ligand GVP Encoder 使用真实 3D 坐标
- ✅ PhysChem Modality Dropout (避免过度依赖单一特征)
- ✅ 移除 Distance-Aware Attention (避免数据泄漏风险)
- ✅ Dice Loss + Focal Loss 组合损失函数
- ✅ 使用 BioLiP 标准的 VDW 半径距离阈值
- ✅ 支持80维配体特征 (35D V2 + 45D global)
- ✅ torch.compile 支持 (PyTorch 2.0+)
- ✅ 分布式训练 (DDP via torchrun)
- ✅ 自动混合精度 (AMP)

使用方法:
    # 单GPU
    python train_v6m7.py --config configs/pocket_gvp_v6.yaml

    # 单GPU + AMP
    python train_v6m7.py --config configs/pocket_gvp_v6.yaml --amp

    # 单GPU + torch.compile
    python train_v6m7.py --config configs/pocket_gvp_v6.yaml --compile

    # 双GPU分布式训练 + AMP
    torchrun --nproc_per_node=2 train_v6m7.py \
        --config configs/pocket_gvp_v6.yaml --amp

    # 双GPU + torch.compile + AMP
    torchrun --nproc_per_node=2 train_v6m7.py \
        --config configs/pocket_gvp_v6.yaml --compile --amp
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset_v6 import PocketDatasetV6, create_data_splits_v5, set_seed
from src.data.pocket_dataset import save_splits, load_splits
from src.data.collate import pocket_collate_fn, prepare_model_inputs
from src.models.pocket_gvp_fusion_v6m6 import PocketGVPFusionV6m6
from src.data.pocket_dataset_v6 import PocketDatasetV6
from src.metrics.metrics import compute_segmentation_metrics
from src.metrics.diagnostics import collect_model_diagnostics, compute_feature_statistics


# ============================================================================
# 分布式训练工具函数
# ============================================================================

def setup_distributed():
    """
    初始化分布式训练环境

    Returns:
        rank: 当前进程的rank
        world_size: 总进程数
        local_rank: 本地GPU ID
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单GPU或CPU模式
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        # 初始化进程组
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


# ============================================================================
# 配置和工具函数
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    """
    Setup experiment directory (only on rank 0).

    Args:
        config: Configuration dictionary
        rank: Process rank
        resume_path: Path to checkpoint being resumed from (if any)

    Returns:
        Experiment directory path
    """
    if rank != 0:
        return None

    # If resuming, use the original experiment directory
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            # Extract experiment directory from checkpoint path
            exp_dir = resume_path.parent
            print_rank0(f"Resuming in original directory: {exp_dir}")
            return exp_dir

    # Create new experiment directory
    exp_name = config['experiment']['name']
    # V6m6: Add v6m6 suffix to distinguish from v6/v6m2/v6m3/v6m4
    if not exp_name.endswith('_v6m6'):
        exp_name = exp_name.replace('_v6m4', '_v6m6') if '_v6m4' in exp_name else (exp_name.replace('_v6m3', '_v6m6') if '_v6m3' in exp_name else (exp_name.replace('_v6m2', '_v6m6') if '_v6m2' in exp_name else (exp_name.replace('_v6', '_v6m6') if '_v6' in exp_name else f"{exp_name}_v6m6")))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print_rank0(f"Created new experiment directory: {exp_dir}")
    return exp_dir


def create_model(config: Dict, device: torch.device) -> nn.Module:
    """Create model from config."""
    model_cfg = config['model']

    # Check for ablation flags
    ablation_flags = ['use_gvp', 'use_esm', 'use_physchem_early',
                      'use_physchem_late', 'use_self_attention']
    is_ablation = any(flag in model_cfg for flag in ablation_flags)

    if is_ablation:
        print_rank0("Creating ablation model...")
        model = PocketGVPFusionAblation(model_cfg)
    else:
        print_rank0("Creating V6m6 model (with LayerNorm and GELU in PhysChem Encoder)...")
        model = PocketGVPFusionV6m6(model_cfg)

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model


# ============================================================================
# 损失函数
# ============================================================================

def compute_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    计算 Dice Loss (用于分割任务)

    Args:
        logits: [B, L, C] 模型输出 logits
        labels: [B, L] 真实标签
        mask: [B, L] padding mask
        smooth: 平滑项，避免除零

    Returns:
        dice_loss: 标量损失值
    """
    B, L, C = logits.shape

    # 获取预测概率
    probs = F.softmax(logits, dim=-1)  # [B, L, C]

    # 将标签转换为 one-hot 编码
    labels_one_hot = F.one_hot(labels, num_classes=C).float()  # [B, L, C]

    # 应用 mask
    mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
    probs = probs * mask_expanded
    labels_one_hot = labels_one_hot * mask_expanded

    # 计算每个类别的 Dice 系数
    # Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    intersection = (probs * labels_one_hot).sum(dim=(0, 1))  # [C]
    cardinality = (probs + labels_one_hot).sum(dim=(0, 1))  # [C]

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)  # [C]

    # Dice Loss = 1 - Dice Score
    dice_loss = 1.0 - dice_score.mean()

    return dice_loss


def compute_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float = 0.0  # V6m7: Label smoothing
) -> torch.Tensor:
    """Compute focal loss with class weights, padding masking, and label smoothing."""
    B, L, C = logits.shape
    device = logits.device

    # Flatten
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    # Compute softmax probabilities
    probs = F.softmax(logits_flat, dim=-1)

    # Cross entropy loss with label smoothing (PyTorch built-in)
    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        reduction='none',
        label_smoothing=label_smoothing  # V6m7: Add label smoothing
    )

    # Focal weight
    p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - p_t) ** gamma

    # Apply class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
        sample_weights = class_weights[labels_flat]
        focal_loss = focal_weight * sample_weights * ce_loss
    else:
        focal_loss = focal_weight * ce_loss

    # Apply padding mask
    focal_loss = focal_loss * mask_flat.float()

    # Mean over valid tokens
    n_valid = mask_flat.sum()
    if n_valid > 0:
        loss = focal_loss.sum() / n_valid
    else:
        loss = focal_loss.sum() * 0

    return loss


def compute_combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    label_smoothing: float = 0.0  # V6m7: Label smoothing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算组合损失: Dice Loss + Focal Loss

    Args:
        logits: [B, L, C] 模型输出
        labels: [B, L] 真实标签
        mask: [B, L] padding mask
        class_weights: [C] 类别权重
        focal_gamma: Focal loss gamma 参数
        dice_weight: Dice loss 权重
        focal_weight: Focal loss 权重
        label_smoothing: Label smoothing 参数 (V6m7)

    Returns:
        total_loss: 总损失
        dice_loss: Dice 损失
        focal_loss: Focal 损失
    """
    # 计算 Dice Loss
    dice_loss = compute_dice_loss(logits, labels, mask)

    # 计算 Focal Loss (with label smoothing)
    focal_loss = compute_focal_loss(
        logits, labels, mask, class_weights, focal_gamma,
        label_smoothing=label_smoothing  # V6m7: Pass label smoothing
    )

    # 组合损失
    total_loss = dice_weight * dice_loss + focal_weight * focal_loss

    return total_loss, dice_loss, focal_loss


# ============================================================================
# 训练和评估函数
# ============================================================================


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int = 3
) -> Dict[str, float]:
    """Compute classification metrics."""
    preds = logits.argmax(dim=-1)

    # Flatten and apply mask
    preds_flat = preds[mask]
    labels_flat = labels[mask]

    n_samples = labels_flat.numel()
    if n_samples == 0:
        return {'accuracy': 0.0}

    # Accuracy
    correct = (preds_flat == labels_flat).sum().item()
    accuracy = correct / n_samples

    # Per-class metrics
    metrics = {'accuracy': accuracy}

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

    return metrics


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for different model components.

    Tracks gradient norms for:
    - GVP encoder (structure pathway)
    - Bridge projection (integration layer)
    - Protein context encoder (transformer)
    - Ligand encoder
    - Cross-attention layers
    - Classifier head

    Returns:
        Dict with gradient norms for each component
    """
    stats = {}

    # Helper to compute gradient norm for a module
    def get_grad_norm(module):
        total_norm = 0.0
        n_params = 0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                n_params += 1
        return (total_norm ** 0.5) if n_params > 0 else 0.0

    # Unwrap DDP if needed
    if isinstance(model, DDP):
        model = model.module

    # GVP encoder (structure pathway)
    if hasattr(model, 'gvp_encoder') and model.gvp_encoder is not None:
        stats['grad_norm_gvp_encoder'] = get_grad_norm(model.gvp_encoder)

    # Pairwise bridge (vector -> scalar invariant, no learnable params now)
    if hasattr(model, 'pairwise_bridge') and model.pairwise_bridge is not None:
        bridge_norm = get_grad_norm(model.pairwise_bridge)
        if bridge_norm > 0:  # Only log if has learnable params
            stats['grad_norm_pairwise_bridge'] = bridge_norm

    # Separate projection paths (new architecture)
    if hasattr(model, 'gvp_proj') and model.gvp_proj is not None:
        stats['grad_norm_gvp_proj'] = get_grad_norm(model.gvp_proj)

    if hasattr(model, 'esm_proj') and model.esm_proj is not None:
        stats['grad_norm_esm_proj'] = get_grad_norm(model.esm_proj)

    # Legacy: Bridge projection (old architecture fallback)
    if hasattr(model, 'bridge_proj') and model.bridge_proj is not None:
        stats['grad_norm_bridge_proj'] = get_grad_norm(model.bridge_proj)

    # Protein context encoder (transformer) - may be None when self-attention disabled
    if hasattr(model, 'protein_context_encoder') and model.protein_context_encoder is not None:
        stats['grad_norm_protein_encoder'] = get_grad_norm(model.protein_context_encoder)

    # Ligand encoder
    if hasattr(model, 'ligand_encoder') and model.ligand_encoder is not None:
        stats['grad_norm_ligand_encoder'] = get_grad_norm(model.ligand_encoder)

    # Cross-attention (support both old and new multi-layer format)
    cross_attn_norm = 0.0
    if hasattr(model, 'cross_attention_layers') and model.cross_attention_layers is not None:
        for layer in model.cross_attention_layers:
            cross_attn_norm += get_grad_norm(layer) ** 2
        stats['grad_norm_cross_attn'] = cross_attn_norm ** 0.5
    elif hasattr(model, 'cross_attention') and model.cross_attention is not None:
        stats['grad_norm_cross_attn'] = get_grad_norm(model.cross_attention)

    # Classifier head
    if hasattr(model, 'classifier') and model.classifier is not None:
        stats['grad_norm_classifier'] = get_grad_norm(model.classifier)

    # Compute modality gradient balance ratios
    # GVP pathway: gvp_encoder + gvp_proj
    gvp_total = stats.get('grad_norm_gvp_encoder', 0) + stats.get('grad_norm_gvp_proj', 0)
    # ESM pathway: esm_proj
    esm_total = stats.get('grad_norm_esm_proj', 0)

    # Combined modality ratio
    modality_total = gvp_total + esm_total
    if modality_total > 0:
        stats['grad_ratio_gvp_pathway'] = gvp_total / modality_total
        stats['grad_ratio_esm_pathway'] = esm_total / modality_total

    # Overall GVP ratio (encoder only, for backward compat)
    total_grad = sum(v for v in stats.values() if not v.__class__.__name__.startswith('str'))
    if total_grad > 0 and 'grad_norm_gvp_encoder' in stats:
        stats['grad_ratio_gvp'] = stats['grad_norm_gvp_encoder'] / total_grad

    return stats


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    class_weights: torch.Tensor,
    gamma: float,
    gradient_clip: float,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    log_every: int = 10,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    accumulation_steps: int = 1,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch with combined Dice + Focal loss."""
    model.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0
    total_samples = 0
    all_metrics = []

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Prepare inputs
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_scalar=inputs['protein_scalar'],
                protein_vectors=inputs['protein_vectors'],
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],  # V6m6: 3D coordinates
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

            logits = outputs['logits']

            # V6m3: 使用组合损失
            loss, dice_loss, focal_loss = compute_combined_loss(
                logits, labels, mask, class_weights, gamma,
                dice_weight=dice_weight, focal_weight=focal_weight,
                label_smoothing=0.1  # V6m7: Label smoothing
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with AMP
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % accumulation_steps == 0:
            # ✅ Record gradients BEFORE clipping and optimizer step
            if writer is not None and is_main_process() and (batch_idx + 1) % log_every == 0:
                global_step = epoch * len(dataloader) + batch_idx

                # 梯度统计（在清零之前记录）
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Train/grad_norm', total_norm, global_step)

                # 组件级梯度统计
                grad_stats = compute_gradient_stats(model)
                for key, value in grad_stats.items():
                    writer.add_scalar(f'Gradients/{key}', value, global_step)

            # Gradient clipping
            if gradient_clip > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # Optimizer step
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # Track metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * accumulation_steps * batch_size
        total_dice_loss += dice_loss.item() * batch_size
        total_focal_loss += focal_loss.item() * batch_size
        total_samples += batch_size

        batch_metrics = compute_metrics(logits.detach(), labels, mask)
        all_metrics.append(batch_metrics)

        # Logging
        if (batch_idx + 1) % log_every == 0 and is_main_process():
            elapsed = time.time() - start_time

            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item() * accumulation_steps:.4f} | "
                  f"Dice: {dice_loss.item():.4f} | "
                  f"Focal: {focal_loss.item():.4f} | "
                  f"Acc: {batch_metrics['accuracy']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/batch_loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('Train/dice_loss', dice_loss.item(), global_step)
                writer.add_scalar('Train/focal_loss', focal_loss.item(), global_step)
                writer.add_scalar('Train/batch_acc', batch_metrics['accuracy'], global_step)

                # 健康诊断（GVP范数、门控开放度、注意力熵）
                diagnostics = collect_model_diagnostics(model)
                if 'gvp_scalar_norm' in diagnostics:
                    writer.add_scalar('Health/gvp_scalar_norm', diagnostics['gvp_scalar_norm'], global_step)
                if 'gvp_vector_norm' in diagnostics:
                    writer.add_scalar('Health/gvp_vector_norm', diagnostics['gvp_vector_norm'], global_step)
                if 'gate_openness' in diagnostics:
                    writer.add_scalar('Health/gate_openness', diagnostics['gate_openness'], global_step)
                if 'attn_entropy' in diagnostics:
                    writer.add_scalar('Health/attn_entropy', diagnostics['attn_entropy'], global_step)

                # 特征统计（ESM vs GVP贡献）
                _model = model.module if isinstance(model, DDP) else model
                if hasattr(_model, '_cached_gvp_scalar') and hasattr(_model, '_cached_current_esm'):
                    feature_stats = compute_feature_statistics(
                        esm_embeddings=_model._cached_current_esm,
                        gvp_scalar=_model._cached_gvp_scalar,
                        mask=mask,
                        esm_mask=_model._cached_esm_mask if hasattr(_model, '_cached_esm_mask') else None,
                        gvp_vector=_model._cached_gvp_vector if hasattr(_model, '_cached_gvp_vector') else None,
                        h_geo=_model._cached_h_geo if hasattr(_model, '_cached_h_geo') else None,
                        raw_esm_embeddings=inputs['esm_embeddings'],
                        h_gvp_proj=_model._cached_h_gvp_proj if hasattr(_model, '_cached_h_gvp_proj') else None,
                        h_esm_proj=_model._cached_h_esm_proj if hasattr(_model, '_cached_h_esm_proj') else None
                    )

                    for key, value in feature_stats.items():
                        writer.add_scalar(f'Health/{key}', value, global_step)

    # Aggregate metrics
    avg_loss = total_loss / max(total_samples, 1)
    avg_dice_loss = total_dice_loss / max(total_samples, 1)
    avg_focal_loss = total_focal_loss / max(total_samples, 1)
    avg_metrics = {
        'loss': avg_loss,
        'dice_loss': avg_dice_loss,
        'focal_loss': avg_focal_loss,
        'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
    }

    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    class_weights: torch.Tensor,
    gamma: float,
    device: torch.device,
    use_amp: bool = False,
    evaluate_original: bool = False,  # V5: 是否同时评估原始标签
    dice_weight: float = 0.5,
    focal_weight: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model with segmentation metrics and combined loss.

    V6m3: 使用 Dice + Focal 组合损失
    V5: 支持同时评估 VDW 标签和原始固定阈值标签
    """
    model.eval()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0
    total_samples = 0
    all_metrics = []

    # 累积预测和标签用于计算epoch级别的分割指标
    all_preds = []
    all_labels = []

    # V5: 如果需要评估原始标签
    total_loss_original = 0.0
    total_dice_loss_original = 0.0
    total_focal_loss_original = 0.0
    total_samples_original = 0
    all_metrics_original = []
    all_preds_original = []
    all_labels_original = []

    for batch in dataloader:
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)  # VDW labels
        mask = batch['protein_mask'].to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_scalar=inputs['protein_scalar'],
                protein_vectors=inputs['protein_vectors'],
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],  # V6m6: 3D coordinates
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

            logits = outputs['logits']

            # V6m3: VDW标签的组合loss和metrics
            loss, dice_loss, focal_loss = compute_combined_loss(
                logits, labels, mask, class_weights, gamma,
                dice_weight=dice_weight, focal_weight=focal_weight
            )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_dice_loss += dice_loss.item() * batch_size
        total_focal_loss += focal_loss.item() * batch_size
        total_samples += batch_size

        batch_metrics = compute_metrics(logits, labels, mask)
        all_metrics.append(batch_metrics)

        # 累积预测用于分割指标
        preds = logits.argmax(dim=-1)
        all_preds.append(preds[mask].cpu())
        all_labels.append(labels[mask].cpu())

        # V5: 如果需要评估原始标签
        if evaluate_original and 'labels_original' in batch:
            labels_original = batch['labels_original'].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_original, dice_loss_original, focal_loss_original = compute_combined_loss(
                    logits, labels_original, mask, class_weights, gamma,
                    dice_weight=dice_weight, focal_weight=focal_weight
                )

            total_loss_original += loss_original.item() * batch_size
            total_dice_loss_original += dice_loss_original.item() * batch_size
            total_focal_loss_original += focal_loss_original.item() * batch_size
            total_samples_original += batch_size

            batch_metrics_original = compute_metrics(logits, labels_original, mask)
            all_metrics_original.append(batch_metrics_original)

            all_preds_original.append(preds[mask].cpu())
            all_labels_original.append(labels_original[mask].cpu())

    # Aggregate VDW metrics
    avg_loss = total_loss / max(total_samples, 1)
    avg_dice_loss = total_dice_loss / max(total_samples, 1)
    avg_focal_loss = total_focal_loss / max(total_samples, 1)
    avg_metrics = {
        'loss': avg_loss,
        'dice_loss': avg_dice_loss,
        'focal_loss': avg_focal_loss,
        'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
    }

    # Per-class metrics
    for c in range(3):
        avg_metrics[f'precision_c{c}'] = sum(m[f'precision_c{c}'] for m in all_metrics) / len(all_metrics)
        avg_metrics[f'recall_c{c}'] = sum(m[f'recall_c{c}'] for m in all_metrics) / len(all_metrics)
        avg_metrics[f'f1_c{c}'] = sum(m[f'f1_c{c}'] for m in all_metrics) / len(all_metrics)

    # 计算epoch级别的分割指标（DICE、IoU）
    if all_preds and all_labels:
        epoch_preds = torch.cat(all_preds)
        epoch_labels = torch.cat(all_labels)
        seg_metrics = compute_segmentation_metrics(
            epoch_preds, epoch_labels,
            mask=torch.ones_like(epoch_preds, dtype=torch.bool),  # 已经过滤
            contact_class=0
        )
        avg_metrics['dice_c0'] = seg_metrics['dice']
        avg_metrics['iou_c0'] = seg_metrics['iou']

    # V5: Aggregate original metrics if available
    if evaluate_original and all_metrics_original:
        avg_loss_original = total_loss_original / max(total_samples_original, 1)
        avg_dice_loss_original = total_dice_loss_original / max(total_samples_original, 1)
        avg_focal_loss_original = total_focal_loss_original / max(total_samples_original, 1)
        avg_metrics['loss_original'] = avg_loss_original
        avg_metrics['dice_loss_original'] = avg_dice_loss_original
        avg_metrics['focal_loss_original'] = avg_focal_loss_original
        avg_metrics['accuracy_original'] = sum(m['accuracy'] for m in all_metrics_original) / len(all_metrics_original)

        # Per-class metrics for original
        for c in range(3):
            avg_metrics[f'precision_c{c}_original'] = sum(m[f'precision_c{c}'] for m in all_metrics_original) / len(all_metrics_original)
            avg_metrics[f'recall_c{c}_original'] = sum(m[f'recall_c{c}'] for m in all_metrics_original) / len(all_metrics_original)
            avg_metrics[f'f1_c{c}_original'] = sum(m[f'f1_c{c}'] for m in all_metrics_original) / len(all_metrics_original)

        # Segmentation metrics for original
        if all_preds_original and all_labels_original:
            epoch_preds_original = torch.cat(all_preds_original)
            epoch_labels_original = torch.cat(all_labels_original)
            seg_metrics_original = compute_segmentation_metrics(
                epoch_preds_original, epoch_labels_original,
                mask=torch.ones_like(epoch_preds_original, dtype=torch.bool),
                contact_class=0
            )
            avg_metrics['dice_c0_original'] = seg_metrics_original['dice']
            avg_metrics['iou_c0_original'] = seg_metrics_original['iou']

    return avg_metrics


# ============================================================================
# Main函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Pocket-GVP-Fusion v2 (Optimized)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')

    # Optimization arguments
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    print_rank0(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = config['experiment']['seed']
    set_seed(seed + rank)  # Different seed per rank
    print_rank0(f"Random seed: {seed}")

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    print_rank0(f"Device: {device}")

    # AMP setup
    use_amp = args.amp or config['training'].get('use_amp', False)
    if use_amp:
        print_rank0("Using Automatic Mixed Precision (AMP)")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Setup experiment directory (only rank 0)
    # Pass resume path to reuse original directory
    exp_dir = setup_experiment(config, rank, resume_path=args.resume)
    if rank == 0 and exp_dir:
        print_rank0(f"Experiment directory: {exp_dir}")

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    # Data
    data_cfg = config['data']
    data_root = data_cfg["data_root"]

    # Create or load splits
    if rank == 0:
        split_file = exp_dir / "splits.txt" if exp_dir else Path("splits.txt")

        if data_cfg.get('split_file') and Path(data_cfg['split_file']).exists():
            train_ids, val_ids, test_ids = load_splits(data_cfg['split_file'])
        else:
            # V5: Pass ligand_features_dir to split function
            ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v3')
            train_ids, val_ids, test_ids = create_data_splits_v5(
                data_root,
                train_ratio=data_cfg['train_ratio'],
                val_ratio=data_cfg['val_ratio'],
                test_ratio=data_cfg['test_ratio'],
                seed=seed,
                ligand_features_dir=ligand_features_dir
            )
            save_splits(train_ids, val_ids, test_ids, split_file)

        print_rank0(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    else:
        train_ids = val_ids = test_ids = None

    # Broadcast splits to all ranks
    if world_size > 1:
        # Use object list for broadcasting
        splits = [train_ids, val_ids, test_ids] if rank == 0 else [None, None, None]
        dist.broadcast_object_list(splits, src=0)
        train_ids, val_ids, test_ids = splits

    # Create datasets
    train_cfg = config['training']

    # V5: Get ligand features directory from config
    ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v3')
    distances_file = data_cfg.get('distances_file')
    distances_file_original = data_cfg.get('distances_file_original')

    if is_main_process():
        # Print once (avoid per-dataset noise)
        print_rank0("Data sources:")
        print_rank0(f"  ligand_features_dir: data/{ligand_features_dir}")
        print_rank0(f"  distances_file (vdw): {distances_file or 'DEFAULT (data/residue_ligand_distances_v2.pt)'}")
        if distances_file_original:
            print_rank0(f"  distances_file_original: {distances_file_original}")

    train_dataset = PocketDatasetV6(
        data_root=data_root,
        pdb_ids=train_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,  # for comparison
        ligand_features_dir=ligand_features_dir,  # configurable
        split="train",
        verbose=is_main_process(),
        print_fn=print_rank0
    )

    val_dataset = PocketDatasetV6(
        data_root=data_root,
        pdb_ids=val_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,  # for comparison
        ligand_features_dir=ligand_features_dir,  # configurable
        split="val",
        verbose=False,
        print_fn=print_rank0
    )

    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg['num_workers'],
        collate_fn=pocket_collate_fn,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=pocket_collate_fn,
        pin_memory=True
    )

    # Model
    model = create_model(config, device)

    # torch.compile (PyTorch 2.0+)
    if args.compile:
        print_rank0(f"Compiling model with mode={args.compile_mode}...")
        model = torch.compile(model, mode=args.compile_mode)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print_rank0("Model wrapped with DistributedDataParallel")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=train_cfg['betas']
    )

    # Scheduler
    warmup_epochs = train_cfg['warmup_epochs']
    total_epochs = train_cfg['num_epochs']

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=train_cfg['min_lr']
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        if Path(args.resume).exists():
            print_rank0(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load model state
            if world_size > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load scaler state (for AMP)
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)

            print_rank0(f"✓ Resumed from epoch {checkpoint['epoch'] + 1}")
            print_rank0(f"  Best val loss: {best_val_loss:.4f}")
            print_rank0(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print_rank0(f"⚠️  Checkpoint not found: {args.resume}")
            print_rank0("Starting training from scratch...")
    else:
        print_rank0("Starting training from scratch...")

    # Class weights
    class_weights = torch.tensor(train_cfg['class_weights'], dtype=torch.float32).to(device)

    # TensorBoard (only rank 0)
    if rank == 0 and config['logging']['tensorboard']:
        tb_dir = exp_dir / config['logging']['tensorboard_dir']
        writer = SummaryWriter(tb_dir)
    else:
        writer = None

    # Training loop
    print_rank0("\n" + "=" * 60)
    print_rank0("Starting training...")
    print_rank0("=" * 60)

    for epoch in range(start_epoch, total_epochs):
        print_rank0(f"\nEpoch {epoch + 1}/{total_epochs}")

        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # V6m3: 获取损失权重配置
        dice_weight = train_cfg.get('dice_weight', 0.5)
        focal_weight = train_cfg.get('focal_weight', 0.5)

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            gamma=train_cfg['focal_gamma'],
            gradient_clip=train_cfg['gradient_clip'],
            device=device,
            epoch=epoch,
            writer=writer,
            log_every=config['logging']['log_every'],
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=args.accumulation_steps,
            dice_weight=dice_weight,
            focal_weight=focal_weight
        )

        # Evaluate
        # V5: Evaluate with both VDW and original labels
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            class_weights=class_weights,
            gamma=train_cfg['focal_gamma'],
            device=device,
            use_amp=use_amp,
            evaluate_original=True,  # V5: 同时评估原始标签
            dice_weight=dice_weight,
            focal_weight=focal_weight
        )

        # Scheduler step
        scheduler.step()

        # Logging
        if rank == 0:
            print_rank0(f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice_loss']:.4f}, Focal: {train_metrics['focal_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print_rank0(f"Val Loss (VDW): {val_metrics['loss']:.4f}, Dice: {val_metrics['dice_loss']:.4f}, Focal: {val_metrics['focal_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # V5: 打印原始标签的指标
            if 'loss_original' in val_metrics:
                print_rank0(f"Val Loss (Original): {val_metrics['loss_original']:.4f}, Dice: {val_metrics.get('dice_loss_original', 0):.4f}, Focal: {val_metrics.get('focal_loss_original', 0):.4f}, Acc: {val_metrics['accuracy_original']:.4f}")

            # 打印分割指标
            if 'dice_c0' in val_metrics:
                print_rank0(f"Val DICE (VDW): {val_metrics['dice_c0']:.4f}, IoU: {val_metrics['iou_c0']:.4f}, F1_c0: {val_metrics.get('f1_c0', 0):.4f}")
                if 'dice_c0_original' in val_metrics:
                    print_rank0(f"Val DICE (Original): {val_metrics['dice_c0_original']:.4f}, IoU: {val_metrics['iou_c0_original']:.4f}")

            if writer is not None:
                # V6: 命名空间策略（用于横向对比）
                # Train/ - 训练指标（基于新标签VDW，用于优化）
                # Val/ - 原始标签的验证指标（用于与之前模型对比）
                # Val_new/ - 新标签的验证指标（VDW调整后）
                # Health/ - 健康诊断指标

                # 训练指标（基于新标签VDW）
                writer.add_scalar('Train/epoch_loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/epoch_dice_loss', train_metrics['dice_loss'], epoch)
                writer.add_scalar('Train/epoch_focal_loss', train_metrics['focal_loss'], epoch)
                writer.add_scalar('Train/epoch_acc', train_metrics['accuracy'], epoch)
                writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

                # 原始标签的验证指标 → Val/（用于横向对比）
                if 'loss_original' in val_metrics:
                    writer.add_scalar('Val/epoch_loss', val_metrics['loss_original'], epoch)
                    writer.add_scalar('Val/dice_loss', val_metrics.get('dice_loss_original', 0), epoch)
                    writer.add_scalar('Val/focal_loss', val_metrics.get('focal_loss_original', 0), epoch)
                    writer.add_scalar('Val/epoch_acc', val_metrics['accuracy_original'], epoch)

                    # 每类验证指标 - Original → Val/
                    for k, v in val_metrics.items():
                        if k.endswith('_original'):
                            metric_name = k.replace('_original', '')
                            if k.startswith('f1_') or k.startswith('precision_') or k.startswith('recall_'):
                                writer.add_scalar(f'Val/{metric_name}', v, epoch)

                    # 分割指标 - Original → Val/
                    if 'dice_c0_original' in val_metrics:
                        writer.add_scalar('Val/dice_c0', val_metrics['dice_c0_original'], epoch)
                        writer.add_scalar('Val/iou_c0', val_metrics['iou_c0_original'], epoch)

                # 新标签（VDW）的验证指标 → Val_new/
                writer.add_scalar('Val_new/epoch_loss', val_metrics['loss'], epoch)
                writer.add_scalar('Val_new/dice_loss', val_metrics['dice_loss'], epoch)
                writer.add_scalar('Val_new/focal_loss', val_metrics['focal_loss'], epoch)
                writer.add_scalar('Val_new/epoch_acc', val_metrics['accuracy'], epoch)

                # 每类验证指标 - VDW → Val_new/
                for k, v in val_metrics.items():
                    if k.startswith('f1_') or k.startswith('precision_') or k.startswith('recall_'):
                        if not k.endswith('_original'):
                            writer.add_scalar(f'Val_new/{k}', v, epoch)

                # 分割指标 - VDW → Val_new/
                if 'dice_c0' in val_metrics:
                    writer.add_scalar('Val_new/dice_c0', val_metrics['dice_c0'], epoch)
                    writer.add_scalar('Val_new/iou_c0', val_metrics['iou_c0'], epoch)

                # 分割指标 - Original → Val_original/ (仅当存在时)
                if 'dice_c0_original' in val_metrics:
                    writer.add_scalar('Val_original/dice_c0', val_metrics['dice_c0_original'], epoch)
                    writer.add_scalar('Val_original/iou_c0', val_metrics['iou_c0_original'], epoch)

            # ✅ Save checkpoints with complete state
            # 1. Save latest checkpoint after EVERY epoch
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }
            if scaler is not None:
                latest_checkpoint['scaler_state_dict'] = scaler.state_dict()

            torch.save(latest_checkpoint, exp_dir / 'checkpoint_latest.pt')

            # 2. Save periodic backup every N epochs
            if (epoch + 1) % train_cfg['save_every'] == 0:
                backup_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }
                if scaler is not None:
                    backup_checkpoint['scaler_state_dict'] = scaler.state_dict()

                torch.save(backup_checkpoint, exp_dir / f'checkpoint_epoch_{epoch + 1}.pt')
                print_rank0(f"✓ Saved checkpoint at epoch {epoch + 1}")

            # 3. Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'val_metrics': val_metrics,
                    'config': config
                }
                if scaler is not None:
                    best_checkpoint['scaler_state_dict'] = scaler.state_dict()

                torch.save(best_checkpoint, exp_dir / 'best_model.pt')
                print_rank0(f"✓ Saved best model (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= train_cfg['early_stopping_patience']:
                print_rank0(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Cleanup
    if writer is not None:
        writer.close()

    cleanup_distributed()

    print_rank0("\n" + "=" * 60)
    print_rank0("Training completed!")
    print_rank0("=" * 60)


if __name__ == '__main__':
    main()
