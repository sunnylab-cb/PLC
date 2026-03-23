#!/usr/bin/env python3
"""
优化版训练脚本 for Pocket-GVP-Fusion v2

新增功能:
- ✅ torch.compile 支持 (PyTorch 2.0+)
- ✅ 分布式训练 (DDP via torchrun)
- ✅ 自动混合精度 (AMP)
- ✅ 梯度累积
- ✅ 更好的性能监控

使用方法:
    # 单GPU
    python train_v3_optimized.py --config configs/pocket_gvp_v2_nested.yaml

    # 单GPU + AMP
    python train_v3_optimized.py --config configs/pocket_gvp_v2_nested.yaml --amp

    # 单GPU + torch.compile
    python train_v3_optimized.py --config configs/pocket_gvp_v2_nested.yaml --compile

    # 双GPU分布式训练 + AMP
    torchrun --nproc_per_node=2 train_v3_optimized.py \\
        --config configs/pocket_gvp_v2_nested.yaml --amp

    # 双GPU + torch.compile + AMP
    torchrun --nproc_per_node=2 train_v3_optimized.py \\
        --config configs/pocket_gvp_v2_nested.yaml --compile --amp
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pocket_dataset_v3 import PocketDatasetV3, create_data_splits_v3, set_seed
from src.data.pocket_dataset import save_splits, load_splits
from src.data.collate import pocket_collate_fn, prepare_model_inputs
from src.models.pocket_gvp_fusion import PocketGVPFusion, PocketGVPFusionAblation
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


def setup_experiment(config: Dict, rank: int) -> Path:
    """Setup experiment directory (only on rank 0)."""
    if rank != 0:
        return None

    exp_name = config['experiment']['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

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
        print_rank0("Creating standard model...")
        model = PocketGVPFusion(model_cfg)

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model


# ============================================================================
# 训练和评估函数
# ============================================================================

def compute_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """Compute focal loss with class weights and padding masking."""
    B, L, C = logits.shape
    device = logits.device

    # Flatten
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    # Compute softmax probabilities
    probs = F.softmax(logits_flat, dim=-1)

    # Cross entropy loss
    ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')

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
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch with AMP and gradient accumulation support."""
    model.train()

    total_loss = 0.0
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
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

            logits = outputs['logits']
            loss = compute_focal_loss(logits, labels, mask, class_weights, gamma)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with AMP
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % accumulation_steps == 0:
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
        total_loss += loss.item() * accumulation_steps * mask.sum().item()
        total_samples += mask.sum().item()

        batch_metrics = compute_metrics(logits.detach(), labels, mask)
        all_metrics.append(batch_metrics)

        # Logging
        if (batch_idx + 1) % log_every == 0 and is_main_process():
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(total_samples, 1)

            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item() * accumulation_steps:.4f} | "
                  f"Acc: {batch_metrics['accuracy']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/batch_loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('Train/batch_acc', batch_metrics['accuracy'], global_step)

                # 梯度统计
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
    avg_metrics = {
        'loss': avg_loss,
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
    use_amp: bool = False
) -> Dict[str, float]:
    """Evaluate model with segmentation metrics."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_metrics = []

    # 累积预测和标签用于计算epoch级别的分割指标
    all_preds = []
    all_labels = []

    for batch in dataloader:
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
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

            logits = outputs['logits']
            loss = compute_focal_loss(logits, labels, mask, class_weights, gamma)

        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()

        batch_metrics = compute_metrics(logits, labels, mask)
        all_metrics.append(batch_metrics)

        # 累积预测用于分割指标
        preds = logits.argmax(dim=-1)
        all_preds.append(preds[mask].cpu())
        all_labels.append(labels[mask].cpu())

    # Aggregate
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {
        'loss': avg_loss,
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
    exp_dir = setup_experiment(config, rank)
    if rank == 0:
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
            train_ids, val_ids, test_ids = create_data_splits_v3(
                data_root,
                train_ratio=data_cfg['train_ratio'],
                val_ratio=data_cfg['val_ratio'],
                test_ratio=data_cfg['test_ratio'],
                seed=seed
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

    train_dataset = PocketDatasetV3(
        data_root=data_root,
        pdb_ids=train_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=data_cfg.get('distances_file')
    )

    val_dataset = PocketDatasetV3(
        data_root=data_root,
        pdb_ids=val_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=data_cfg.get('distances_file')
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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
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

    # Class weights
    class_weights = torch.tensor(train_cfg['class_weights'], dtype=torch.float32).to(device)

    # TensorBoard (only rank 0)
    if rank == 0 and config['logging']['tensorboard']:
        tb_dir = exp_dir / config['logging']['tensorboard_dir']
        writer = SummaryWriter(tb_dir)
    else:
        writer = None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print_rank0("\n" + "=" * 60)
    print_rank0("Starting training...")
    print_rank0("=" * 60)

    for epoch in range(total_epochs):
        print_rank0(f"\nEpoch {epoch + 1}/{total_epochs}")

        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)

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
            accumulation_steps=args.accumulation_steps
        )

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            class_weights=class_weights,
            gamma=train_cfg['focal_gamma'],
            device=device,
            use_amp=use_amp
        )

        # Scheduler step
        scheduler.step()

        # Logging
        if rank == 0:
            print_rank0(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print_rank0(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # 打印分割指标
            if 'dice_c0' in val_metrics:
                print_rank0(f"Val DICE: {val_metrics['dice_c0']:.4f}, IoU: {val_metrics['iou_c0']:.4f}, F1_c0: {val_metrics.get('f1_c0', 0):.4f}")

            if writer is not None:
                # Epoch级别的损失和准确率
                writer.add_scalar('Train/epoch_loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/epoch_acc', train_metrics['accuracy'], epoch)
                writer.add_scalar('Val/epoch_loss', val_metrics['loss'], epoch)
                writer.add_scalar('Val/epoch_acc', val_metrics['accuracy'], epoch)
                writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

                # 每类验证指标（precision、recall、F1）
                for k, v in val_metrics.items():
                    if k.startswith('f1_') or k.startswith('precision_') or k.startswith('recall_'):
                        writer.add_scalar(f'Val/{k}', v, epoch)

                # 分割指标（DICE、IoU）
                if 'dice_c0' in val_metrics:
                    writer.add_scalar('Val/dice_c0', val_metrics['dice_c0'], epoch)
                    writer.add_scalar('Val/iou_c0', val_metrics['iou_c0'], epoch)

            # Save checkpoint
            if (epoch + 1) % train_cfg['save_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()

                torch.save(checkpoint, exp_dir / f'checkpoint_epoch_{epoch + 1}.pt')

            # Best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }
                torch.save(checkpoint, exp_dir / 'best_model.pt')
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
