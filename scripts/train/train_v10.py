#!/usr/bin/env python3
"""
Training script for Pocket-GVP-Fusion V10.

V10 updates:
- protein complete_features_v4
- ligand_features_v5
- dynamic collate for updated protein vector/scalar dimensions
- model input dimensions updated for the new feature pipelines
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pocket_dataset_v8 import PocketDatasetV8, set_seed
from src.data.pocket_dataset import save_splits, load_splits, load_splits_with_cluster_size
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v10
from src.models.pocket_gvp_fusion_v10 import PocketGVPFusionV10
from src.data.pocket_dataset_v6 import create_data_splits_v5
from src.metrics.metrics import compute_segmentation_metrics, compute_enhanced_metrics
from src.metrics.diagnostics import collect_model_diagnostics, compute_feature_statistics
from src.losses import compute_combined_iou_loss, compute_iou_metric
from src.losses.boundary_loss import compute_combined_loss_with_boundary
from src.utils.performance_monitor import PerformanceMonitor, DataLoaderProfiler


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    if rank != 0:
        return None

    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            exp_dir = resume_path.parent
            print_rank0(f"Resuming in original directory: {exp_dir}")
            return exp_dir

    raw_exp_name = config['experiment']['name']
    version_tokens = {'v6', 'v7', 'v7m1', 'v7m2', 'v7m3', 'v8', 'v9', 'v9m2', 'v9m3', 'v10'}
    exp_tokens = [token for token in raw_exp_name.split('_') if token]
    normalized_tokens = [token for token in exp_tokens if token not in version_tokens]
    exp_name = '_'.join(normalized_tokens + ['v10'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print_rank0(f"Created new experiment directory: {exp_dir}")
    return exp_dir


def create_model(config: Dict, device: torch.device, use_gradient_checkpointing: bool = None, use_compile: bool = None) -> nn.Module:
    model_cfg = config['model']

    if use_gradient_checkpointing is not None:
        model_cfg['use_gradient_checkpointing'] = use_gradient_checkpointing
        print_rank0(f"Gradient checkpointing: {use_gradient_checkpointing} (from CLI)")
    elif 'use_gradient_checkpointing' in model_cfg:
        print_rank0(f"Gradient checkpointing: {model_cfg['use_gradient_checkpointing']} (from config)")
    else:
        model_cfg['use_gradient_checkpointing'] = False
        print_rank0("Gradient checkpointing: False (default)")

    if use_compile is not None:
        enable_compile = use_compile
        print_rank0(f"Torch.compile: {enable_compile} (from CLI)")
    elif 'use_compile' in model_cfg:
        enable_compile = model_cfg['use_compile']
        print_rank0(f"Torch.compile: {enable_compile} (from config)")
    else:
        enable_compile = False
        print_rank0("Torch.compile: False (default)")

    print_rank0("Creating V10 model (complete_features_v4 + ligand_features_v5)...")
    model = PocketGVPFusionV10(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    if enable_compile:
        print_rank0("Compiling model with torch.compile (dynamic=True, mode='reduce-overhead')...")
        print_rank0("Note: First epoch will be slow due to compilation overhead")
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        print_rank0("Model compiled successfully")

    return model


class ClusterWeightedLoss:
    def __init__(self, cluster_sizes: Dict[str, int], epsilon: float = 2.0):
        self.cluster_sizes = cluster_sizes
        self.epsilon = epsilon
        self.weights = {}
        for pdb_id, size in cluster_sizes.items():
            self.weights[pdb_id] = 1.0 / torch.log(torch.tensor(float(size) + epsilon))

    def get_batch_weights(self, pdb_ids: list) -> torch.Tensor:
        return torch.tensor([
            self.weights[pdb_id].item() if pdb_id in self.weights else 1.0
            for pdb_id in pdb_ids
        ], dtype=torch.float32)

    def print_statistics(self):
        values = list(self.weights.values())
        numeric_values = [float(v.item()) for v in values]
        print_rank0(f"  Samples with weights: {len(numeric_values)}")
        print_rank0(f"  Weight range: [{min(numeric_values):.4f}, {max(numeric_values):.4f}]")
        print_rank0(f"  Mean weight: {sum(numeric_values) / len(numeric_values):.4f}")


def compute_dice_loss(logits, labels, mask, smooth=1.0):
    probs = F.softmax(logits, dim=-1)
    n_classes = logits.size(-1)
    labels_one_hot = F.one_hot(labels, num_classes=n_classes).float()
    mask_expanded = mask.unsqueeze(-1).float()
    probs = probs * mask_expanded
    labels_one_hot = labels_one_hot * mask_expanded
    intersection = (probs * labels_one_hot).sum(dim=(0, 1))
    union = probs.sum(dim=(0, 1)) + labels_one_hot.sum(dim=(0, 1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def compute_focal_loss(logits, labels, mask, class_weights, gamma=2.0, label_smoothing=0.0):
    B, L, C = logits.shape
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
        label_smoothing=label_smoothing
    )

    probs = F.softmax(logits_flat, dim=-1)
    pt = probs[torch.arange(len(labels_flat), device=labels_flat.device), labels_flat]
    focal_factor = (1 - pt) ** gamma

    if class_weights is not None:
        alpha = class_weights[labels_flat]
        loss = alpha * focal_factor * ce_loss
    else:
        loss = focal_factor * ce_loss

    loss = loss * mask_flat.float()
    return loss.sum() / mask_flat.float().sum().clamp_min(1.0)


def compute_combined_loss(logits, labels, mask, class_weights, gamma, dice_weight=0.4, focal_weight=0.4, label_smoothing=0.0):
    dice_loss = compute_dice_loss(logits, labels, mask)
    focal_loss = compute_focal_loss(logits, labels, mask, class_weights, gamma, label_smoothing=label_smoothing)
    total_loss = dice_weight * dice_loss + focal_weight * focal_loss
    return total_loss, dice_loss, focal_loss


def compute_combined_loss_with_sample_weights(logits, labels, mask, class_weights, sample_weights, gamma, dice_weight=0.4, focal_weight=0.4, label_smoothing=0.0):
    weighted_mask = mask.float() * sample_weights[:, None].to(mask.device)
    dice_loss = compute_dice_loss(logits, labels, weighted_mask > 0)

    B, L, C = logits.shape
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    sample_weight_flat = sample_weights[:, None].expand(B, L).reshape(-1).to(logits.device)
    mask_flat = mask.reshape(-1)
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
        label_smoothing=label_smoothing
    )
    probs = F.softmax(logits_flat, dim=-1)
    pt = probs[torch.arange(len(labels_flat), device=labels_flat.device), labels_flat]
    focal_factor = (1 - pt) ** gamma
    alpha = class_weights[labels_flat] if class_weights is not None else 1.0
    focal_loss = alpha * focal_factor * ce_loss * sample_weight_flat * mask_flat.float()
    focal_loss = focal_loss.sum() / (sample_weight_flat * mask_flat.float()).sum().clamp_min(1.0)
    total_loss = dice_weight * dice_loss + focal_weight * focal_loss
    return total_loss, dice_loss, focal_loss


def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, device, epoch, class_weights,
    gamma=2.0, dice_weight=0.4, focal_weight=0.4, log_every=10,
    writer=None, use_amp=False, iou_weight=0.0, cluster_weighted_loss=None,
    use_boundary_loss=False, boundary_weight=0.2, boundary_multiplier=2.0,
    boundary_distance_threshold=8.0, label_smoothing=0.15,
    enable_profiling=False, exp_dir: Optional[Path] = None,
    accumulation_steps: int = 1, use_bf16: bool = False, gradient_clip: float = 1.0
):
    model.train()
    # Tensor accumulators — avoid per-batch CPU-GPU sync via .item()
    total_loss = torch.tensor(0.0, device=device)
    total_dice_loss = torch.tensor(0.0, device=device)
    total_focal_loss = torch.tensor(0.0, device=device)
    total_iou_loss = torch.tensor(0.0, device=device)
    total_boundary_loss = torch.tensor(0.0, device=device)
    total_boundary_ratio = 0.0
    total_samples = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    optimizer.zero_grad(set_to_none=True)

    perf_monitor = PerformanceMonitor(log_dir=exp_dir / 'performance_logs') if (enable_profiling and exp_dir and is_main_process()) else None

    for batch_idx, batch in enumerate(dataloader):
        if perf_monitor:
            perf_monitor.start('data_loading')

        inputs = prepare_model_inputs_v10(batch, device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask = batch['protein_mask'].to(device, non_blocking=True)

        if perf_monitor:
            perf_monitor.end('data_loading')
            perf_monitor.start('forward')

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_physchem=inputs['protein_physchem'],
                protein_spatial_scalar=inputs['protein_spatial_scalar'],
                protein_spatial_vector=inputs['protein_spatial_vector'],
                protein_gvp_vectors=inputs['protein_gvp_vectors'],
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )
            logits = outputs['logits']

            if perf_monitor:
                perf_monitor.end('forward')
                perf_monitor.start('loss_computation')

            if use_boundary_loss:
                loss, dice_loss, focal_loss, boundary_loss_val, boundary_ratio = compute_combined_loss_with_boundary(
                    logits=logits,
                    labels=labels,
                    coords=inputs['protein_coords'],
                    mask=mask,
                    class_weights=class_weights,
                    focal_gamma=gamma,
                    dice_weight=dice_weight,
                    focal_weight=focal_weight,
                    boundary_weight=boundary_weight,
                    boundary_multiplier=boundary_multiplier,
                    distance_threshold=boundary_distance_threshold,
                    label_smoothing=label_smoothing
                )
            elif cluster_weighted_loss is not None:
                pdb_ids = batch['pdb_ids']
                sample_weights = cluster_weighted_loss.get_batch_weights(pdb_ids)
                loss, dice_loss, focal_loss = compute_combined_loss_with_sample_weights(
                    logits, labels, mask, class_weights, sample_weights, gamma,
                    dice_weight=dice_weight, focal_weight=focal_weight,
                    label_smoothing=label_smoothing
                )
                boundary_loss_val = torch.tensor(0.0, device=device)
                boundary_ratio = 0.0
            else:
                loss, dice_loss, focal_loss = compute_combined_loss(
                    logits, labels, mask, class_weights, gamma,
                    dice_weight=dice_weight, focal_weight=focal_weight,
                    label_smoothing=label_smoothing
                )
                boundary_loss_val = torch.tensor(0.0, device=device)
                boundary_ratio = 0.0

            iou_loss = torch.tensor(0.0, device=device)
            if iou_weight > 0:
                iou_loss = compute_combined_iou_loss(
                    logits, labels, mask,
                    d_max=10.0, power=1.5, alpha=0.2
                )
                loss = loss + iou_weight * iou_loss

        if perf_monitor:
            perf_monitor.end('loss_computation')
            perf_monitor.start('backward')

        loss_scaled = loss / accumulation_steps
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if perf_monitor:
            perf_monitor.end('backward')

        batch_size = labels.size(0)
        total_loss += loss.detach() * batch_size
        total_dice_loss += dice_loss.detach() * batch_size
        total_focal_loss += focal_loss.detach() * batch_size
        total_iou_loss += iou_loss.detach() * batch_size
        total_boundary_loss += boundary_loss_val.detach() * batch_size
        total_boundary_ratio += boundary_ratio * batch_size
        total_samples += batch_size

        if batch_idx % log_every == 0 and is_main_process():
            print_rank0(
                f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)} | "
                f"Loss={loss.item():.4f} Dice={dice_loss.item():.4f} Focal={focal_loss.item():.4f}"
            )
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/batch_loss', loss.item(), global_step)
                writer.add_scalar('Train/batch_dice_loss', dice_loss.item(), global_step)
                writer.add_scalar('Train/batch_focal_loss', focal_loss.item(), global_step)
                writer.flush()

    if scheduler is not None:
        scheduler.step()

    avg_loss = (total_loss / max(total_samples, 1)).item()
    avg_dice_loss = (total_dice_loss / max(total_samples, 1)).item()
    avg_focal_loss = (total_focal_loss / max(total_samples, 1)).item()
    avg_iou_loss = (total_iou_loss / max(total_samples, 1)).item()
    avg_boundary_loss = (total_boundary_loss / max(total_samples, 1)).item()
    avg_boundary_ratio = total_boundary_ratio / max(total_samples, 1)

    if writer and is_main_process():
        writer.add_scalar('Train/loss', avg_loss, epoch)
        writer.add_scalar('Train/dice_loss', avg_dice_loss, epoch)
        writer.add_scalar('Train/focal_loss', avg_focal_loss, epoch)
        writer.add_scalar('Train/iou_loss', avg_iou_loss, epoch)
        writer.add_scalar('Train/boundary_loss', avg_boundary_loss, epoch)
        writer.add_scalar('Train/boundary_ratio', avg_boundary_ratio, epoch)
        writer.flush()

    return {
        'loss': avg_loss,
        'dice_loss': avg_dice_loss,
        'focal_loss': avg_focal_loss,
        'iou_loss': avg_iou_loss,
        'boundary_loss': avg_boundary_loss,
        'boundary_ratio': avg_boundary_ratio,
    }


@torch.no_grad()
def validate(model, dataloader, device, class_weights, gamma=2.0, label_smoothing=0.15, writer=None, epoch=None):
    model.eval()
    all_metrics = []
    total_loss = total_samples = 0

    for batch in dataloader:
        inputs = prepare_model_inputs_v10(batch, device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask = batch['protein_mask'].to(device, non_blocking=True)

        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_physchem=inputs['protein_physchem'],
            protein_spatial_scalar=inputs['protein_spatial_scalar'],
            protein_spatial_vector=inputs['protein_spatial_vector'],
            protein_gvp_vectors=inputs['protein_gvp_vectors'],
            protein_coords=inputs['protein_coords'],
            protein_mask=inputs['protein_mask'],
            ligand_x=inputs['ligand_x'],
            ligand_pos=inputs['ligand_pos'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_to_protein=inputs['ligand_to_protein'],
            ligand_mask=inputs['ligand_mask']
        )

        logits = outputs['logits']
        loss, _, _ = compute_combined_loss(
            logits, labels, mask, class_weights, gamma,
            dice_weight=0.4, focal_weight=0.4,
            label_smoothing=label_smoothing
        )
        preds = logits.argmax(dim=-1)
        seg_metrics = compute_segmentation_metrics(preds, labels, mask)
        cls_metrics = compute_enhanced_metrics(logits, labels, mask, n_classes=logits.size(-1))
        metrics = {**cls_metrics, **seg_metrics}
        all_metrics.append(metrics)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_metrics = {k: float(sum(m[k] for m in all_metrics) / len(all_metrics)) for k in all_metrics[0]}
    mean_metrics['loss'] = total_loss / max(total_samples, 1)

    if writer and epoch is not None and is_main_process():
        for k, v in mean_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.flush()

    return mean_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Pocket-GVP-Fusion V10")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config['experiment']['seed']
    set_seed(seed)

    # Enable cuDNN/MIOpen benchmark for fixed-size operations
    # Note: For variable-length inputs (dynamic padding), this may cause overhead
    # Set to False if training is slower than expected
    torch.backends.cudnn.benchmark = config['training'].get('cudnn_benchmark', False)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    exp_dir = setup_experiment(config, rank, resume_path=args.resume)

    use_amp = args.amp or config['training'].get('use_amp', False)
    if use_amp:
        print_rank0("Using Automatic Mixed Precision (AMP)")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    data_cfg = config['data']
    data_root = data_cfg["data_root"]
    cluster_sizes = None

    if rank == 0:
        split_file = exp_dir / "splits.txt" if exp_dir else Path("splits.txt")
        if data_cfg.get('split_file') and Path(data_cfg['split_file']).exists():
            source_split_file = data_cfg['split_file']
            print_rank0(f"Loading splits from: {source_split_file}")
            try:
                train_ids, val_ids, test_ids, cluster_sizes = load_splits_with_cluster_size(source_split_file)
                print_rank0(f"✓ Loaded cluster size information for {len(cluster_sizes)} samples")
            except Exception as e:
                print_rank0(f"⚠ Cluster sizes not available, using standard loading: {e}")
                train_ids, val_ids, test_ids = load_splits(source_split_file)
                cluster_sizes = None
            save_splits(train_ids, val_ids, test_ids, split_file)
            print_rank0(f"Saved splits copy to: {split_file}")
        else:
            ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v5')
            train_ids, val_ids, test_ids = create_data_splits_v5(
                data_root,
                train_ratio=data_cfg['train_ratio'],
                val_ratio=data_cfg['val_ratio'],
                test_ratio=data_cfg['test_ratio'],
                seed=seed,
                ligand_features_dir=ligand_features_dir
            )
            save_splits(train_ids, val_ids, test_ids, split_file)
            print_rank0(f"Created and saved new splits to: {split_file}")
            cluster_sizes = None
        print_rank0(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    else:
        train_ids = val_ids = test_ids = None

    if world_size > 1:
        splits = [train_ids, val_ids, test_ids, cluster_sizes] if rank == 0 else [None, None, None, None]
        dist.broadcast_object_list(splits, src=0)
        train_ids, val_ids, test_ids, cluster_sizes = splits

    train_cfg = config['training']
    ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v5')
    protein_features_dir = data_cfg.get('protein_features_dir', 'complete_features_v4')
    nma_features_dir = data_cfg.get('nma_features_dir', protein_features_dir)
    nma_n_confs = data_cfg.get('nma_n_confs', 1)
    nma_max_conf = data_cfg.get('nma_max_conf', 0)
    distances_file = data_cfg.get('distances_file')
    distances_file_original = data_cfg.get('distances_file_original')

    if is_main_process():
        print_rank0("Data sources:")
        print_rank0(f"  protein_features_dir: data/{protein_features_dir}")
        print_rank0(f"  nma_features_dir: data/{nma_features_dir}")
        print_rank0(f"  nma_n_confs: {nma_n_confs}")
        print_rank0(f"  ligand_features_dir: data/{ligand_features_dir}")

    train_dataset = PocketDatasetV8(
        data_root=data_root,
        pdb_ids=train_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,
        ligand_features_dir=ligand_features_dir,
        protein_features_dir=protein_features_dir,
        nma_features_dir=nma_features_dir,
        nma_n_confs=nma_n_confs,
        nma_max_conf=nma_max_conf,
        use_nma=(nma_n_confs > 1),
        split="train",
        verbose=is_main_process(),
        print_fn=print_rank0
    )

    val_dataset = PocketDatasetV8(
        data_root=data_root,
        pdb_ids=val_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,
        ligand_features_dir=ligand_features_dir,
        protein_features_dir=protein_features_dir,
        nma_features_dir=nma_features_dir,
        nma_n_confs=nma_n_confs,
        nma_max_conf=nma_max_conf,
        use_nma=False,
        split="val",
        verbose=is_main_process(),
        print_fn=print_rank0
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg['num_workers'],
        collate_fn=pocket_collate_fn_v10,
        pin_memory=train_cfg.get('pin_memory', True),
        persistent_workers=train_cfg['num_workers'] > 0,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if train_cfg['num_workers'] > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_cfg['num_workers'],
        collate_fn=pocket_collate_fn_v10,
        pin_memory=train_cfg.get('pin_memory', True),
        persistent_workers=train_cfg['num_workers'] > 0,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if train_cfg['num_workers'] > 0 else None,
    )

    cluster_weighted_loss = None
    if cluster_sizes is not None:
        print_rank0("\n" + "=" * 60)
        print_rank0("Initializing Cluster-Weighted Loss")
        print_rank0("=" * 60)
        cluster_weighted_loss = ClusterWeightedLoss(cluster_sizes, epsilon=2.0)
        if rank == 0:
            cluster_weighted_loss.print_statistics()
    else:
        print_rank0("\n⚠ Cluster sizes not available, using standard loss (no sample weighting)")

    gradient_checkpointing_arg = args.gradient_checkpointing if args.gradient_checkpointing else None
    compile_arg = args.compile if args.compile else None
    model = create_model(config, device, use_gradient_checkpointing=gradient_checkpointing_arg, use_compile=compile_arg)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print_rank0("Model wrapped with DistributedDataParallel")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith('.bias') or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {'params': decay_params, 'weight_decay': train_cfg['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=train_cfg['learning_rate'],
        betas=tuple(train_cfg['betas']),
    )

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(train_cfg['warmup_epochs'], 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(train_cfg['num_epochs'] - train_cfg['warmup_epochs'], 1), eta_min=train_cfg['min_lr'])
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[train_cfg['warmup_epochs']]
    )

    # TensorBoard writer
    if is_main_process() and config['logging'].get('tensorboard', True):
        tb_dir = exp_dir / config['logging']['tensorboard_dir']
        writer = SummaryWriter(str(tb_dir))
    else:
        writer = None

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print_rank0(f"Resumed from epoch {start_epoch}")
        print_rank0(f"  Best val loss: {best_val_loss:.4f}")

    class_weights = torch.tensor(train_cfg['class_weights'], dtype=torch.float32).to(device)

    try:
        for epoch in range(start_epoch, train_cfg['num_epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                class_weights=class_weights,
                gamma=train_cfg['focal_gamma'],
                dice_weight=train_cfg['dice_weight'],
                focal_weight=train_cfg['focal_weight'],
                log_every=config['logging']['log_every'],
                writer=writer,
                use_amp=use_amp,
                iou_weight=train_cfg.get('iou_weight', 0.0),
                cluster_weighted_loss=cluster_weighted_loss,
                use_boundary_loss=train_cfg.get('use_boundary_loss', False),
                boundary_weight=train_cfg.get('boundary_weight', 0.2),
                boundary_multiplier=train_cfg.get('boundary_multiplier', 2.0),
                boundary_distance_threshold=train_cfg.get('boundary_distance_threshold', 8.0),
                label_smoothing=train_cfg.get('label_smoothing', 0.15),
                enable_profiling=train_cfg.get('enable_profiling', False),
                exp_dir=exp_dir,
                accumulation_steps=train_cfg.get('accumulation_steps', 1),
                use_bf16=train_cfg.get('use_bf16', False),
                gradient_clip=train_cfg.get('gradient_clip', 1.0),
            )

            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                class_weights=class_weights,
                gamma=train_cfg['focal_gamma'],
                label_smoothing=train_cfg.get('label_smoothing', 0.15),
                writer=writer,
                epoch=epoch,
            )

            if is_main_process():
                print_rank0(
                    f"[Epoch {epoch}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
                )

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'config': config,
                }
                if use_amp:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()

                if epoch % train_cfg['save_every'] == 0:
                    torch.save(checkpoint, exp_dir / f'checkpoint_epoch_{epoch}.pt')

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    checkpoint['best_val_loss'] = best_val_loss
                    torch.save(checkpoint, exp_dir / 'best_model.pt')
                    print_rank0(f"✓ Saved best model (val_loss={best_val_loss:.4f})")
                else:
                    patience_counter += 1

                if patience_counter >= train_cfg['early_stopping_patience']:
                    print_rank0(f"Early stopping triggered at epoch {epoch}")
                    break
    finally:
        if writer:
            writer.close()
        cleanup_distributed()


if __name__ == '__main__':
    main()
