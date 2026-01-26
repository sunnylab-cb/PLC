#!/usr/bin/env python3
"""
Training script for Pocket-GVP-Fusion v2.

Trains the model on pre-extracted protein-ligand features with:
- TensorBoard logging
- Focal Loss with class weights and padding masking
- Cosine annealing learning rate schedule with warmup
- Gradient clipping
- Early stopping and checkpointing

Usage:
    python train.py --config configs/pocket_gvp_v2.yaml
    python train.py --config configs/pocket_gvp_v2.yaml --resume outputs/latest/checkpoint.pt
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset import (
    PocketDataset, 
    create_data_splits, 
    save_splits, 
    load_splits,
    set_seed,
    create_dataloader
)
from src.data.collate import pocket_collate_fn, prepare_model_inputs, compute_class_weights
from src.models.pocket_gvp_fusion import PocketGVPFusion, PocketGVPFusionSimplified
from src.metrics.metrics import compute_all_publishable_metrics, compute_segmentation_metrics
from src.metrics.diagnostics import collect_model_diagnostics, check_health_warnings


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: Dict) -> Path:
    """Create experiment directory with timestamp."""
    exp_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    return exp_dir


def create_model(config: Dict, device: torch.device) -> nn.Module:
    """Create model from config."""
    model_config = config['model']
    
    # Use full PocketGVPFusion model
    model = PocketGVPFusion(model_config)
    
    model = model.to(device)
    
    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    return model


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer from config."""
    train_cfg = config['training']
    
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=tuple(train_cfg['betas'])
    )
    
    return optimizer


def create_scheduler(optimizer, config: Dict, num_training_steps: int):
    """Create learning rate scheduler with warmup."""
    train_cfg = config['training']
    
    warmup_steps = train_cfg['warmup_epochs']
    total_epochs = train_cfg['num_epochs']
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_steps,
        eta_min=train_cfg['min_lr']
    )
    
    # Combine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


def compute_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute focal loss with class weights and padding masking.
    
    Args:
        logits: [B, L, C] - predicted logits
        labels: [B, L] - ground truth labels
        mask: [B, L] - padding mask (True = valid)
        class_weights: [C] - per-class weights
        gamma: focal loss gamma parameter
        
    Returns:
        Scalar loss value
    """
    B, L, C = logits.shape
    device = logits.device
    
    # Flatten
    logits_flat = logits.view(-1, C)  # [B*L, C]
    labels_flat = labels.view(-1)      # [B*L]
    mask_flat = mask.view(-1).float()  # [B*L]
    
    # Compute softmax probabilities
    probs = F.softmax(logits_flat, dim=-1)  # [B*L, C]
    
    # Cross entropy loss
    ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # [B*L]
    
    # Focal weight: (1 - p_t)^gamma
    p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*L]
    focal_weight = (1 - p_t) ** gamma
    
    # Apply class weights
    class_weights = class_weights.to(device)
    sample_weights = class_weights[labels_flat]  # [B*L]
    
    # Combined loss
    focal_loss = focal_weight * sample_weights * ce_loss
    
    # Apply padding mask
    focal_loss = focal_loss * mask_flat
    
    # Mean over valid tokens
    n_valid = mask_flat.sum()
    if n_valid > 0:
        loss = focal_loss.sum() / n_valid
    else:
        loss = focal_loss.sum() * 0  # Returns 0, avoids NaN
    
    return loss


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int = 3
) -> Dict[str, float]:
    """Compute classification metrics."""
    preds = logits.argmax(dim=-1)  # [B, L]
    
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



def check_tensor_nan(tensor: torch.Tensor, name: str, batch_idx: int = -1):
    """Check if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"!! NaN/Inf detected in {name} at batch {batch_idx} !!")
        print(f"  Shape: {tensor.shape}")
        print(f"  Range: [{tensor.min()}, {tensor.max()}]")
        print(f"  Mean: {tensor.mean()}")
        return True
    return False


def train_one_epoch(
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
    debug: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
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
        
        # Debug: Check inputs
        if debug:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    if check_tensor_nan(v, f"Input {k}", batch_idx):
                        raise ValueError(f"NaN found in input {k}")
        
        # Forward pass
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
        
        # Debug: Check outputs
        if debug:
             if check_tensor_nan(logits, "Logits", batch_idx):
                 raise ValueError("NaN found in logits")
        
        # Compute loss
        loss = compute_focal_loss(logits, labels, mask, class_weights, gamma)
        
        # Debug: Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"!! Loss is {loss.item()} at batch {batch_idx} !!")
            print("Checking inputs again...")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    print(f"{k}: min={v.min()}, max={v.max()}, nan={torch.isnan(v).any()}")
            print("Checking logits...")
            print(f"Logits: min={logits.min()}, max={logits.max()}")
            raise ValueError("Loss is NaN")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient monitoring
        if writer is not None and (batch_idx % log_every == 0 or debug):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/grad_norm', total_norm, global_step)
            
            if debug and total_norm > 10.0:
                print(f"High gradient norm: {total_norm:.4f} at batch {batch_idx}")
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()
        
        batch_metrics = compute_metrics(logits.detach(), labels, mask)
        all_metrics.append(batch_metrics)
        
        # Logging
        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(total_samples, 1)
            
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {batch_metrics['accuracy']:.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            # TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/batch_loss', loss.item(), global_step)
                writer.add_scalar('Train/batch_acc', batch_metrics['accuracy'], global_step)
                
                # Training dynamics / module health diagnostics
                diagnostics = collect_model_diagnostics(model)
                if 'gvp_scalar_norm' in diagnostics:
                    writer.add_scalar('Health/gvp_scalar_norm', diagnostics['gvp_scalar_norm'], global_step)
                if 'gvp_vector_norm' in diagnostics:
                    writer.add_scalar('Health/gvp_vector_norm', diagnostics['gvp_vector_norm'], global_step)
                if 'gate_openness' in diagnostics:
                    writer.add_scalar('Health/gate_openness', diagnostics['gate_openness'], global_step)
                if 'attn_entropy' in diagnostics:
                    writer.add_scalar('Health/attn_entropy', diagnostics['attn_entropy'], global_step)
                
                # Check for health warnings
                if debug:
                    warnings = check_health_warnings(diagnostics)
                    for warning in warnings:
                        print(f"  [HEALTH WARNING] {warning}")
    
    # Aggregate metrics
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics)
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    class_weights: torch.Tensor,
    gamma: float,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on a dataset with publishable metrics."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_metrics = []
    
    # Accumulators for epoch-level segmentation metrics
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)
        
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
        
        # Accumulate predictions for epoch-level segmentation metrics
        preds = logits.argmax(dim=-1)
        all_preds.append(preds[mask].cpu())
        all_labels.append(labels[mask].cpu())
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics)
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = avg_loss
    
    # Compute epoch-level segmentation metrics (DICE, IoU)
    if all_preds and all_labels:
        epoch_preds = torch.cat(all_preds)
        epoch_labels = torch.cat(all_labels)
        seg_metrics = compute_segmentation_metrics(
            epoch_preds, epoch_labels, 
            mask=torch.ones_like(epoch_preds, dtype=torch.bool),  # Already filtered
            contact_class=0
        )
        avg_metrics['dice_c0'] = seg_metrics['dice']
        avg_metrics['iou_c0'] = seg_metrics['iou']
    
    return avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    exp_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    # Save latest
    torch.save(checkpoint, exp_dir / "checkpoints" / "latest.pt")
    
    # Save periodic
    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint, exp_dir / "checkpoints" / f"epoch_{epoch + 1}.pt")
    
    # Save best
    if is_best:
        torch.save(checkpoint, exp_dir / "checkpoints" / "best.pt")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    checkpoint_path: str
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(checkpoint_path)
    
    # Convert legacy state dict for backward compatibility
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'convert_legacy_state_dict'):
        state_dict = model.convert_legacy_state_dict(state_dict)
    
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint.get('best_val_loss', float('inf'))


def main():
    parser = argparse.ArgumentParser(description='Train Pocket-GVP-Fusion v2')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--device', type=str, default=None, help='Device override (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (anomaly detection, nan checks)')
    args = parser.parse_args()
    
    # Debug mode
    if args.debug:
        print("Debug mode enabled: torch.autograd.set_detect_anomaly(True)")
        torch.autograd.set_detect_anomaly(True)

    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed = config['experiment']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup experiment directory
    exp_dir = setup_experiment(config)
    print(f"Experiment directory: {exp_dir}")
    
    # Data
    data_cfg = config['data']
    feature_dir = data_cfg['feature_dir']
    
    # Check if features exist
    if not Path(feature_dir).exists():
        print(f"Warning: Feature directory '{feature_dir}' does not exist!")
        print("Please run feature extraction first or update the config.")
        print("Continuing with dummy setup for testing...")
        
        # Create dummy directory for structure testing
        Path(feature_dir).mkdir(parents=True, exist_ok=True)
    
    # Create or load splits
    split_file = exp_dir / "splits.txt"
    
    if data_cfg.get('split_file') and Path(data_cfg['split_file']).exists():
        train_ids, val_ids, test_ids = load_splits(data_cfg['split_file'])
        print(f"Loaded splits from {data_cfg['split_file']}")
    else:
        train_ids, val_ids, test_ids = create_data_splits(
            feature_dir,
            train_ratio=data_cfg['train_ratio'],
            val_ratio=data_cfg['val_ratio'],
            test_ratio=data_cfg['test_ratio'],
            seed=seed
        )
        save_splits(train_ids, val_ids, test_ids, split_file)
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create dataloaders
    train_cfg = config['training']
    
    train_loader = create_dataloader(
        feature_dir=feature_dir,
        pdb_ids=train_ids,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        seed=seed
    )
    
    val_loader = create_dataloader(
        feature_dir=feature_dir,
        pdb_ids=val_ids,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        seed=seed
    )
    
    # Model
    model = create_model(config, device)
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_loader) * train_cfg['num_epochs']
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Class weights
    class_weights = torch.tensor(train_cfg['class_weights'], dtype=torch.float32, device=device)
    gamma = train_cfg['focal_gamma']
    
    # TensorBoard
    writer = None
    if config['logging'].get('tensorboard', True):
        tb_dir = exp_dir / config['logging']['tensorboard_dir']
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard: {tb_dir}")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    # Early stopping
    patience = train_cfg.get('early_stopping_patience', 15)
    no_improve_count = 0
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{train_cfg['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            gamma=gamma,
            gradient_clip=train_cfg['gradient_clip'],
            device=device,
            epoch=epoch,
            writer=writer,
            log_every=config['logging']['log_every'],
            debug=args.debug
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            class_weights=class_weights,
            gamma=gamma,
            device=device
        )
        
        # Log
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Log publishable metrics
        if 'dice_c0' in val_metrics:
            print(f"Val   - DICE: {val_metrics['dice_c0']:.4f}, IoU: {val_metrics['iou_c0']:.4f}, F1_c0: {val_metrics.get('f1_c0', 0):.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # TensorBoard epoch logging
        if writer is not None:
            writer.add_scalar('Train/epoch_loss', train_metrics['loss'], epoch)
            writer.add_scalar('Train/epoch_acc', train_metrics['accuracy'], epoch)
            writer.add_scalar('Val/epoch_loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/epoch_acc', val_metrics['accuracy'], epoch)
            writer.add_scalar('Train/learning_rate', current_lr, epoch)
            
            # Per-class metrics
            for k, v in val_metrics.items():
                if k.startswith('f1_') or k.startswith('precision_') or k.startswith('recall_'):
                    writer.add_scalar(f'Val/{k}', v, epoch)
            
            # Publishable metrics (DICE, IoU)
            if 'dice_c0' in val_metrics:
                writer.add_scalar('Val/dice_c0', val_metrics['dice_c0'], epoch)
                writer.add_scalar('Val/iou_c0', val_metrics['iou_c0'], epoch)
        
        # Checkpointing
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            no_improve_count = 0
            print(f"  New best validation loss: {best_val_loss:.4f}")
        else:
            no_improve_count += 1
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            best_val_loss, exp_dir, is_best
        )
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
    
    # Cleanup
    if writer is not None:
        writer.close()
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {exp_dir / 'checkpoints'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
