#!/usr/bin/env python3
"""
Attention Visualization and Analysis Script for Pocket-GVP-Fusion v2.

Visualizes:
- Self-attention maps from Protein Context Encoder
- Cross-attention maps from Protein-Ligand Fusion
- Correlation between attention scores and spatial distance

Usage:
    python visualize_attention.py --checkpoint outputs/experiment/checkpoints/best.pt \
                                   --config outputs/experiment/config.yaml \
                                   --output_dir attention_vis/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset import PocketDataset, load_splits, set_seed, create_dataloader
from src.data.collate import prepare_model_inputs
from src.models.pocket_gvp_fusion import PocketGVPFusion


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: Dict, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model_config = config['model']
    model = PocketGVPFusion(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def collect_attention_weights(model) -> Dict[str, List[torch.Tensor]]:
    """
    Collect attention weights from all attention layers.
    
    Returns:
        Dict with 'self_attention' and 'cross_attention' lists
    """
    weights = {
        'self_attention': [],
        'cross_attention': []
    }
    
    # Self-attention from Protein Context Encoder
    if hasattr(model, 'protein_context_encoder'):
        for layer in model.protein_context_encoder.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, '_last_attn_weights'):
                w = layer.attn._last_attn_weights
                if w is not None:
                    weights['self_attention'].append(w.cpu())
    
    # Cross-attention from Protein-Ligand Fusion
    if hasattr(model, 'cross_attention'):
        ca = model.cross_attention
        if hasattr(ca, 'cross_attn') and hasattr(ca.cross_attn, '_last_attn_weights'):
            w = ca.cross_attn._last_attn_weights
            if w is not None:
                weights['cross_attention'].append(w.cpu())
    
    return weights


def compute_pairwise_distances(coords: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between residues.
    
    Args:
        coords: [L, 3] Cα coordinates
        mask: [L] validity mask
        
    Returns:
        [L_valid, L_valid] distance matrix
    """
    coords_valid = coords[mask].cpu().numpy()
    n = coords_valid.shape[0]
    
    # Pairwise distance matrix
    diff = coords_valid[:, np.newaxis, :] - coords_valid[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    
    return distances


def visualize_attention_map(
    attn_weights: np.ndarray,
    title: str,
    save_path: str,
    labels: Optional[np.ndarray] = None,
    max_size: int = 100
):
    """
    Visualize a single attention map as a heatmap.
    
    Args:
        attn_weights: [L_q, L_k] attention weights (averaged over heads)
        title: Plot title
        save_path: Path to save figure
        labels: Optional ground truth labels for annotation
        max_size: Maximum sequence length to display
    """
    # Truncate if too large
    L_q, L_k = attn_weights.shape
    if L_q > max_size or L_k > max_size:
        attn_weights = attn_weights[:max_size, :max_size]
        truncated = True
    else:
        truncated = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use log scale for better visualization
    im = ax.imshow(
        attn_weights, 
        cmap='viridis',
        aspect='auto',
        norm=colors.LogNorm(vmin=max(attn_weights.min(), 1e-6), vmax=attn_weights.max())
    )
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title + (' (truncated)' if truncated else ''))
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_multihead_attention(
    attn_weights: np.ndarray,
    title: str,
    save_path: str,
    n_heads_to_show: int = 4
):
    """
    Visualize multiple attention heads.
    
    Args:
        attn_weights: [H, L_q, L_k] attention weights per head
        title: Plot title
        save_path: Path to save figure
        n_heads_to_show: Number of heads to display
    """
    n_heads = min(attn_weights.shape[0], n_heads_to_show)
    
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i >= attn_weights.shape[0]:
            break
            
        # Truncate for display
        head_weights = attn_weights[i][:50, :50]
        
        im = ax.imshow(head_weights, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_attention_vs_distance(
    attn_weights: np.ndarray,
    distances: np.ndarray,
    title: str,
    save_path: str,
    sample_fraction: float = 0.1
):
    """
    Analyze and visualize the relationship between attention scores and spatial distance.
    
    Args:
        attn_weights: [L, L] attention weights (averaged over heads)
        distances: [L, L] pairwise distances
        title: Plot title
        save_path: Path to save figure
        sample_fraction: Fraction of points to sample for scatter plot
    """
    # Flatten and filter out diagonal (self-attention)
    L = attn_weights.shape[0]
    mask = ~np.eye(L, dtype=bool)
    
    attn_flat = attn_weights[mask].flatten()
    dist_flat = distances[mask].flatten()
    
    # Remove any NaN/Inf
    valid = np.isfinite(attn_flat) & np.isfinite(dist_flat) & (attn_flat > 0)
    attn_flat = attn_flat[valid]
    dist_flat = dist_flat[valid]
    
    if len(attn_flat) == 0:
        print(f"Warning: No valid attention-distance pairs for {title}")
        return None
    
    # Sample for scatter plot
    n_samples = int(len(attn_flat) * sample_fraction)
    n_samples = max(min(n_samples, 5000), min(len(attn_flat), 500))  # Between 500 and 5000
    idx = np.random.choice(len(attn_flat), n_samples, replace=False)
    
    # Compute correlation
    correlation, p_value = stats.pearsonr(attn_flat, dist_flat)
    spearman_corr, spearman_p = stats.spearmanr(attn_flat, dist_flat)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot with log attention
    ax1 = axes[0]
    scatter = ax1.scatter(
        dist_flat[idx], 
        attn_flat[idx], 
        alpha=0.3, 
        s=5,
        c=np.log10(attn_flat[idx] + 1e-10),
        cmap='viridis'
    )
    ax1.set_xlabel('Spatial Distance (Å)')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title(f'{title}\nPearson r={correlation:.3f}, Spearman ρ={spearman_corr:.3f}')
    ax1.set_yscale('log')
    
    # Add trend line
    z = np.polyfit(dist_flat, np.log10(attn_flat + 1e-10), 1)
    p = np.poly1d(z)
    x_line = np.linspace(dist_flat.min(), dist_flat.max(), 100)
    ax1.plot(x_line, 10 ** p(x_line), 'r--', linewidth=2, label='Trend')
    ax1.legend()
    
    # Binned analysis
    ax2 = axes[1]
    
    # Bin by distance
    distance_bins = np.arange(0, min(50, dist_flat.max()), 5)
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(len(distance_bins) - 1):
        bin_mask = (dist_flat >= distance_bins[i]) & (dist_flat < distance_bins[i+1])
        if bin_mask.sum() > 10:
            bin_centers.append((distance_bins[i] + distance_bins[i+1]) / 2)
            bin_means.append(attn_flat[bin_mask].mean())
            bin_stds.append(attn_flat[bin_mask].std())
    
    if bin_centers:
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=3)
        ax2.set_xlabel('Spatial Distance (Å)')
        ax2.set_ylabel('Mean Attention Weight')
        ax2.set_title('Attention vs Distance (Binned)')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'pearson_r': correlation,
        'pearson_p': p_value,
        'spearman_rho': spearman_corr,
        'spearman_p': spearman_p,
        'n_pairs': len(attn_flat)
    }


def analyze_attention_by_label(
    attn_weights: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str
):
    """
    Analyze attention patterns grouped by residue labels.
    
    Args:
        attn_weights: [L, L] attention weights
        labels: [L] class labels (0=Contact, 1=Near, 2=Far)
        title: Plot title
        save_path: Path to save figure
    """
    class_names = ['Contact', 'Near', 'Far']
    n_classes = 3
    
    L = attn_weights.shape[0]
    
    # Compute mean attention FROM each class TO each class
    class_attention = np.zeros((n_classes, n_classes))
    class_counts = np.zeros((n_classes, n_classes))
    
    for i in range(L):
        for j in range(L):
            if i != j:
                from_class = labels[i]
                to_class = labels[j]
                class_attention[from_class, to_class] += attn_weights[i, j]
                class_counts[from_class, to_class] += 1
    
    # Normalize
    class_attention = class_attention / (class_counts + 1e-8)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(class_attention, cmap='YlOrRd')
    
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Attending TO')
    ax.set_ylabel('Attending FROM')
    ax.set_title(f'{title}\nMean Attention by Class')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f'{class_attention[i, j]:.4f}',
                          ha='center', va='center', color='black')
    
    plt.colorbar(im, ax=ax, label='Mean Attention')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def run_visualization(
    model,
    dataloader,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 5
):
    """
    Run attention visualization on samples.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Compute device
        output_dir: Output directory for visualizations
        n_samples: Number of samples to visualize
    """
    model.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_distance_correlations = {
        'self_attention': [],
        'cross_attention': []
    }
    
    sample_count = 0
    for batch in dataloader:
        if sample_count >= n_samples:
            break
            
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)
        pdb_ids = batch['pdb_ids']
        coords = inputs['protein_coords']
        
        # Forward pass (this populates attention weight caches)
        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_scalar=inputs['protein_scalar'],
            protein_vectors=inputs['protein_vectors'],
            protein_coords=coords,
            protein_mask=inputs['protein_mask'],
            ligand_x=inputs['ligand_x'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_to_protein=inputs['ligand_to_protein'],
            ligand_mask=inputs['ligand_mask']
        )
        
        # Collect attention weights
        attn_weights = collect_attention_weights(model)
        
        # Process each sample in batch
        B = len(pdb_ids)
        for b in range(B):
            if sample_count >= n_samples:
                break
                
            pdb_id = pdb_ids[b]
            mask_b = mask[b]
            n_valid = mask_b.sum().item()
            labels_b = labels[b, mask_b].cpu().numpy()
            coords_b = coords[b]
            
            print(f"\nProcessing {pdb_id} ({n_valid} residues)...")
            
            sample_dir = output_dir / pdb_id
            sample_dir.mkdir(exist_ok=True)
            
            # Compute pairwise distances
            distances = compute_pairwise_distances(coords_b, mask_b)
            
            # Self-attention analysis
            if attn_weights['self_attention']:
                # Average over layers and heads for this sample
                for layer_idx, layer_attn in enumerate(attn_weights['self_attention']):
                    # layer_attn: [B, H, L, L]
                    attn_b = layer_attn[b, :, :n_valid, :n_valid].numpy()  # [H, L, L]
                    attn_avg = attn_b.mean(axis=0)  # Average over heads
                    
                    # Visualize attention map
                    visualize_attention_map(
                        attn_avg,
                        f'{pdb_id} - Self-Attention Layer {layer_idx}',
                        sample_dir / f'self_attn_layer{layer_idx}_map.png',
                        labels=labels_b
                    )
                    
                    # Visualize multi-head
                    visualize_multihead_attention(
                        attn_b,
                        f'{pdb_id} - Self-Attention Layer {layer_idx} (Per Head)',
                        sample_dir / f'self_attn_layer{layer_idx}_heads.png'
                    )
                    
                    # Attention vs distance analysis (only for last layer)
                    if layer_idx == len(attn_weights['self_attention']) - 1:
                        corr_result = analyze_attention_vs_distance(
                            attn_avg,
                            distances,
                            f'{pdb_id} - Self-Attention vs Distance',
                            sample_dir / 'self_attn_vs_distance.png'
                        )
                        if corr_result:
                            all_distance_correlations['self_attention'].append(corr_result)
                        
                        # Analysis by label
                        analyze_attention_by_label(
                            attn_avg,
                            labels_b,
                            f'{pdb_id} - Self-Attention',
                            sample_dir / 'self_attn_by_label.png'
                        )
            
            # Cross-attention analysis
            if attn_weights['cross_attention']:
                for layer_idx, layer_attn in enumerate(attn_weights['cross_attention']):
                    # layer_attn: [B, H, L_protein, L_ligand]
                    L_lig = layer_attn.shape[3]
                    attn_b = layer_attn[b, :, :n_valid, :].numpy()  # [H, L_prot, L_lig]
                    attn_avg = attn_b.mean(axis=0)  # [L_prot, L_lig]
                    
                    # Visualize cross-attention map
                    fig, ax = plt.subplots(figsize=(max(8, L_lig * 0.3), 6))
                    im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
                    ax.set_xlabel('Ligand Atom Index')
                    ax.set_ylabel('Protein Residue Index')
                    ax.set_title(f'{pdb_id} - Cross-Attention (Protein → Ligand)')
                    plt.colorbar(im, ax=ax, label='Attention Weight')
                    plt.tight_layout()
                    plt.savefig(sample_dir / 'cross_attn_map.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Cross-attention by residue label
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    class_names = ['Contact', 'Near', 'Far']
                    class_colors = ['#d62728', '#ff7f0e', '#2ca02c']
                    
                    for c in range(3):
                        class_mask = labels_b == c
                        if class_mask.sum() > 0:
                            mean_attn = attn_avg[class_mask].mean(axis=0)
                            ax.plot(mean_attn, label=class_names[c], color=class_colors[c], linewidth=2)
                    
                    ax.set_xlabel('Ligand Atom Index')
                    ax.set_ylabel('Mean Attention Weight')
                    ax.set_title(f'{pdb_id} - Cross-Attention by Residue Class')
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(sample_dir / 'cross_attn_by_class.png', dpi=150, bbox_inches='tight')
                    plt.close()
            
            sample_count += 1
    
    # Aggregate analysis
    print(f"\n{'='*60}")
    print("AGGREGATE ANALYSIS")
    print('='*60)
    
    # Self-attention distance correlation summary
    if all_distance_correlations['self_attention']:
        corrs = all_distance_correlations['self_attention']
        mean_pearson = np.mean([c['pearson_r'] for c in corrs])
        mean_spearman = np.mean([c['spearman_rho'] for c in corrs])
        
        print(f"\nSelf-Attention vs Distance:")
        print(f"  Mean Pearson correlation: {mean_pearson:.4f}")
        print(f"  Mean Spearman correlation: {mean_spearman:.4f}")
        
        # Save summary plot
        fig, ax = plt.subplots(figsize=(8, 5))
        pearson_vals = [c['pearson_r'] for c in corrs]
        spearman_vals = [c['spearman_rho'] for c in corrs]
        
        x = np.arange(len(corrs))
        width = 0.35
        ax.bar(x - width/2, pearson_vals, width, label='Pearson r', color='steelblue')
        ax.bar(x + width/2, spearman_vals, width, label='Spearman ρ', color='coral')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Correlation')
        ax.set_title('Self-Attention vs Spatial Distance Correlation')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregate_distance_correlation.png', dpi=150)
        plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, default='attention_vis', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which split to use')
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
        return
    
    train_ids, val_ids, test_ids = load_splits(split_file)
    
    if args.split == 'train':
        pdb_ids = train_ids
    elif args.split == 'val':
        pdb_ids = val_ids
    else:
        pdb_ids = test_ids
    
    print(f"Using {args.split} split: {len(pdb_ids)} proteins")
    
    # Create dataloader
    data_cfg = config['data']
    dataloader = create_dataloader(
        feature_dir=data_cfg['feature_dir'],
        pdb_ids=pdb_ids[:args.n_samples * 2],  # Load a few extra in case of issues
        batch_size=1,  # Process one at a time for cleaner visualization
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        seed=config['experiment']['seed']
    )
    
    # Run visualization
    output_dir = Path(args.output_dir)
    run_visualization(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        n_samples=args.n_samples
    )


if __name__ == '__main__':
    main()
