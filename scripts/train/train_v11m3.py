#!/usr/bin/env python3
"""
训练脚本 for Pocket-GVP-Fusion V11m3

V11m3 更新 (基于 V11m2):
- Uni-Mol global embedding → ligand atom feature augmentation:
  [B,512] Uni-Mol CLS 經小 MLP 投影後按 ligand_batch repeat 到每個 atom，
  concat 到 ligand_x，作為 isolated lab，不影響 V11/V11m2。

V11m2 更新 (基于 V11):
- Ligand self-attention 3D distance bias (use_lig_dist_bias=True):
  Per-layer RBF bias on ligand atom-atom pairs, symmetric to protein DistanceBias.
  Global token row/col = 0. Zero-init, safe.
- b_lig1 AttnRes anchor (use_lig_attnres_b1=True):
  Post-self-attn h_lig_ctx 加入配体侧 AttnRes 作为第 2 个 anchor。
  InterBlockAttnRes.proj 无需改动（Linear(d,1)，与 source 数量无关）。
- 序数标签平滑 (ordinal_label_smoothing=True):
  仅平滑到相邻类别（Contact↔Near，Near↔Far），不影响 Tversky、dist_aux 和 metrics。

继承 V11 特性:
- Ligand-internal self-attention (n_ligand_self_attn_layers)
- Per-layer distance bias (AlphaFold2/ESMFold style)
- Distance regression auxiliary head
- Confident Negative Clipping (CNC)
- Dynamic Ligand Global Token
- per-sample Contact-only Tversky Loss + Focal Loss

使用方法:
    python scripts/train/train_v11m3.py --config configs/pocket_gvp_v11m3.yaml
    torchrun --nproc_per_node=2 scripts/train/train_v11m3.py --config configs/pocket_gvp_v11m3.yaml
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

from src.data.pocket_dataset_v8 import PocketDatasetV8, set_seed
from src.data.pocket_dataset import save_splits, load_splits, load_splits_with_cluster_size
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
from src.models.pocket_gvp_fusion_v11m3 import PocketGVPFusionV11m3
from src.data.pocket_dataset_v6 import create_data_splits_v5
from src.metrics.metrics import compute_segmentation_metrics, compute_enhanced_metrics
from src.metrics.diagnostics import collect_model_diagnostics, compute_feature_statistics
from src.losses import compute_combined_iou_loss, compute_iou_metric  # V7m2: IoU Loss
from src.losses.boundary_loss import compute_boundary_loss, compute_combined_loss_with_boundary  # V9m2: Boundary Loss
from src.losses.confident_negative_clip import compute_cnc_weights, compute_cnc_stats  # V10m7: CNC
from src.utils.performance_monitor import PerformanceMonitor, DataLoaderProfiler  # V9m2: Performance monitoring


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
# V11m3: Uni-Mol 嵌入快取
# ============================================================================

def load_unimol_cache(data_root: str, unimol_dir: str) -> Dict[str, torch.Tensor]:
    """
    一次性載入 Uni-Mol CLS embeddings 到 RAM。
    目錄結構: {data_root}/{unimol_dir}/{yr_range}/{pdb_id}/{pdb_id}_unimol.pt
    返回: {pdb_id: tensor [512]} float32 CPU。
    """
    root = Path(data_root) / unimol_dir
    if not root.exists():
        raise RuntimeError(
            f"[Uni-Mol] 目錄不存在: {root}。V11m3 啟用 Uni-Mol ligand input 時不允許零向量 fallback。"
        )

    cache: Dict[str, torch.Tensor] = {}
    n_failed = 0
    n_zero = 0
    n_nonfinite = 0
    bad_shape = []
    for fpath in root.rglob("*_unimol.pt"):
        pdb_id = fpath.stem.replace("_unimol", "")
        try:
            obj = torch.load(fpath, map_location="cpu", weights_only=True)
            embed = obj.get("embedding") if isinstance(obj, dict) else obj
            if isinstance(embed, torch.Tensor) and embed.shape == (512,):
                embed = embed.float()
                if not torch.isfinite(embed).all():
                    n_nonfinite += 1
                    n_failed += 1
                elif embed.abs().sum() <= 1e-6:
                    n_zero += 1
                    n_failed += 1
                else:
                    cache[pdb_id] = embed
            else:
                bad_shape.append(
                    f"{fpath}: {type(embed).__name__}, shape={getattr(embed, 'shape', None)}"
                )
                n_failed += 1
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print_rank0(f"[Uni-Mol] 載入失敗 {fpath.name}: {e}")

    if bad_shape:
        raise RuntimeError(
            "[Uni-Mol] 壞 cache：embedding shape/type invalid. "
            f"examples={bad_shape[:5]}。請重新提取 Uni-Mol embeddings。"
        )
    if n_zero > 0 or n_nonfinite > 0:
        raise RuntimeError(
            f"[Uni-Mol] 壞 cache：zero={n_zero}, nonfinite={n_nonfinite}. "
            "禁止用全零/非有限 embedding 繼續訓練。"
        )
    if n_failed > 5:
        print_rank0(f"[Uni-Mol] ... 共 {n_failed} 筆載入失敗（僅顯示前 5 筆）")
    if not cache:
        raise RuntimeError(f"[Uni-Mol] 未載入任何有效 embedding: {root}")
    print_rank0(f"[Uni-Mol] 載入 {len(cache)} 個有效嵌入，{n_failed} 筆失敗")
    return cache


def get_batch_unimol(
    pdb_ids: list,
    cache: Dict[str, torch.Tensor],
    d_unimol: int = 512,
    device: torch.device = None,
) -> Optional[torch.Tensor]:
    """從 cache 取 batch Uni-Mol embeddings [B,512]；cache 空則返回 None。"""
    if not cache:
        return None
    missing = [pid for pid in pdb_ids if pid not in cache]
    if missing:
        raise RuntimeError(
            f"[Uni-Mol] batch 中有 {len(missing)} 個 PDB 缺少 embedding，例如 {missing[:5]}。"
            "請補齊或重新提取 Uni-Mol embeddings。"
        )
    t = torch.stack([cache[pid] for pid in pdb_ids], dim=0)
    return t.to(device, non_blocking=True) if device is not None else t


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
    raw_exp_name = config['experiment']['name']
    # Normalize experiment name: strip known version tokens, then append current version
    version_tokens = {'v6', 'v7', 'v7m1', 'v7m2', 'v7m3', 'v8', 'v9', 'v9m2', 'v10', 'v10t', 'v10m3', 'v10m4', 'v10m5', 'v10m6', 'v10m7', 'v10m8', 'v11', 'v11m1', 'v11m2', 'v11m3'}
    exp_tokens = [token for token in raw_exp_name.split('_') if token]
    normalized_tokens = [token for token in exp_tokens if token not in version_tokens]
    exp_name = '_'.join(normalized_tokens + ['v11m3'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print_rank0(f"Created new experiment directory: {exp_dir}")
    return exp_dir


def create_model(config: Dict, device: torch.device, use_gradient_checkpointing: bool = None, use_compile: bool = None) -> nn.Module:
    """Create model from config.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        use_gradient_checkpointing: Override gradient checkpointing setting
        use_compile: Override torch.compile setting (V7m1)
    """
    model_cfg = config['model']

    # V7: Gradient checkpointing - prioritize config file, allow CLI override
    if use_gradient_checkpointing is not None:
        # CLI argument overrides config file
        model_cfg['use_gradient_checkpointing'] = use_gradient_checkpointing
        print_rank0(f"Gradient checkpointing: {use_gradient_checkpointing} (from CLI)")
    elif 'use_gradient_checkpointing' in model_cfg:
        # Use config file setting
        print_rank0(f"Gradient checkpointing: {model_cfg['use_gradient_checkpointing']} (from config)")
    else:
        # Default to False if not specified
        model_cfg['use_gradient_checkpointing'] = False
        print_rank0("Gradient checkpointing: False (default)")

    # V7m1: Torch.compile support - prioritize CLI, then config, then default
    if use_compile is not None:
        # CLI argument overrides config file
        enable_compile = use_compile
        print_rank0(f"Torch.compile: {enable_compile} (from CLI)")
    elif 'use_compile' in model_cfg:
        # Use config file setting
        enable_compile = model_cfg['use_compile']
        print_rank0(f"Torch.compile: {enable_compile} (from config)")
    else:
        # Default to False if not specified
        enable_compile = False
        print_rank0("Torch.compile: False (default)")

    # Check for ablation flags
    ablation_flags = ['use_gvp', 'use_esm', 'use_physchem_early',
                      'use_physchem_late', 'use_self_attention']
    is_ablation = any(flag in model_cfg for flag in ablation_flags)

    if is_ablation:
        print_rank0("Creating ablation model...")
        model = PocketGVPFusionAblation(model_cfg)
    else:
        n_lig_sa   = model_cfg.get('n_ligand_self_attn_layers', 0)
        dist_bias  = model_cfg.get('use_lig_dist_bias', True)
        b_lig1     = model_cfg.get('use_lig_attnres_b1', True)
        unimol_on = model_cfg.get('use_unimol_ligand_input', True)
        unimol_proj = model_cfg.get('d_unimol_ligand_proj', 64)
        print_rank0(f"Creating V11m3 model (ligand self-attn layers={n_lig_sa}, "
                    f"lig_dist_bias={dist_bias}, b_lig1_attnres={b_lig1}, "
                    f"unimol_ligand_input={unimol_on}, unimol_proj={unimol_proj})...")
        model = PocketGVPFusionV11m3(model_cfg)

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # V7m1: Apply torch.compile if enabled
    if enable_compile:
        print_rank0("Compiling model with torch.compile (dynamic=True, mode='reduce-overhead')...")
        print_rank0("Note: First epoch will be slow due to compilation overhead")
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        print_rank0("Model compiled successfully")

    return model


# ============================================================================
# 损失函数
# ============================================================================

class ClusterWeightedLoss:
    """
    基于簇大小的动态样本权重 Loss

    对小簇（孤儿蛋白）施加更高的权重，提高模型对低相似度样本的泛化能力。

    权重公式:
        weight = 1 / log(cluster_size + ε)

    其中 ε 是平滑项，防止 log(1) = 0
    """

    def __init__(self, cluster_sizes: Dict[str, int], epsilon: float = 2.0):
        """
        Args:
            cluster_sizes: Dict mapping pdb_id -> cluster_size
            epsilon: 平滑项，防止权重过大 (default: 2.0)
        """
        self.cluster_sizes = cluster_sizes
        self.epsilon = epsilon

        # 预计算所有样本的权重
        self.weights = {}
        for pdb_id, size in cluster_sizes.items():
            # weight = 1 / log(size + ε)
            weight = 1.0 / torch.log(torch.tensor(size + epsilon, dtype=torch.float32))
            self.weights[pdb_id] = weight.item()

        # 统计信息
        weights_list = list(self.weights.values())
        self.min_weight = min(weights_list)
        self.max_weight = max(weights_list)
        self.mean_weight = sum(weights_list) / len(weights_list)

    def get_batch_weights(self, pdb_ids: list) -> torch.Tensor:
        """
        获取一个 batch 的样本权重

        Args:
            pdb_ids: List of PDB IDs in the batch

        Returns:
            weights: [B] tensor of sample weights
        """
        weights = [self.weights.get(pdb_id, 1.0) for pdb_id in pdb_ids]
        return torch.tensor(weights, dtype=torch.float32)

    def print_statistics(self):
        """打印权重统计信息"""
        print(f"\n{'='*60}")
        print(f"Cluster-Weighted Loss Statistics:")
        print(f"{'='*60}")
        print(f"  Total samples: {len(self.weights)}")
        print(f"  Weight range: [{self.min_weight:.4f}, {self.max_weight:.4f}]")
        print(f"  Mean weight: {self.mean_weight:.4f}")
        print(f"  Epsilon: {self.epsilon}")

        # 找出权重最高的样本（最小簇）
        top_5 = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 highest-weighted samples (smallest clusters):")
        for pdb_id, weight in top_5:
            cluster_size = self.cluster_sizes[pdb_id]
            print(f"    {pdb_id}: weight={weight:.4f}, cluster_size={cluster_size}")

        # 找出权重最低的样本（最大簇）
        bottom_5 = sorted(self.weights.items(), key=lambda x: x[1])[:5]
        print(f"\n  Top 5 lowest-weighted samples (largest clusters):")
        for pdb_id, weight in bottom_5:
            cluster_size = self.cluster_sizes[pdb_id]
            print(f"    {pdb_id}: weight={weight:.4f}, cluster_size={cluster_size}")
        print(f"{'='*60}\n")


def compute_tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    V10m4: Per-sample Contact-only Tversky Loss.

    Tversky(Contact) = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    相比 global batch Dice 的改进：
    - Per-sample 聚合：避免 batch 内少数类被多数类稀释
    - Contact-only：专注 binding site，梯度方向明确
    - alpha < beta：允许适量 FP，强调不漏检（recall 优先）
    - smooth=1.0：训练早期梯度稳定，避免梯度爆炸

    Args:
        logits: [B, L, C]
        labels: [B, L]
        mask:   [B, L]  padding mask (True = valid)
        alpha:  FP 惩罚系数（低 = 允许更多 FP）
        beta:   FN 惩罚系数（高 = 减少漏检）
        smooth: 平滑项（推荐 1.0）
    """
    probs     = F.softmax(logits, dim=-1)   # [B, L, C]
    p_contact = probs[:, :, 0]              # [B, L]
    t_contact = (labels == 0).float()       # [B, L]
    m         = mask.float()               # [B, L]

    TP = (p_contact * t_contact * m).sum(dim=1)               # [B]
    FP = (p_contact * (1.0 - t_contact) * m).sum(dim=1)       # [B]
    FN = ((1.0 - p_contact) * t_contact * m).sum(dim=1)       # [B]

    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return (1.0 - tversky_coef).mean()


def compute_weighted_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    V10m4: Per-sample class-frequency Weighted Dice Loss.

    相比旧 global batch Dice 的改进：
    - Per-sample 聚合：每个蛋白独立计算 Dice，再 batch mean
    - class_weights 加权：Contact 权重高，Far 权重低，不受 Far 主导
    - smooth=1.0：训练早期稳定

    Args:
        logits:        [B, L, C]
        labels:        [B, L]
        mask:          [B, L]
        class_weights: [C]  与 focal loss 共享同一组权重
        smooth:        平滑项（推荐 1.0）
    """
    B, L, C = logits.shape
    device   = logits.device

    probs         = F.softmax(logits, dim=-1)                          # [B, L, C]
    labels_one_hot = F.one_hot(labels, num_classes=C).float()          # [B, L, C]
    m             = mask.unsqueeze(-1).float()                         # [B, L, 1]

    # Per-sample per-class Dice
    intersection   = (probs * labels_one_hot * m).sum(dim=1)           # [B, C]
    pred_sum       = (probs * m).sum(dim=1)                            # [B, C]
    target_sum     = (labels_one_hot * m).sum(dim=1)                   # [B, C]
    dice_per_class = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)  # [B, C]

    # Weighted average over classes (normalize weights to sum=1)
    w = class_weights.to(device)
    w = w / w.sum()                                                    # [C], normalized
    weighted_dice = (dice_per_class * w.unsqueeze(0)).sum(dim=1)       # [B]

    return (1.0 - weighted_dice).mean()


def compute_combined_loss_v4(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float,
    focal_weight: float = 1.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_smooth: float = 1.0,
    weighted_dice_weight: float = 0.0,
    weighted_dice_smooth: float = 1.0,
    label_smoothing: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    cnc_weights: Optional[torch.Tensor] = None,
    ordinal_label_smoothing: bool = False,  # V11m2: restrict smoothing to adjacent classes
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    V10m7 统一组合损失入口（基于 V10m4，新增 CNC 支持）。

    total = focal_weight * focal
           + tversky_weight * tversky          (Contact-only, per-sample)
           + weighted_dice_weight * w_dice     (all-class weighted, per-sample)

    当 tversky_weight=0 且 weighted_dice_weight=0 时退化为纯 Focal（最稳定）。

    Args:
        sample_weights: [B] 样本权重（cluster dedup），None = 等权
        cnc_weights:    [B, L] per-residue CNC 权重（V10m7），None = 不启用
                        对 Focal: 直接乘以 per-residue loss
                        对 Tversky: 乘以 FP 项（降低被 clip 残基的 false positive 惩罚）
                        对 Dice: 乘以 c_2 残基的 pred_sum 贡献
    Returns:
        (total_loss, tversky_loss, focal_loss)
        tversky_loss 仅在 tversky_weight > 0 时有意义（否则为 0 tensor）
    """
    B, L, C = logits.shape
    device   = logits.device

    # Smart gating: skip entire computation branch when weight == 0
    _need_focal  = focal_weight > 0
    _need_tversky = tversky_weight > 0
    _need_wdice  = weighted_dice_weight > 0

    # Shared tensors (computed lazily only when needed)
    _probs = _labels_oh = None

    def _get_shared():
        nonlocal _probs, _labels_oh
        if _probs is None:
            _labels_oh = F.one_hot(labels, num_classes=C).float()
            log_p = F.log_softmax(logits, dim=-1)
            _probs = (log_p, log_p.exp(), _labels_oh)
        return _probs

    # ── Focal Loss ──────────────────────────────────────────────────────────
    if _need_focal:
        log_probs, probs, labels_one_hot = _get_shared()

        if label_smoothing > 0:
            if ordinal_label_smoothing and C == 3:
                # V11m2 Ordinal smoothing: Contact/Near/Far are ordinal labels.
                # Mass goes only to adjacent classes:
                #   Contact(0) → [1-ε, ε,   0  ]
                #   Near   (1) → [ε/2, 1-ε, ε/2]
                #   Far    (2) → [0,   ε,   1-ε ]
                # Tversky, dist_aux, and metrics use hard labels — unaffected.
                sm = logits.new_tensor([
                    [1.0 - label_smoothing, label_smoothing,       0.0                  ],
                    [label_smoothing / 2,   1.0 - label_smoothing, label_smoothing / 2  ],
                    [0.0,                   label_smoothing,        1.0 - label_smoothing],
                ])  # [3, 3]
                smooth_labels = sm[labels.view(-1)].view_as(labels_one_hot)  # [B, L, 3]
            else:
                smooth_labels = labels_one_hot * (1.0 - label_smoothing) + label_smoothing / C
        else:
            smooth_labels = labels_one_hot

        p_t          = (probs * labels_one_hot).sum(dim=-1, keepdim=True)      # [B, L, 1]
        focal_w      = (1.0 - p_t) ** focal_gamma                              # [B, L, 1]
        ce_loss      = -(smooth_labels * log_probs).sum(dim=-1)                # [B, L]
        weighted_ce  = ce_loss * focal_w.squeeze(-1)                           # [B, L]

        cls_w = class_weights.to(device).view(1, 1, C)
        cls_w_per_pos = (labels_one_hot * cls_w).sum(dim=-1)                   # [B, L]
        weighted_ce   = weighted_ce * cls_w_per_pos                            # [B, L]

        masked_focal = weighted_ce * mask.float()                              # [B, L]
        # V10m7: CNC — reduce focal loss for clipped c_2 residues
        if cnc_weights is not None:
            masked_focal = masked_focal * cnc_weights                          # [B, L]
        n_valid      = mask.float().sum(dim=1).clamp(min=1.0)                  # [B]
        focal_per_sample = masked_focal.sum(dim=1) / n_valid                   # [B]
    else:
        focal_per_sample = torch.zeros(B, device=device)

    # ── Tversky Loss ─────────────────────────────────────────────────────────
    if _need_tversky:
        _, probs, _ = _get_shared()
        p_contact = probs[:, :, 0]
        t_contact = (labels == 0).float()
        m         = mask.float()
        TP = (p_contact * t_contact * m).sum(dim=1)
        # V10m7: CNC — reduce FP contribution of clipped c_2 residues
        if cnc_weights is not None:
            FP = (p_contact * (1.0 - t_contact) * m * cnc_weights).sum(dim=1)
        else:
            FP = (p_contact * (1.0 - t_contact) * m).sum(dim=1)
        FN = ((1.0 - p_contact) * t_contact * m).sum(dim=1)
        tv_coef     = (TP + tversky_smooth) / (TP + tversky_alpha * FP + tversky_beta * FN + tversky_smooth)
        tversky_per_sample = 1.0 - tv_coef                                 # [B]
    else:
        tversky_per_sample = torch.zeros(B, device=device)

    # ── Weighted Dice Loss ───────────────────────────────────────────────────
    if _need_wdice:
        _, probs, labels_one_hot = _get_shared()
        m_exp      = mask.unsqueeze(-1).float()
        intersect  = (probs * labels_one_hot * m_exp).sum(dim=1)           # [B, C]
        # V10m7: CNC — reduce pred_sum contribution of clipped c_2 residues
        if cnc_weights is not None:
            cnc_exp = cnc_weights.unsqueeze(-1)                            # [B, L, 1]
            pred_sum = (probs * m_exp * cnc_exp).sum(dim=1)               # [B, C]
        else:
            pred_sum = (probs * m_exp).sum(dim=1)                         # [B, C]
        tgt_sum    = (labels_one_hot * m_exp).sum(dim=1)                   # [B, C]
        dice_cls   = (2.0 * intersect + weighted_dice_smooth) / (pred_sum + tgt_sum + weighted_dice_smooth)
        w_norm     = class_weights.to(device) / class_weights.to(device).sum()
        wdice_per_sample = 1.0 - (dice_cls * w_norm.unsqueeze(0)).sum(dim=1)  # [B]
    else:
        wdice_per_sample = torch.zeros(B, device=device)

    # ── Apply sample weights & combine ───────────────────────────────────────
    if sample_weights is not None:
        sw = sample_weights.to(device).view(B)
    else:
        sw = torch.ones(B, device=device)

    focal_loss   = (focal_per_sample * sw).mean()   if _need_focal   else torch.tensor(0.0, device=device)
    tversky_loss = (tversky_per_sample * sw).mean() if _need_tversky else torch.tensor(0.0, device=device)
    wdice_loss   = (wdice_per_sample * sw).mean()   if _need_wdice  else torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    if _need_focal:   total_loss = total_loss + focal_weight * focal_loss
    if _need_tversky: total_loss = total_loss + tversky_weight * tversky_loss
    if _need_wdice:   total_loss = total_loss + weighted_dice_weight * wdice_loss

    return total_loss, tversky_loss, focal_loss


def compute_dist_aux_loss(
    pred_dist: torch.Tensor,
    target_dist: torch.Tensor,
    mask: torch.Tensor,
    max_dist: float = 20.0,
) -> torch.Tensor:
    """
    V10m8: Distance regression auxiliary loss.

    Uses Huber loss with proximity-weighted scheme: closer residues
    get higher weight since boundary precision matters most there.

    Args:
        pred_dist:    [B, L] predicted distances from Softplus head
        target_dist:  [B, L] ground-truth residue-ligand distances
        mask:         [B, L] validity mask
        max_dist:     clamp target distances for numerical stability

    Returns:
        Scalar loss
    """
    m = mask.float()
    # Clamp target to avoid extreme values from padding
    t = target_dist.clamp(max=max_dist)

    # Huber loss (smooth L1): robust to outliers at large distances
    huber = F.smooth_l1_loss(pred_dist, t, reduction='none')  # [B, L]

    # Proximity weighting: w = 1 / (1 + d), so contact residues matter more
    with torch.no_grad():
        prox_weight = 1.0 / (1.0 + t)  # [B, L], range (0, 1]

    weighted_loss = huber * prox_weight * m  # [B, L]
    n_valid = m.sum().clamp(min=1.0)
    return weighted_loss.sum() / n_valid


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


def compute_combined_loss_with_sample_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    focal_gamma: float,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算带有样本权重的组合损失: Dice Loss + Focal Loss

    V7m3: 添加 sample_weights 参数，基于簇大小动态调整每个样本的损失权重
    V9m2: 优化为向量化实现，避免逐样本循环（性能提升10-20x）

    Args:
        logits: [B, L, C] 模型输出
        labels: [B, L] 真实标签
        mask: [B, L] padding mask
        class_weights: [C] 类别权重
        sample_weights: [B] 样本权重（基于簇大小）
        focal_gamma: Focal loss gamma 参数
        dice_weight: Dice loss 权重
        focal_weight: Focal loss 权重
        label_smoothing: Label smoothing 参数

    Returns:
        total_loss: 总损失
        dice_loss: Dice 损失
        focal_loss: Focal 损失
    """
    B, L, C = logits.shape
    device = logits.device

    # 将 sample_weights 移到正确的设备并reshape为 [B]
    sample_weights = sample_weights.to(device).view(B)

    # === Dice Loss (向量化版本) ===
    # Softmax + one-hot encoding
    probs = torch.softmax(logits, dim=-1)  # [B, L, C]
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=C).float()  # [B, L, C]

    # Apply mask: [B, L, 1]
    mask_expanded = mask.unsqueeze(-1).float()

    # Compute intersection and union per sample
    intersection = (probs * labels_one_hot * mask_expanded).sum(dim=1)  # [B, C]
    pred_sum = (probs * mask_expanded).sum(dim=1)  # [B, C]
    target_sum = (labels_one_hot * mask_expanded).sum(dim=1)  # [B, C]

    # Dice coefficient per sample per class
    dice_coef = (2.0 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)  # [B, C]
    dice_loss_per_sample = 1.0 - dice_coef.mean(dim=1)  # [B]

    # Apply sample weights and average
    dice_loss_weighted = (dice_loss_per_sample * sample_weights).mean()

    # === Focal Loss (向量化版本) ===
    # Cross entropy with label smoothing
    if label_smoothing > 0:
        # Smooth labels: [B, L, C]
        smooth_labels = labels_one_hot * (1 - label_smoothing) + label_smoothing / C
    else:
        smooth_labels = labels_one_hot

    # Log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, L, C]

    # Focal weight: (1 - p_t)^gamma
    # p_t is probability of true class
    p_t = (probs * labels_one_hot).sum(dim=-1, keepdim=True)  # [B, L, 1]
    focal_weight = (1 - p_t) ** focal_gamma  # [B, L, 1]

    # Class weights: [C] -> [1, 1, C]
    class_weights_expanded = class_weights.view(1, 1, C)

    # Compute focal loss per position: [B, L]
    ce_loss = -(smooth_labels * log_probs).sum(dim=-1)  # [B, L]
    weighted_ce = ce_loss * focal_weight.squeeze(-1)  # [B, L]

    # Apply class weights (average weight for each position based on true label)
    class_weight_per_pos = (labels_one_hot * class_weights_expanded).sum(dim=-1)  # [B, L]
    weighted_ce = weighted_ce * class_weight_per_pos  # [B, L]

    # Apply mask and compute per-sample loss
    masked_loss = weighted_ce * mask.float()  # [B, L]
    focal_loss_per_sample = masked_loss.sum(dim=1) / (mask.sum(dim=1).float() + 1e-8)  # [B]

    # Apply sample weights and average
    focal_loss_weighted = (focal_loss_per_sample * sample_weights).mean()

    # 组合损失
    total_loss = dice_weight * dice_loss_weighted + focal_weight * focal_loss_weighted

    return total_loss, dice_loss_weighted, focal_loss_weighted


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

    # V11: Ligand internal self-attention
    if hasattr(model, 'ligand_self_attn') and model.ligand_self_attn is not None:
        stats['grad_norm_ligand_self_attn'] = get_grad_norm(model.ligand_self_attn)

    # V11m3: Uni-Mol ligand feature injection path
    if hasattr(model, 'unimol_ligand_proj') and model.unimol_ligand_proj is not None:
        stats['grad_norm_unimol_ligand_proj'] = get_grad_norm(model.unimol_ligand_proj)
    alpha = getattr(model, 'unimol_ligand_alpha', None)
    if isinstance(alpha, torch.nn.Parameter) and alpha.grad is not None:
        stats['grad_norm_unimol_ligand_alpha'] = alpha.grad.detach().float().norm(2).item()

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


def compute_unimol_stats(model: nn.Module, unimol_embed: Optional[torch.Tensor]) -> Dict[str, float]:
    """V11m3: monitor Uni-Mol ligand feature coverage and injected magnitude."""
    if isinstance(model, DDP):
        model = model.module
    stats: Dict[str, float] = {}
    if unimol_embed is None:
        stats['unimol_coverage'] = 0.0
        return stats

    with torch.no_grad():
        ue = unimol_embed.detach().float()
        row_valid = torch.isfinite(ue).all(dim=-1) & (ue.abs().sum(dim=-1) > 1e-6)
        stats['unimol_coverage'] = row_valid.float().mean().item()
        if row_valid.any():
            valid_ue = ue[row_valid]
            stats['unimol_raw_norm'] = (
                valid_ue.norm(dim=-1).mean() / (valid_ue.shape[-1] ** 0.5)
            ).item()

        delta = getattr(model, '_cached_unimol_ligand_delta', None)
        if delta is not None and delta.numel() > 0:
            delta = delta.detach().float()
            delta_norm = delta.norm(dim=-1).mean() / (delta.shape[-1] ** 0.5)
            stats['unimol_ligand_delta_norm'] = delta_norm.item()

            alpha = getattr(model, 'unimol_ligand_alpha', None)
            if alpha is not None:
                stats['unimol_ligand_alpha'] = float(alpha.detach().float().item())

            proj_dim = getattr(model, 'd_unimol_ligand_proj', delta.shape[-1])
            raw_dim = getattr(model, 'raw_ligand_atom_dim', None)
            if raw_dim is not None:
                # Approximate dimensional share of injected Uni-Mol channels.
                stats['unimol_ligand_channel_ratio'] = float(proj_dim) / float(raw_dim + proj_dim)

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
    focal_weight: float = 1.0,           # V10m4
    tversky_weight: float = 0.0,         # V10m4: Contact-only Tversky
    tversky_alpha: float = 0.3,          # V10m4
    tversky_beta: float = 0.7,           # V10m4
    tversky_smooth: float = 1.0,         # V10m4
    weighted_dice_weight: float = 0.0,   # V10m4: class-weighted Dice
    weighted_dice_smooth: float = 1.0,   # V10m4
    iou_weight: float = 0.0,  # V7m2: IoU loss weight
    iou_component_weight: float = 1.0,  # V7m2: IoU component weight
    iou_dist_weight: float = 0.2,  # V7m2: Distance penalty weight
    iou_power: float = 1.5,  # V7m2: Distance penalty power
    iou_d_max: float = 10.0,  # V7m2: Distance normalization factor
    cluster_weighted_loss: Optional[ClusterWeightedLoss] = None,  # V7m3: Cluster-weighted loss
    use_boundary_loss: bool = False,  # V9m2: enable boundary loss
    boundary_weight: float = 0.2,  # V9m2: boundary loss weight
    boundary_multiplier: float = 2.0,  # V9m2: boundary residue weight multiplier
    boundary_distance_threshold: float = 8.0,  # V9m2: boundary detection distance
    label_smoothing: float = 0.15,  # V9m2: label smoothing (default 0.15)
    enable_profiling: bool = False,  # V9m2: enable performance profiling
    exp_dir: Optional[Path] = None,  # V9m2: experiment directory for logs
    use_bf16: bool = False,  # V10: use BF16 instead of FP16
    cnc_enabled: bool = False,           # V10m7: Confident Negative Clipping
    cnc_clip_thresh: float = 0.85,       # V10m7
    cnc_clip_ratio: float = 0.1,         # V10m7
    cnc_warmup_epochs: int = 5,          # V10m7
    cnc_max_clip_frac: float = 0.3,      # V10m7
    dist_aux_enabled: bool = False,      # V10m8: Distance regression aux head
    dist_aux_weight: float = 0.1,        # V10m8: weight for dist aux loss
    dist_aux_max_dist: float = 20.0,     # V10m8: clamp distance
    ordinal_label_smoothing: bool = False,  # V11m2: ordinal-aware label smoothing
    unimol_cache: Optional[Dict[str, torch.Tensor]] = None,  # V11m3
    d_unimol: int = 512,                                      # V11m3
) -> Dict[str, float]:
    """Train for one epoch. V11m3: V11m2 + Uni-Mol ligand input."""
    model.train()

    # V9m2: 初始化性能监控器
    perf_monitor = None
    if enable_profiling:
        perf_monitor = PerformanceMonitor(window_size=100, log_every=log_every)
        if exp_dir:
            perf_monitor.set_log_file(log_dir=str(exp_dir), exp_name=exp_dir.name)

    # V10: Tensor accumulators for reduced CPU-GPU sync
    total_loss = torch.tensor(0.0, device=device)
    total_tversky_loss = torch.tensor(0.0, device=device)   # V10m4
    total_focal_loss = torch.tensor(0.0, device=device)
    total_iou_loss = torch.tensor(0.0, device=device)
    total_iou_component = torch.tensor(0.0, device=device)
    total_iou_dist = torch.tensor(0.0, device=device)
    total_boundary_loss = torch.tensor(0.0, device=device)
    total_boundary_ratio = 0.0
    total_cnc_clipped = 0       # V10m7
    total_cnc_valid = 0         # V10m7
    total_dist_aux_loss = torch.tensor(0.0, device=device)  # V10m8
    total_samples = 0
    all_metrics = []

    # V10: BF16 support
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # V9m2: 开始数据加载计时
        if perf_monitor:
            perf_monitor.start('data_loading')

        # Prepare inputs
        _unimol = get_batch_unimol(batch['pdb_ids'], unimol_cache or {}, d_unimol, device)
        inputs = prepare_model_inputs_v11(batch, device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask = batch['protein_mask'].to(device, non_blocking=True)

        # V9m2: 结束数据加载，开始前向传播
        if perf_monitor:
            perf_monitor.end('data_loading')
            perf_monitor.start('forward')

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_physchem=inputs['protein_physchem'],  # V8: 62D 物理化学特征
                protein_spatial_scalar=inputs['protein_spatial_scalar'],  # V8: 72D 空间标量特征
                protein_spatial_vector=inputs['protein_spatial_vector'],  # V8: 6×3D 空间矢量特征
                protein_gvp_vectors=inputs['protein_gvp_vectors'],  # V8: 4×3D GVP 矢量
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                chain_ids=inputs.get('chain_ids'),
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_mask=inputs['ligand_mask'],
                unimol_embed=_unimol
            )

            logits = outputs['logits']

            # V9m2: 结束前向传播，开始损失计算
            if perf_monitor:
                perf_monitor.end('forward')
                perf_monitor.start('loss_computation')

            # V10m4: 统一使用 compute_combined_loss_v4
            # 支持 cluster sample_weights；boundary loss 仍可选叠加
            sw = None
            if cluster_weighted_loss is not None:
                pdb_ids = batch['pdb_ids']
                sw = cluster_weighted_loss.get_batch_weights(pdb_ids)

            # V10m7: Confident Negative Clipping weights
            cnc_w = None
            if cnc_enabled:
                cnc_w = compute_cnc_weights(
                    logits=logits.detach(),
                    labels=labels,
                    mask=mask,
                    clip_thresh=cnc_clip_thresh,
                    clip_ratio=cnc_clip_ratio,
                    warmup_epochs=cnc_warmup_epochs,
                    current_epoch=epoch,
                    max_clip_frac=cnc_max_clip_frac,
                )
                cnc_batch_stats = compute_cnc_stats(cnc_w, mask)
                total_cnc_clipped += cnc_batch_stats['n_clipped']
                total_cnc_valid += cnc_batch_stats['n_valid']

            loss, tversky_loss_val, focal_loss = compute_combined_loss_v4(
                logits=logits,
                labels=labels,
                mask=mask,
                class_weights=class_weights,
                focal_gamma=gamma,
                focal_weight=focal_weight,
                tversky_weight=tversky_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_smooth=tversky_smooth,
                weighted_dice_weight=weighted_dice_weight,
                weighted_dice_smooth=weighted_dice_smooth,
                label_smoothing=label_smoothing,
                sample_weights=sw,
                cnc_weights=cnc_w,
                ordinal_label_smoothing=ordinal_label_smoothing,  # V11m2
            )

            # V9m2: Boundary loss（可选叠加）
            if use_boundary_loss:
                _, boundary_loss_val, boundary_ratio = compute_combined_loss_with_boundary(
                    logits=logits,
                    labels=labels,
                    coords=inputs['protein_coords'],
                    mask=mask,
                    class_weights=class_weights,
                    focal_gamma=gamma,
                    dice_weight=0.0,
                    focal_weight=1.0,
                    boundary_weight=boundary_weight,
                    boundary_multiplier=boundary_multiplier,
                    distance_threshold=boundary_distance_threshold,
                    label_smoothing=label_smoothing
                )
                # Only add boundary component, not re-do focal
                loss = loss + boundary_weight * boundary_loss_val
            else:
                boundary_loss_val = torch.tensor(0.0, device=device)
                boundary_ratio = 0.0

            # V7m2: IoU Loss（可选叠加）
            if iou_weight > 0:
                iou_loss, iou_component, iou_dist = compute_combined_iou_loss(
                    logits=logits,
                    labels=labels,
                    coords=inputs['protein_coords'],
                    mask=mask,
                    iou_weight=iou_component_weight,
                    dist_weight=iou_dist_weight,
                    power=iou_power,
                    d_max=iou_d_max
                )
                loss = loss + iou_weight * iou_loss
            else:
                iou_loss = torch.tensor(0.0, device=device)
                iou_component = torch.tensor(0.0, device=device)
                iou_dist = torch.tensor(0.0, device=device)

            # V10m8: Distance regression auxiliary loss
            if dist_aux_enabled and 'pred_dist' in outputs and 'raw_distances' in batch:
                target_dist = batch['raw_distances'].to(device, non_blocking=True)
                dist_aux_loss = compute_dist_aux_loss(
                    pred_dist=outputs['pred_dist'],
                    target_dist=target_dist,
                    mask=mask,
                    max_dist=dist_aux_max_dist,
                )
                loss = loss + dist_aux_weight * dist_aux_loss
            else:
                dist_aux_loss = torch.tensor(0.0, device=device)

            # 防御式处理：确保所有损失为标量
            if torch.is_tensor(tversky_loss_val) and tversky_loss_val.ndim > 0:
                tversky_loss_val = tversky_loss_val.mean()
            if torch.is_tensor(focal_loss) and focal_loss.ndim > 0:
                focal_loss = focal_loss.mean()
            if torch.is_tensor(iou_loss) and iou_loss.ndim > 0:
                iou_loss = iou_loss.mean()
            if torch.is_tensor(iou_component) and iou_component.ndim > 0:
                iou_component = iou_component.mean()
            if torch.is_tensor(iou_dist) and iou_dist.ndim > 0:
                iou_dist = iou_dist.mean()
            if torch.is_tensor(boundary_loss_val) and boundary_loss_val.ndim > 0:
                boundary_loss_val = boundary_loss_val.mean()
            if torch.is_tensor(loss) and loss.ndim > 0:
                loss = loss.mean()

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # V9m2: 结束损失计算，开始反向传播
        if perf_monitor:
            perf_monitor.end('loss_computation')
            perf_monitor.start('backward')

        # V10: Gradient accumulation (loss already divided by accumulation_steps above)
        loss_scaled = loss
        if use_amp and scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # V9m2: 结束反向传播
        if perf_monitor:
            perf_monitor.end('backward')

        # V10: Optimizer step (with gradient accumulation, including last batch)
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # V9m2: 开始优化器步骤
            if perf_monitor:
                perf_monitor.start('optimizer_step')

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

            optimizer.zero_grad(set_to_none=True)

            # V9m2: 结束优化器步骤
            if perf_monitor:
                perf_monitor.end('optimizer_step')

        # V9m2: 记录GPU统计并检查是否需要打印
        if perf_monitor:
            perf_monitor.step()
            if perf_monitor.should_log():
                perf_monitor.log_stats(epoch=epoch, batch_idx=batch_idx)
                # 诊断建议
                perf_monitor.diagnose_bottleneck()

        # V10: Track metrics with tensor accumulators
        batch_size = labels.size(0)
        total_loss += loss.detach() * accumulation_steps * batch_size
        total_tversky_loss += tversky_loss_val.detach() * batch_size   # V10m4
        total_focal_loss += focal_loss.detach() * batch_size
        total_iou_loss += iou_loss.detach() * batch_size
        total_iou_component += iou_component.detach() * batch_size
        total_iou_dist += iou_dist.detach() * batch_size
        if dist_aux_enabled:
            total_dist_aux_loss += dist_aux_loss.detach() * batch_size
        if use_boundary_loss:
            total_boundary_loss += boundary_loss_val.detach() * batch_size
            total_boundary_ratio += boundary_ratio * batch_size
        total_samples += batch_size

        batch_metrics = compute_metrics(logits.detach(), labels, mask)
        all_metrics.append(batch_metrics)

        # Logging
        if (batch_idx + 1) % log_every == 0 and is_main_process():
            elapsed = time.time() - start_time

            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item() * accumulation_steps:.4f} | "
                  f"Tversky: {tversky_loss_val.item():.4f} | "
                  f"Focal: {focal_loss.item():.4f} | "
                  f"Acc: {batch_metrics['accuracy']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/batch_loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('Train/tversky_loss', tversky_loss_val.item(), global_step)   # V10m4
                writer.add_scalar('Train/focal_loss', focal_loss.item(), global_step)
                writer.add_scalar('Train/batch_acc', batch_metrics['accuracy'], global_step)

                # V10m7: CNC stats
                if cnc_enabled and cnc_w is not None:
                    writer.add_scalar('CNC/batch_clip_fraction', cnc_batch_stats['clip_fraction'], global_step)
                    writer.add_scalar('CNC/batch_n_clipped', cnc_batch_stats['n_clipped'], global_step)

                # V10m8: dist aux loss
                if dist_aux_enabled:
                    writer.add_scalar('Train/dist_aux_loss', dist_aux_loss.item(), global_step)

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

                # V11m3: Uni-Mol ligand feature contribution diagnostics.
                unimol_stats = compute_unimol_stats(_model, _unimol)
                for key, value in unimol_stats.items():
                    writer.add_scalar(f'Health/{key}', value, global_step)

    # V10: Aggregate metrics - convert tensors to scalars
    avg_loss          = (total_loss / max(total_samples, 1)).item()
    avg_tversky_loss  = (total_tversky_loss / max(total_samples, 1)).item()   # V10m4
    avg_focal_loss    = (total_focal_loss / max(total_samples, 1)).item()
    avg_iou_loss      = (total_iou_loss / max(total_samples, 1)).item()
    avg_iou_component = (total_iou_component / max(total_samples, 1)).item()
    avg_iou_dist      = (total_iou_dist / max(total_samples, 1)).item()
    avg_boundary_loss = (total_boundary_loss / max(total_samples, 1)).item()
    avg_boundary_ratio = total_boundary_ratio / max(total_samples, 1)
    avg_metrics = {
        'loss':         avg_loss,
        'tversky_loss': avg_tversky_loss,   # V10m4 (replaces dice_loss)
        'dice_loss':    avg_tversky_loss,   # backward-compat alias for logging
        'focal_loss':   avg_focal_loss,
        'iou_loss':     avg_iou_loss,
        'iou_component':avg_iou_component,
        'iou_dist':     avg_iou_dist,
        'accuracy':     sum(m['accuracy'] for m in all_metrics) / len(all_metrics),
        'cnc_clip_fraction': total_cnc_clipped / max(total_cnc_valid, 1),  # V10m7
        'dist_aux_loss': (total_dist_aux_loss / max(total_samples, 1)).item(),  # V10m8
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
    evaluate_original: bool = False,
    focal_weight: float = 1.0,          # V10m4
    tversky_weight: float = 0.0,        # V10m4
    tversky_alpha: float = 0.3,         # V10m4
    tversky_beta: float = 0.7,          # V10m4
    tversky_smooth: float = 1.0,        # V10m4
    weighted_dice_weight: float = 0.0,  # V10m4
    weighted_dice_smooth: float = 1.0,  # V10m4
    unimol_cache: Optional[Dict[str, torch.Tensor]] = None,  # V11m3
    d_unimol: int = 512,                                      # V11m3
) -> Dict[str, float]:
    """
    Evaluate model with segmentation metrics. V10m4: Tversky + Focal loss.
    V5: 支持同时评估原始固定阈值标签
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
        _unimol = get_batch_unimol(batch['pdb_ids'], unimol_cache or {}, d_unimol, device)
        inputs = prepare_model_inputs_v11(batch, device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)  # VDW labels
        mask = batch['protein_mask'].to(device, non_blocking=True)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_physchem=inputs['protein_physchem'],  # V8: 62D 物理化学特征
                protein_spatial_scalar=inputs['protein_spatial_scalar'],  # V8: 72D 空间标量特征
                protein_spatial_vector=inputs['protein_spatial_vector'],  # V8: 6×3D 空间矢量特征
                protein_gvp_vectors=inputs['protein_gvp_vectors'],  # V8: 4×3D GVP 矢量
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                chain_ids=inputs.get('chain_ids'),
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_mask=inputs['ligand_mask'],
                unimol_embed=_unimol
            )

            logits = outputs['logits']

            # V10m4: 统一使用 compute_combined_loss_v4
            loss, tversky_loss_val, focal_loss = compute_combined_loss_v4(
                logits=logits,
                labels=labels,
                mask=mask,
                class_weights=class_weights,
                focal_gamma=gamma,
                focal_weight=focal_weight,
                tversky_weight=tversky_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_smooth=tversky_smooth,
                weighted_dice_weight=weighted_dice_weight,
                weighted_dice_smooth=weighted_dice_smooth,
            )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_dice_loss += tversky_loss_val.item() * batch_size   # V10m4 alias
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
                loss_original, tversky_orig, focal_loss_original = compute_combined_loss_v4(
                    logits=logits,
                    labels=labels_original,
                    mask=mask,
                    class_weights=class_weights,
                    focal_gamma=gamma,
                    focal_weight=focal_weight,
                    tversky_weight=tversky_weight,
                    tversky_alpha=tversky_alpha,
                    tversky_beta=tversky_beta,
                    tversky_smooth=tversky_smooth,
                    weighted_dice_weight=weighted_dice_weight,
                    weighted_dice_smooth=weighted_dice_smooth,
                )

            total_loss_original += loss_original.item() * batch_size
            total_dice_loss_original += tversky_orig.item() * batch_size
            total_focal_loss_original += focal_loss_original.item() * batch_size
            total_samples_original += batch_size

            batch_metrics_original = compute_metrics(logits, labels_original, mask)
            all_metrics_original.append(batch_metrics_original)

            all_preds_original.append(preds[mask].cpu())
            all_labels_original.append(labels_original[mask].cpu())

    # Aggregate metrics
    avg_loss      = total_loss / max(total_samples, 1)
    avg_dice_loss = total_dice_loss / max(total_samples, 1)   # tversky alias
    avg_focal_loss= total_focal_loss / max(total_samples, 1)
    avg_metrics = {
        'loss':       avg_loss,
        'dice_loss':  avg_dice_loss,   # V10m4: actually tversky, alias for logging compat
        'focal_loss': avg_focal_loss,
        'accuracy':   sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
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
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Use gradient checkpointing for memory efficiency (V7)')

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

    # V10: Enable cuDNN/MIOpen benchmark (configurable)
    torch.backends.cudnn.benchmark = config['training'].get('cudnn_benchmark', False)

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
    cluster_sizes = None  # V7m3: Store cluster sizes for weighted loss
    if rank == 0:
        split_file = exp_dir / "splits.txt" if exp_dir else Path("splits.txt")

        if data_cfg.get('split_file') and Path(data_cfg['split_file']).exists():
            # Load from specified split file
            source_split_file = data_cfg['split_file']
            print_rank0(f"Loading splits from: {source_split_file}")

            # V7m3: Try to load with cluster sizes
            try:
                train_ids, val_ids, test_ids, cluster_sizes = load_splits_with_cluster_size(source_split_file)
                print_rank0(f"✓ Loaded cluster size information for {len(cluster_sizes)} samples")
            except Exception as e:
                # Fallback to standard loading if cluster sizes not available
                print_rank0(f"⚠ Cluster sizes not available, using standard loading: {e}")
                train_ids, val_ids, test_ids = load_splits(source_split_file)
                cluster_sizes = None

            # Always save a copy to experiment directory for verification
            save_splits(train_ids, val_ids, test_ids, split_file)
            print_rank0(f"Saved splits copy to: {split_file}")
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
            print_rank0(f"Created and saved new splits to: {split_file}")
            cluster_sizes = None  # No cluster sizes for random splits

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

    ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v4')
    protein_features_dir = data_cfg.get('protein_features_dir', 'complete_features_v3')
    nma_features_dir = data_cfg.get('nma_features_dir', 'complete_features_v3_nma')  # V9
    nma_n_confs = data_cfg.get('nma_n_confs', 3)   # V9: default 3
    nma_max_conf = data_cfg.get('nma_max_conf', 9)  # V9: default 9
    distances_file = data_cfg.get('distances_file')
    distances_file_original = data_cfg.get('distances_file_original')

    # V10m3: d_center label thresholds (Contact < 5.0Å, Near 5.0-8.0Å, Far > 8.0Å)
    label_contact_thresh = data_cfg.get('label_contact_thresh', 5.0)
    label_near_thresh = data_cfg.get('label_near_thresh', 8.0)

    if is_main_process():
        print_rank0("Data sources:")
        print_rank0(f"  protein_features_dir: data/{protein_features_dir}")
        print_rank0(f"  nma_features_dir: data/{nma_features_dir}")
        print_rank0(f"  nma_n_confs: {nma_n_confs} (train only)")
        print_rank0(f"  ligand_features_dir: data/{ligand_features_dir}")
        print_rank0(f"  distances_file (d_center): {distances_file or 'DEFAULT'}")
        print_rank0(f"  label_thresholds: Contact<{label_contact_thresh}Å, Near<{label_near_thresh}Å")
        if distances_file_original:
            print_rank0(f"  distances_file_original: {distances_file_original}")

    # V11m3: Uni-Mol cache. Config can omit this key; the old pocket YAML still works
    # if data/unimol_embeddings follows the standard extractor layout.
    use_unimol_ligand_input = config['model'].get('use_unimol_ligand_input', True)
    d_unimol = int(config['model'].get('d_unimol', 512))
    if use_unimol_ligand_input:
        unimol_dir = data_cfg.get('unimol_embeddings_dir', 'unimol_embeddings')
        if is_main_process():
            print_rank0(f"  unimol_embeddings_dir: data/{unimol_dir}")
        unimol_cache = load_unimol_cache(data_root, unimol_dir)
    else:
        unimol_cache = {}

    train_dataset = PocketDatasetV8(  # V10m3: d_center labels (5.0/8.0Å)
        data_root=data_root,
        pdb_ids=train_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,
        ligand_features_dir=ligand_features_dir,
        protein_features_dir=protein_features_dir,
        nma_features_dir=nma_features_dir,  # V9
        nma_n_confs=nma_n_confs,            # V9
        nma_max_conf=nma_max_conf,          # V9
        use_nma=True,                       # V9: enable NMA augmentation for train
        label_contact_thresh=label_contact_thresh,
        label_near_thresh=label_near_thresh,
        split="train",
        verbose=is_main_process(),
        print_fn=print_rank0
    )

    val_dataset = PocketDatasetV8(  # V10m3: d_center labels (5.0/8.0Å)
        data_root=data_root,
        pdb_ids=val_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        distances_file_original=distances_file_original,
        ligand_features_dir=ligand_features_dir,
        protein_features_dir=protein_features_dir,
        nma_features_dir=nma_features_dir,
        label_contact_thresh=label_contact_thresh,
        label_near_thresh=label_near_thresh,
        use_nma=False,                      # V9: val always uses conf0 only
        split="val",
        verbose=False,
        print_fn=print_rank0
    )

    # V9: Check if datasets are empty
    if len(train_dataset) == 0:
        error_msg = (
            "\n" + "="*60 + "\n"
            "❌ 错误: 训练数据集为空！\n"
            "="*60 + "\n"
            "可能原因:\n"
            "1. NMA 特征文件尚未提取\n"
            "2. 特征目录路径不正确\n"
            "\n"
            "解决方法:\n"
            "1. 提取 V4 NMA 特征:\n"
            "   python preparation/protein/extract_complete_features_v4.py \\\n"
            "     --input data/processed/ \\\n"
            f"     --output data/{nma_features_dir}/ \\\n"
            "     --n-confs 10 --workers 32\n"
            "\n"
            f"2. 验证特征文件: ls -la data/{nma_features_dir}/1981-2000/\n"
            "="*60
        )
        print_rank0(error_msg)
        sys.exit(1)

    if len(val_dataset) == 0:
        print_rank0("\n⚠️  警告: 验证数据集为空！")

    print_rank0(f"\n✅ 数据集加载成功:")
    print_rank0(f"   训练集: {len(train_dataset)} 样本")
    print_rank0(f"   验证集: {len(val_dataset)} 样本")

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
    pin_memory = train_cfg.get('pin_memory', True)
    prefetch_factor = train_cfg.get('prefetch_factor', 2)
    num_workers = train_cfg['num_workers']
    # persistent_workers: config-driven (default: True when num_workers > 0).
    # Set to false in config when debugging or on memory-constrained machines.
    persistent_workers = train_cfg.get('persistent_workers', num_workers > 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    # V7m3: Initialize Cluster-Weighted Loss
    cluster_weighted_loss = None
    if cluster_sizes is not None:
        print_rank0("\n" + "="*60)
        print_rank0("Initializing Cluster-Weighted Loss (V7m3)")
        print_rank0("="*60)
        cluster_weighted_loss = ClusterWeightedLoss(cluster_sizes, epsilon=2.0)
        if rank == 0:
            cluster_weighted_loss.print_statistics()
    else:
        print_rank0("\n⚠ Cluster sizes not available, using standard loss (no sample weighting)")

    # Model
    # V7: Pass gradient_checkpointing flag only if explicitly set via CLI
    gradient_checkpointing_arg = args.gradient_checkpointing if args.gradient_checkpointing else None
    # V7m1: Pass compile flag (will use dynamic=True inside create_model)
    compile_arg = args.compile if args.compile else None
    model = create_model(config, device, use_gradient_checkpointing=gradient_checkpointing_arg, use_compile=compile_arg)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print_rank0("Model wrapped with DistributedDataParallel")

    # Optimizer
    # V9: param groups to exclude LayerNorm and bias from weight decay
    no_decay_keywords = ['bias', 'norm.weight', 'norm.bias', 'LayerNorm.weight', 'LayerNorm.bias']
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(kw in n for kw in no_decay_keywords)
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(kw in n for kw in no_decay_keywords)
    ]
    optimizer = AdamW(
        [
            {'params': decay_params,    'weight_decay': train_cfg['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=train_cfg['learning_rate'],
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

        # V10m4: 获取损失权重配置
        focal_weight          = train_cfg.get('focal_weight', 1.0)
        tversky_weight        = train_cfg.get('tversky_weight', 0.0)
        tversky_alpha         = train_cfg.get('tversky_alpha', 0.3)
        tversky_beta          = train_cfg.get('tversky_beta', 0.7)
        tversky_smooth        = train_cfg.get('tversky_smooth', 1.0)
        weighted_dice_weight  = train_cfg.get('weighted_dice_weight', 0.0)
        weighted_dice_smooth  = train_cfg.get('weighted_dice_smooth', 1.0)

        # V7m2: 获取 IoU Loss 配置
        iou_weight = train_cfg.get('iou_weight', 0.0)
        iou_component_weight = train_cfg.get('iou_component_weight', 1.0)
        iou_dist_weight = train_cfg.get('iou_dist_weight', 0.2)
        iou_power = train_cfg.get('iou_power', 1.5)
        iou_d_max = train_cfg.get('iou_d_max', 10.0)

        # V10m7: CNC 配置
        cnc_cfg = train_cfg.get('cnc', {})
        cnc_enabled = cnc_cfg.get('enabled', False)
        cnc_clip_thresh = cnc_cfg.get('clip_thresh', 0.85)
        cnc_clip_ratio = cnc_cfg.get('clip_ratio', 0.1)
        cnc_warmup_epochs = cnc_cfg.get('warmup_epochs', 5)
        cnc_max_clip_frac = cnc_cfg.get('max_clip_frac', 0.3)
        if epoch == start_epoch and cnc_enabled:
            print_rank0(f"  CNC enabled: thresh={cnc_clip_thresh}, ratio={cnc_clip_ratio}, "
                        f"warmup={cnc_warmup_epochs}, max_clip_frac={cnc_max_clip_frac}")

        # V10m8: Distance regression aux head config
        dist_aux_cfg = train_cfg.get('dist_aux', {})
        dist_aux_enabled = dist_aux_cfg.get('enabled', False)
        dist_aux_weight = dist_aux_cfg.get('weight', 0.1)
        dist_aux_max_dist = dist_aux_cfg.get('max_dist', 20.0)
        if epoch == start_epoch and dist_aux_enabled:
            print_rank0(f"  Dist aux enabled: weight={dist_aux_weight}, max_dist={dist_aux_max_dist}")

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
            focal_weight=focal_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_smooth=tversky_smooth,
            weighted_dice_weight=weighted_dice_weight,
            weighted_dice_smooth=weighted_dice_smooth,
            iou_weight=iou_weight,
            iou_component_weight=iou_component_weight,
            iou_dist_weight=iou_dist_weight,
            iou_power=iou_power,
            iou_d_max=iou_d_max,
            cluster_weighted_loss=cluster_weighted_loss,
            use_boundary_loss=train_cfg.get('use_boundary_loss', False),
            boundary_weight=train_cfg.get('boundary_weight', 0.2),
            boundary_multiplier=train_cfg.get('boundary_multiplier', 2.0),
            boundary_distance_threshold=train_cfg.get('boundary_distance_threshold', 8.0),
            label_smoothing=train_cfg.get('label_smoothing', 0.05),   # V10m4: 0.05 default
            enable_profiling=train_cfg.get('enable_profiling', False),
            exp_dir=exp_dir,
            use_bf16=train_cfg.get('use_bf16', False),
            cnc_enabled=cnc_enabled,
            cnc_clip_thresh=cnc_clip_thresh,
            cnc_clip_ratio=cnc_clip_ratio,
            cnc_warmup_epochs=cnc_warmup_epochs,
            cnc_max_clip_frac=cnc_max_clip_frac,
            dist_aux_enabled=dist_aux_enabled,
            dist_aux_weight=dist_aux_weight,
            dist_aux_max_dist=dist_aux_max_dist,
            ordinal_label_smoothing=train_cfg.get('ordinal_label_smoothing', False),  # V11m2
            unimol_cache=unimol_cache,
            d_unimol=d_unimol,
        )

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            class_weights=class_weights,
            gamma=train_cfg['focal_gamma'],
            device=device,
            use_amp=use_amp,
            evaluate_original=True,
            focal_weight=focal_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_smooth=tversky_smooth,
            weighted_dice_weight=weighted_dice_weight,
            weighted_dice_smooth=weighted_dice_smooth,
            unimol_cache=unimol_cache,
            d_unimol=d_unimol,
        )

        # Scheduler step
        scheduler.step()

        # Logging
        if rank == 0:
            cnc_frac_str = f", CNC clip: {train_metrics['cnc_clip_fraction']:.3f}" if cnc_enabled else ""
            dist_aux_str = f", DistAux: {train_metrics['dist_aux_loss']:.4f}" if dist_aux_enabled else ""
            print_rank0(f"Train Loss: {train_metrics['loss']:.4f}, Tversky: {train_metrics['tversky_loss']:.4f}, Focal: {train_metrics['focal_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}{cnc_frac_str}{dist_aux_str}")
            print_rank0(f"Val Loss (d_center): {val_metrics['loss']:.4f}, Tversky: {val_metrics['dice_loss']:.4f}, Focal: {val_metrics['focal_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # 打印分割指标
            if 'dice_c0' in val_metrics:
                print_rank0(f"Val DICE (d_center): {val_metrics['dice_c0']:.4f}, IoU: {val_metrics['iou_c0']:.4f}, F1_c0: {val_metrics.get('f1_c0', 0):.4f}")

            if writer is not None:
                # V10m3: 命名空间策略
                # Train/ - 训练指标（d_center labels）
                # Val/   - 验证指标（d_center labels，与 Train/ 对称，可直接横向对比）
                # Health/ - 健康诊断指标

                # V10m7: CNC epoch-level stats
                if cnc_enabled:
                    writer.add_scalar('CNC/epoch_clip_fraction', train_metrics['cnc_clip_fraction'], epoch)

                # V10m8: dist aux epoch stats
                if dist_aux_enabled:
                    writer.add_scalar('Train/epoch_dist_aux_loss', train_metrics['dist_aux_loss'], epoch)

                # 训练指标
                writer.add_scalar('Train/epoch_loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/epoch_dice_loss', train_metrics['dice_loss'], epoch)
                writer.add_scalar('Train/epoch_focal_loss', train_metrics['focal_loss'], epoch)
                writer.add_scalar('Train/epoch_acc', train_metrics['accuracy'], epoch)
                writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

                # 验证指标（d_center labels）→ Val/
                writer.add_scalar('Val/epoch_loss', val_metrics['loss'], epoch)
                writer.add_scalar('Val/dice_loss', val_metrics['dice_loss'], epoch)
                writer.add_scalar('Val/focal_loss', val_metrics['focal_loss'], epoch)
                writer.add_scalar('Val/epoch_acc', val_metrics['accuracy'], epoch)

                # 每类验证指标
                for k, v in val_metrics.items():
                    if k.startswith('f1_') or k.startswith('precision_') or k.startswith('recall_'):
                        if not k.endswith('_original'):
                            writer.add_scalar(f'Val/{k}', v, epoch)

                # 分割指标
                if 'dice_c0' in val_metrics:
                    writer.add_scalar('Val/dice_c0', val_metrics['dice_c0'], epoch)
                    writer.add_scalar('Val/iou_c0', val_metrics['iou_c0'], epoch)

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
