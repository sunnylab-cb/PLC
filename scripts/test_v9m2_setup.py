#!/usr/bin/env python3
"""
测试V9m2模型设置

验证：
1. Flash Attention可用性
2. V9m2模型可以正常实例化
3. Boundary Loss可以正常计算
4. 配置文件正确加载
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

# 测试Flash Attention
print("=" * 60)
print("测试 Flash Attention")
print("=" * 60)
try:
    from flash_attn import flash_attn_func
    print("✓ Flash Attention 已安装")
    import flash_attn
    print(f"  版本: {flash_attn.__version__}")
except ImportError as e:
    print(f"✗ Flash Attention 未安装: {e}")

# 测试V9m2模型
print("\n" + "=" * 60)
print("测试 V9m2 模型")
print("=" * 60)
try:
    from src.models.pocket_gvp_fusion_v9m2 import PocketGVPFusionV9m2

    config = {
        'd_model': 512,
        'n_heads': 8,
        'use_flash_attn': True,
        'stochastic_depth_prob': 0.1,
        'attn_dropout': 0.1,
        'classifier_dropout': 0.2
    }

    model = PocketGVPFusionV9m2(config)
    print("✓ V9m2 模型实例化成功")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

except Exception as e:
    print(f"✗ V9m2 模型实例化失败: {e}")
    import traceback
    traceback.print_exc()

# 测试Boundary Loss
print("\n" + "=" * 60)
print("测试 Boundary Loss")
print("=" * 60)
try:
    from src.losses.boundary_loss import compute_boundary_loss, compute_boundary_mask

    # 创建测试数据
    B, N, C = 2, 100, 3
    logits = torch.randn(B, N, C)
    labels = torch.randint(0, C, (B, N))
    coords = torch.randn(B, N, 3) * 10  # 模拟Cα坐标
    mask = torch.ones(B, N, dtype=torch.bool)

    # 计算boundary mask
    boundary_mask = compute_boundary_mask(labels, coords, mask, distance_threshold=8.0)
    print(f"✓ Boundary mask 计算成功")
    print(f"  边界残基比例: {boundary_mask.sum().item() / mask.sum().item():.2%}")

    # 计算boundary loss
    loss, boundary_loss, non_boundary_loss, boundary_ratio = compute_boundary_loss(
        logits, labels, coords, mask,
        boundary_weight=2.0,
        distance_threshold=8.0,
        label_smoothing=0.15
    )
    print(f"✓ Boundary loss 计算成功")
    print(f"  总损失: {loss.item():.4f}")
    print(f"  边界损失: {boundary_loss.item():.4f}")
    print(f"  非边界损失: {non_boundary_loss.item():.4f}")
    print(f"  边界比例: {boundary_ratio:.2%}")

except Exception as e:
    print(f"✗ Boundary loss 计算失败: {e}")
    import traceback
    traceback.print_exc()

# 测试配置文件
print("\n" + "=" * 60)
print("测试 V9m2 配置文件")
print("=" * 60)
try:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pocket_gvp_v9m2.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("✓ 配置文件加载成功")
    print(f"  实验名称: {config['experiment']['name']}")
    print(f"  Flash Attention: {config['model'].get('use_flash_attn', False)}")
    print(f"  Label Smoothing: {config['training'].get('label_smoothing', 0.1)}")
    print(f"  Boundary Loss: {config['training'].get('use_boundary_loss', False)}")
    print(f"  Boundary Weight: {config['training'].get('boundary_weight', 0.0)}")

except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试Transformer Layers
print("\n" + "=" * 60)
print("测试 Transformer Layers (Flash Attention)")
print("=" * 60)
try:
    from src.models.transformer_layers import GatedMultiHeadAttention, FLASH_ATTN_AVAILABLE

    print(f"  Flash Attention 可用: {FLASH_ATTN_AVAILABLE}")

    # 测试标准attention
    attn_std = GatedMultiHeadAttention(
        d_model=512,
        n_heads=8,
        dropout=0.1,
        use_flash_attn=False
    )
    print("✓ 标准 Attention 初始化成功")

    # 测试flash attention
    if FLASH_ATTN_AVAILABLE:
        attn_flash = GatedMultiHeadAttention(
            d_model=512,
            n_heads=8,
            dropout=0.1,
            use_flash_attn=True
        )
        print("✓ Flash Attention 初始化成功")

        # 前向传播测试
        x = torch.randn(2, 100, 512)
        mask = torch.ones(2, 100, dtype=torch.bool)

        with torch.no_grad():
            out_std = attn_std(x, mask=mask)
            out_flash = attn_flash(x, mask=mask)

        print(f"  标准输出形状: {out_std.shape}")
        print(f"  Flash输出形状: {out_flash.shape}")
    else:
        print("  Flash Attention 不可用，跳过测试")

except Exception as e:
    print(f"✗ Transformer layers 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
