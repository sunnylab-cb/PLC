#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 V3 特征提取脚本

验证：
1. 特征维度正确
2. 无 NaN 值
3. 矢量归一化正确
4. 特殊残基（GLY）处理正确
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preparation.protein.extract_complete_features_v3 import (
    extract_complete_features_v3,
    safe_normalize
)


def test_safe_normalize():
    """测试安全归一化函数"""
    print("\n测试 safe_normalize()...")

    # 测试正常向量
    v1 = np.array([3.0, 4.0, 0.0])
    v1_norm = safe_normalize(v1)
    assert np.allclose(np.linalg.norm(v1_norm), 1.0), "归一化失败"
    print(f"  ✓ 正常向量: {v1} -> {v1_norm}")

    # 测试零向量（避坑指南 #1）
    v2 = np.array([0.0, 0.0, 0.0])
    v2_norm = safe_normalize(v2)
    assert np.allclose(v2_norm, 0.0), "零向量处理失败"
    print(f"  ✓ 零向量: {v2} -> {v2_norm}")

    # 测试极小向量
    v3 = np.array([1e-10, 1e-10, 1e-10])
    v3_norm = safe_normalize(v3)
    assert np.allclose(v3_norm, 0.0), "极小向量处理失败"
    print(f"  ✓ 极小向量: {v3} -> {v3_norm}")

    print("  ✓ safe_normalize() 测试通过！")


def test_feature_extraction(pdb_path: Path):
    """测试特征提取"""
    print(f"\n测试特征提取: {pdb_path.name}...")

    # 提取特征
    features = extract_complete_features_v3(pdb_path, mode='complex')

    # 检查特征维度
    physchem = features['physchem_features']
    spatial_scalar = features['spatial_scalar_features']
    spatial_vector = features['spatial_vector_features']
    n_residues = features['n_residues']

    print(f"\n特征维度检查:")
    print(f"  残基数量: {n_residues}")
    print(f"  物理化学特征: {physchem.shape} (期望: [{n_residues}, 62])")
    print(f"  空间标量特征: {spatial_scalar.shape} (期望: [{n_residues}, 72])")
    print(f"  空间矢量特征: {spatial_vector.shape} (期望: [{n_residues}, 6, 3])")

    assert physchem.shape == (n_residues, 62), f"物理化学特征维度错误"
    assert spatial_scalar.shape == (n_residues, 72), f"空间标量特征维度错误"
    assert spatial_vector.shape == (n_residues, 6, 3), f"空间矢量特征维度错误"
    print("  ✓ 特征维度正确！")

    # 检查 NaN（避坑指南 #1）
    print(f"\nNaN 检查:")
    physchem_nan = torch.isnan(physchem).sum().item()
    scalar_nan = torch.isnan(spatial_scalar).sum().item()
    vector_nan = torch.isnan(spatial_vector).sum().item()

    print(f"  物理化学特征 NaN 数量: {physchem_nan}")
    print(f"  空间标量特征 NaN 数量: {scalar_nan}")
    print(f"  空间矢量特征 NaN 数量: {vector_nan}")

    assert physchem_nan == 0, "物理化学特征包含 NaN！"
    assert scalar_nan == 0, "空间标量特征包含 NaN！"
    assert vector_nan == 0, "空间矢量特征包含 NaN！"
    print("  ✓ 无 NaN 值！")

    # 检查矢量归一化
    print(f"\n矢量归一化检查:")
    vector_norms = torch.norm(spatial_vector, dim=-1)  # [N, 6]

    for i in range(6):
        vector_name = [
            "Sidechain Base",
            "Sidechain CoM",
            "Aromatic Normal",
            "H-bond Donor",
            "H-bond Acceptor",
            "Cavity Pointing"
        ][i]

        norms = vector_norms[:, i]
        non_zero_mask = norms > 1e-6
        non_zero_norms = norms[non_zero_mask]

        if len(non_zero_norms) > 0:
            mean_norm = non_zero_norms.mean().item()
            max_norm = non_zero_norms.max().item()
            min_norm = non_zero_norms.min().item()

            print(f"  {vector_name}: mean={mean_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}")

            # 非零向量应该接近归一化（模长 ≈ 1.0）
            assert 0.95 <= mean_norm <= 1.05, f"{vector_name} 归一化失败"
        else:
            print(f"  {vector_name}: 全部为零向量")

    print("  ✓ 矢量归一化正确！")

    # 检查特殊残基（GLY）
    print(f"\n特殊残基检查:")
    residue_names = features['residue_names']
    gly_indices = [i for i, name in enumerate(residue_names) if name == 'GLY']

    if len(gly_indices) > 0:
        print(f"  找到 {len(gly_indices)} 个 GLY 残基")
        for idx in gly_indices[:3]:  # 只显示前 3 个
            v_base = spatial_vector[idx, 0].numpy()
            v_com = spatial_vector[idx, 1].numpy()
            print(f"    GLY {idx}: v_base={v_base}, v_com={v_com}")

            # GLY 的 v_base 和 v_com 应该相同（都使用虚拟 Cβ）
            assert np.allclose(v_base, v_com, atol=1e-5), f"GLY {idx} 的 v_base 和 v_com 不一致"

        print("  ✓ GLY 残基处理正确！")
    else:
        print("  未找到 GLY 残基")

    # 统计信息
    print(f"\n特征统计:")
    print(f"  物理化学特征范围: [{physchem.min():.4f}, {physchem.max():.4f}]")
    print(f"  空间标量特征范围: [{spatial_scalar.min():.4f}, {spatial_scalar.max():.4f}]")
    print(f"  空间矢量特征范围: [{spatial_vector.min():.4f}, {spatial_vector.max():.4f}]")

    print(f"\n✓ 所有测试通过！")


def main():
    # 测试安全归一化函数
    test_safe_normalize()

    # 测试特征提取
    # 需要提供一个测试 PDB 文件
    test_pdb = Path("/home/husrcf/Code/ProtBind/New/1981-2000/1a0q/1a0q_protein.pdb")

    if test_pdb.exists():
        test_feature_extraction(test_pdb)
    else:
        print(f"\n警告: 测试 PDB 文件不存在: {test_pdb}")
        print("请提供一个有效的 PDB 文件路径进行测试")


if __name__ == '__main__':
    main()
