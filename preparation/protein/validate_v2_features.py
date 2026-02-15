#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证V2蛋白质特征质量

检查项：
1. 特征维度正确性
2. 数值范围合理性
3. 对齐信息完整性
4. 与ESM3特征对齐（如果可用）
"""

import torch
import numpy as np
from pathlib import Path
import sys

def validate_v2_features(feature_file: Path, esm3_file: Path = None):
    """验证V2特征文件"""

    print("="*70)
    print(f"验证文件: {feature_file.name}")
    print("="*70)

    # 加载V2特征
    data = torch.load(feature_file, weights_only=False)

    # 1. 检查必需字段
    required_fields = [
        'complete_features_v2',
        'physicochemical_features',
        'surface_features',
        'structural_geometry_features',
        'pocketness_features',
        'flexibility_features',
        'environmental_features',
        'residue_index',
        'chain_id',
        'sequence',
        'ca_coords'
    ]

    print("\n1. 字段完整性检查:")
    missing_fields = []
    for field in required_fields:
        if field in data:
            print(f"  ✅ {field}")
        else:
            print(f"  ❌ {field} (缺失)")
            missing_fields.append(field)

    if missing_fields:
        print(f"\n⚠️  缺失字段: {missing_fields}")
        return False

    # 2. 检查特征维度
    print("\n2. 特征维度检查:")
    expected_dims = {
        'complete_features_v2': 94,
        'physicochemical_features': 42,
        'surface_features': 14,
        'structural_geometry_features': 26,
        'pocketness_features': 3,
        'flexibility_features': 3,
        'environmental_features': 6
    }

    all_dims_correct = True
    for key, expected_dim in expected_dims.items():
        actual_dim = data[key].shape[1]
        status = "✅" if actual_dim == expected_dim else "❌"
        print(f"  {status} {key}: {actual_dim} (期望 {expected_dim})")
        if actual_dim != expected_dim:
            all_dims_correct = False

    # 3. 检查数值范围
    print("\n3. 数值范围检查:")

    # 检查是否有NaN或Inf
    complete_feats = data['complete_features_v2'].numpy()
    has_nan = np.isnan(complete_feats).any()
    has_inf = np.isinf(complete_feats).any()

    print(f"  NaN值: {'❌ 存在' if has_nan else '✅ 无'}")
    print(f"  Inf值: {'❌ 存在' if has_inf else '✅ 无'}")

    # 显示统计信息
    print(f"\n  complete_features_v2统计:")
    print(f"    最小值: {complete_feats.min():.4f}")
    print(f"    最大值: {complete_feats.max():.4f}")
    print(f"    平均值: {complete_feats.mean():.4f}")
    print(f"    标准差: {complete_feats.std():.4f}")

    # 检查新增特征的范围
    print(f"\n  新增特征范围:")

    # Pocketness features (应该在0-1范围)
    pocket_feats = data['pocketness_features'].numpy()
    print(f"    Pocketness: [{pocket_feats.min():.4f}, {pocket_feats.max():.4f}]")

    # Flexibility features (应该在0-1范围)
    flex_feats = data['flexibility_features'].numpy()
    print(f"    Flexibility: [{flex_feats.min():.4f}, {flex_feats.max():.4f}]")

    # Environmental features
    env_feats = data['environmental_features'].numpy()
    print(f"    Environmental: [{env_feats.min():.4f}, {env_feats.max():.4f}]")

    # 4. 检查对齐信息
    print("\n4. 对齐信息检查:")
    n_residues = data['n_residues']
    seq_len = len(data['sequence'])
    res_idx_len = len(data['residue_index'])
    chain_id_len = len(data['chain_id'])

    print(f"  残基数量: {n_residues}")
    print(f"  序列长度: {seq_len}")
    print(f"  residue_index长度: {res_idx_len}")
    print(f"  chain_id长度: {chain_id_len}")

    alignment_ok = (n_residues == seq_len == res_idx_len == chain_id_len)
    print(f"  对齐一致性: {'✅ 一致' if alignment_ok else '❌ 不一致'}")

    # 显示前5个残基
    print(f"\n  前5个残基:")
    print(f"  {'索引':<8} {'链':<6} {'残基':<8} {'单字母':<8}")
    print("  " + "-"*40)
    for i in range(min(5, n_residues)):
        print(f"  {data['residue_index'][i]:<8} {data['chain_id'][i]:<6} "
              f"{data['residue_name_3'][i]:<8} {data['residue_name_1'][i]:<8}")

    # 5. 与ESM3对齐检查（如果提供）
    if esm3_file and esm3_file.exists():
        print("\n5. ESM3对齐检查:")
        try:
            esm3_data = torch.load(esm3_file, weights_only=False)

            esm3_n_res = len(esm3_data.get('residue_index', []))
            esm3_seq = esm3_data.get('sequence', '')

            print(f"  ESM3残基数: {esm3_n_res}")
            print(f"  V2残基数: {n_residues}")
            print(f"  数量匹配: {'✅' if esm3_n_res == n_residues else '❌'}")

            if esm3_seq:
                seq_match = (esm3_seq == data['sequence'])
                print(f"  序列匹配: {'✅' if seq_match else '❌'}")

                if not seq_match:
                    print(f"    ESM3序列: {esm3_seq[:50]}...")
                    print(f"    V2序列:   {data['sequence'][:50]}...")

        except Exception as e:
            print(f"  ⚠️  无法加载ESM3文件: {e}")

    # 总结
    print("\n" + "="*70)
    if all_dims_correct and not has_nan and not has_inf and alignment_ok:
        print("✅ 验证通过！所有检查项正常")
        return True
    else:
        print("⚠️  验证发现问题，请检查上述错误")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 validate_v2_features.py <v2_feature_file> [esm3_feature_file]")
        sys.exit(1)

    v2_file = Path(sys.argv[1])
    esm3_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not v2_file.exists():
        print(f"❌ 文件不存在: {v2_file}")
        sys.exit(1)

    success = validate_v2_features(v2_file, esm3_file)
    sys.exit(0 if success else 1)
