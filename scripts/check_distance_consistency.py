#!/usr/bin/env python3
"""
检查VDW距离文件和原始距离文件的残基数一致性

分析不匹配的样例，找出缺失的字段和数据差异
"""

import torch
from pathlib import Path
from collections import defaultdict
import sys

def load_distance_file(path):
    """加载距离文件"""
    print(f"加载文件: {path}")
    data = torch.load(path, weights_only=False)
    print(f"  包含 {len(data)} 个复合物")
    return data

def analyze_sample(pdb_id, vdw_data, orig_data):
    """详细分析单个样本"""
    print(f"\n{'='*80}")
    print(f"详细分析: {pdb_id}")
    print(f"{'='*80}")

    # VDW数据
    print(f"\n【VDW距离数据】")
    if isinstance(vdw_data, dict):
        print(f"  类型: dict")
        print(f"  键: {list(vdw_data.keys())}")
        for key, value in vdw_data.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: {value}")
    else:
        print(f"  类型: {type(vdw_data)}")
        if isinstance(vdw_data, torch.Tensor):
            print(f"  shape: {vdw_data.shape}")

    # 原始数据
    print(f"\n【原始距离数据】")
    if isinstance(orig_data, dict):
        print(f"  类型: dict")
        print(f"  键: {list(orig_data.keys())}")
        for key, value in orig_data.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: {value}")
    else:
        print(f"  类型: {type(orig_data)}")
        if isinstance(orig_data, torch.Tensor):
            print(f"  shape: {orig_data.shape}")

    # 比较
    print(f"\n【差异分析】")
    if isinstance(vdw_data, dict) and isinstance(orig_data, dict):
        vdw_keys = set(vdw_data.keys())
        orig_keys = set(orig_data.keys())

        common_keys = vdw_keys & orig_keys
        vdw_only = vdw_keys - orig_keys
        orig_only = orig_keys - vdw_keys

        print(f"  共同键: {sorted(common_keys)}")
        if vdw_only:
            print(f"  仅VDW有: {sorted(vdw_only)}")
        if orig_only:
            print(f"  仅原始有: {sorted(orig_only)}")

        # 比较distances字段
        if 'distances' in common_keys:
            vdw_dist = vdw_data['distances']
            orig_dist = orig_data['distances']
            print(f"\n  distances字段比较:")
            print(f"    VDW: {vdw_dist.shape}")
            print(f"    原始: {orig_dist.shape}")
            if vdw_dist.shape != orig_dist.shape:
                print(f"    ⚠️ 形状不匹配！")
                print(f"    差异: {vdw_dist.shape[0] - orig_dist.shape[0]} 个残基")

        # 比较distances_vdw_adjusted字段（如果存在）
        if 'distances_vdw_adjusted' in vdw_data:
            vdw_adj = vdw_data['distances_vdw_adjusted']
            print(f"\n  VDW特有字段:")
            print(f"    distances_vdw_adjusted: {vdw_adj.shape}")
            print(f"    min: {vdw_adj.min().item():.2f}Å")
            print(f"    max: {vdw_adj.max().item():.2f}Å")
            print(f"    mean: {vdw_adj.mean().item():.2f}Å")

def main():
    # 文件路径
    vdw_file = Path("data/residue_ligand_distances_vdw.pt")
    orig_file = Path("data/residue_ligand_distances_poc_style.pt")

    if not vdw_file.exists():
        print(f"错误: VDW距离文件不存在: {vdw_file}")
        sys.exit(1)

    if not orig_file.exists():
        print(f"错误: 原始距离文件不存在: {orig_file}")
        sys.exit(1)

    # 加载数据
    print("="*80)
    print("加载距离文件")
    print("="*80)
    vdw_distances = load_distance_file(vdw_file)
    orig_distances = load_distance_file(orig_file)

    # 找出共同的PDB ID
    vdw_ids = set(vdw_distances.keys())
    orig_ids = set(orig_distances.keys())

    common_ids = vdw_ids & orig_ids
    vdw_only_ids = vdw_ids - orig_ids
    orig_only_ids = orig_ids - vdw_ids

    print(f"\n{'='*80}")
    print("数据集概览")
    print(f"{'='*80}")
    print(f"VDW距离文件: {len(vdw_ids)} 个复合物")
    print(f"原始距离文件: {len(orig_ids)} 个复合物")
    print(f"共同复合物: {len(common_ids)} 个")
    if vdw_only_ids:
        print(f"仅VDW有: {len(vdw_only_ids)} 个")
        print(f"  示例: {list(vdw_only_ids)[:5]}")
    if orig_only_ids:
        print(f"仅原始有: {len(orig_only_ids)} 个")
        print(f"  示例: {list(orig_only_ids)[:5]}")

    # 检查残基数一致性
    print(f"\n{'='*80}")
    print("检查残基数一致性")
    print(f"{'='*80}")

    mismatches = []
    length_diff_stats = defaultdict(int)

    for pdb_id in sorted(common_ids):
        vdw_data = vdw_distances[pdb_id]
        orig_data = orig_distances[pdb_id]

        # 获取distances字段的长度
        if isinstance(vdw_data, dict) and 'distances' in vdw_data:
            vdw_len = len(vdw_data['distances'])
        elif isinstance(vdw_data, torch.Tensor):
            vdw_len = len(vdw_data)
        else:
            vdw_len = None

        if isinstance(orig_data, dict) and 'distances' in orig_data:
            orig_len = len(orig_data['distances'])
        elif isinstance(orig_data, torch.Tensor):
            orig_len = len(orig_data)
        else:
            orig_len = None

        if vdw_len is not None and orig_len is not None:
            if vdw_len != orig_len:
                diff = vdw_len - orig_len
                mismatches.append({
                    'pdb_id': pdb_id,
                    'vdw_len': vdw_len,
                    'orig_len': orig_len,
                    'diff': diff
                })
                length_diff_stats[diff] += 1

    print(f"检查了 {len(common_ids)} 个共同复合物")
    print(f"发现 {len(mismatches)} 个不匹配样本 ({len(mismatches)/len(common_ids)*100:.2f}%)")

    if mismatches:
        print(f"\n长度差异分布:")
        for diff in sorted(length_diff_stats.keys()):
            count = length_diff_stats[diff]
            print(f"  差异 {diff:+4d} 残基: {count:4d} 个样本 ({count/len(mismatches)*100:.1f}%)")

        # 显示前10个不匹配样本
        print(f"\n前10个不匹配样本:")
        print(f"{'PDB ID':<10} {'VDW长度':>10} {'原始长度':>10} {'差异':>8}")
        print("-" * 42)
        for item in mismatches[:10]:
            print(f"{item['pdb_id']:<10} {item['vdw_len']:>10} {item['orig_len']:>10} {item['diff']:>+8}")

        # 详细分析几个样例
        print(f"\n{'='*80}")
        print("详细分析不匹配样例")
        print(f"{'='*80}")

        # 选择3个有代表性的样例
        samples_to_analyze = []

        # 1. VDW更长的样例
        longer_samples = [m for m in mismatches if m['diff'] > 0]
        if longer_samples:
            samples_to_analyze.append(longer_samples[0]['pdb_id'])

        # 2. 原始更长的样例
        shorter_samples = [m for m in mismatches if m['diff'] < 0]
        if shorter_samples:
            samples_to_analyze.append(shorter_samples[0]['pdb_id'])

        # 3. 差异最大的样例
        max_diff_sample = max(mismatches, key=lambda x: abs(x['diff']))
        if max_diff_sample['pdb_id'] not in samples_to_analyze:
            samples_to_analyze.append(max_diff_sample['pdb_id'])

        for pdb_id in samples_to_analyze[:3]:
            analyze_sample(pdb_id, vdw_distances[pdb_id], orig_distances[pdb_id])
    else:
        print("\n✓ 所有样本的残基数都一致！")

    # 保存不匹配列表
    if mismatches:
        output_file = Path("data/distance_mismatches.txt")
        with open(output_file, 'w') as f:
            f.write("# 距离文件残基数不匹配列表\n")
            f.write(f"# 总共 {len(mismatches)} 个不匹配样本\n")
            f.write(f"# 格式: PDB_ID VDW_LENGTH ORIG_LENGTH DIFF\n\n")
            for item in mismatches:
                f.write(f"{item['pdb_id']} {item['vdw_len']} {item['orig_len']} {item['diff']:+d}\n")
        print(f"\n不匹配列表已保存到: {output_file}")

if __name__ == "__main__":
    main()
