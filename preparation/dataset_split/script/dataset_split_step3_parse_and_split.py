#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3：解析MMseqs2聚类结果并划分数据集

使用贪心算法确保：
1. 同一簇的所有样本分配到同一集合
2. 最终样本比例接近 8:1:1 (训练:验证:测试)
3. 严格的序列去冗余（30%一致性）
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import json


def parse_mmseqs_clusters(cluster_tsv: Path) -> Dict[str, List[str]]:
    """
    解析MMseqs2聚类结果

    Args:
        cluster_tsv: cluster_res_cluster.tsv文件路径

    Returns:
        {cluster_rep: [member1, member2, ...]}
    """
    clusters = defaultdict(list)

    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            cluster_rep = parts[0]  # 簇代表序列
            member = parts[1]       # 簇成员

            clusters[cluster_rep].append(member)

    return dict(clusters)


def greedy_split_clusters(
    clusters: Dict[str, List[str]],
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    使用贪心算法划分簇到训练/验证/测试集

    策略：
    1. 按簇大小（样本数）从大到小排序
    2. 依次将每个簇分配给当前样本量缺口最大的集合
    3. 确保最终比例接近目标比例

    Args:
        clusters: {cluster_rep: [members]}
        target_ratios: (train, val, test) 目标比例
        seed: 随机种子

    Returns:
        {
            'train': [pdb_ids],
            'val': [pdb_ids],
            'test': [pdb_ids]
        }
    """
    random.seed(seed)

    # 计算总样本数
    total_samples = sum(len(members) for members in clusters.values())
    print(f"\n总样本数: {total_samples}")
    print(f"总簇数: {len(clusters)}")
    print(f"目标比例: Train={target_ratios[0]:.1%}, Val={target_ratios[1]:.1%}, Test={target_ratios[2]:.1%}")

    # 目标样本数
    target_train = int(total_samples * target_ratios[0])
    target_val = int(total_samples * target_ratios[1])
    target_test = total_samples - target_train - target_val

    print(f"目标样本数: Train={target_train}, Val={target_val}, Test={target_test}")

    # 按簇大小排序（从大到小）
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print(f"\n簇大小统计:")
    cluster_sizes = [len(members) for _, members in sorted_clusters]
    print(f"  最大簇: {max(cluster_sizes)} 个样本")
    print(f"  最小簇: {min(cluster_sizes)} 个样本")
    print(f"  平均簇大小: {sum(cluster_sizes)/len(cluster_sizes):.1f} 个样本")
    print(f"  中位数簇大小: {sorted(cluster_sizes)[len(cluster_sizes)//2]} 个样本")

    # 初始化集合
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    current_counts = {
        'train': 0,
        'val': 0,
        'test': 0
    }

    # 贪心分配
    print(f"\n开始贪心分配...")
    for cluster_rep, members in sorted_clusters:
        cluster_size = len(members)

        # 计算每个集合的缺口（目标 - 当前）
        gaps = {
            'train': target_train - current_counts['train'],
            'val': target_val - current_counts['val'],
            'test': target_test - current_counts['test']
        }

        # 选择缺口最大的集合
        assigned_split = max(gaps.items(), key=lambda x: x[1])[0]

        # 分配整个簇
        splits[assigned_split].extend(members)
        current_counts[assigned_split] += cluster_size

    # 打印最终统计
    print(f"\n最终分配结果:")
    for split_name in ['train', 'val', 'test']:
        count = current_counts[split_name]
        ratio = count / total_samples
        print(f"  {split_name.capitalize()}: {count} 样本 ({ratio:.2%})")

    # 计算与目标的偏差
    print(f"\n与目标的偏差:")
    deviations = {
        'train': abs(current_counts['train'] - target_train),
        'val': abs(current_counts['val'] - target_val),
        'test': abs(current_counts['test'] - target_test)
    }
    for split_name, dev in deviations.items():
        dev_pct = dev / total_samples * 100
        print(f"  {split_name.capitalize()}: ±{dev} 样本 (±{dev_pct:.2f}%)")

    return splits


def save_splits(
    splits: Dict[str, List[str]],
    output_dir: Path,
    data_dir: Path
):
    """
    保存数据集划分结果

    Args:
        splits: {'train': [pdb_ids], 'val': [pdb_ids], 'test': [pdb_ids]}
        output_dir: 输出目录
        data_dir: 原始数据目录
    """
    print(f"\n保存数据集划分...")

    # 1. 保存统一格式的splits文件（类似filtered_splits.txt）
    splits_file = output_dir / "mmseqs_splits.txt"
    total_samples = sum(len(ids) for ids in splits.values())

    with open(splits_file, 'w') as f:
        # 写入头部注释
        f.write("# Train/Val/Test splits (MMseqs2 30% sequence identity clustering)\n")
        f.write(f"# Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\n")
        f.write(f"# Total: {total_samples}\n")
        f.write(f"# Clustering: --min-seq-id 0.3 -c 0.8\n")
        f.write("\n")

        # 写入TRAIN
        f.write("# TRAIN\n")
        for pdb_id in sorted(splits['train']):
            f.write(f"train\t{pdb_id}\n")

        # 写入VAL
        f.write("\n# VAL\n")
        for pdb_id in sorted(splits['val']):
            f.write(f"val\t{pdb_id}\n")

        # 写入TEST
        f.write("\n# TEST\n")
        for pdb_id in sorted(splits['test']):
            f.write(f"test\t{pdb_id}\n")

    print(f"  ✅ mmseqs_splits.txt: {total_samples} 个样本")

    # 2. 保存单独的ID列表（兼容性）
    for split_name, pdb_ids in splits.items():
        output_file = output_dir / f"{split_name}_ids.txt"
        with open(output_file, 'w') as f:
            for pdb_id in sorted(pdb_ids):
                f.write(f"{pdb_id}\n")
        print(f"  ✅ {split_name}_ids.txt: {len(pdb_ids)} 个样本")

    # 3. 保存JSON格式（包含文件路径）
    splits_with_paths = {}
    for split_name, pdb_ids in splits.items():
        splits_with_paths[split_name] = []
        for pdb_id in pdb_ids:
            # 查找对应的文件路径
            pdb_files = list(data_dir.rglob(f"{pdb_id}_protein.pdb"))
            if pdb_files:
                rel_path = pdb_files[0].relative_to(data_dir)
                splits_with_paths[split_name].append({
                    'pdb_id': pdb_id,
                    'protein_pdb': str(rel_path),
                    'ligand_sdf': str(rel_path).replace('_protein.pdb', '_ligand.sdf')
                })

    json_file = output_dir / "dataset_splits.json"
    with open(json_file, 'w') as f:
        json.dump(splits_with_paths, f, indent=2)
    print(f"  ✅ dataset_splits.json")

    # 4. 保存统计信息
    stats = {
        'total_samples': total_samples,
        'train': {
            'count': len(splits['train']),
            'ratio': len(splits['train']) / total_samples
        },
        'val': {
            'count': len(splits['val']),
            'ratio': len(splits['val']) / total_samples
        },
        'test': {
            'count': len(splits['test']),
            'ratio': len(splits['test']) / total_samples
        }
    }

    stats_file = output_dir / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ split_statistics.json")

    print(f"\n所有文件已保存到: {output_dir}")
    print(f"\n主要输出文件:")
    print(f"  - mmseqs_splits.txt: 统一格式的划分文件（推荐使用）")
    print(f"  - train/val/test_ids.txt: 单独的ID列表")
    print(f"  - dataset_splits.json: JSON格式（包含文件路径）")


def main():
    """主函数"""
    print("="*70)
    print("步骤3：解析MMseqs2聚类结果并划分数据集")
    print("="*70)

    # 配置路径
    data_dir = Path("/home/husrcf/Code/ProtBind/New")
    output_dir = Path("/home/husrcf/Code/ProtBind/dataset_split")

    cluster_tsv = output_dir / "cluster_res_cluster.tsv"

    if not cluster_tsv.exists():
        print(f"\n❌ 聚类结果文件不存在: {cluster_tsv}")
        print(f"\n请先运行MMseqs2聚类:")
        print(f"  cd {output_dir}")
        print(f"  mmseqs easy-cluster dataset_proteins.fasta cluster_res tmp_dir \\")
        print(f"      --min-seq-id 0.3 -c 0.8 --cov-mode 1 --threads 16")
        return

    # 1. 解析聚类结果
    print(f"\n解析聚类结果: {cluster_tsv}")
    clusters = parse_mmseqs_clusters(cluster_tsv)
    print(f"  找到 {len(clusters)} 个簇")

    # 2. 贪心划分
    splits = greedy_split_clusters(
        clusters,
        target_ratios=(0.8, 0.1, 0.1),  # 8:1:1
        seed=42
    )

    # 3. 保存结果
    save_splits(splits, output_dir, data_dir)

    print("\n" + "="*70)
    print("✅ 数据集划分完成！")
    print("="*70)
    print(f"\n输出文件:")
    print(f"  - train_ids.txt: 训练集PDB ID列表")
    print(f"  - val_ids.txt: 验证集PDB ID列表")
    print(f"  - test_ids.txt: 测试集PDB ID列表")
    print(f"  - dataset_splits.json: 完整划分信息（包含文件路径）")
    print(f"  - split_statistics.json: 统计信息")


if __name__ == "__main__":
    main()
