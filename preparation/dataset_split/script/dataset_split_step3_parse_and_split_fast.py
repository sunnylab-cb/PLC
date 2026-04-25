#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3：解析MMseqs2聚类结果并划分数据集（优化版）

优化：
1. 预先构建文件路径索引（避免重复rglob）
2. 使用多线程加速
3. 添加进度条
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import json
from tqdm import tqdm


def parse_mmseqs_clusters(cluster_tsv: Path) -> Dict[str, List[str]]:
    """解析MMseqs2聚类结果"""
    clusters = defaultdict(list)

    print(f"\n解析聚类结果: {cluster_tsv}")
    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cluster_rep = parts[0]
            member = parts[1]
            clusters[cluster_rep].append(member)

    return dict(clusters)


def greedy_split_clusters(
    clusters: Dict[str, List[str]],
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, int]]:
    """使用贪心算法划分簇

    Returns:
        splits: {'train': [pdb_ids], 'val': [pdb_ids], 'test': [pdb_ids]}
        cluster_map: {pdb_id: cluster_representative_id}
        cluster_sizes: {cluster_representative_id: cluster_size}
    """
    random.seed(seed)

    total_samples = sum(len(members) for members in clusters.values())
    print(f"\n总样本数: {total_samples}")
    print(f"总簇数: {len(clusters)}")
    print(f"目标比例: Train={target_ratios[0]:.1%}, Val={target_ratios[1]:.1%}, Test={target_ratios[2]:.1%}")

    target_train = int(total_samples * target_ratios[0])
    target_val = int(total_samples * target_ratios[1])
    target_test = total_samples - target_train - target_val

    print(f"目标样本数: Train={target_train}, Val={target_val}, Test={target_test}")

    # 按簇大小排序
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

    # 构建簇映射表 {pdb_id: cluster_representative}
    cluster_map = {}

    # 构建簇大小表 {cluster_representative: cluster_size}
    cluster_sizes = {cluster_rep: len(members) for cluster_rep, members in clusters.items()}

    # 贪心分配
    print(f"\n贪心分配簇...")
    for cluster_rep, members in tqdm(sorted_clusters, desc="分配簇"):
        cluster_size = len(members)

        # 计算缺口
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

        # 记录簇映射关系
        for member in members:
            cluster_map[member] = cluster_rep

    # 打印最终统计
    print(f"\n最终分配结果:")
    for split_name in ['train', 'val', 'test']:
        count = current_counts[split_name]
        ratio = count / total_samples
        print(f"  {split_name.capitalize()}: {count} 样本 ({ratio:.2%})")

    return splits, cluster_map, cluster_sizes


def build_file_index(data_dir: Path) -> Dict[str, str]:
    """
    预先构建文件路径索引（一次性rglob，避免重复查找）

    Returns:
        {pdb_id: relative_path}
    """
    print(f"\n构建文件路径索引...")
    file_index = {}

    pdb_files = list(data_dir.rglob("*_protein.pdb"))
    print(f"  找到 {len(pdb_files)} 个PDB文件")

    for pdb_file in tqdm(pdb_files, desc="索引文件"):
        pdb_id = pdb_file.stem.replace('_protein', '')
        rel_path = pdb_file.relative_to(data_dir)
        file_index[pdb_id] = str(rel_path)

    print(f"  ✅ 索引完成: {len(file_index)} 个文件")
    return file_index


def save_splits(
    splits: Dict[str, List[str]],
    output_dir: Path,
    file_index: Dict[str, str],
    cluster_map: Dict[str, str],
    cluster_sizes: Dict[str, int]
):
    """
    保存数据集划分结果（使用预构建的索引）

    Args:
        splits: {'train': [pdb_ids], 'val': [pdb_ids], 'test': [pdb_ids]}
        output_dir: 输出目录
        file_index: {pdb_id: relative_path} 预构建的文件索引
        cluster_map: {pdb_id: cluster_representative_id} 簇映射表
        cluster_sizes: {cluster_representative_id: cluster_size} 簇大小表
    """
    print(f"\n保存数据集划分...")

    total_samples = sum(len(ids) for ids in splits.values())

    # 1. 保存统一格式的splits文件（类似filtered_splits.txt，向下兼容）
    splits_file = output_dir / "mmseqs_splits_strict.txt"

    print(f"  写入 mmseqs_splits_strict.txt (两列格式，向下兼容)...")
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

    print(f"  ✅ mmseqs_splits_strict.txt: {total_samples} 个样本 (两列格式)")

    # 2. 保存CSV格式文件（四列：set, id, cluster_representative, cluster_size）
    csv_file = output_dir / "mmseqs_splits_strict.csv"

    print(f"  写入 mmseqs_splits_strict.csv (四列格式，包含簇代表和簇大小)...")
    with open(csv_file, 'w') as f:
        # 写入头部注释
        f.write("# Train/Val/Test splits (MMseqs2 30% sequence identity clustering)\n")
        f.write(f"# Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\n")
        f.write(f"# Total: {total_samples}\n")
        f.write(f"# Clustering: --min-seq-id 0.3 -c 0.8\n")
        f.write("\n")

        # 写入TRAIN
        f.write("# TRAIN\n")
        for pdb_id in sorted(splits['train']):
            cluster_rep = cluster_map.get(pdb_id, pdb_id)
            cluster_size = cluster_sizes.get(cluster_rep, 1)
            f.write(f"train\t{pdb_id}\t{cluster_rep}\t{cluster_size}\n")

        # 写入VAL
        f.write("\n# VAL\n")
        for pdb_id in sorted(splits['val']):
            cluster_rep = cluster_map.get(pdb_id, pdb_id)
            cluster_size = cluster_sizes.get(cluster_rep, 1)
            f.write(f"val\t{pdb_id}\t{cluster_rep}\t{cluster_size}\n")

        # 写入TEST
        f.write("\n# TEST\n")
        for pdb_id in sorted(splits['test']):
            cluster_rep = cluster_map.get(pdb_id, pdb_id)
            cluster_size = cluster_sizes.get(cluster_rep, 1)
            f.write(f"test\t{pdb_id}\t{cluster_rep}\t{cluster_size}\n")

    print(f"  ✅ mmseqs_splits_strict.csv: {total_samples} 个样本 (四列格式)")


    # 2. 保存单独的ID列表（向下兼容）
    for split_name, pdb_ids in splits.items():
        output_file = output_dir / f"{split_name}_ids.txt"
        with open(output_file, 'w') as f:
            for pdb_id in sorted(pdb_ids):
                f.write(f"{pdb_id}\n")
        print(f"  ✅ {split_name}_ids.txt: {len(pdb_ids)} 个样本")

    # 3. 保存JSON格式（使用预构建的索引，快速查找）
    print(f"  生成 dataset_splits.json...")
    splits_with_paths = {}

    for split_name, pdb_ids in splits.items():
        splits_with_paths[split_name] = []
        for pdb_id in tqdm(pdb_ids, desc=f"处理{split_name}", leave=False):
            if pdb_id in file_index:
                protein_path = file_index[pdb_id]
                ligand_path = protein_path.replace('_protein.pdb', '_ligand.sdf')
                cluster_rep = cluster_map.get(pdb_id, pdb_id)
                cluster_size = cluster_sizes.get(cluster_rep, 1)
                splits_with_paths[split_name].append({
                    'pdb_id': pdb_id,
                    'protein_pdb': protein_path,
                    'ligand_sdf': ligand_path,
                    'cluster_representative': cluster_rep,
                    'cluster_size': cluster_size
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
    print(f"  - mmseqs_splits_strict.txt: 两列格式（向下兼容）")
    print(f"  - mmseqs_splits_strict.csv: 四列格式（包含簇代表ID和簇大小）")
    print(f"  - train/val/test_ids.txt: 单独的ID列表")
    print(f"  - dataset_splits.json: JSON格式（包含文件路径、簇代表和簇大小）")


def main():
    """主函数"""
    # 配置路径
    data_dir = Path("/home/husrcf/Code/ProtBind/New")
    output_dir = Path("/home/husrcf/Code/ProtBind/dataset_split")
    cluster_tsv = output_dir / "cluster_res_cluster.tsv"

    print("="*70)
    print("步骤3：解析聚类结果并划分数据集（优化版）")
    print("="*70)

    # 1. 解析聚类结果
    clusters = parse_mmseqs_clusters(cluster_tsv)

    # 2. 贪心划分
    splits, cluster_map, cluster_sizes = greedy_split_clusters(clusters, target_ratios=(0.8, 0.1, 0.1))

    # 3. 预先构建文件索引（关键优化！）
    file_index = build_file_index(data_dir)

    # 4. 保存划分结果（使用索引，快速查找）
    save_splits(splits, output_dir, file_index, cluster_map, cluster_sizes)

    print("\n" + "="*70)
    print("✅ 数据集划分完成！")
    print("="*70)


if __name__ == "__main__":
    main()
