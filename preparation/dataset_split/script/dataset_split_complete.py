#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分完整流程（两阶段，支持resume）

阶段1：提取序列 -> FASTA文件
阶段2：MMseqs2聚类 + 解析 + 划分数据集

支持：
- 多线程并行
- tqdm进度条
- Resume功能（可以从任意阶段继续）
"""

from pathlib import Path
from typing import Dict, List, Tuple
import sys
import multiprocessing as mp
import subprocess
import json
from collections import defaultdict
import random

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from biotite.structure.io.pdb import PDBFile
    import biotite.structure as bs
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    print("❌ Biotite未安装，请运行: conda install -c conda-forge biotite")
    sys.exit(1)

# 氨基酸三字母到单字母映射
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# ============================================================================
# 阶段1：提取序列
# ============================================================================

def extract_sequence_from_pdb(pdb_path: Path) -> Tuple[str, str]:
    """从PDB文件提取序列"""
    pdb_id = pdb_path.stem.replace('_protein', '')
    try:
        pdb_file = PDBFile.read(str(pdb_path))
        structure = pdb_file.get_structure(model=1)
        atom_array = structure[bs.filter_amino_acids(structure) & ~structure.hetero]
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]

        sequence = []
        for residue in bs.residue_iter(ca_atoms):
            res_name = residue.res_name[0]
            aa_1 = AA_3TO1.get(res_name, 'X')
            sequence.append(aa_1)

        return (pdb_id, ''.join(sequence))
    except:
        return (pdb_id, None)


def stage1_extract_sequences(
    data_dir: Path,
    output_fasta: Path,
    n_workers: int = 8,
    force: bool = False
) -> Dict[str, str]:
    """
    阶段1：提取序列

    Returns:
        {pdb_id: sequence}
    """
    print("\n" + "="*70)
    print("阶段1：提取蛋白质序列")
    print("="*70)

    # 检查是否已存在
    if output_fasta.exists() and not force:
        print(f"\n✅ FASTA文件已存在: {output_fasta}")
        print(f"   加载现有序列...")

        # 从FASTA文件加载序列
        sequences = {}
        with open(output_fasta, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                sequences[current_id] = ''.join(current_seq)

        print(f"   已加载 {len(sequences)} 条序列")
        return sequences

    # 查找所有PDB文件
    print(f"\n查找PDB文件...")
    pdb_files = sorted(data_dir.rglob("*_protein.pdb"))
    print(f"找到 {len(pdb_files)} 个蛋白质PDB文件")

    # 并行提取序列
    print(f"\n使用 {n_workers} 个进程并行提取序列...")

    sequences = {}
    duplicates = {}
    success_count = 0
    fail_count = 0

    with mp.Pool(n_workers) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(
                pool.imap(extract_sequence_from_pdb, pdb_files),
                total=len(pdb_files),
                desc="提取序列",
                unit="文件"
            ))
        else:
            results = pool.map(extract_sequence_from_pdb, pdb_files)

    for pdb_id, sequence in results:
        if sequence:
            sequences[pdb_id] = sequence
            success_count += 1
            if sequence in duplicates:
                duplicates[sequence].append(pdb_id)
            else:
                duplicates[sequence] = [pdb_id]
        else:
            fail_count += 1

    print(f"\n提取完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  唯一序列数: {len(duplicates)}")

    # 写入FASTA文件
    print(f"\n写入FASTA文件: {output_fasta}")
    with open(output_fasta, 'w') as f:
        for pdb_id, sequence in sequences.items():
            f.write(f">{pdb_id}\n")
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')

    print(f"✅ FASTA文件已保存: {len(sequences)} 条序列")

    # 保存映射文件
    mapping_file = output_fasta.with_suffix('.mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("pdb_id\tsequence_length\tsequence\n")
        for pdb_id, sequence in sequences.items():
            f.write(f"{pdb_id}\t{len(sequence)}\t{sequence}\n")

    print(f"✅ 映射文件已保存: {mapping_file}")

    return sequences


# ============================================================================
# 阶段2：MMseqs2聚类 + 解析 + 划分
# ============================================================================

def stage2_mmseqs_clustering(
    fasta_file: Path,
    output_prefix: Path,
    tmp_dir: Path,
    n_threads: int = 16,
    force: bool = False
) -> Path:
    """
    阶段2a：MMseqs2聚类

    Returns:
        cluster_tsv文件路径
    """
    print("\n" + "="*70)
    print("阶段2a：MMseqs2聚类")
    print("="*70)

    cluster_tsv = Path(str(output_prefix) + "_cluster.tsv")

    # 检查是否已存在
    if cluster_tsv.exists() and not force:
        print(f"\n✅ 聚类结果已存在: {cluster_tsv}")
        return cluster_tsv

    # 检查MMseqs2
    try:
        result = subprocess.run(['mmseqs', 'version'], capture_output=True, text=True)
        print(f"\nMMseqs2版本: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\n❌ MMseqs2未安装")
        print("   安装: conda install -c bioconda mmseqs2")
        sys.exit(1)

    # 运行MMseqs2
    print(f"\n运行MMseqs2聚类...")
    print(f"  参数:")
    print(f"    --min-seq-id 0.3  (30%序列一致性)")
    print(f"    -c 0.8            (80%比对覆盖率)")
    print(f"    --threads {n_threads}")

    cmd = [
        'mmseqs', 'easy-cluster',
        str(fasta_file),
        str(output_prefix),
        str(tmp_dir),
        '--min-seq-id', '0.3',
        '-c', '0.8',
        '--cov-mode', '1',
        '--threads', str(n_threads),
        '--cluster-mode', '0',
        '--cluster-steps', '3'
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 聚类完成: {cluster_tsv}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ MMseqs2聚类失败: {e}")
        sys.exit(1)

    return cluster_tsv


def parse_mmseqs_clusters(cluster_tsv: Path) -> Dict[str, List[str]]:
    """解析MMseqs2聚类结果"""
    clusters = defaultdict(list)
    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cluster_rep, member = parts
                clusters[cluster_rep].append(member)
    return dict(clusters)


def greedy_split_clusters(
    clusters: Dict[str, List[str]],
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Dict[str, List[str]]:
    """贪心算法划分簇"""
    random.seed(seed)

    total_samples = sum(len(members) for members in clusters.values())
    target_train = int(total_samples * target_ratios[0])
    target_val = int(total_samples * target_ratios[1])
    target_test = total_samples - target_train - target_val

    print(f"\n总样本数: {total_samples}")
    print(f"总簇数: {len(clusters)}")
    print(f"目标: Train={target_train}, Val={target_val}, Test={target_test}")

    # 按簇大小排序
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    splits = {'train': [], 'val': [], 'test': []}
    current_counts = {'train': 0, 'val': 0, 'test': 0}

    # 贪心分配
    if TQDM_AVAILABLE:
        iterator = tqdm(sorted_clusters, desc="划分簇", unit="簇")
    else:
        iterator = sorted_clusters

    for cluster_rep, members in iterator:
        cluster_size = len(members)
        gaps = {
            'train': target_train - current_counts['train'],
            'val': target_val - current_counts['val'],
            'test': target_test - current_counts['test']
        }
        assigned_split = max(gaps.items(), key=lambda x: x[1])[0]
        splits[assigned_split].extend(members)
        current_counts[assigned_split] += cluster_size

    # 打印结果
    print(f"\n最终分配:")
    for split_name in ['train', 'val', 'test']:
        count = current_counts[split_name]
        ratio = count / total_samples
        print(f"  {split_name.capitalize()}: {count} 样本 ({ratio:.2%})")

    return splits


def save_splits(splits: Dict[str, List[str]], output_dir: Path, data_dir: Path):
    """保存划分结果"""
    print(f"\n保存数据集划分...")

    # 1. 保存ID列表
    for split_name, pdb_ids in splits.items():
        output_file = output_dir / f"{split_name}_ids.txt"
        with open(output_file, 'w') as f:
            for pdb_id in sorted(pdb_ids):
                f.write(f"{pdb_id}\n")
        print(f"  ✅ {split_name}_ids.txt: {len(pdb_ids)} 个样本")

    # 2. 保存JSON
    splits_with_paths = {}
    for split_name, pdb_ids in splits.items():
        splits_with_paths[split_name] = []
        for pdb_id in pdb_ids:
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

    # 3. 保存统计
    stats = {
        'total_samples': sum(len(ids) for ids in splits.values()),
        'train': {'count': len(splits['train']), 'ratio': len(splits['train']) / sum(len(ids) for ids in splits.values())},
        'val': {'count': len(splits['val']), 'ratio': len(splits['val']) / sum(len(ids) for ids in splits.values())},
        'test': {'count': len(splits['test']), 'ratio': len(splits['test']) / sum(len(ids) for ids in splits.values())}
    }

    stats_file = output_dir / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ split_statistics.json")


def stage2_parse_and_split(
    cluster_tsv: Path,
    output_dir: Path,
    data_dir: Path,
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):
    """阶段2b：解析聚类并划分数据集"""
    print("\n" + "="*70)
    print("阶段2b：解析聚类并划分数据集")
    print("="*70)

    # 解析聚类
    print(f"\n解析聚类结果: {cluster_tsv}")
    clusters = parse_mmseqs_clusters(cluster_tsv)
    print(f"  解析到 {len(clusters)} 个簇")

    # 贪心划分
    splits = greedy_split_clusters(clusters, target_ratios)

    # 保存结果
    save_splits(splits, output_dir, data_dir)


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="数据集划分完整流程（两阶段）")
    parser.add_argument("--data_dir", type=str, default="/home/husrcf/Code/ProtBind/New")
    parser.add_argument("--output_dir", type=str, default="/home/husrcf/Code/ProtBind/dataset_split")
    parser.add_argument("--workers", type=int, default=8, help="序列提取并行数")
    parser.add_argument("--threads", type=int, default=16, help="MMseqs2线程数")
    parser.add_argument("--force_extract", action='store_true', help="强制重新提取序列")
    parser.add_argument("--force_cluster", action='store_true', help="强制重新聚类")
    parser.add_argument("--stage", type=str, choices=['1', '2', 'all'], default='all',
                        help="运行阶段: 1=仅提取序列, 2=仅聚类和划分, all=全部")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    fasta_file = output_dir / "dataset_proteins.fasta"
    cluster_prefix = output_dir / "cluster_res"
    tmp_dir = output_dir / "tmp_mmseqs"
    tmp_dir.mkdir(exist_ok=True)

    print("="*70)
    print("数据集划分流程（两阶段，支持Resume）")
    print("="*70)
    print(f"\n配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  序列提取并行数: {args.workers}")
    print(f"  MMseqs2线程数: {args.threads}")

    # 阶段1：提取序列
    if args.stage in ['1', 'all']:
        sequences = stage1_extract_sequences(
            data_dir, fasta_file, args.workers, args.force_extract
        )

    # 阶段2：聚类和划分
    if args.stage in ['2', 'all']:
        # 2a: MMseqs2聚类
        cluster_tsv = stage2_mmseqs_clustering(
            fasta_file, cluster_prefix, tmp_dir, args.threads, args.force_cluster
        )

        # 2b: 解析和划分
        stage2_parse_and_split(cluster_tsv, output_dir, data_dir)

    print("\n" + "="*70)
    print("✅ 数据集划分完成！")
    print("="*70)
    print(f"\n输出文件位于: {output_dir}")
    print(f"  - train_ids.txt")
    print(f"  - val_ids.txt")
    print(f"  - test_ids.txt")
    print(f"  - dataset_splits.json")
    print(f"  - split_statistics.json")


if __name__ == "__main__":
    main()
