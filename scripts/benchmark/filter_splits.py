#!/usr/bin/env python3
"""
从现有的 splits 文件中排除指定的 PDB IDs（如 benchmark 数据集）

使用方法:
    python scripts/filter_splits.py \
        --input outputs/pocket_gvp_fusion_v6m7_20260217_011531/splits.txt \
        --exclude configs/spilts/excluded_samples.txt \
        --output configs/spilts/filtered_splits.txt
"""

import argparse
from pathlib import Path
from typing import Set, List, Tuple


def load_excluded_ids(exclude_file: str) -> Set[str]:
    """
    加载需要排除的 PDB IDs

    Args:
        exclude_file: 排除列表文件路径（每行一个 PDB ID）

    Returns:
        排除的 PDB IDs 集合
    """
    excluded = set()
    with open(exclude_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            excluded.add(line.lower())  # 转小写以便匹配
    return excluded


def load_splits(split_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    加载现有的 splits 文件

    Args:
        split_file: splits 文件路径

    Returns:
        (train_ids, val_ids, test_ids)
    """
    train_ids, val_ids, test_ids = [], [], []

    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) == 2:
                split, pdb_id = parts
                if split == 'train':
                    train_ids.append(pdb_id)
                elif split == 'val':
                    val_ids.append(pdb_id)
                elif split == 'test':
                    test_ids.append(pdb_id)

    return train_ids, val_ids, test_ids


def filter_splits(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    excluded: Set[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    从 splits 中移除被排除的 IDs

    Args:
        train_ids: 训练集 IDs
        val_ids: 验证集 IDs
        test_ids: 测试集 IDs
        excluded: 需要排除的 IDs 集合

    Returns:
        过滤后的 (train_ids, val_ids, test_ids)
    """
    train_filtered = [pid for pid in train_ids if pid.lower() not in excluded]
    val_filtered = [pid for pid in val_ids if pid.lower() not in excluded]
    test_filtered = [pid for pid in test_ids if pid.lower() not in excluded]

    return train_filtered, val_filtered, test_filtered


def save_splits(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    output_file: str,
    original_counts: Tuple[int, int, int] = None
):
    """
    保存过滤后的 splits 文件

    Args:
        train_ids: 训练集 IDs
        val_ids: 验证集 IDs
        test_ids: 测试集 IDs
        output_file: 输出文件路径
        original_counts: 原始数据集大小 (train, val, test)
    """
    with open(output_file, 'w') as f:
        f.write("# Train/Val/Test splits (filtered to exclude benchmark datasets)\n")
        f.write(f"# Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}\n")

        if original_counts:
            orig_train, orig_val, orig_test = original_counts
            f.write(f"# Original counts: Train: {orig_train}, Val: {orig_val}, Test: {orig_test}\n")
            f.write(f"# Excluded: Train: {orig_train - len(train_ids)}, "
                   f"Val: {orig_val - len(val_ids)}, Test: {orig_test - len(test_ids)}\n")

        f.write("\n# TRAIN\n")
        for pdb_id in train_ids:
            f.write(f"train\t{pdb_id}\n")

        f.write("\n# VAL\n")
        for pdb_id in val_ids:
            f.write(f"val\t{pdb_id}\n")

        f.write("\n# TEST\n")
        for pdb_id in test_ids:
            f.write(f"test\t{pdb_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 splits 文件中排除指定的 PDB IDs"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的 splits 文件路径'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        required=True,
        help='排除列表文件路径（每行一个 PDB ID）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出的 splits 文件路径'
    )

    args = parser.parse_args()

    # 加载排除列表
    print(f"Loading excluded IDs from: {args.exclude}")
    excluded = load_excluded_ids(args.exclude)
    print(f"  Found {len(excluded)} IDs to exclude")

    # 加载原始 splits
    print(f"\nLoading original splits from: {args.input}")
    train_ids, val_ids, test_ids = load_splits(args.input)
    orig_counts = (len(train_ids), len(val_ids), len(test_ids))
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"  Total: {sum(orig_counts)}")

    # 过滤 splits
    print(f"\nFiltering splits...")
    train_filtered, val_filtered, test_filtered = filter_splits(
        train_ids, val_ids, test_ids, excluded
    )

    # 统计排除数量
    n_excluded_train = len(train_ids) - len(train_filtered)
    n_excluded_val = len(val_ids) - len(val_filtered)
    n_excluded_test = len(test_ids) - len(test_filtered)
    n_excluded_total = n_excluded_train + n_excluded_val + n_excluded_test

    print(f"  Excluded from train: {n_excluded_train}")
    print(f"  Excluded from val: {n_excluded_val}")
    print(f"  Excluded from test: {n_excluded_test}")
    print(f"  Total excluded: {n_excluded_total}")

    print(f"\nFiltered splits:")
    print(f"  Train: {len(train_filtered)}, Val: {len(val_filtered)}, Test: {len(test_filtered)}")
    print(f"  Total: {len(train_filtered) + len(val_filtered) + len(test_filtered)}")

    # 保存过滤后的 splits
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving filtered splits to: {args.output}")
    save_splits(
        train_filtered, val_filtered, test_filtered,
        args.output, orig_counts
    )
    print("Done!")


if __name__ == '__main__':
    main()
