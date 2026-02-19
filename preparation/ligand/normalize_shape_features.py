#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归一化现有 ligand_features_v4 文件中的 shape features

对以下维度进行 z-score 归一化：
- Dim 0: PMI1 (主惯性矩1)
- Dim 1: PMI2 (主惯性矩2)
- Dim 2: PMI3 (主惯性矩3)
- Dim 4: Radius of Gyration (回转半径)
- Dim 7: Span L (主轴跨度-长)
- Dim 8: Span W (主轴跨度-宽)
- Dim 9: Span H (主轴跨度-高)
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse


# Normalization parameters computed from 19,037 samples
NORM_PARAMS = {
    0: (3879.901123, 6650.009766),    # PMI1
    1: (13954.526367, 28178.578125),  # PMI2
    2: (15758.538086, 30137.193359),  # PMI3
    4: (4.633801, 1.638986),          # Radius of Gyration
    7: (14.585329, 5.682965),         # Span L
    8: (8.701250, 3.143703),          # Span W
    9: (5.480688, 2.575974),          # Span H
}


def normalize_shape_features(shape_features: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to shape features.

    Args:
        shape_features: [10] array

    Returns:
        Normalized shape features
    """
    normalized = shape_features.copy()
    for dim, (mean, std) in NORM_PARAMS.items():
        if std > 0:
            normalized[dim] = (normalized[dim] - mean) / std
    return normalized


def process_single_file(pt_file: Path, overwrite: bool = True) -> tuple:
    """
    Normalize shape features in a single .pt file.

    Returns:
        (success, message)
    """
    try:
        # Load file
        data = torch.load(pt_file, weights_only=False)

        # Check if already normalized (heuristic: if max < 100, likely normalized)
        shape_feats = data['shape_features'].numpy()
        if shape_feats.max() < 100 and not overwrite:
            return (True, f"SKIP {pt_file.stem} (already normalized)")

        # Normalize shape features
        shape_feats_normalized = normalize_shape_features(shape_feats)

        # Update atom_features (last 10 dims are shape features)
        atom_feats = data['atom_features'].numpy()  # [N_atoms, 96]
        n_atoms = atom_feats.shape[0]

        # Replace shape part (86:96) with normalized values
        shape_part_normalized = np.tile(shape_feats_normalized, (n_atoms, 1))
        atom_feats[:, 86:96] = shape_part_normalized

        # Update data
        data['shape_features'] = torch.from_numpy(shape_feats_normalized)
        data['atom_features'] = torch.from_numpy(atom_feats)

        # Save
        torch.save(data, pt_file)

        return (True, f"{pt_file.stem}: OK")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{pt_file.stem}: ERROR - {str(e)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="归一化 ligand_features_v4 中的 shape features")
    parser.add_argument('--features-dir', type=Path, default='data/ligand_features_v4',
                        help='ligand_features_v4 目录')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行 worker 数量')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数（用于测试）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已归一化的文件')

    args = parser.parse_args()

    features_dir = args.features_dir.resolve()

    if not features_dir.exists():
        print(f"错误: 特征目录不存在: {features_dir}")
        return

    # 查找所有 .pt 文件
    year_ranges = ['1981-2000', '2001-2010', '2011-2019']
    pt_files = []

    for year_range in year_ranges:
        year_dir = features_dir / year_range
        if year_dir.exists():
            pt_files.extend(sorted(year_dir.rglob('*_ligand.pt')))

    if args.max_files:
        pt_files = pt_files[:args.max_files]

    print(f"找到 {len(pt_files)} 个文件")
    print(f"特征目录: {features_dir}")
    print(f"使用 {args.workers} 个 worker")
    print()
    print("归一化参数:")
    for dim, (mean, std) in NORM_PARAMS.items():
        print(f"  Dim {dim}: mean={mean:.2f}, std={std:.2f}")
    print()

    # 并行处理
    process_func = partial(process_single_file, overwrite=args.overwrite)

    success_count = 0
    skip_count = 0
    fail_count = 0

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, pt_files),
                total=len(pt_files),
                desc="归一化文件"
            ))
    else:
        results = []
        for pt_file in tqdm(pt_files, desc="归一化文件"):
            results.append(process_func(pt_file))

    # 统计结果
    failed_files = []
    for success, message in results:
        if success:
            if "SKIP" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            failed_files.append(message)

    print(f"\n{'='*70}")
    print(f"完成统计:")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(pt_files)}")
    print(f"{'='*70}")

    if failed_files:
        print("\n失败的文件:")
        for msg in failed_files[:20]:  # 只显示前20个
            print(f"  {msg}")
        if len(failed_files) > 20:
            print(f"  ... 还有 {len(failed_files) - 20} 个失败")


if __name__ == "__main__":
    main()
