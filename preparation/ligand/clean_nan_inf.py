#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 ligand_features_v4 中的 NaN 和 Inf 值

策略：
1. 第一遍：收集所有有效值，计算每个维度的均值
2. 第二遍：将 NaN 和 Inf 替换为对应维度的均值
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse


def collect_statistics(pt_files):
    """
    第一遍：收集所有维度的统计信息
    """
    print("第一遍：收集统计信息...")

    # 收集每个维度的有效值
    dim_values = [[] for _ in range(96)]

    for pt_file in tqdm(pt_files, desc="收集统计"):
        try:
            data = torch.load(pt_file, weights_only=False)
            atom_feats = data['atom_features'].numpy()  # [N_atoms, 96]

            for dim in range(96):
                values = atom_feats[:, dim]
                # 只收集有效值（非 NaN 且非 Inf）
                valid_mask = np.isfinite(values)
                valid_values = values[valid_mask]
                if len(valid_values) > 0:
                    dim_values[dim].extend(valid_values.tolist())
        except Exception as e:
            continue

    # 计算每个维度的均值
    dim_means = np.zeros(96, dtype=np.float32)
    for dim in range(96):
        if len(dim_values[dim]) > 0:
            dim_means[dim] = np.mean(dim_values[dim])
        else:
            dim_means[dim] = 0.0  # 如果没有有效值，使用 0

    return dim_means


def clean_single_file(pt_file: Path, dim_means: np.ndarray) -> tuple:
    """
    清理单个文件中的 NaN 和 Inf

    Returns:
        (success, message)
    """
    try:
        # Load file
        data = torch.load(pt_file, weights_only=False)
        atom_feats = data['atom_features'].numpy()  # [N_atoms, 96]

        # 检查是否有 NaN 或 Inf
        has_nan = np.isnan(atom_feats).any()
        has_inf = np.isinf(atom_feats).any()

        if not has_nan and not has_inf:
            return (True, f"SKIP {pt_file.stem} (no NaN/Inf)")

        # 替换 NaN 和 Inf
        nan_count = 0
        inf_count = 0

        for dim in range(96):
            values = atom_feats[:, dim]

            # 替换 NaN
            nan_mask = np.isnan(values)
            if nan_mask.any():
                nan_count += nan_mask.sum()
                atom_feats[nan_mask, dim] = dim_means[dim]

            # 替换 Inf
            inf_mask = np.isinf(values)
            if inf_mask.any():
                inf_count += inf_mask.sum()
                atom_feats[inf_mask, dim] = dim_means[dim]

        # 更新数据
        data['atom_features'] = torch.from_numpy(atom_feats)

        # 保存
        torch.save(data, pt_file)

        return (True, f"{pt_file.stem}: Fixed {nan_count} NaN, {inf_count} Inf")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{pt_file.stem}: ERROR - {str(e)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="清理 ligand_features_v4 中的 NaN 和 Inf")
    parser.add_argument('--features-dir', type=Path, default='data/ligand_features_v4',
                        help='ligand_features_v4 目录')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行 worker 数量')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数（用于测试）')

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
    print()

    # 第一遍：收集统计信息
    dim_means = collect_statistics(pt_files)

    print()
    print("每个维度的均值（用于替换 NaN/Inf）:")
    print("Dimensions with potential NaN/Inf (35-36, 41-85):")
    for dim in list(range(35, 41)) + list(range(41, 86)):
        print(f"  Dim {dim}: mean={dim_means[dim]:.4f}")
    print()

    # 第二遍：清理文件
    print(f"第二遍：清理 NaN 和 Inf (使用 {args.workers} 个 worker)...")

    process_func = partial(clean_single_file, dim_means=dim_means)

    success_count = 0
    skip_count = 0
    fail_count = 0
    fixed_count = 0

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, pt_files),
                total=len(pt_files),
                desc="清理文件"
            ))
    else:
        results = []
        for pt_file in tqdm(pt_files, desc="清理文件"):
            results.append(process_func(pt_file))

    # 统计结果
    failed_files = []
    for success, message in results:
        if success:
            if "SKIP" in message:
                skip_count += 1
            else:
                success_count += 1
                fixed_count += 1
                print(f"  {message}")
        else:
            fail_count += 1
            failed_files.append(message)

    print(f"\n{'='*70}")
    print(f"完成统计:")
    print(f"  修复: {fixed_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(pt_files)}")
    print(f"{'='*70}")

    if failed_files:
        print("\n失败的文件:")
        for msg in failed_files[:20]:
            print(f"  {msg}")
        if len(failed_files) > 20:
            print(f"  ... 还有 {len(failed_files) - 20} 个失败")


if __name__ == "__main__":
    main()
