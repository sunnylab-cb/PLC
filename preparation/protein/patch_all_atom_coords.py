#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速补丁脚本：为现有的 complete_features_v2 文件添加 all_atom_coords 和 all_atom_names
"""

import torch
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preparation.protein.extract_physchem_from_pdb import parse_pdb_protein


def patch_single_file(pt_file: Path, pdb_root: Path, year_ranges: List[str]) -> tuple:
    """
    为单个 .pt 文件添加 all_atom_coords 和 all_atom_names

    Returns:
        (success, message)
    """
    try:
        pdb_id = pt_file.stem.replace('_protein', '')

        # 查找对应的 PDB 文件
        pdb_file = None
        for year_range in year_ranges:
            test_path = pdb_root / year_range / pdb_id / f'{pdb_id}_protein.pdb'
            if test_path.exists():
                pdb_file = test_path
                break

        if pdb_file is None:
            return (False, f"{pdb_id}: PDB file not found")

        # 加载现有的 .pt 文件
        data = torch.load(pt_file, weights_only=False)

        # 检查是否已经有这些字段
        if 'all_atom_coords' in data and 'all_atom_names' in data:
            return (True, f"{pdb_id}: SKIP (already has fields)")

        # 从 PDB 文件解析 all_atom_coords
        mode = data.get('mode', 'complex')
        pdb_info = parse_pdb_protein(pdb_file, chain_id=None, mode=mode)

        # 添加缺失的字段
        data['all_atom_coords'] = pdb_info['all_atom_coords']
        data['all_atom_names'] = pdb_info['all_atom_names']

        # 保存回文件
        torch.save(data, pt_file)

        return (True, f"{pdb_id}: OK")

    except Exception as e:
        return (False, f"{pt_file.stem}: ERROR - {str(e)[:100]}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="快速补丁：添加 all_atom_coords 到现有文件")
    parser.add_argument('--features-dir', type=Path, required=True,
                        help='complete_features_v2 目录')
    parser.add_argument('--pdb-root', type=Path, required=True,
                        help='PDB 文件根目录')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行 worker 数量')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数（用于测试）')

    args = parser.parse_args()

    features_dir = args.features_dir.resolve()
    pdb_root = args.pdb_root.resolve()

    if not features_dir.exists():
        print(f"错误: 特征目录不存在: {features_dir}")
        return

    if not pdb_root.exists():
        print(f"错误: PDB 目录不存在: {pdb_root}")
        return

    # 查找所有 .pt 文件
    year_ranges = ['1981-2000', '2001-2010', '2011-2019']
    pt_files = []

    for year_range in year_ranges:
        year_dir = features_dir / year_range
        if year_dir.exists():
            pt_files.extend(sorted(year_dir.rglob('*_protein.pt')))

    if args.max_files:
        pt_files = pt_files[:args.max_files]

    print(f"找到 {len(pt_files)} 个文件")
    print(f"特征目录: {features_dir}")
    print(f"PDB 目录: {pdb_root}")
    print(f"使用 {args.workers} 个 worker\n")

    # 并行处理
    process_func = partial(
        patch_single_file,
        pdb_root=pdb_root,
        year_ranges=year_ranges
    )

    success_count = 0
    skip_count = 0
    fail_count = 0

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, pt_files),
                total=len(pt_files),
                desc="补丁文件"
            ))
    else:
        results = []
        for pt_file in tqdm(pt_files, desc="补丁文件"):
            results.append(process_func(pt_file))

    # 统计结果
    for success, message in results:
        if success:
            if "SKIP" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            print(f"❌ {message}")

    print(f"\n{'='*70}")
    print(f"完成统计:")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(pt_files)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
