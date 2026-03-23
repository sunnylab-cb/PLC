#!/usr/bin/env python3
"""
更新现有 V3 特征文件，添加缺失的 ca_coords 和 all_atom_coords 字段

从对应的 V2 特征文件中复制这些字段
"""

import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def update_single_file(args_tuple):
    """更新单个 V3 特征文件"""
    v3_file, v2_file = args_tuple

    try:
        # 加载 V3 和 V2 特征
        v3_data = torch.load(v3_file, weights_only=False)
        v2_data = torch.load(v2_file, weights_only=False)

        # 检查是否已经有这些字段
        if 'ca_coords' in v3_data and 'all_atom_coords' in v3_data:
            return (v3_file.stem, True, "已存在")

        # 从 V2 复制缺失的字段
        v3_data['ca_coords'] = v2_data['ca_coords']
        v3_data['all_atom_coords'] = v2_data['all_atom_coords']

        # 保存更新后的文件
        torch.save(v3_data, v3_file)

        return (v3_file.stem, True, None)

    except Exception as e:
        return (v3_file.stem, False, str(e))


def main():
    v3_dir = Path("data/complete_features_v3")
    v2_dir = Path("data/complete_features_v2")

    if not v3_dir.exists():
        print(f"错误: V3 目录不存在: {v3_dir}")
        return

    if not v2_dir.exists():
        print(f"错误: V2 目录不存在: {v2_dir}")
        return

    # 查找所有 V3 特征文件
    year_ranges = ['1981-2000', '2001-2010', '2011-2019']
    file_pairs = []

    for year_range in year_ranges:
        v3_year_dir = v3_dir / year_range
        v2_year_dir = v2_dir / year_range

        if not v3_year_dir.exists():
            continue

        for pdb_dir in v3_year_dir.iterdir():
            if not pdb_dir.is_dir():
                continue

            pdb_id = pdb_dir.name
            v3_file = pdb_dir / f"{pdb_id}_protein.pt"
            v2_file = v2_year_dir / pdb_id / f"{pdb_id}_protein.pt"

            if v3_file.exists() and v2_file.exists():
                file_pairs.append((v3_file, v2_file))

    if len(file_pairs) == 0:
        print("没有找到需要更新的文件")
        return

    print(f"找到 {len(file_pairs)} 个 V3 特征文件")
    print("开始更新...")

    # 并行处理
    n_workers = min(mp.cpu_count(), 32)
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(update_single_file, file_pairs),
            total=len(file_pairs),
            desc="更新特征"
        ))

    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count
    already_exists = sum(1 for _, success, msg in results if success and msg == "已存在")

    print(f"\n完成:")
    print(f"  成功: {success_count}")
    print(f"  已存在: {already_exists}")
    print(f"  失败: {failed_count}")

    if failed_count > 0:
        print(f"\n失败的文件:")
        for pdb_id, success, error in results:
            if not success:
                print(f"  {pdb_id}: {error}")


if __name__ == "__main__":
    main()
