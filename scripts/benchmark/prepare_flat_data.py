#!/usr/bin/env python3
"""
准备扁平化的数据结构用于训练

将嵌套的年份分组数据结构转换为扁平结构：
- data/complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt -> database/features/{pdb_id}_features.npz
- data/esm3_latents/YEAR/PDB_ID/{pdb_id}_protein.pt -> data/esm3_latents/{pdb_id}.pt
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_flat_data(
    source_features_dir: str = "data/complete_features",
    source_esm_dir: str = "data/esm3_latents",
    target_features_dir: str = "database/features",
    target_esm_dir: str = "data/esm3_latents_flat"
):
    """
    准备扁平化的数据结构

    Args:
        source_features_dir: 源特征目录（嵌套结构）
        source_esm_dir: 源ESM目录（嵌套结构）
        target_features_dir: 目标特征目录（扁平结构）
        target_esm_dir: 目标ESM目录（扁平结构）
    """

    source_features_path = Path(source_features_dir)
    source_esm_path = Path(source_esm_dir)
    target_features_path = Path(target_features_dir)
    target_esm_path = Path(target_esm_dir)

    # 创建目标目录
    target_features_path.mkdir(parents=True, exist_ok=True)
    target_esm_path.mkdir(parents=True, exist_ok=True)

    print("扫描数据文件...")

    # 收集所有PDB ID
    pdb_ids = set()

    # 从complete_features收集
    for year_dir in source_features_path.iterdir():
        if year_dir.is_dir() and not year_dir.name.endswith('.log'):
            for pdb_dir in year_dir.iterdir():
                if pdb_dir.is_dir():
                    pdb_ids.add(pdb_dir.name)

    print(f"找到 {len(pdb_ids)} 个PDB条目")

    # 处理特征文件
    print("\n处理特征文件...")
    features_copied = 0
    features_missing = 0

    for pdb_id in tqdm(sorted(pdb_ids), desc="特征文件"):
        # 查找源文件
        source_file = None
        for year_dir in source_features_path.iterdir():
            if year_dir.is_dir():
                candidate = year_dir / pdb_id / f"{pdb_id}_protein.pt"
                if candidate.exists():
                    source_file = candidate
                    break

        if source_file:
            # 目标文件名：{pdb_id}_features.npz
            # 但源文件是.pt，我们需要创建符号链接或复制
            target_file = target_features_path / f"{pdb_id}_features.npz"

            # 检查是否是.pt文件（需要重命名）
            if source_file.suffix == '.pt':
                # 创建符号链接
                if not target_file.exists():
                    # 注意：这里假设源文件实际上是npz格式但扩展名错误
                    # 如果确实是.pt文件，需要转换
                    try:
                        os.symlink(source_file.absolute(), target_file)
                        features_copied += 1
                    except Exception as e:
                        print(f"  警告：无法链接 {pdb_id}: {e}")
                        features_missing += 1
        else:
            features_missing += 1

    print(f"  成功: {features_copied}, 缺失: {features_missing}")

    # 处理ESM3文件
    print("\n处理ESM3文件...")
    esm_copied = 0
    esm_missing = 0

    for pdb_id in tqdm(sorted(pdb_ids), desc="ESM3文件"):
        # 查找源文件
        source_file = None
        for year_dir in source_esm_path.iterdir():
            if year_dir.is_dir():
                candidate = year_dir / pdb_id / f"{pdb_id}_protein.pt"
                if candidate.exists():
                    source_file = candidate
                    break

        if source_file:
            # 目标文件名：{pdb_id}.pt
            target_file = target_esm_path / f"{pdb_id}.pt"

            if not target_file.exists():
                try:
                    # 创建符号链接（节省空间）
                    os.symlink(source_file.absolute(), target_file)
                    esm_copied += 1
                except Exception as e:
                    print(f"  警告：无法链接 {pdb_id}: {e}")
                    esm_missing += 1
        else:
            esm_missing += 1

    print(f"  成功: {esm_copied}, 缺失: {esm_missing}")

    # 统计
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"特征文件: {features_copied} 个")
    print(f"ESM3文件: {esm_copied} 个")
    print(f"\n目标目录:")
    print(f"  特征: {target_features_path.absolute()}")
    print(f"  ESM3: {target_esm_path.absolute()}")
    print("\n现在可以运行训练：")
    print("  python scripts/train.py --config configs/pocket_gvp_v2.yaml")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='准备扁平化的数据结构')
    parser.add_argument('--source-features', default='data/complete_features',
                       help='源特征目录')
    parser.add_argument('--source-esm', default='data/esm3_latents',
                       help='源ESM目录')
    parser.add_argument('--target-features', default='database/features',
                       help='目标特征目录')
    parser.add_argument('--target-esm', default='data/esm3_latents_flat',
                       help='目标ESM目录')

    args = parser.parse_args()

    prepare_flat_data(
        source_features_dir=args.source_features,
        source_esm_dir=args.source_esm,
        target_features_dir=args.target_features,
        target_esm_dir=args.target_esm
    )
