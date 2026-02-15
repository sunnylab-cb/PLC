#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取配体特征 V2 (增强版)

从P-L目录读取配体文件，提取35维per-atom特征
✅ 修复版本：保留原始实验坐标用于距离计算
✅ 添加多种坐标版本和分子描述符
"""

from __future__ import annotations

import argparse
from pathlib import Path
import multiprocessing as mp
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from ligand_features_v2 import extract_ligand_features_from_file


def extract_ligand_features_single(
    ligand_file: Path,
    ligand_root: Path,
    output_dir: Path,
    num_conformers: int,
    overwrite: bool,
    no_conformers: bool = False
) -> tuple:
    """
    处理单个配体文件
    """
    try:
        # 构建输出路径
        rel_path = ligand_file.relative_to(ligand_root)
        out_path = (output_dir / rel_path).with_suffix('.pt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否已存在
        if out_path.exists() and not overwrite:
            return (True, f"SKIP {rel_path}")

        # 提取特征
        features = extract_ligand_features_from_file(
            str(ligand_file),
            include_conformers=(not no_conformers and num_conformers > 0),
            num_conformers=num_conformers
        )

        # 转换为torch张量
        data = {
            # 原子级别特征
            'atom_features': torch.from_numpy(features['atom_features']),

            # 坐标信息（多个版本）
            'coords': torch.from_numpy(features['coords']),
            'coords_heavy_current': torch.from_numpy(features['coords_heavy_current']),

            # 电子效应特征
            'gasteiger_charges': torch.from_numpy(features['gasteiger_charges']),
            'logp_contribs': torch.from_numpy(features['logp_contribs']),
            'tpsa_contribs': torch.from_numpy(features['tpsa_contribs']),

            # 柔性特征
            'rmsf': torch.from_numpy(features['rmsf']),
            'flexibility_flag': torch.from_numpy(features['flexibility_flag']),

            # 元数据
            'num_atoms': features['num_atoms'],
            'num_heavy_atoms': features['num_heavy_atoms'],
            'has_original_coords': features['has_original_coords'],

            # 分子级别描述符
            'molecular_descriptors': features['molecular_descriptors'],

            # 路径
            'ligand_path': str(ligand_file)
        }

        # ✅ 保存原始实验坐标（关键！）
        if features['coords_original'] is not None:
            data['coords_original'] = torch.from_numpy(features['coords_original'])

        # 保存
        torch.save(data, out_path)

        ligand_name = ligand_file.stem.replace('_ligand', '')
        return (True, f"{ligand_name} ({features['num_atoms']} atoms)")

    except Exception as e:
        ligand_name = ligand_file.stem.replace('_ligand', '')
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{ligand_name}: {str(e)[:100]}\n{error_msg[:200]}")


def batch_extract_ligand_features(
    ligand_root: Path,
    output_dir: Path,
    pattern: str = "*_ligand.*",
    num_conformers: int = 50,
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None,
    no_conformers: bool = False
) -> None:
    """
    批量提取配体特征
    """
    # 查找所有配体文件，优先选择.sdf格式
    ligand_files_dict = {}  # 使用字典去重，key=蛋白ID

    # 优先级：.sdf > .mol2 > .pdb > .mol
    for ext in ['.sdf', '.mol2', '.pdb', '.mol']:
        for file_path in ligand_root.rglob(f"*_ligand{ext}"):
            # 提取蛋白ID（去掉_ligand和扩展名）
            protein_id = file_path.stem.replace('_ligand', '')
            # 只保留第一个找到的（优先级高的）
            if protein_id not in ligand_files_dict:
                ligand_files_dict[protein_id] = file_path

    ligand_files = sorted(ligand_files_dict.values())

    if max_files:
        ligand_files = ligand_files[:max_files]

    if not ligand_files:
        print(f"No ligand files found in {ligand_root}")
        return

    print(f"Found {len(ligand_files)} ligand files")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(ligand_files))

    print(f"Using {n_workers} worker processes")
    print(f"Extracting 35D per-atom features with {num_conformers} conformers")

    if n_workers == 1:
        # 单进程模式
        success_count = 0
        fail_count = 0
        skip_count = 0

        iterator = tqdm(ligand_files, desc="Processing", unit="file") if TQDM_AVAILABLE else ligand_files

        for idx, ligand_file in enumerate(iterator, 1):
            success, message = extract_ligand_features_single(
                ligand_file, ligand_root, output_dir, num_conformers, overwrite, no_conformers
            )

            if TQDM_AVAILABLE:
                if "SKIP" in message:
                    skip_count += 1
                else:
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"[{idx}/{len(ligand_files)}] {status} {message}")

            if success:
                if "SKIP" in message:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1

    else:
        # 多进程模式
        success_count = 0
        fail_count = 0
        skip_count = 0

        tasks = [(f, ligand_root, output_dir, num_conformers, overwrite, no_conformers) for f in ligand_files]

        with mp.Pool(processes=n_workers) as pool:
            if TQDM_AVAILABLE:
                results = pool.starmap(extract_ligand_features_single, tasks)
                iterator = tqdm(results, total=len(tasks), desc="Processing", unit="file")

                for idx, (success, message) in enumerate(iterator, 1):
                    if "SKIP" in message:
                        skip_count += 1
                    elif success:
                        success_count += 1
                    else:
                        fail_count += 1
                    iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                results = pool.starmap(extract_ligand_features_single, tasks)

                for idx, (success, message) in enumerate(results, 1):
                    status = "✓" if success else "✗"
                    print(f"[{idx}/{len(ligand_files)}] {status} {message}")

                    if success:
                        if "SKIP" in message:
                            skip_count += 1
                        else:
                            success_count += 1
                    else:
                        fail_count += 1

    print(f"\n=== Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total:   {len(ligand_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="批量提取配体特征 (35D per-atom)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="配体文件根目录 (P-L)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录 (ligand_features)"
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=50,
        help="生成的构象数量 (默认50，设为0则只使用实验结构)"
    )
    parser.add_argument(
        "--no-conformers",
        action="store_true",
        help="不生成构象，只使用实验结构（更快，但无柔性分析）"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="最大处理文件数"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的文件"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认=CPU核心数）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Processing ligands from: {input_path}")

    batch_extract_ligand_features(
        ligand_root=input_path,
        output_dir=output_path,
        num_conformers=args.conformers,
        max_files=args.max,
        overwrite=args.overwrite,
        n_workers=args.workers,
        no_conformers=args.no_conformers
    )


if __name__ == "__main__":
    main()
