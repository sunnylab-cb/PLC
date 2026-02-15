#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取完整特征：物理化学特征(42维) + 表面特征(14维) = 56维
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict
import warnings
import multiprocessing as mp

import numpy as np
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 导入特征提取模块
from extract_physchem_from_pdb import parse_pdb_protein
from physicochemical_features import extract_physicochemical_features
from surface_features import extract_surface_features
from structural_geometry_features import extract_structural_geometry_features


def extract_complete_features(pdb_path: Path, chain_id: str = None, mode: str = 'complex') -> Dict:
    """
    提取完整的82维特征（42维物理化学 + 14维表面 + 26维结构几何）

    Args:
        pdb_path: PDB文件路径
        chain_id: 链ID（仅在mode='chain'时使用）
        mode: 'chain' = 单链模式, 'complex' = 多链模式

    Returns:
        包含82维特征的字典
    """
    # 1. 解析PDB文件
    pdb_info = parse_pdb_protein(pdb_path, chain_id, mode)

    sequence = pdb_info['sequence']
    residue_names = pdb_info['residue_names']
    residue_indices = pdb_info['residue_indices']
    chain_ids = pdb_info['chain_ids']
    ca_coords = pdb_info['ca_coords']
    n_residues = pdb_info['n_residues']

    # 2. 提取物理化学特征（42维）
    print(f"  提取物理化学特征...")
    physchem_features = extract_physicochemical_features(
        pdb_path, sequence, residue_names, residue_indices, chain_ids
    )

    # 3. 提取表面特征（14维）
    print(f"  提取表面特征...")
    surface_feats = extract_surface_features(ca_coords, residue_names)

    # 4. 提取结构几何特征（26维）
    print(f"  提取结构几何特征...")
    geom_features = extract_structural_geometry_features(
        pdb_path, ca_coords, residue_indices, chain_ids
    )

    # 5. 拼接特征: 42 + 14 + 26 = 82
    complete_features = np.concatenate([
        physchem_features, 
        surface_feats, 
        geom_features
    ], axis=1)

    # 6. 构建输出字典
    output = {
        'complete_features': complete_features,  # [N_residues, 82]
        'physicochemical_features': physchem_features,  # [N_residues, 42]
        'surface_features': surface_feats,  # [N_residues, 14]
        'geometry_features': geom_features,  # [N_residues, 26]
        'sequence': sequence,
        'residue_names': residue_names,
        'residue_indices': residue_indices,
        'chain_ids': chain_ids,
        'ca_coords': ca_coords,  # [N_residues, 3] - 保留用于快速访问
        'all_atom_coords': pdb_info['all_atom_coords'],  # List[np.ndarray] - 每个残基的所有原子
        'all_atom_names': pdb_info['all_atom_names'],    # List[List[str]] - 每个残基的原子名称
        'pdb_path': str(pdb_path),
        'n_residues': n_residues,
        'mode': mode
    }

    return output


def process_single_file(
    pdb_path: Path,
    pdb_root: Path,
    output_dir: Path,
    mode: str,
    overwrite: bool
) -> tuple:
    """
    处理单个PDB文件（用于多进程）
    """
    try:
        # 构建输出路径
        rel_path = pdb_path.relative_to(pdb_root)
        out_path = (output_dir / rel_path).with_suffix('.pt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否已存在
        if out_path.exists() and not overwrite:
            return (True, f"SKIP {rel_path} (already exists)")

        # 提取特征
        data = extract_complete_features(pdb_path, chain_id=None, mode=mode)

        # 转换为torch张量
        data['complete_features'] = torch.from_numpy(data['complete_features'])
        data['physicochemical_features'] = torch.from_numpy(data['physicochemical_features'])
        data['surface_features'] = torch.from_numpy(data['surface_features'])
        data['geometry_features'] = torch.from_numpy(data['geometry_features'])
        data['ca_coords'] = torch.from_numpy(data['ca_coords'])

        # 转换所有原子坐标为torch张量列表
        data['all_atom_coords'] = [torch.from_numpy(coords) for coords in data['all_atom_coords']]

        # 保存
        torch.save(data, out_path)

        return (True, f"OK   {rel_path} ({data['n_residues']} residues, 82D)")

    except Exception as e:
        return (False, f"FAIL {pdb_path.relative_to(pdb_root)}: {e}")


def batch_extract_complete_features(
    pdb_root: Path,
    output_dir: Path,
    pattern: str = "*_protein.pdb",
    mode: str = 'complex',
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None
) -> None:
    """
    批量提取完整特征（82维）
    """
    # 查找所有PDB文件
    pdb_files = sorted(pdb_root.rglob(pattern))
    if max_files:
        pdb_files = pdb_files[:max_files]

    if not pdb_files:
        print(f"No PDB files found in {pdb_root} with pattern {pattern}")
        return

    print(f"Found {len(pdb_files)} PDB files")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(pdb_files))

    print(f"Using {n_workers} worker processes")
    print(f"Extracting 82D features (42D physchem + 14D surface + 26D geometry)")

    if n_workers == 1:
        # 单进程模式
        success_count = 0
        fail_count = 0
        skip_count = 0

        iterator = tqdm(pdb_files, desc="Processing", unit="file") if TQDM_AVAILABLE else pdb_files

        for pdb_path in iterator:
            success, message = process_single_file(
                pdb_path, pdb_root, output_dir, mode, overwrite
            )

            if TQDM_AVAILABLE:
                if "SKIP" in message:
                    iterator.set_postfix({"status": "SKIP", "success": success_count, "fail": fail_count, "skip": skip_count + 1})
                elif success:
                    iterator.set_postfix({"status": "OK", "success": success_count + 1, "fail": fail_count, "skip": skip_count})
                else:
                    iterator.set_postfix({"status": "FAIL", "success": success_count, "fail": fail_count + 1, "skip": skip_count})
            else:
                print(f"[{success_count + fail_count + skip_count + 1}/{len(pdb_files)}] {message}")

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

        tasks = [(pdb_path, pdb_root, output_dir, mode, overwrite) for pdb_path in pdb_files]

        with mp.Pool(processes=n_workers) as pool:
            if TQDM_AVAILABLE:
                results = pool.starmap(process_single_file, tasks)
                iterator = tqdm(results, total=len(tasks), desc="Processing", unit="file")

                for success, message in iterator:
                    if "SKIP" in message:
                        skip_count += 1
                        iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
                    elif success:
                        success_count += 1
                        iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
                    else:
                        fail_count += 1
                        iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                results = pool.starmap(process_single_file, tasks)

                for idx, (success, message) in enumerate(results, 1):
                    print(f"[{idx}/{len(pdb_files)}] {message}")

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
    print(f"Total:   {len(pdb_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="提取完整特征：物理化学(42D) + 表面(14D) = 56D"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="PDB文件或目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_protein.pdb",
        help="PDB文件匹配模式"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="complex",
        choices=["complex", "chain"],
        help="提取模式：complex=多链，chain=单链"
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

    if input_path.is_file():
        # 单个文件处理
        print(f"Processing single file: {input_path}")
        data = extract_complete_features(input_path, None, args.mode)

        # 保存
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{input_path.stem}_complete.pt"

        data['complete_features'] = torch.from_numpy(data['complete_features'])
        data['physicochemical_features'] = torch.from_numpy(data['physicochemical_features'])
        data['surface_features'] = torch.from_numpy(data['surface_features'])
        data['ca_coords'] = torch.from_numpy(data['ca_coords'])

        # 转换所有原子坐标为torch张量列表
        data['all_atom_coords'] = [torch.from_numpy(coords) for coords in data['all_atom_coords']]

        torch.save(data, out_file)
        print(f"Saved: {out_file}")
        print(f"Features: {data['complete_features'].shape}")

    else:
        # 目录批处理
        print(f"Processing directory: {input_path}")
        batch_extract_complete_features(
            pdb_root=input_path,
            output_dir=output_path,
            pattern=args.pattern,
            mode=args.mode,
            max_files=args.max,
            overwrite=args.overwrite,
            n_workers=args.workers
        )


if __name__ == "__main__":
    main()
