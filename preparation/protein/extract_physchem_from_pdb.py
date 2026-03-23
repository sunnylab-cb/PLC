#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始PDB文件提取物理化学特征
独立于ESM3 latent提取流程

输出格式：
{
    'physicochemical_features': [N_residues, 42],
    'sequence': "ACDEFGH...",
    'residue_names': ['ALA', 'CYS', ...],
    'chain_id': 'A',
    'residue_indices': [1, 2, 3, ...],
    'pdb_path': '/path/to/protein.pdb'
}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import multiprocessing as mp
from functools import partial

import numpy as np
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available. Install with: pip install tqdm")

# 导入物理化学特征提取函数
from physicochemical_features import (
    extract_physicochemical_features,
    AA_3TO1
)

try:
    from biotite.structure.io.pdb import PDBFile
    import biotite.structure as bs
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    warnings.warn("Biotite not installed. Cannot parse PDB files.")


def parse_pdb_protein(pdb_path: Path, chain_id: str = None, mode: str = 'complex') -> Dict:
    """
    从PDB文件解析蛋白质信息（支持多链）

    Args:
        pdb_path: PDB文件路径
        chain_id: 链ID（如果为None且mode='chain'，选择第一条链；如果mode='complex'，处理所有链）
        mode: 'chain' = 单链模式, 'complex' = 多链模式（与ESM3一致）

    Returns:
        字典包含序列、残基名称、坐标等信息
    """
    if not BIOTITE_AVAILABLE:
        raise ImportError("Biotite is required. Install: pip install biotite")

    # 读取PDB文件
    pdb_file = PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)

    # 只保留氨基酸，去除HETATM
    atom_array = structure[bs.filter_amino_acids(structure) & ~structure.hetero]

    if mode == 'chain':
        # 单链模式：只处理一条链
        if chain_id is None:
            chains = np.unique(atom_array.chain_id)
            if len(chains) == 0:
                raise ValueError(f"No protein chains found in {pdb_path}")
            chain_id = chains[0]
            print(f"  Auto-selected chain: {chain_id}")

        # 过滤指定链
        chain_mask = atom_array.chain_id == chain_id
        atom_array = atom_array[chain_mask]

        if len(atom_array) == 0:
            raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

    elif mode == 'complex':
        # 多链模式：处理所有链（与ESM3一致）
        print(f"  Complex mode: processing all chains")
        chains = np.unique(atom_array.chain_id)
        print(f"  Found chains: {', '.join(chains)}")

    # 提取C-alpha原子
    ca_mask = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]

    # 提取残基信息
    residue_names = []
    residue_indices = []
    insertion_codes = []
    chain_ids = []
    sequence = []

    for residue in bs.residue_iter(ca_atoms):
        res_name = residue.res_name[0]  # 三字母代码
        res_id = residue.res_id[0]
        res_chain = residue.chain_id[0]
        ins_code = residue.ins_code[0] if hasattr(residue, 'ins_code') else ''

        residue_names.append(res_name)
        residue_indices.append(int(res_id))
        insertion_codes.append(str(ins_code).strip())
        chain_ids.append(res_chain)

        # 转换为单字母代码
        aa_1 = AA_3TO1.get(res_name, 'X')
        sequence.append(aa_1)

    sequence_str = ''.join(sequence)

    # 提取C-alpha坐标（保留用于快速访问）
    ca_coords = ca_atoms.coord

    # 提取每个残基的所有原子坐标和名称（骨架 + 侧链）
    all_atom_coords = []  # List[np.ndarray]: 每个残基的所有原子坐标
    all_atom_names = []   # List[List[str]]: 每个残基的所有原子名称

    for idx, residue in enumerate(bs.residue_iter(ca_atoms)):
        # 获取当前残基的所有原子
        res_id = residue.res_id[0]
        res_chain = residue.chain_id[0]
        ins = insertion_codes[idx]

        # 在原始atom_array中找到该残基的所有原子（含 insertion code 精确匹配）
        res_mask = (
            (atom_array.res_id == res_id) &
            (atom_array.chain_id == res_chain) &
            (atom_array.ins_code == ins)
        )
        res_atoms = atom_array[res_mask]

        # 提取该残基的所有重原子（排除氢原子）
        # 注意：biotite的filter_amino_acids已经过滤了HETATM，这里只需要排除氢
        heavy_atom_mask = res_atoms.element != 'H'
        heavy_atoms = res_atoms[heavy_atom_mask]

        # 存储坐标和原子名称
        res_coords = heavy_atoms.coord  # [n_atoms, 3]
        res_atom_names = [atom.atom_name.strip() for atom in heavy_atoms]

        all_atom_coords.append(res_coords)
        all_atom_names.append(res_atom_names)

    return {
        'sequence': sequence_str,
        'residue_names': residue_names,
        'residue_indices': residue_indices,
        'insertion_codes': insertion_codes,  # 每个残基的insertion code（列表）
        'chain_ids': chain_ids,  # 每个残基的chain_id（列表）
        'ca_coords': ca_coords,  # [N, 3] - 保留用于快速访问
        'all_atom_coords': all_atom_coords,  # List[np.ndarray] - 每个残基的所有原子
        'all_atom_names': all_atom_names,    # List[List[str]] - 每个残基的原子名称
        'pdb_path': str(pdb_path),
        'n_residues': len(residue_names),
        'mode': mode
    }


def extract_physchem_from_pdb(pdb_path: Path, chain_id: str = None, mode: str = 'complex') -> Dict:
    """
    从PDB文件提取完整的物理化学特征（per-residue，支持多链）

    Args:
        pdb_path: PDB文件路径
        chain_id: 链ID（仅在mode='chain'时使用）
        mode: 'chain' = 单链模式, 'complex' = 多链模式（默认，与ESM3一致）

    Returns:
        包含42维物理化学特征的字典
    """
    # 1. 解析PDB文件
    pdb_info = parse_pdb_protein(pdb_path, chain_id, mode)

    sequence = pdb_info['sequence']
    residue_names = pdb_info['residue_names']
    residue_indices = pdb_info['residue_indices']
    chain_ids = pdb_info['chain_ids']
    n_residues = pdb_info['n_residues']

    # 2. 提取物理化学特征（per-residue）
    # 注意：SASA计算需要整个复合物的上下文，所以传入None作为chain_id
    physchem_features = extract_physicochemical_features(
        pdb_path, sequence, residue_names, residue_indices, chain_ids
    )

    # 3. 构建输出字典（与ESM3格式一致）
    output = {
        'physicochemical_features': physchem_features,
        'sequence': sequence,
        'residue_names': residue_names,
        'residue_indices': residue_indices,
        'chain_ids': chain_ids,  # 列表：每个残基的chain_id
        'ca_coords': pdb_info['ca_coords'],
        'all_atom_coords': pdb_info['all_atom_coords'],  # 新增：所有原子坐标
        'all_atom_names': pdb_info['all_atom_names'],    # 新增：所有原子名称
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
) -> Tuple[bool, str]:
    """
    处理单个PDB文件（用于多进程）

    Returns:
        (success, message)
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
        data = extract_physchem_from_pdb(pdb_path, chain_id=None, mode=mode)

        # 转换为torch张量
        data['physicochemical_features'] = torch.from_numpy(data['physicochemical_features'])
        data['ca_coords'] = torch.from_numpy(data['ca_coords'])

        # 保存
        torch.save(data, out_path)

        return (True, f"OK   {rel_path} ({data['n_residues']} residues)")

    except Exception as e:
        return (False, f"FAIL {pdb_path.relative_to(pdb_root)}: {e}")


def batch_extract_from_pdb_directory(
    pdb_root: Path,
    output_dir: Path,
    pattern: str = "*_protein.pdb",
    mode: str = 'complex',
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None
) -> None:
    """
    批量从PDB目录提取物理化学特征（支持多进程）

    Args:
        pdb_root: PDB文件根目录
        output_dir: 输出目录
        pattern: PDB文件匹配模式
        mode: 'chain' 或 'complex'
        max_files: 最大处理文件数
        overwrite: 是否覆盖已存在的文件
        n_workers: 进程数（None=自动检测CPU核心数）
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

    if n_workers == 1:
        # 单进程模式（用于调试）
        success_count = 0
        fail_count = 0
        skip_count = 0

        # 使用tqdm进度条
        iterator = tqdm(pdb_files, desc="Processing", unit="file") if TQDM_AVAILABLE else pdb_files

        for pdb_path in iterator:
            success, message = process_single_file(
                pdb_path, pdb_root, output_dir, mode, overwrite
            )

            if TQDM_AVAILABLE:
                # 更新tqdm描述
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

        # 创建参数列表
        tasks = [(pdb_path, pdb_root, output_dir, mode, overwrite) for pdb_path in pdb_files]

        # 使用进程池
        with mp.Pool(processes=n_workers) as pool:
            # 使用imap_unordered以便实时显示进度
            if TQDM_AVAILABLE:
                # 使用tqdm包装结果迭代器
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
                # 无tqdm，使用原始输出
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
        description="从原始PDB文件提取物理化学特征"
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
        help="PDB文件匹配模式（用于目录批处理）"
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="链ID（如果不指定，自动选择第一条链）"
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
    parser.add_argument(
        "--mode",
        type=str,
        default="complex",
        choices=["complex", "chain"],
        help="提取模式：complex=多链（与ESM3对齐），chain=单链"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # 判断是单个文件还是目录
    if input_path.is_file():
        # 单个文件处理
        print(f"Processing single file: {input_path}")
        data = extract_physchem_from_pdb(input_path, args.chain, args.mode)

        # 保存
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{input_path.stem}_physchem.pt"

        data['physicochemical_features'] = torch.from_numpy(data['physicochemical_features'])
        data['ca_coords'] = torch.from_numpy(data['ca_coords'])

        torch.save(data, out_file)
        print(f"Saved: {out_file}")

    else:
        # 目录批处理
        print(f"Processing directory: {input_path}")
        batch_extract_from_pdb_directory(
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
