#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取完整特征 V2：物理化学特征(42维) + 表面特征(14维) + 结构几何特征(26维) + 增强特征(12维) = 94维

新增功能：
1. 增强特征V2 (12维)
   - 显式几何"口袋度"特征 (3维)
   - 柔性与置信度特征 (3维)
   - 平滑化环境物理化学特征 (6维)

2. Per-amino acid对齐机制
   - 确保特征与残基索引严格对应
   - 处理缺失残基和插入码
   - 与ESM3 latent对齐
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
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
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from physicochemical_features import extract_physicochemical_features
from PLC.preparation.protein.extract_physchem_from_pdb import parse_pdb_protein
from PLC.preparation.protein.surface_features import extract_surface_features
from PLC.preparation.protein.structural_geometry_features import extract_structural_geometry_features
from PLC.preparation.protein.enhanced_features_v2 import (
    compute_pocketness_features,
    compute_flexibility_features,
    compute_environmental_features
)


def extract_complete_features_v2(
    pdb_path: Path,
    chain_id: str = None,
    mode: str = 'complex',
    esm3_reference: Dict = None
) -> Dict:
    """
    提取完整的94维特征（42维物理化学 + 14维表面 + 26维结构几何 + 12维增强）

    Args:
        pdb_path: PDB文件路径
        chain_id: 链ID（仅在mode='chain'时使用）
        mode: 'chain' = 单链模式, 'complex' = 多链模式
        esm3_reference: ESM3特征字典（用于对齐），包含residue_index, chain_id等

    Returns:
        包含94维特征的字典，确保与ESM3对齐
    """
    print(f"\n处理: {pdb_path.name}")

    # 1. 解析PDB文件
    pdb_info = parse_pdb_protein(pdb_path, chain_id, mode)

    sequence = pdb_info['sequence']
    residue_names = pdb_info['residue_names']
    residue_indices = pdb_info['residue_indices']
    chain_ids = pdb_info['chain_ids']
    ca_coords = pdb_info['ca_coords']
    n_residues = pdb_info['n_residues']

    print(f"  解析PDB: {n_residues} 个残基")

    # 2. 如果提供了ESM3参考，进行对齐检查
    if esm3_reference is not None:
        print(f"  对齐检查: ESM3有 {len(esm3_reference['residue_index'])} 个残基")

        # 检查残基数量是否匹配
        if n_residues != len(esm3_reference['residue_index']):
            warnings.warn(
                f"残基数量不匹配: PDB={n_residues}, ESM3={len(esm3_reference['residue_index'])}"
            )

        # 检查序列是否匹配
        esm3_seq = esm3_reference.get('sequence', '')
        if esm3_seq and sequence != esm3_seq:
            warnings.warn(f"序列不匹配:\n  PDB: {sequence[:50]}...\n  ESM3: {esm3_seq[:50]}...")

    # 3. 提取物理化学特征（42维）
    print(f"  提取物理化学特征...")
    physchem_features = extract_physicochemical_features(
        pdb_path, sequence, residue_names, residue_indices, chain_ids
    )

    # 4. 提取表面特征（14维）
    print(f"  提取表面特征...")
    surface_feats = extract_surface_features(ca_coords, residue_names)

    # 5. 提取结构几何特征（26维）
    print(f"  提取结构几何特征...")
    geom_features = extract_structural_geometry_features(
        pdb_path, ca_coords, residue_indices, chain_ids
    )

    # 6. 提取增强特征V2（12维）
    print(f"  提取增强特征V2...")

    # 6.1 显式几何"口袋度"特征 (3维)
    pocketness_feats = compute_pocketness_features(ca_coords)

    # 6.2 柔性与置信度特征 (3维)
    # 需要SASA值（从surface_feats中提取第一列）
    sasa_values = surface_feats[:, 0] if surface_feats.shape[1] > 0 else np.zeros(n_residues)
    flexibility_feats = compute_flexibility_features(str(pdb_path), ca_coords, sasa_values)

    # 6.3 平滑化环境物理化学特征 (6维)
    env_feats = compute_environmental_features(ca_coords, physchem_features, residue_names)

    # 7. 拼接所有特征: 42 + 14 + 26 + 3 + 3 + 6 = 94
    complete_features_v2 = np.concatenate([
        physchem_features,   # [N, 42]
        surface_feats,       # [N, 14]
        geom_features,       # [N, 26]
        pocketness_feats,    # [N, 3]
        flexibility_feats,   # [N, 3]
        env_feats            # [N, 6]
    ], axis=1)

    print(f"  ✅ 完成: {complete_features_v2.shape} (应为 [N, 94])")

    # 8. 构建输出字典（包含对齐信息）
    output = {
        # 主要特征
        'complete_features_v2': complete_features_v2,  # [N_residues, 94]

        # 分组特征（用于分析）
        'physicochemical_features': physchem_features,  # [N_residues, 42]
        'surface_features': surface_feats,              # [N_residues, 14]
        'structural_geometry_features': geom_features,  # [N_residues, 26]
        'pocketness_features': pocketness_feats,        # [N_residues, 3]
        'flexibility_features': flexibility_feats,      # [N_residues, 3]
        'environmental_features': env_feats,            # [N_residues, 6]

        # 对齐信息（关键！）
        'residue_index': residue_indices,  # List[int]: 残基索引
        'chain_id': chain_ids,             # List[str]: 链ID
        'residue_name_3': residue_names,   # List[str]: 三字母残基名
        'residue_name_1': list(sequence),  # List[str]: 单字母残基名
        'sequence': sequence,              # str: 完整序列

        # 坐标信息
        'ca_coords': ca_coords,            # [N_residues, 3]
        'all_atom_coords': pdb_info['all_atom_coords'],  # List[np.ndarray]

        # 元数据
        'n_residues': n_residues,
        'mode': mode,
        'pdb_path': str(pdb_path),
        'feature_version': 'v2',
        'feature_dim': 94
    }

    return output


def align_with_esm3(
    protein_features: Dict,
    esm3_features: Dict
) -> Dict:
    """
    将蛋白质特征与ESM3特征对齐

    Args:
        protein_features: 从PDB提取的特征
        esm3_features: ESM3 latent特征

    Returns:
        对齐后的特征字典
    """
    # 检查残基索引和链ID是否匹配
    pdb_residues = list(zip(protein_features['chain_id'], protein_features['residue_index']))
    esm3_residues = list(zip(esm3_features['chain_id'], esm3_features['residue_index']))

    if pdb_residues != esm3_residues:
        warnings.warn("残基索引不完全匹配，尝试对齐...")

        # 创建映射
        esm3_to_pdb = {}
        for i, res in enumerate(esm3_residues):
            if res in pdb_residues:
                pdb_idx = pdb_residues.index(res)
                esm3_to_pdb[i] = pdb_idx

        # 重新排列PDB特征以匹配ESM3顺序
        n_esm3 = len(esm3_residues)
        feature_dim = protein_features['complete_features_v2'].shape[1]

        aligned_features = np.zeros((n_esm3, feature_dim), dtype=np.float32)
        aligned_ca_coords = np.zeros((n_esm3, 3), dtype=np.float32)

        for esm3_idx, pdb_idx in esm3_to_pdb.items():
            aligned_features[esm3_idx] = protein_features['complete_features_v2'][pdb_idx]
            aligned_ca_coords[esm3_idx] = protein_features['ca_coords'][pdb_idx]

        protein_features['complete_features_v2'] = aligned_features
        protein_features['ca_coords'] = aligned_ca_coords
        protein_features['n_residues'] = n_esm3
        protein_features['aligned'] = True

        print(f"  ✅ 对齐完成: {len(esm3_to_pdb)}/{n_esm3} 个残基匹配")
    else:
        protein_features['aligned'] = True
        print(f"  ✅ 残基完全匹配，无需对齐")

    return protein_features


def process_single_protein(
    pdb_file: Path,
    output_dir: Path,
    mode: str,
    overwrite: bool,
    esm3_dir: Path = None
) -> Tuple[bool, str]:
    """
    处理单个蛋白质文件

    Args:
        pdb_file: PDB文件路径
        output_dir: 输出目录
        mode: 'chain' 或 'complex'
        overwrite: 是否覆盖已存在的文件
        esm3_dir: ESM3特征目录（用于对齐）

    Returns:
        (success, message)
    """
    try:
        # 构建输出路径
        rel_path = pdb_file.relative_to(pdb_file.parent.parent.parent)
        out_path = output_dir / rel_path.parent / f"{pdb_file.stem}_protein_v2.pt"

        # 检查是否已存在
        if out_path.exists() and not overwrite:
            return (True, f"SKIP {pdb_file.stem}")

        # 加载ESM3特征（如果提供）
        esm3_reference = None
        if esm3_dir is not None:
            esm3_file = esm3_dir / rel_path.parent / f"{pdb_file.stem}_protein.pt"
            if esm3_file.exists():
                esm3_reference = torch.load(esm3_file, weights_only=False)

        # 提取特征
        features = extract_complete_features_v2(
            pdb_file,
            chain_id=None,
            mode=mode,
            esm3_reference=esm3_reference
        )

        # 如果有ESM3参考，进行对齐
        if esm3_reference is not None:
            features = align_with_esm3(features, esm3_reference)

        # 转换为torch张量并保存
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'complete_features_v2': torch.from_numpy(features['complete_features_v2']),
            'physicochemical_features': torch.from_numpy(features['physicochemical_features']),
            'surface_features': torch.from_numpy(features['surface_features']),
            'structural_geometry_features': torch.from_numpy(features['structural_geometry_features']),
            'pocketness_features': torch.from_numpy(features['pocketness_features']),
            'flexibility_features': torch.from_numpy(features['flexibility_features']),
            'environmental_features': torch.from_numpy(features['environmental_features']),
            'ca_coords': torch.from_numpy(features['ca_coords']),
            'residue_index': features['residue_index'],
            'chain_id': features['chain_id'],
            'residue_name_3': features['residue_name_3'],
            'residue_name_1': features['residue_name_1'],
            'sequence': features['sequence'],
            'n_residues': features['n_residues'],
            'mode': features['mode'],
            'pdb_path': features['pdb_path'],
            'feature_version': features['feature_version'],
            'feature_dim': features['feature_dim'],
            'aligned': features.get('aligned', False)
        }

        torch.save(data, out_path)

        return (True, f"{pdb_file.stem} ({features['n_residues']} residues, 94-dim)")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{pdb_file.stem}: {str(e)[:100]}\n{error_msg[:200]}")


def batch_extract_complete_features_v2(
    pdb_root: Path,
    output_dir: Path,
    pattern: str = "*_protein.pdb",
    mode: str = 'complex',
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None,
    esm3_dir: Path = None
) -> None:
    """
    批量提取完整特征V2

    Args:
        pdb_root: PDB文件根目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        mode: 'chain' 或 'complex'
        max_files: 最大处理文件数
        overwrite: 是否覆盖已存在的文件
        n_workers: 并行worker数量
        esm3_dir: ESM3特征目录（用于对齐）
    """
    # 查找所有PDB文件
    pdb_files = sorted(pdb_root.rglob(pattern))

    if max_files:
        pdb_files = pdb_files[:max_files]

    print(f"找到 {len(pdb_files)} 个PDB文件")
    print(f"输出目录: {output_dir}")
    print(f"模式: {mode}")
    print(f"特征维度: 94 (42+14+26+12)")
    if esm3_dir:
        print(f"ESM3参考目录: {esm3_dir}")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    print(f"使用 {n_workers} 个worker进程\n")

    # 并行处理
    from functools import partial
    process_func = partial(
        process_single_protein,
        output_dir=output_dir,
        mode=mode,
        overwrite=overwrite,
        esm3_dir=esm3_dir
    )

    success_count = 0
    fail_count = 0
    skip_count = 0

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(process_func, pdb_files),
                    total=len(pdb_files),
                    desc="提取特征"
                ))
            else:
                results = pool.map(process_func, pdb_files)
    else:
        results = []
        iterator = tqdm(pdb_files, desc="提取特征") if TQDM_AVAILABLE else pdb_files
        for pdb_file in iterator:
            results.append(process_func(pdb_file))

    # 统计结果
    for success, message in results:
        if success:
            if "SKIP" in message:
                skip_count += 1
            else:
                success_count += 1
                print(f"✅ {message}")
        else:
            fail_count += 1
            print(f"❌ {message}")

    print(f"\n{'='*70}")
    print(f"完成统计:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过: {skip_count}")
    print(f"  总计: {len(pdb_files)}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="提取完整蛋白质特征V2 (94维)")
    parser.add_argument('--input', type=Path, required=True, help='输入PDB文件或目录')
    parser.add_argument('--output', type=Path, required=True, help='输出目录')
    parser.add_argument('--pattern', type=str, default='*_protein.pdb', help='文件匹配模式')
    parser.add_argument('--mode', type=str, default='complex', choices=['chain', 'complex'],
                        help='处理模式: chain=单链, complex=多链')
    parser.add_argument('--max', type=int, default=None, help='最大处理文件数')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
    parser.add_argument('--workers', type=int, default=None, help='并行worker数量')
    parser.add_argument('--esm3-dir', type=Path, default=None, help='ESM3特征目录（用于对齐）')

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    if input_path.is_file():
        # 单文件处理
        print(f"处理单个文件: {input_path}")

        # 加载ESM3参考（如果提供）
        esm3_reference = None
        if args.esm3_dir:
            esm3_file = args.esm3_dir / f"{input_path.stem}_protein.pt"
            if esm3_file.exists():
                esm3_reference = torch.load(esm3_file, weights_only=False)

        data = extract_complete_features_v2(
            input_path,
            None,
            args.mode,
            esm3_reference
        )

        # 保存
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{input_path.stem}_protein_v2.pt"

        torch_data = {
            'complete_features_v2': torch.from_numpy(data['complete_features_v2']),
            'physicochemical_features': torch.from_numpy(data['physicochemical_features']),
            'surface_features': torch.from_numpy(data['surface_features']),
            'structural_geometry_features': torch.from_numpy(data['structural_geometry_features']),
            'pocketness_features': torch.from_numpy(data['pocketness_features']),
            'flexibility_features': torch.from_numpy(data['flexibility_features']),
            'environmental_features': torch.from_numpy(data['environmental_features']),
            'ca_coords': torch.from_numpy(data['ca_coords']),
            'residue_index': data['residue_index'],
            'chain_id': data['chain_id'],
            'residue_name_3': data['residue_name_3'],
            'residue_name_1': data['residue_name_1'],
            'sequence': data['sequence'],
            'n_residues': data['n_residues'],
            'mode': data['mode'],
            'pdb_path': data['pdb_path'],
            'feature_version': data['feature_version'],
            'feature_dim': data['feature_dim']
        }

        torch.save(torch_data, out_file)
        print(f"保存: {out_file}")
        print(f"特征: {data['complete_features_v2'].shape}")

    else:
        # 目录批处理
        print(f"处理目录: {input_path}")
        batch_extract_complete_features_v2(
            pdb_root=input_path,
            output_dir=output_path,
            pattern=args.pattern,
            mode=args.mode,
            max_files=args.max,
            overwrite=args.overwrite,
            n_workers=args.workers,
            esm3_dir=args.esm3_dir
        )


if __name__ == "__main__":
    main()
