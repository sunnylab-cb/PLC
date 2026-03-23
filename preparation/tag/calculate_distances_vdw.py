#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于范德华半径的蛋白质-配体距离计算 (BioLiP标准) - PLK版本

使用自适应物理阈值而非固定的5.5/8.0 Å阈值：
    d_min < r_vdw(P) + r_vdw(L) + 0.5 Å

这种方法考虑了原子的实际大小，更科学地定义结合位点。

改进：
1. 使用实验原始坐标（从PDB/SDF文件直接提取）
2. 过滤氢原子（与训练时一致）
3. 基于VDW半径的物理阈值
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import multiprocessing as mp
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

# 范德华半径查找表 (Bondi半径, 单位: Å)
VDW_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'S': 1.80,
    'P': 1.80,
    'F': 1.47,
    'Cl': 1.75,
    'Br': 1.85,
    'I': 1.98,
    'Se': 1.90,
    'B': 1.92,
    'Si': 2.10,
    'As': 1.85,
    'default': 1.70  # 未知原子类型的默认值
}

# 容差值 (Å)
TOLERANCE = 0.5


def get_vdw_radius(atom_symbol: str) -> float:
    """
    获取原子的范德华半径

    Args:
        atom_symbol: 原子符号 (如 'C', 'N', 'O')

    Returns:
        范德华半径 (Å)
    """
    # 移除可能的数字和特殊字符，只保留元素符号
    clean_symbol = ''.join(c for c in atom_symbol if c.isalpha())

    # 首字母大写，其余小写
    if len(clean_symbol) > 0:
        clean_symbol = clean_symbol[0].upper() + clean_symbol[1:].lower()

    return VDW_RADII.get(clean_symbol, VDW_RADII['default'])


def extract_protein_atoms_from_pdb(protein_pdb: str) -> Tuple[np.ndarray, List[str]]:
    """
    从蛋白质PDB文件提取所有重原子坐标和元素符号

    Args:
        protein_pdb: 蛋白质PDB文件路径

    Returns:
        coords: [N_residues, max_atoms, 3] 每个残基的重原子坐标
        atom_symbols: List[List[str]] 每个残基的原子符号
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)

    if len(structure) == 0:
        return [], []

    model = structure[0]  # 只处理第一个模型

    residue_coords = []
    residue_symbols = []

    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':  # 跳过HETATM
                continue

            # 获取所有重原子
            coords = []
            symbols = []
            for atom in residue.get_atoms():
                if atom.element != 'H':  # 过滤氢原子
                    coords.append(atom.get_coord())
                    symbols.append(atom.element)

            if len(coords) > 0:
                residue_coords.append(np.array(coords))
                residue_symbols.append(symbols)

    return residue_coords, residue_symbols


def extract_ligand_coords_from_pdb(ligand_pdb: str) -> Tuple[np.ndarray, List[str]]:
    """
    从配体PDB文件提取坐标和元素符号（过滤氢原子）

    Args:
        ligand_pdb: 配体PDB文件路径

    Returns:
        coords: [M, 3] 配体重原子坐标
        symbols: List[str] 原子符号
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', ligand_pdb)

    coords = []
    symbols = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue.get_atoms():
                    if atom.element != 'H':  # 过滤氢原子
                        coords.append(atom.get_coord())
                        symbols.append(atom.element)

    return np.array(coords), symbols


def extract_ligand_from_pt(ligand_pt: Path) -> Tuple[np.ndarray, List[str]]:
    """
    从.pt特征文件提取配体坐标和原子符号（过滤氢原子）

    Args:
        ligand_pt: 配体特征文件路径

    Returns:
        coords: [M, 3] 配体重原子坐标
        symbols: List[str] 原子符号
    """
    data = torch.load(ligand_pt, weights_only=False)

    # 提取坐标
    coords = data['coords'].numpy()

    # 提取原子符号
    if 'atom_symbols' in data:
        symbols = data['atom_symbols']
    else:
        # 从atom_features推断（前11维是原子类型的one-hot编码）
        atom_features = data['atom_features']
        atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Other']

        symbols = []
        for feat in atom_features:
            atom_type_onehot = feat[:11]
            idx = torch.argmax(atom_type_onehot).item()
            symbols.append(atom_types[idx])

    # 过滤氢原子
    heavy_mask = np.array([s != 'H' for s in symbols])
    coords = coords[heavy_mask]
    symbols = [s for s, m in zip(symbols, heavy_mask) if m]

    return coords, symbols


def calculate_vdw_based_distances(
    protein_coords: List[np.ndarray],
    protein_symbols: List[List[str]],
    ligand_coords: np.ndarray,
    ligand_symbols: List[str],
    tolerance: float = TOLERANCE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于范德华半径计算蛋白质-配体距离

    Args:
        protein_coords: List of [n_atoms, 3] 每个残基的重原子坐标
        protein_symbols: List of List[str] 每个残基的原子符号
        ligand_coords: [M, 3] 配体重原子坐标
        ligand_symbols: List[str] 配体原子符号
        tolerance: 容差值 (Å)

    Returns:
        min_distances: [N_res] 每个残基到配体的最小欧几里得距离 (Å)
        min_vdw_distances: [N_res] 每个残基到配体的最小VDW距离
            VDW距离 = distance - (r_vdw_protein + r_vdw_ligand)
            - 负值: 原子重叠/穿透 (强相互作用，如氢键、盐桥)
            - 0附近: 恰好VDW接触
            - 正值: 有间隙 (弱相互作用或无相互作用)
        is_binding: [N_res] 是否为结合位点
            判断标准: distance < (r_vdw_protein + r_vdw_ligand + tolerance)
            即 VDW距离 < tolerance (0.5Å)
    """
    n_residues = len(protein_coords)
    n_ligand_atoms = len(ligand_coords)

    min_distances = np.full(n_residues, np.inf, dtype=np.float32)
    min_vdw_distances = np.full(n_residues, np.inf, dtype=np.float32)
    is_binding = np.zeros(n_residues, dtype=bool)

    # 对每个残基
    for i in range(n_residues):
        residue_coords = protein_coords[i]
        residue_symbols = protein_symbols[i] if i < len(protein_symbols) else []

        if residue_coords is None or len(residue_coords) == 0:
            continue

        # 对残基的每个原子
        for atom_idx, p_coord in enumerate(residue_coords):
            # 获取原子符号
            if atom_idx < len(residue_symbols):
                p_symbol = residue_symbols[atom_idx]
            else:
                p_symbol = 'C'

            p_vdw = get_vdw_radius(p_symbol)

            # 计算到所有配体原子的距离
            for j in range(n_ligand_atoms):
                l_coord = ligand_coords[j]
                l_symbol = ligand_symbols[j] if j < len(ligand_symbols) else 'C'
                l_vdw = get_vdw_radius(l_symbol)

                # 欧几里得距离
                dist = np.linalg.norm(p_coord - l_coord)

                # VDW半径和 (不含tolerance)
                vdw_radii_sum = p_vdw + l_vdw

                # VDW距离 = 实际距离 - VDW半径和
                # 负值: 穿透/重叠 (强相互作用)
                # 0附近: 恰好VDW接触
                # 正值: 有间隙
                vdw_dist = dist - vdw_radii_sum

                # 更新最小欧几里得距离
                if dist < min_distances[i]:
                    min_distances[i] = dist

                # 更新最小VDW距离
                if vdw_dist < min_vdw_distances[i]:
                    min_vdw_distances[i] = vdw_dist

                # 结合位点判断: 使用tolerance作为宽松阈值
                # is_binding = True 当 distance < (r_vdw_P + r_vdw_L + tolerance)
                vdw_threshold_with_tolerance = vdw_radii_sum + tolerance
                if dist < vdw_threshold_with_tolerance:
                    is_binding[i] = True

    return min_distances, min_vdw_distances, is_binding


def process_single_complex(args_tuple) -> Tuple[bool, str, Optional[Dict]]:
    """
    处理单个蛋白质-配体复合物

    Returns:
        (success, message, result_dict)
    """
    complex_dir, year_dir, data_root_path = args_tuple
    complex_id = complex_dir.name

    try:
        # 查找蛋白和配体文件
        protein_pdb = complex_dir / f'{complex_id}_protein.pdb'
        ligand_pdb = complex_dir / f'{complex_id}_ligand.pdb'

        # 尝试其他配体格式
        if not ligand_pdb.exists():
            ligand_pdb = complex_dir / f'{complex_id}_ligand.sdf'
        if not ligand_pdb.exists():
            ligand_pdb = complex_dir / f'{complex_id}_ligand.mol2'

        if not protein_pdb.exists() or not ligand_pdb.exists():
            return (False, f"{complex_id}: 文件不存在", None)

        # 提取蛋白质坐标和原子符号
        protein_coords, protein_symbols = extract_protein_atoms_from_pdb(str(protein_pdb))

        if len(protein_coords) == 0:
            return (False, f"{complex_id}: 无法提取蛋白质坐标", None)

        # 提取配体坐标和原子符号
        if ligand_pdb.suffix == '.pdb':
            ligand_coords, ligand_symbols = extract_ligand_coords_from_pdb(str(ligand_pdb))
        else:
            # 对于SDF/MOL2，从.pt文件读取（但要过滤氢原子）
            # 尝试多个可能的路径
            possible_paths = [
                data_root_path / 'PLK' / 'ligand_features_v3' / year_dir / complex_id / f'{complex_id}_ligand.pt',
                data_root_path / 'PLK' / 'ligand_features' / year_dir / complex_id / f'{complex_id}_ligand.pt',
            ]

            ligand_feat_file = None
            for path in possible_paths:
                if path.exists():
                    ligand_feat_file = path
                    break

            if ligand_feat_file:
                ligand_coords, ligand_symbols = extract_ligand_from_pt(ligand_feat_file)
            else:
                return (False, f"{complex_id}: 配体特征文件不存在", None)

        if len(ligand_coords) == 0:
            return (False, f"{complex_id}: 无法提取配体坐标", None)

        # 计算基于VDW的距离
        min_distances, min_vdw_distances, is_binding = calculate_vdw_based_distances(
            protein_coords,
            protein_symbols,
            ligand_coords,
            ligand_symbols
        )

        # 统计
        n_binding = is_binding.sum()
        n_total = len(is_binding)

        result = {
            'pdb_id': complex_id,
            'distances': torch.from_numpy(min_distances),  # 欧几里得距离
            'vdw_distances': torch.from_numpy(min_vdw_distances),  # VDW距离 (distance - threshold)
            'is_binding_vdw': torch.from_numpy(is_binding),  # 布尔标记
            'n_binding': int(n_binding),
            'n_total': int(n_total),
            'binding_ratio': float(n_binding / n_total) if n_total > 0 else 0.0
        }

        return (True, f"{complex_id}: {n_binding}/{n_total} 结合残基 ({result['binding_ratio']:.1%})", result)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{complex_id}: {str(e)[:100]}", None)


def batch_calculate_distances(
    data_root: str,
    output_file: str,
    num_workers: int = 64
):
    """
    批量计算所有复合物的残基-配体距离 (多进程版)
    """
    data_root_path = Path(data_root)
    pl_dir = data_root_path / 'P-L'

    distances_dict = {}
    tasks = []

    # 1. 收集所有任务
    print("收集任务...")
    for year_dir in ['1981-2000', '2001-2010', '2011-2019']:
        year_path = pl_dir / year_dir
        if not year_path.exists():
            continue

        for complex_dir in year_path.iterdir():
            if not complex_dir.is_dir():
                continue
            tasks.append((complex_dir, year_dir, data_root_path))

    print(f"总共有 {len(tasks)} 个复合物待处理")
    print(f"使用 {num_workers} 个进程进行计算...")
    print(f"使用 BioLiP 标准: d < r_vdw(P) + r_vdw(L) + {TOLERANCE} Å")

    # 2. 并行处理
    success_count = 0
    fail_count = 0

    if num_workers == 1:
        # 单进程
        for task in tqdm(tasks, desc="计算距离"):
            success, message, result = process_single_complex(task)
            if success and result:
                distances_dict[result['pdb_id']] = result
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 10:
                    print(f"  ✗ {message}")
    else:
        # 多进程
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_complex, tasks),
                total=len(tasks),
                desc="计算距离"
            ))

            for success, message, result in results:
                if success and result:
                    distances_dict[result['pdb_id']] = result
                    success_count += 1
                else:
                    fail_count += 1
                    if fail_count <= 10:
                        print(f"  ✗ {message}")

    # 保存结果
    torch.save(distances_dict, output_file)

    print(f"\n=== 总结 ===")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(tasks)}")
    print(f"\n结果已保存到: {output_file}")

    # 统计信息
    if distances_dict:
        binding_ratios = [r['binding_ratio'] for r in distances_dict.values()]
        print(f"\n结合位点比例统计:")
        print(f"  平均: {np.mean(binding_ratios):.1%}")
        print(f"  中位数: {np.median(binding_ratios):.1%}")
        print(f"  最小: {np.min(binding_ratios):.1%}")
        print(f"  最大: {np.max(binding_ratios):.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="基于范德华半径计算蛋白质-配体距离 (BioLiP标准) - PLK版本"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/husrcf/Code/ProtBind",
        help="数据根目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/husrcf/Code/ProtBind/PLC/data/residue_ligand_distances_vdw.pt",
        help="输出文件路径"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="并行进程数"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    batch_calculate_distances(
        data_root=str(data_root),
        output_file=args.output,
        num_workers=args.workers
    )
