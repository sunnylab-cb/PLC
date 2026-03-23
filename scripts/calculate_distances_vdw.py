#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于范德华半径的蛋白质-配体距离计算 (BioLiP标准)

使用自适应物理阈值而非固定的3.5 Å阈值：
    d_min < r_vdw(P) + r_vdw(L) + 0.5 Å

这种方法考虑了原子的实际大小，更科学地定义结合位点。
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import multiprocessing as mp

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

# 默认特征目录（与 V8 训练保持一致）
DEFAULT_PROTEIN_FEATURES_DIR = 'complete_features_v3'
DEFAULT_LIGAND_FEATURES_DIR = 'ligand_features_v4'


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


def extract_atom_symbols_from_protein(protein_file: Path) -> List[str]:
    """
    从蛋白质特征文件中提取原子符号

    Args:
        protein_file: 蛋白质特征文件路径 (.pt)

    Returns:
        原子符号列表 (每个残基的重原子)
    """
    try:
        data = torch.load(protein_file, weights_only=False)

        # 尝试从不同字段获取原子符号
        if 'atom_symbols' in data:
            return data['atom_symbols']
        elif 'all_atom_symbols' in data:
            # all_atom_symbols 是列表的列表，每个残基一个列表
            atom_symbols = []
            for residue_atoms in data['all_atom_symbols']:
                if residue_atoms and len(residue_atoms) > 0:
                    # 使用每个残基的第一个重原子（通常是主链原子）
                    atom_symbols.append(residue_atoms[0])
                else:
                    atom_symbols.append('C')  # 默认为碳
            return atom_symbols
        else:
            # 如果没有原子符号信息，使用默认值
            n_residues = len(data.get('ca_coords', data.get('complete_features', [])))
            return ['C'] * n_residues  # 默认所有原子为碳

    except Exception as e:
        print(f"  ⚠️  无法提取原子符号 {protein_file.name}: {e}")
        return []


def extract_atom_symbols_from_ligand(ligand_file: Path) -> List[str]:
    """
    从配体特征文件中提取原子符号

    Args:
        ligand_file: 配体特征文件路径 (.pt)

    Returns:
        原子符号列表
    """
    try:
        data = torch.load(ligand_file, weights_only=False)

        if 'atom_symbols' in data:
            return data['atom_symbols']
        else:
            # 从atom_features推断（前11维是原子类型的one-hot编码）
            atom_features = data['atom_features']
            atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Other']

            atom_symbols = []
            for feat in atom_features:
                # 前11维是原子类型
                atom_type_onehot = feat[:11]
                idx = torch.argmax(atom_type_onehot).item()
                atom_symbols.append(atom_types[idx])

            return atom_symbols

    except Exception as e:
        print(f"  ⚠️  无法提取配体原子符号 {ligand_file.name}: {e}")
        return []


def calculate_vdw_based_distances(
    protein_all_atom_coords: List,
    protein_all_atom_names: List,
    ligand_coords: np.ndarray,
    ligand_atom_symbols: List[str],
    tolerance: float = TOLERANCE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于范德华半径计算蛋白质-配体距离 (使用所有原子)

    Args:
        protein_all_atom_coords: List of [n_atoms, 3] 每个残基的所有原子坐标
        protein_all_atom_names: List of atom names for each residue
        ligand_coords: [M, 3] 配体原子坐标
        ligand_atom_symbols: 配体原子符号列表
        tolerance: 容差值 (Å)

    Returns:
        min_distances: [N_res] 每个残基到配体的最小实际距离
        min_distances_vdw_adjusted: [N_res] 扣除VDW半径后的最小距离
        is_binding: [N_res] 是否为结合位点 (基于VDW阈值)
    """
    n_residues = len(protein_all_atom_coords)
    n_ligand_atoms = len(ligand_coords)

    min_distances = np.full(n_residues, np.inf, dtype=np.float32)
    min_distances_vdw_adjusted = np.full(n_residues, np.inf, dtype=np.float32)
    is_binding = np.zeros(n_residues, dtype=bool)

    # 对每个残基
    for i in range(n_residues):
        residue_coords = protein_all_atom_coords[i]
        residue_atom_names = protein_all_atom_names[i] if i < len(protein_all_atom_names) else []

        if residue_coords is None or len(residue_coords) == 0:
            continue

        # 转换为numpy数组
        if isinstance(residue_coords, torch.Tensor):
            residue_coords = residue_coords.numpy()

        # 对残基的每个原子
        for atom_idx, p_coord in enumerate(residue_coords):
            # 获取原子符号（从原子名称中提取元素符号）
            if atom_idx < len(residue_atom_names):
                atom_name = residue_atom_names[atom_idx]
                # 原子名称通常以元素符号开头，如 'CA', 'CB', 'N', 'O'
                p_symbol = ''.join(c for c in atom_name if c.isalpha())[:2]
            else:
                p_symbol = 'C'

            p_vdw = get_vdw_radius(p_symbol)

            # 计算到所有配体原子的距离
            for j in range(n_ligand_atoms):
                l_coord = ligand_coords[j]
                l_symbol = ligand_atom_symbols[j] if j < len(ligand_atom_symbols) else 'C'
                l_vdw = get_vdw_radius(l_symbol)

                # 欧几里得距离
                dist = np.linalg.norm(p_coord - l_coord)

                # 扣除VDW半径后的距离
                dist_vdw_adjusted = dist - (p_vdw + l_vdw)

                # 更新最小实际距离
                if dist < min_distances[i]:
                    min_distances[i] = dist

                # 更新扣除VDW半径后的最小距离
                if dist_vdw_adjusted < min_distances_vdw_adjusted[i]:
                    min_distances_vdw_adjusted[i] = dist_vdw_adjusted

                # VDW阈值判断
                vdw_threshold = p_vdw + l_vdw + tolerance
                if dist < vdw_threshold:
                    is_binding[i] = True

    return min_distances, min_distances_vdw_adjusted, is_binding


def process_single_complex(args_tuple) -> Tuple[bool, str, Optional[Dict]]:
    """
    处理单个蛋白质-配体复合物

    Returns:
        (success, message, result_dict)
    """
    pdb_id, data_root, year_range, protein_features_dir, ligand_features_dir = args_tuple

    try:
        data_root = Path(data_root)

        # 文件路径
        protein_file = data_root / protein_features_dir / year_range / pdb_id / f'{pdb_id}_protein.pt'
        ligand_file = data_root / ligand_features_dir / year_range / pdb_id / f'{pdb_id}_ligand.pt'

        if not protein_file.exists() or not ligand_file.exists():
            return (False, f"{pdb_id}: 文件不存在", None)

        # 加载数据
        protein_data = torch.load(protein_file, weights_only=False)
        ligand_data = torch.load(ligand_file, weights_only=False)

        # 提取坐标 - 使用所有原子坐标而不是只用Cα
        protein_all_atom_coords = protein_data['all_atom_coords']  # List of [n_atoms, 3]
        protein_all_atom_names = protein_data.get('all_atom_names', [])  # List of atom names
        ligand_coords = ligand_data['coords'].numpy()  # [M, 3]

        # 提取配体原子符号
        ligand_symbols = extract_atom_symbols_from_ligand(ligand_file)

        # 确保符号列表长度匹配
        if len(ligand_symbols) < len(ligand_coords):
            ligand_symbols.extend(['C'] * (len(ligand_coords) - len(ligand_symbols)))

        # 计算基于VDW的距离（使用所有原子）
        min_distances, min_distances_vdw_adjusted, is_binding = calculate_vdw_based_distances(
            protein_all_atom_coords,
            protein_all_atom_names,
            ligand_coords,
            ligand_symbols
        )

        # 统计
        n_binding = is_binding.sum()
        n_total = len(is_binding)

        result = {
            'pdb_id': pdb_id,
            'distances': torch.from_numpy(min_distances),  # 实际距离
            'distances_vdw_adjusted': torch.from_numpy(min_distances_vdw_adjusted),  # 扣除VDW半径后的距离
            'is_binding_vdw': torch.from_numpy(is_binding),
            'n_binding': int(n_binding),
            'n_total': int(n_total),
            'binding_ratio': float(n_binding / n_total) if n_total > 0 else 0.0
        }

        return (True, f"{pdb_id}: {n_binding}/{n_total} 结合残基 ({result['binding_ratio']:.1%})", result)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{pdb_id}: {str(e)[:100]}", None)


def batch_calculate_distances(
    data_root: Path,
    output_file: Path,
    n_workers: int = None,
    max_complexes: int = None,
    specific_ids: list = None,
    protein_features_dir: str = DEFAULT_PROTEIN_FEATURES_DIR,
    ligand_features_dir: str = DEFAULT_LIGAND_FEATURES_DIR,
):
    """
    批量计算所有复合物的距离

    Args:
        data_root: 数据根目录
        output_file: 输出文件路径
        n_workers: 并行进程数
        max_complexes: 最大处理复合物数
        specific_ids: 指定要处理的PDB ID列表（如果提供，只处理这些ID）
    """
    # 收集所有复合物ID
    year_ranges = ['1981-2000', '2001-2010', '2011-2019']
    complex_list = []

    if specific_ids:
        # 只处理指定的ID
        print(f"处理指定的 {len(specific_ids)} 个复合物: {', '.join(specific_ids)}")
        for pdb_id in specific_ids:
            # 查找该ID所在的年份范围
            found = False
            for year_range in year_ranges:
                protein_file = data_root / protein_features_dir / year_range / pdb_id / f'{pdb_id}_protein.pt'
                if protein_file.exists():
                    complex_list.append((pdb_id, str(data_root), year_range, protein_features_dir, ligand_features_dir))
                    found = True
                    break
            if not found:
                print(f"警告: 未找到 {pdb_id}")
    else:
        # 处理所有复合物
        for year_range in year_ranges:
            protein_year_dir = data_root / protein_features_dir / year_range
            if not protein_year_dir.exists():
                continue

            for complex_dir in protein_year_dir.iterdir():
                if not complex_dir.is_dir():
                    continue

                pdb_id = complex_dir.name
                complex_list.append((pdb_id, str(data_root), year_range, protein_features_dir, ligand_features_dir))

        if max_complexes:
            complex_list = complex_list[:max_complexes]

    print(f"找到 {len(complex_list)} 个复合物")
    print(f"蛋白特征目录: data/{protein_features_dir}")
    print(f"配体特征目录: data/{ligand_features_dir}")

    # 确定进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(complex_list))

    print(f"使用 {n_workers} 个工作进程")
    print(f"使用 BioLiP 标准: d < r_vdw(P) + r_vdw(L) + {TOLERANCE} Å")

    # 处理
    results_dict = {}
    success_count = 0
    fail_count = 0

    if n_workers == 1:
        # 单进程
        for task in tqdm(complex_list, desc="计算距离"):
            success, message, result = process_single_complex(task)
            if success and result:
                results_dict[result['pdb_id']] = result
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 10:
                    print(f"  ✗ {message}")
    else:
        # 多进程
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_complex, complex_list),
                total=len(complex_list),
                desc="计算距离"
            ))

            for success, message, result in results:
                if success and result:
                    results_dict[result['pdb_id']] = result
                    success_count += 1
                else:
                    fail_count += 1
                    if fail_count <= 10:
                        print(f"  ✗ {message}")

    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results_dict, output_file)

    print(f"\n=== 总结 ===")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(complex_list)}")
    print(f"\n结果已保存到: {output_file}")

    # 统计信息
    if results_dict:
        binding_ratios = [r['binding_ratio'] for r in results_dict.values()]
        print(f"\n结合位点比例统计:")
        print(f"  平均: {np.mean(binding_ratios):.1%}")
        print(f"  中位数: {np.median(binding_ratios):.1%}")
        print(f"  最小: {np.min(binding_ratios):.1%}")
        print(f"  最大: {np.max(binding_ratios):.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="基于范德华半径计算蛋白质-配体距离 (BioLiP标准)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="数据根目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/residue_ligand_distances_vdw.pt",
        help="输出文件路径"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="最大处理复合物数（用于测试）"
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs='+',
        default=None,
        help="指定要处理的PDB ID列表（例如: --ids 1lvk 2pll 3vjs）"
    )
    parser.add_argument(
        "--protein-features-dir",
        type=str,
        default=DEFAULT_PROTEIN_FEATURES_DIR,
        help=f"蛋白特征目录名（默认: {DEFAULT_PROTEIN_FEATURES_DIR}）"
    )
    parser.add_argument(
        "--ligand-features-dir",
        type=str,
        default=DEFAULT_LIGAND_FEATURES_DIR,
        help=f"配体特征目录名（默认: {DEFAULT_LIGAND_FEATURES_DIR}）"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_file = Path(args.output)

    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    batch_calculate_distances(
        data_root=data_root,
        output_file=output_file,
        n_workers=args.workers,
        max_complexes=args.max,
        specific_ids=args.ids,
        protein_features_dir=args.protein_features_dir,
        ligand_features_dir=args.ligand_features_dir,
    )


if __name__ == "__main__":
    main()
