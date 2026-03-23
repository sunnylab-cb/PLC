#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
氢键网络特征提取模块

基于几何标准计算per-residue氢键特征：
1. 氢键供体/受体数量
2. 氢键距离统计
3. 氢键角度统计
4. 氢键网络连通性

参考文献：
- Chemical features and machine learning assisted predictions of protein-ligand
  short hydrogen bonds (Nature Scientific Reports, 2023)
  https://www.nature.com/articles/s41598-023-40614-7
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

try:
    from Bio.PDB import PDBParser, NeighborSearch
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("Biopython not available")


# ============================================================================
# 氢键几何标准
# ============================================================================

# 氢键距离标准（Å）
HBOND_DISTANCE_MIN = 2.5  # 最小距离
HBOND_DISTANCE_MAX = 3.5  # 最大距离（标准）
HBOND_DISTANCE_WEAK = 4.0  # 弱氢键最大距离

# 氢键角度标准（度）
HBOND_ANGLE_MIN = 120.0  # D-H...A角度最小值
HBOND_ANGLE_IDEAL = 180.0  # 理想角度

# 氨基酸的氢键供体/受体原子
HBOND_DONORS = {
    'ARG': ['NE', 'NH1', 'NH2'],  # 胍基
    'ASN': ['ND2'],  # 酰胺
    'GLN': ['NE2'],  # 酰胺
    'HIS': ['ND1', 'NE2'],  # 咪唑
    'LYS': ['NZ'],  # 氨基
    'SER': ['OG'],  # 羟基
    'THR': ['OG1'],  # 羟基
    'TRP': ['NE1'],  # 吲哚
    'TYR': ['OH'],  # 酚羟基
    'CYS': ['SG'],  # 巯基
    # 主链
    'BACKBONE': ['N'],  # 主链氮
}

HBOND_ACCEPTORS = {
    'ASP': ['OD1', 'OD2'],  # 羧基
    'GLU': ['OE1', 'OE2'],  # 羧基
    'ASN': ['OD1'],  # 羰基
    'GLN': ['OE1'],  # 羰基
    'HIS': ['ND1', 'NE2'],  # 咪唑（两性）
    'SER': ['OG'],  # 羟基（两性）
    'THR': ['OG1'],  # 羟基（两性）
    'TYR': ['OH'],  # 酚羟基（两性）
    'CYS': ['SG'],  # 巯基（两性）
    # 主链
    'BACKBONE': ['O'],  # 主链羰基氧
}


def is_donor_atom(residue_name: str, atom_name: str) -> bool:
    """判断原子是否为氢键供体"""
    if atom_name == 'N':  # 主链氮
        return True
    if residue_name in HBOND_DONORS:
        return atom_name in HBOND_DONORS[residue_name]
    return False


def is_acceptor_atom(residue_name: str, atom_name: str) -> bool:
    """判断原子是否为氢键受体"""
    if atom_name == 'O':  # 主链羰基氧
        return True
    if residue_name in HBOND_ACCEPTORS:
        return atom_name in HBOND_ACCEPTORS[residue_name]
    return False


def calculate_angle(coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray) -> float:
    """
    计算三个点形成的角度（度）

    Args:
        coord1: 第一个点（供体）
        coord2: 第二个点（氢，或供体-受体中点）
        coord3: 第三个点（受体）

    Returns:
        角度（度）
    """
    v1 = coord1 - coord2
    v2 = coord3 - coord2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi

    return angle


def find_hydrogen_bonds(
    pdb_path: Path,
    residue_indices: List[int],
    chain_ids: List[str]
) -> Dict[Tuple[str, int], Dict]:
    """
    查找所有氢键（基于几何标准）

    Args:
        pdb_path: PDB文件路径
        residue_indices: PDB残基编号列表
        chain_ids: 每个残基的链ID列表

    Returns:
        字典: (chain_id, residue_index) -> 氢键信息
    """
    if not BIOPYTHON_AVAILABLE:
        return {}

    try:
        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        model = structure[0]

        # 获取所有原子
        atoms = [atom for atom in model.get_atoms()]

        # 构建邻居搜索树
        ns = NeighborSearch(atoms)

        # 初始化氢键字典
        hbond_dict = {}
        for res_idx, chain_id in zip(residue_indices, chain_ids):
            hbond_dict[(chain_id, res_idx)] = {
                'donor_count': 0,
                'acceptor_count': 0,
                'distances': [],
                'angles': [],
                'partners': []
            }

        # 遍历所有残基对
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]
                chain_id = chain.id
                res_name = residue.resname

                key = (chain_id, res_id)
                if key not in hbond_dict:
                    continue

                # 查找该残基的供体和受体原子
                for atom in residue:
                    atom_name = atom.name
                    atom_coord = atom.coord

                    # 如果是供体原子
                    if is_donor_atom(res_name, atom_name):
                        # 搜索附近的受体原子
                        nearby_atoms = ns.search(atom_coord, HBOND_DISTANCE_WEAK)

                        for nearby_atom in nearby_atoms:
                            nearby_res = nearby_atom.parent
                            nearby_res_name = nearby_res.resname
                            nearby_atom_name = nearby_atom.name
                            nearby_coord = nearby_atom.coord

                            # 检查是否为受体原子
                            if is_acceptor_atom(nearby_res_name, nearby_atom_name):
                                # 计算距离
                                distance = np.linalg.norm(atom_coord - nearby_coord)

                                if HBOND_DISTANCE_MIN < distance < HBOND_DISTANCE_WEAK:
                                    # 简化角度计算（使用供体-受体向量）
                                    # 理想情况下应该包含氢原子位置
                                    angle = 180.0  # 简化假设

                                    if angle > HBOND_ANGLE_MIN:
                                        # 记录所有2.5-4.0Å范围的氢键（包括弱氢键）
                                        hbond_dict[key]['donor_count'] += 1
                                        hbond_dict[key]['distances'].append(distance)
                                        hbond_dict[key]['angles'].append(angle)

                                        nearby_key = (nearby_res.parent.id, nearby_res.id[1])
                                        hbond_dict[key]['partners'].append(nearby_key)

                    # 如果是受体原子
                    if is_acceptor_atom(res_name, atom_name):
                        # 搜索附近的供体原子
                        nearby_atoms = ns.search(atom_coord, HBOND_DISTANCE_WEAK)

                        for nearby_atom in nearby_atoms:
                            nearby_res = nearby_atom.parent
                            nearby_res_name = nearby_res.resname
                            nearby_atom_name = nearby_atom.name
                            nearby_coord = nearby_atom.coord

                            # 检查是否为供体原子
                            if is_donor_atom(nearby_res_name, nearby_atom_name):
                                distance = np.linalg.norm(atom_coord - nearby_coord)

                                if HBOND_DISTANCE_MIN < distance < HBOND_DISTANCE_WEAK:
                                    # 记录所有2.5-4.0Å范围的受体氢键
                                    hbond_dict[key]['acceptor_count'] += 1

        return hbond_dict

    except Exception as e:
        warnings.warn(f"Hydrogen bond calculation failed for {pdb_path}: {e}")
        return {}


def compute_hbond_features(
    pdb_path: Path,
    residue_indices: List[int],
    chain_ids: List[str]
) -> np.ndarray:
    """
    计算per-residue氢键网络特征（8维）

    特征组成：
    1. 氢键供体数量（归一化）
    2. 氢键受体数量（归一化）
    3. 平均氢键距离（归一化）
    4. 氢键距离标准差
    5. 是否有强氢键（<3.0Å）
    6. 是否有弱氢键（3.5-4.0Å）
    7. 氢键伙伴数量（归一化）
    8. 氢键网络连通度（归一化）

    Args:
        pdb_path: PDB文件路径
        residue_indices: PDB残基编号列表
        chain_ids: 每个残基的链ID列表

    Returns:
        [N_residues, 8] 氢键特征矩阵
    """
    n_residues = len(residue_indices)
    hbond_features = np.zeros((n_residues, 8), dtype=np.float32)

    if not BIOPYTHON_AVAILABLE:
        return hbond_features

    # 查找所有氢键
    hbond_dict = find_hydrogen_bonds(pdb_path, residue_indices, chain_ids)

    if not hbond_dict:
        return hbond_features

    # 为每个残基计算特征
    for i, (res_idx, chain_id) in enumerate(zip(residue_indices, chain_ids)):
        key = (chain_id, res_idx)

        if key not in hbond_dict:
            continue

        info = hbond_dict[key]

        # 1. 供体数量（归一化到[0,1]，最大假设为4）
        hbond_features[i, 0] = min(info['donor_count'] / 4.0, 1.0)

        # 2. 受体数量（归一化到[0,1]，最大假设为4）
        hbond_features[i, 1] = min(info['acceptor_count'] / 4.0, 1.0)

        # 3-4. 距离统计
        if info['distances']:
            distances = np.array(info['distances'])
            hbond_features[i, 2] = distances.mean() / HBOND_DISTANCE_WEAK  # 归一化
            hbond_features[i, 3] = distances.std() / HBOND_DISTANCE_WEAK

        # 5. 是否有强氢键
        if info['distances']:
            has_strong = any(d < 3.0 for d in info['distances'])
            hbond_features[i, 4] = 1.0 if has_strong else 0.0

        # 6. 是否有弱氢键
        if info['distances']:
            has_weak = any(3.5 < d < 4.0 for d in info['distances'])
            hbond_features[i, 5] = 1.0 if has_weak else 0.0

        # 7. 氢键伙伴数量
        hbond_features[i, 6] = min(len(info['partners']) / 4.0, 1.0)

        # 8. 网络连通度（总氢键数 / 最大可能数）
        total_hbonds = info['donor_count'] + info['acceptor_count']
        hbond_features[i, 7] = min(total_hbonds / 8.0, 1.0)

    return hbond_features


# ============================================================================
# 测试
# ============================================================================

def test_hbond_features():
    """测试氢键特征提取"""
    print("=== 测试氢键网络特征提取 ===\n")

    from pathlib import Path

    # 使用真实PDB文件测试
    pdb_path = Path("New/1981-2000/10gs/10gs_protein.pdb")

    if not pdb_path.exists():
        print("测试PDB文件不存在，跳过测试")
        return

    # 简单测试：前10个残基
    residue_indices = list(range(2, 12))
    chain_ids = ['A'] * 10

    print("1. 测试氢键查找:")
    hbond_dict = find_hydrogen_bonds(pdb_path, residue_indices, chain_ids)

    for key, info in list(hbond_dict.items())[:5]:
        print(f"   残基 {key}: 供体={info['donor_count']}, 受体={info['acceptor_count']}")
    print()

    print("2. 测试氢键特征编码:")
    hbond_features = compute_hbond_features(pdb_path, residue_indices, chain_ids)
    print(f"   特征shape: {hbond_features.shape}")
    print(f"   特征统计: mean={hbond_features.mean():.4f}, std={hbond_features.std():.4f}")
    print(f"   非零比例: {(hbond_features != 0).mean():.2%}")
    print()

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_hbond_features()
