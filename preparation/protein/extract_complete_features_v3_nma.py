#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取完整特征 V3：物理化学特征(62维) + 空间标量特征(72维) + 空间矢量特征(6×3维)

V3 核心改进：
1. 特征分类重组：
   - 物理化学特征 (62维)：纯化学属性，进 PhysChem MLP Encoder
   - 空间标量特征 (72维)：几何标量，进 GVP node_s
   - 空间矢量特征 (6×3维)：方向矢量，进 GVP node_v

2. 新增空间矢量特征 (6个)：
   - Sidechain Base Vector: 侧链基础伸展方向
   - Sidechain CoM Vector: 侧链质心方向（药效团）
   - Aromatic Normal Vector: 芳香环π体系法向量
   - H-bond Donor Vector: 氢键供体方向
   - H-bond Acceptor Vector: 氢键受体方向
   - Cavity Pointing Vector: 空腔/溶剂逃逸方向

3. 安全性增强：
   - 安全归一化函数（避免 NaN）
   - 多链复合物质心计算
   - 零向量处理

作者：V3 Feature Extraction
日期：2026-02-24
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp

import numpy as np
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from physicochemical_features import extract_physicochemical_features
from preparation.protein.extract_physchem_from_pdb import parse_pdb_protein
from preparation.protein.surface_features import extract_surface_features
from preparation.protein.structural_geometry_features import extract_structural_geometry_features
from preparation.protein.enhanced_features_v2 import (
    compute_pocketness_features,
    compute_flexibility_features,
    compute_environmental_features
)


# ============================================================================
# 核心工具函数：安全归一化（避坑指南 #1）
# ============================================================================

def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    安全的向量归一化函数，避免 NaN 爆炸

    Args:
        v: 输入向量 [3] 或 [N, 3]
        eps: 最小模长阈值

    Returns:
        归一化后的向量，如果模长 < eps 则返回零向量
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # 避免除零：如果模长太小，返回零向量
    mask = (norm > eps).astype(np.float32)
    return np.where(mask, v / (norm + eps), 0.0)


def compute_angle(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算两个向量的夹角（弧度）

    Args:
        v1, v2: 输入向量 [3]
        eps: 数值稳定性阈值

    Returns:
        夹角（弧度），范围 [0, π]
    """
    v1_norm = safe_normalize(v1, eps)
    v2_norm = safe_normalize(v2, eps)

    # 避免数值误差导致 arccos 超出 [-1, 1]
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


# ============================================================================
# 氨基酸分类常量
# ============================================================================

AMINO_ACIDS = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
    'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG',
    'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

NONPOLAR_RESIDUES = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'GLY']
POLAR_RESIDUES = ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN']
CHARGED_RESIDUES = ['LYS', 'ARG', 'HIS', 'ASP', 'GLU']

AROMATIC_RESIDUES = {
    'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # 6-membered ring
    'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
}

HBOND_DONOR_RESIDUES = ['SER', 'THR', 'TYR', 'CYS', 'ASN', 'GLN', 'ARG', 'LYS', 'HIS', 'TRP']
HBOND_ACCEPTOR_RESIDUES = ['ASP', 'GLU', 'SER', 'THR', 'TYR', 'ASN', 'GLN', 'HIS']

HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
HYDROPHILIC_RESIDUES = ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN', 'LYS', 'ARG', 'HIS', 'ASP', 'GLU']


# ============================================================================
# 物理化学特征提取（62 维）
# ============================================================================

def extract_physchem_features_v3(
    pdb_info: Dict,
    v2_physchem: np.ndarray,
    v2_sequence: np.ndarray,
    v2_hbond: np.ndarray,
    v2_structural_geometry: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    提取物理化学特征 V3（62 维）

    组成：
    - V2 Basic (0-19): 20 维
    - V2 Sequence (26-33): 8 维
    - V2 H-bond Network (34-41): 8 维
    - 新增：残基类型 One-hot: 20 维
    - 新增：二级结构 One-hot: 3 维
    - 新增：极性分类 One-hot: 3 维

    Args:
        pdb_info: PDB 解析信息
        v2_physchem: V2 的物理化学特征 (42维)
        v2_sequence: V2 的序列特征 (8维)
        v2_hbond: V2 的氢键特征 (8维)
        v2_structural_geometry: V2 的结构几何特征 (26维)，前8维为DSSP one-hot

    Returns:
        physchem_features: [N, 62]
    """
    residue_names = pdb_info['residue_names']
    n_residues = len(residue_names)

    # 初始化特征矩阵
    physchem_features = np.zeros((n_residues, 62), dtype=np.float32)

    # 1. 复用 V2 的 Basic 特征 (0-19)
    physchem_features[:, 0:20] = v2_physchem[:, 0:20]

    # 2. 复用 V2 的 Sequence 特征 (26-33) → 放到 20-27
    physchem_features[:, 20:28] = v2_sequence

    # 3. 复用 V2 的 H-bond Network 特征 (34-41) → 放到 28-35
    physchem_features[:, 28:36] = v2_hbond

    # 4. 新增：残基类型 One-hot (20 维) → 36-55
    for i, res_name in enumerate(residue_names):
        if res_name in AMINO_ACIDS:
            idx = AMINO_ACIDS.index(res_name)
            physchem_features[i, 36 + idx] = 1.0

    # 5. 新增：二级结构 One-hot (3 维) → 56-58
    # 从 V2 structural geometry 的 DSSP 8类聚合为 3类：Helix/Sheet/Coil
    if v2_structural_geometry is not None and v2_structural_geometry.shape[1] >= 8:
        dssp8 = v2_structural_geometry[:, :8]
        ss3_scores = np.stack([
            dssp8[:, [0, 3, 4]].sum(axis=1),  # Helix: H/G/I
            dssp8[:, [1, 2]].sum(axis=1),     # Sheet: B/E
            dssp8[:, [5, 6, 7]].sum(axis=1),  # Coil: T/S/-
        ], axis=1)

        valid_mask = ss3_scores.sum(axis=1) > 0
        if np.any(valid_mask):
            ss3_idx = np.argmax(ss3_scores[valid_mask], axis=1)
            physchem_features[valid_mask, 56:59] = 0.0
            physchem_features[valid_mask, 56 + ss3_idx] = 1.0

        if np.any(~valid_mask):
            physchem_features[~valid_mask, 56:59] = np.array([0, 0, 1], dtype=np.float32)
    else:
        # 无DSSP信息时回退为 Coil
        physchem_features[:, 56:59] = np.array([0, 0, 1], dtype=np.float32)

    # 6. 新增：极性分类 One-hot (3 维) → 59-61
    for i, res_name in enumerate(residue_names):
        if res_name in NONPOLAR_RESIDUES:
            physchem_features[i, 59:62] = [1, 0, 0]  # Nonpolar
        elif res_name in POLAR_RESIDUES:
            physchem_features[i, 59:62] = [0, 1, 0]  # Polar
        elif res_name in CHARGED_RESIDUES:
            physchem_features[i, 59:62] = [0, 0, 1]  # Charged

    return physchem_features


# ============================================================================
# 空间标量特征提取（72 维）
# ============================================================================

def extract_spatial_scalar_features_v3(
    pdb_info: Dict,
    v2_sasa: np.ndarray,
    v2_surface: np.ndarray,
    v2_structural_geometry: np.ndarray,
    v2_enhanced: np.ndarray
) -> np.ndarray:
    """
    提取空间标量特征 V3（72 维）

    组成：
    - V2 SASA (20-25): 6 维
    - V2 Surface (42-55): 14 维
    - V2 Structural Geometry (56-81): 26 维
    - V2 Enhanced V2 (82-93): 12 维
    - 新增：局部密度统计: 4 维
    - 新增：到质心距离: 1 维
    - 新增：表面法向量模长: 1 维
    - 新增：局部电荷密度: 4 维
    - 新增：局部疏水性: 4 维

    Args:
        pdb_info: PDB 解析信息
        v2_sasa: V2 的 SASA 特征 (6维)
        v2_surface: V2 的 Surface 特征 (14维)
        v2_structural_geometry: V2 的 Structural Geometry 特征 (26维)
        v2_enhanced: V2 的 Enhanced 特征 (12维)

    Returns:
        spatial_scalar_features: [N, 72]
    """
    ca_coords = pdb_info['ca_coords']  # [N, 3]
    residue_names = pdb_info['residue_names']
    n_residues = len(residue_names)

    # 初始化特征矩阵
    spatial_scalar = np.zeros((n_residues, 72), dtype=np.float32)

    # 1. 复用 V2 的 SASA 特征 (6维) → 0-5
    spatial_scalar[:, 0:6] = v2_sasa

    # 2. 复用 V2 的 Surface 特征 (14维) → 6-19
    spatial_scalar[:, 6:20] = v2_surface

    # 3. 复用 V2 的 Structural Geometry 特征 (26维) → 20-45
    spatial_scalar[:, 20:46] = v2_structural_geometry

    # 4. 复用 V2 的 Enhanced 特征 (12维) → 46-57
    spatial_scalar[:, 46:58] = v2_enhanced

    # 5. 新增：局部密度统计 (4维) → 58-61
    density_features = compute_local_density(ca_coords)
    spatial_scalar[:, 58:62] = density_features

    # 6. 新增：到质心距离 (1维) → 62
    distance_to_center = compute_distance_to_center(ca_coords)
    spatial_scalar[:, 62] = distance_to_center

    # 7. 新增：表面法向量模长 (1维) → 63
    surface_normal_magnitude = compute_surface_normal_magnitude(ca_coords)
    spatial_scalar[:, 63] = surface_normal_magnitude

    # 8. 新增：局部电荷密度 (4维) → 64-67
    charge_density = compute_local_charge_density(ca_coords, residue_names)
    spatial_scalar[:, 64:68] = charge_density

    # 9. 新增：局部疏水性 (4维) → 68-71
    hydrophobicity = compute_local_hydrophobicity(ca_coords, residue_names)
    spatial_scalar[:, 68:72] = hydrophobicity

    return spatial_scalar


def compute_local_density(ca_coords: np.ndarray, radii: List[float] = [5.0, 10.0, 15.0]) -> np.ndarray:
    """
    计算局部密度统计（4 维）

    Args:
        ca_coords: Cα 坐标 [N, 3]
        radii: 半径列表

    Returns:
        density_features: [N, 4]
            - [0-2]: 5Å, 10Å, 15Å 半径内的归一化密度
            - [3]: 密度梯度 (10Å - 5Å)
    """
    n_residues = len(ca_coords)
    density_features = np.zeros((n_residues, 4), dtype=np.float32)

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :],
        axis=-1
    )  # [N, N]

    for i, radius in enumerate(radii):
        # 计算每个残基在该半径内的邻居数量
        neighbors_count = np.sum((dist_matrix > 0) & (dist_matrix < radius), axis=1)

        # 归一化：除以球体体积
        volume = (4.0 / 3.0) * np.pi * (radius ** 3)
        density = neighbors_count / volume

        # 归一化到 [0, 1]
        max_density = density.max() if density.max() > 0 else 1.0
        density_features[:, i] = density / max_density

    # 密度梯度 (10Å - 5Å)
    density_features[:, 3] = density_features[:, 1] - density_features[:, 0]

    return density_features


def compute_distance_to_center(ca_coords: np.ndarray) -> np.ndarray:
    """
    计算到蛋白质质心的距离（1 维）

    避坑指南 #2：多链复合物的质心计算
    确保使用所有链的 Cα 原子计算全局质心

    Args:
        ca_coords: Cα 坐标 [N, 3]

    Returns:
        distance_norm: [N] 归一化距离
    """
    # 计算全局质心（所有链）
    protein_center = ca_coords.mean(axis=0)  # [3]

    # 计算每个残基到质心的距离
    distances = np.linalg.norm(ca_coords - protein_center, axis=1)  # [N]

    # 归一化到 [0, 1]
    max_distance = distances.max() if distances.max() > 0 else 1.0
    distance_norm = distances / max_distance

    return distance_norm


def compute_surface_normal_magnitude(ca_coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """
    计算表面法向量模长（1 维）

    通过 PCA 计算局部表面法向量，模长反映表面暴露度

    Args:
        ca_coords: Cα 坐标 [N, 3]
        radius: 邻居搜索半径

    Returns:
        normal_magnitude: [N] 法向量模长（0=埋藏，1=表面）
    """
    n_residues = len(ca_coords)
    normal_magnitude = np.zeros(n_residues, dtype=np.float32)

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :],
        axis=-1
    )

    for i in range(n_residues):
        # 找到邻居
        neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
        neighbors_coords = ca_coords[neighbors_mask]

        if len(neighbors_coords) < 3:
            # 邻居太少，无法计算 PCA
            normal_magnitude[i] = 0.0
            continue

        # 中心化坐标
        coords_centered = neighbors_coords - ca_coords[i]

        # 计算协方差矩阵
        cov_matrix = coords_centered.T @ coords_centered

        # 特征值分解
        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        # 法向量模长：1 - (最小特征值 / 最大特征值)
        # 表面残基：最小特征值很小 → normal_magnitude 接近 1
        # 埋藏残基：特征值相近 → normal_magnitude 接近 0
        if eigenvalues.max() > 1e-8:
            normal_magnitude[i] = 1.0 - (eigenvalues.min() / eigenvalues.max())
        else:
            normal_magnitude[i] = 0.0

    return normal_magnitude


def compute_local_charge_density(
    ca_coords: np.ndarray,
    residue_names: List[str],
    radii: List[float] = [5.0, 10.0]
) -> np.ndarray:
    """
    计算局部电荷密度（4 维）

    Args:
        ca_coords: Cα 坐标 [N, 3]
        residue_names: 残基名称列表
        radii: 半径列表

    Returns:
        charge_density: [N, 4]
            - [0-1]: 5Å, 10Å 半径内的正电荷密度
            - [2-3]: 5Å, 10Å 半径内的负电荷密度
    """
    n_residues = len(ca_coords)
    charge_density = np.zeros((n_residues, 4), dtype=np.float32)

    # 定义电荷
    positive_residues = ['LYS', 'ARG', 'HIS']
    negative_residues = ['ASP', 'GLU']

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :],
        axis=-1
    )

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            # 找到邻居
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors_residues = [residue_names[k] for k in range(n_residues) if neighbors_mask[k]]

            # 计算正/负电荷数量
            positive_count = sum([1 for res in neighbors_residues if res in positive_residues])
            negative_count = sum([1 for res in neighbors_residues if res in negative_residues])

            # 归一化：除以球体体积
            volume = (4.0 / 3.0) * np.pi * (radius ** 3)
            charge_density[i, j] = positive_count / volume
            charge_density[i, j + 2] = negative_count / volume

    # 归一化到 [0, 1]
    for col in range(4):
        max_val = charge_density[:, col].max()
        if max_val > 0:
            charge_density[:, col] /= max_val

    return charge_density


def compute_local_hydrophobicity(
    ca_coords: np.ndarray,
    residue_names: List[str],
    radii: List[float] = [5.0, 10.0]
) -> np.ndarray:
    """
    计算局部疏水性（4 维）

    Args:
        ca_coords: Cα 坐标 [N, 3]
        residue_names: 残基名称列表
        radii: 半径列表

    Returns:
        hydrophobicity: [N, 4]
            - [0-1]: 5Å, 10Å 半径内的疏水残基比例
            - [2-3]: 5Å, 10Å 半径内的亲水残基比例
    """
    n_residues = len(ca_coords)
    hydrophobicity = np.zeros((n_residues, 4), dtype=np.float32)

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :],
        axis=-1
    )

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            # 找到邻居
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors_residues = [residue_names[k] for k in range(n_residues) if neighbors_mask[k]]

            if len(neighbors_residues) == 0:
                continue

            # 计算疏水/亲水残基比例
            hydrophobic_count = sum([1 for res in neighbors_residues if res in HYDROPHOBIC_RESIDUES])
            hydrophilic_count = sum([1 for res in neighbors_residues if res in HYDROPHILIC_RESIDUES])

            hydrophobicity[i, j] = hydrophobic_count / len(neighbors_residues)
            hydrophobicity[i, j + 2] = hydrophilic_count / len(neighbors_residues)

    return hydrophobicity


# ============================================================================
# 空间矢量特征提取（6 个矢量 × 3 维）
# ============================================================================

def extract_spatial_vector_features_v3(
    pdb_path: Path,
    pdb_info: Dict,
    chain_id: str = None
) -> np.ndarray:
    """
    提取空间矢量特征 V3（6 个矢量 × 3 维）

    矢量列表：
    [0] Sidechain Base Vector: 侧链基础伸展方向
    [1] Sidechain CoM Vector: 侧链质心方向（药效团）
    [2] Aromatic Normal Vector: 芳香环π体系法向量
    [3] H-bond Donor Vector: 氢键供体方向
    [4] H-bond Acceptor Vector: 氢键受体方向
    [5] Cavity Pointing Vector: 空腔/溶剂逃逸方向

    Args:
        pdb_path: PDB 文件路径
        pdb_info: PDB 解析信息
        chain_id: 链 ID

    Returns:
        spatial_vector_features: [N, 6, 3]
    """
    ca_coords = pdb_info['ca_coords']
    residue_names = pdb_info['residue_names']
    residue_indices = pdb_info['residue_indices']
    insertion_codes = pdb_info.get('insertion_codes', [''] * len(residue_names))
    chain_ids = pdb_info['chain_ids']
    n_residues = len(residue_names)

    # 初始化矢量特征矩阵
    spatial_vectors = np.zeros((n_residues, 6, 3), dtype=np.float32)

    # 解析 PDB 文件以获取所有原子坐标
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
    except Exception as e:
        warnings.warn(f"无法解析 PDB 文件 {pdb_path}: {e}")
        return spatial_vectors

    # 遍历每个残基（严格对齐：含 insertion code 精确匹配）
    for i, (res_name, res_idx, chain_id_i, ins_code) in enumerate(
        zip(residue_names, residue_indices, chain_ids, insertion_codes)
    ):
        # 获取主链原子坐标
        ca_coord = ca_coords[i]

        # 获取残基对象
        chain_key = str(chain_id_i).strip()
        if chain_key not in structure[0]:
            raise KeyError(f"链不存在: chain={chain_key}, residue={res_name}{res_idx}")

        chain = structure[0][chain_key]

        # 用 insertion code 精确查找残基：(' ', resseq, icode)
        icode = ins_code if ins_code else ' '
        try:
            residue = chain[(' ', int(res_idx), icode)]
        except Exception:
            same_seq_candidates = [r.get_id() for r in chain.get_residues() if int(r.get_id()[1]) == int(res_idx)]
            raise ValueError(
                f"残基对齐失败: chain={chain_key}, res_idx={res_idx}, icode='{icode}', "
                f"expected_key=(' ', {int(res_idx)}, '{icode}'), candidates={same_seq_candidates[:5]}"
            )

        # 额外一致性检查：残基名必须匹配
        if residue.get_resname().strip() != str(res_name).strip():
            raise ValueError(
                f"残基名不匹配: chain={chain_key}, res_idx={res_idx}, icode='{icode}', "
                f"expected={res_name}, got={residue.get_resname()}"
            )

        n_coord = residue['N'].get_coord() if 'N' in residue else None
        c_coord = residue['C'].get_coord() if 'C' in residue else None
        o_coord = residue['O'].get_coord() if 'O' in residue else None

        # 矢量 0: Sidechain Base Vector
        spatial_vectors[i, 0] = compute_sidechain_base_vector(
            residue, res_name, ca_coord, n_coord, c_coord
        )

        # 矢量 1: Sidechain CoM Vector
        spatial_vectors[i, 1] = compute_sidechain_com_vector(
            residue, res_name, ca_coord, spatial_vectors[i, 0]
        )

        # 矢量 2: Aromatic Normal Vector
        spatial_vectors[i, 2] = compute_aromatic_normal_vector(
            residue, res_name, ca_coord
        )

        # 矢量 3: H-bond Donor Vector
        spatial_vectors[i, 3] = compute_hbond_donor_vector(
            residue, res_name, spatial_vectors[i, 1]
        )

        # 矢量 4: H-bond Acceptor Vector
        spatial_vectors[i, 4] = compute_hbond_acceptor_vector(
            residue, res_name, c_coord, o_coord, spatial_vectors[i, 1]
        )

        # 矢量 5: Cavity Pointing Vector
        spatial_vectors[i, 5] = compute_cavity_pointing_vector(
            ca_coord, ca_coords
        )

    return spatial_vectors


def compute_sidechain_base_vector(
    residue,
    res_name: str,
    ca_coord: np.ndarray,
    n_coord: Optional[np.ndarray],
    c_coord: Optional[np.ndarray]
) -> np.ndarray:
    """
    计算侧链基础伸展矢量

    定义：normalize(Cβ - Cα)
    Glycine 特殊处理：计算虚拟 Cβ

    Args:
        residue: Bio.PDB 残基对象
        res_name: 残基名称
        ca_coord: Cα 坐标
        n_coord: N 坐标
        c_coord: C 坐标

    Returns:
        v_base: [3] 归一化矢量
    """
    if res_name != 'GLY':
        # 标准残基：使用 Cβ
        if 'CB' in residue:
            cb_coord = residue['CB'].get_coord()
            v_base = cb_coord - ca_coord
            return safe_normalize(v_base)

    # Glycine 或缺少 Cβ：计算虚拟 Cβ
    if n_coord is not None and c_coord is not None:
        v1 = safe_normalize(ca_coord - n_coord)
        v2 = safe_normalize(ca_coord - c_coord)
        bisector = safe_normalize(v1 + v2)

        # 虚拟 Cβ 在角平分线反方向，距离 1.5 Å
        cb_virtual = ca_coord - bisector * 1.5
        v_base = cb_virtual - ca_coord
        return safe_normalize(v_base)

    # 无法计算，返回零向量
    return np.zeros(3, dtype=np.float32)


def compute_sidechain_com_vector(
    residue,
    res_name: str,
    ca_coord: np.ndarray,
    v_base: np.ndarray
) -> np.ndarray:
    """
    计算侧链质心矢量（药效团方向）

    定义：normalize(CoM_sidechain - Cα)

    Args:
        residue: Bio.PDB 残基对象
        res_name: 残基名称
        ca_coord: Cα 坐标
        v_base: 基础侧链矢量（用于 GLY/ALA）

    Returns:
        v_com: [3] 归一化矢量
    """
    # 提取侧链重原子（排除主链 N, CA, C, O, H）
    sidechain_atoms = []
    for atom in residue.get_atoms():
        atom_name = atom.get_name()
        element = atom.element

        # 排除主链原子和氢原子
        if atom_name not in ['N', 'CA', 'C', 'O'] and element != 'H':
            sidechain_atoms.append(atom.get_coord())

    if len(sidechain_atoms) > 0:
        # 计算侧链质心
        com_sidechain = np.mean(sidechain_atoms, axis=0)
        v_com = com_sidechain - ca_coord
        return safe_normalize(v_com)
    else:
        # GLY 或 ALA：使用基础矢量
        return v_base


def compute_aromatic_normal_vector(
    residue,
    res_name: str,
    ca_coord: np.ndarray
) -> np.ndarray:
    """
    计算芳香环π体系法向量

    定义：芳香环平面的法向量（通过 PCA 计算）

    Args:
        residue: Bio.PDB 残基对象
        res_name: 残基名称
        ca_coord: Cα 坐标

    Returns:
        v_aromatic: [3] 归一化法向量
    """
    if res_name not in AROMATIC_RESIDUES:
        return np.zeros(3, dtype=np.float32)

    # 提取芳香环原子
    ring_atom_names = AROMATIC_RESIDUES[res_name]
    ring_coords = []

    for atom_name in ring_atom_names:
        if atom_name in residue:
            ring_coords.append(residue[atom_name].get_coord())

    if len(ring_coords) < 3:
        # 原子不足，无法计算平面
        return np.zeros(3, dtype=np.float32)

    ring_coords = np.array(ring_coords)

    # 中心化坐标
    ring_center = ring_coords.mean(axis=0)
    coords_centered = ring_coords - ring_center

    # 计算协方差矩阵
    cov_matrix = coords_centered.T @ coords_centered

    # 特征值分解
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 最小特征值对应的特征向量 = 法向量
        min_idx = np.argmin(eigenvalues)
        normal_vector = eigenvectors[:, min_idx].real

        # 统一方向：指向 Cα 的一侧
        if np.dot(normal_vector, ca_coord - ring_center) < 0:
            normal_vector = -normal_vector

        return safe_normalize(normal_vector)

    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32)


def compute_hbond_donor_vector(
    residue,
    res_name: str,
    v_com: np.ndarray
) -> np.ndarray:
    """
    计算氢键供体矢量

    简化版本：使用侧链质心方向作为近似

    Args:
        residue: Bio.PDB 残基对象
        res_name: 残基名称
        v_com: 侧链质心矢量

    Returns:
        v_donor: [3] 归一化矢量
    """
    if res_name in HBOND_DONOR_RESIDUES:
        # 简化：使用侧链质心方向
        return v_com
    else:
        return np.zeros(3, dtype=np.float32)


def compute_hbond_acceptor_vector(
    residue,
    res_name: str,
    c_coord: Optional[np.ndarray],
    o_coord: Optional[np.ndarray],
    v_com: np.ndarray
) -> np.ndarray:
    """
    计算氢键受体矢量

    简化版本：
    - 侧链受体：使用侧链质心反方向
    - 主链 C=O：使用 O - C 方向

    Args:
        residue: Bio.PDB 残基对象
        res_name: 残基名称
        c_coord: C 坐标
        o_coord: O 坐标
        v_com: 侧链质心矢量

    Returns:
        v_acceptor: [3] 归一化矢量
    """
    if res_name in HBOND_ACCEPTOR_RESIDUES:
        # 侧链受体：质心反方向
        return -v_com
    elif c_coord is not None and o_coord is not None:
        # 主链 C=O 作为受体
        v_acceptor = o_coord - c_coord
        return safe_normalize(v_acceptor)
    else:
        return np.zeros(3, dtype=np.float32)


def compute_cavity_pointing_vector(
    ca_coord: np.ndarray,
    all_ca_coords: np.ndarray,
    radius: float = 10.0
) -> np.ndarray:
    """
    计算空腔/溶剂逃逸矢量

    定义：normalize(Cα - CoM_neighbors)

    Args:
        ca_coord: 当前 Cα 坐标
        all_ca_coords: 所有 Cα 坐标 [N, 3]
        radius: 邻居搜索半径

    Returns:
        v_cavity: [3] 归一化矢量
    """
    # 计算到所有其他 Cα 的距离
    distances = np.linalg.norm(all_ca_coords - ca_coord, axis=1)

    # 找到邻居（排除自己）
    neighbors_mask = (distances > 0) & (distances < radius)
    neighbors_coords = all_ca_coords[neighbors_mask]

    if len(neighbors_coords) == 0:
        # 孤立残基
        return np.zeros(3, dtype=np.float32)

    # 计算邻居质心
    com_neighbors = neighbors_coords.mean(axis=0)

    # 逃逸矢量
    v_cavity = ca_coord - com_neighbors
    return safe_normalize(v_cavity)


# ============================================================================
# 主函数：提取完整特征 V3
# ============================================================================

def extract_complete_features_v3(
    pdb_path: Path,
    chain_id: str = None,
    mode: str = 'complex'
) -> Dict:
    """
    提取完整特征 V3

    Args:
        pdb_path: PDB 文件路径
        chain_id: 链 ID（仅在 mode='chain' 时使用）
        mode: 'chain' = 单链模式, 'complex' = 多链模式

    Returns:
        特征字典：
        {
            'physchem_features': [N, 62],
            'spatial_scalar_features': [N, 72],
            'spatial_vector_features': [N, 6, 3],
            'residue_names': List[str],
            'residue_numbers': List[int],
            'chain_ids': List[str],
            'n_residues': int
        }
    """
    print(f"\n处理: {pdb_path.name}")

    # 1. 解析 PDB 文件
    pdb_info = parse_pdb_protein(pdb_path, chain_id, mode)
    n_residues = pdb_info['n_residues']
    print(f"  解析 PDB: {n_residues} 个残基")

    # 2. 提取 V2 特征（作为基础）
    print(f"  提取 V2 基础特征...")

    # 物理化学特征 (42维)
    v2_physchem_full = extract_physicochemical_features(
        pdb_path,
        pdb_info['sequence'],
        pdb_info['residue_names'],
        pdb_info['residue_indices'],
        pdb_info['chain_ids']
    )  # [N, 42]

    # 表面特征 (14维)
    v2_surface = extract_surface_features(
        pdb_info['ca_coords'],
        pdb_info['residue_names']
    )  # [N, 14]

    # 结构几何特征 (26维)
    v2_structural_geometry = extract_structural_geometry_features(
        pdb_path,
        pdb_info['ca_coords'],
        pdb_info['residue_indices'],
        pdb_info['chain_ids']
    )  # [N, 26]

    # 增强特征 V2 (12维)
    pocketness = compute_pocketness_features(pdb_info['ca_coords'])  # [N, 3]

    # 从 v2_surface 中提取 SASA 值（假设前几维是 SASA）
    sasa_values = v2_surface[:, 0]  # 使用第一维作为 SASA
    flexibility = compute_flexibility_features(
        str(pdb_path),
        pdb_info['ca_coords'],
        sasa_values
    )  # [N, 3]

    environmental = compute_environmental_features(
        pdb_info['ca_coords'],
        v2_physchem_full,
        pdb_info['residue_names']
    )  # [N, 6]

    v2_enhanced = np.concatenate([pocketness, flexibility, environmental], axis=1)  # [N, 12]

    # 拆分 V2 物理化学特征
    v2_basic = v2_physchem_full[:, 0:20]  # Basic (0-19)
    v2_sasa = v2_physchem_full[:, 20:26]  # SASA (20-25)
    v2_sequence = v2_physchem_full[:, 26:34]  # Sequence (26-33)
    v2_hbond = v2_physchem_full[:, 34:42]  # H-bond (34-41)

    # 3. 提取 V3 物理化学特征 (62维)
    print(f"  提取 V3 物理化学特征 (62维)...")
    physchem_features = extract_physchem_features_v3(
        pdb_info, v2_basic, v2_sequence, v2_hbond, v2_structural_geometry
    )

    # 4. 提取 V3 空间标量特征 (72维)
    print(f"  提取 V3 空间标量特征 (72维)...")
    spatial_scalar_features = extract_spatial_scalar_features_v3(
        pdb_info, v2_sasa, v2_surface, v2_structural_geometry, v2_enhanced
    )

    # 5. 提取 V3 空间矢量特征 (6×3维)
    print(f"  提取 V3 空间矢量特征 (6×3维)...")
    spatial_vector_features = extract_spatial_vector_features_v3(
        pdb_path, pdb_info, chain_id
    )

    # 6. 检查 NaN（避坑指南 #1）
    if np.isnan(physchem_features).any():
        warnings.warn(f"物理化学特征包含 NaN！")
    if np.isnan(spatial_scalar_features).any():
        warnings.warn(f"空间标量特征包含 NaN！")
    if np.isnan(spatial_vector_features).any():
        warnings.warn(f"空间矢量特征包含 NaN！")

    # 7. 返回特征字典
    return {
        'physchem_features': torch.from_numpy(physchem_features).float(),
        'spatial_scalar_features': torch.from_numpy(spatial_scalar_features).float(),
        'spatial_vector_features': torch.from_numpy(spatial_vector_features).float(),
        'ca_coords': torch.from_numpy(pdb_info['ca_coords']).float(),  # V8: 添加 ca_coords
        'all_atom_coords': pdb_info['all_atom_coords'],  # V8: 添加 all_atom_coords (用于计算 GVP vectors)
        'residue_names': pdb_info['residue_names'],
        'residue_indices': pdb_info['residue_indices'],
        'insertion_codes': pdb_info.get('insertion_codes', [''] * n_residues),
        'chain_ids': pdb_info['chain_ids'],
        'n_residues': n_residues
    }


def process_single_pdb(args_tuple):
    """
    处理单个 PDB 文件（用于多进程）

    Args:
        args_tuple: (pdb_path, output_dir, chain_id, mode)

    Returns:
        (pdb_id, success, error_msg)
    """
    pdb_path, output_dir, chain_id, mode = args_tuple
    pdb_id = pdb_path.stem[:-8] if pdb_path.stem.endswith('_protein') else pdb_path.stem

    try:
        # 提取特征
        features = extract_complete_features_v3(pdb_path, chain_id, mode)

        # 从输入路径中提取年份范围信息
        # 假设输入路径格式为: .../YEAR_RANGE/PDB_ID/xxx_protein.pdb
        year_range = None
        for part in pdb_path.parts:
            if '-' in part and any(c.isdigit() for c in part):
                # 检查是否为年份范围格式 (e.g., "1981-2000")
                parts_split = part.split('-')
                if len(parts_split) == 2 and parts_split[0].isdigit() and parts_split[1].isdigit():
                    year_range = part
                    break

        # 创建嵌套的输出目录结构: output_dir/YEAR_RANGE/PDB_ID/
        if year_range:
            # 使用嵌套结构
            pdb_output_dir = output_dir / year_range / pdb_id
        else:
            # 如果无法确定年份范围，使用扁平结构
            pdb_output_dir = output_dir / pdb_id

        pdb_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存为 .pt 文件，使用与 V2 相同的命名格式
        output_path = pdb_output_dir / f"{pdb_id}_protein.pt"
        torch.save(features, output_path)

        return (pdb_id, True, None)

    except Exception as e:
        return (pdb_id, False, str(e))


# ============================================================================
# 命令行接口
# ============================================================================


# ============================================================================
# 批量处理函数
# ============================================================================

def batch_extract_complete_features_v3(
    pdb_root: Path,
    output_dir: Path,
    pattern: str = "*_protein.pdb",
    mode: str = 'complex',
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None
) -> None:
    """
    批量提取完整特征 V3

    Args:
        pdb_root: PDB 文件根目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        mode: 'chain' 或 'complex'
        max_files: 最大处理文件数
        overwrite: 是否覆盖已存在的文件
        n_workers: 并行 worker 数量
    """
    # 查找所有 PDB 文件（递归搜索）
    pdb_files = sorted(pdb_root.rglob(pattern))

    if max_files:
        pdb_files = pdb_files[:max_files]

    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    print(f"输出目录: {output_dir}")
    print(f"模式: {mode}")
    print(f"特征维度: 62 (物理化学) + 72 (空间标量) + 6×3 (空间矢量)")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    print(f"使用 {n_workers} 个 worker 进程\n")

    # 过滤已存在的文件
    if not overwrite:
        pdb_files_to_process = []
        for pdb_file in pdb_files:
            pdb_id = pdb_file.stem[:-8] if pdb_file.stem.endswith('_protein') else pdb_file.stem

            # 从输入路径中提取年份范围信息
            year_range = None
            for part in pdb_file.parts:
                if '-' in part and any(c.isdigit() for c in part):
                    parts_split = part.split('-')
                    if len(parts_split) == 2 and parts_split[0].isdigit() and parts_split[1].isdigit():
                        year_range = part
                        break

            # 检查输出文件是否存在
            if year_range:
                out_file = output_dir / year_range / pdb_id / f"{pdb_id}_protein.pt"
            else:
                out_file = output_dir / pdb_id / f"{pdb_id}_protein.pt"

            if not out_file.exists():
                pdb_files_to_process.append(pdb_file)

        print(f"跳过 {len(pdb_files) - len(pdb_files_to_process)} 个已存在的文件")
        pdb_files = pdb_files_to_process

    if len(pdb_files) == 0:
        print("没有需要处理的文件")
        return

    # 准备参数
    args_list = [
        (pdb_file, output_dir, None, mode)
        for pdb_file in pdb_files
    ]

    # 并行处理
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(process_single_pdb, args_list),
                    total=len(args_list),
                    desc="提取特征 V3"
                ))
            else:
                results = pool.map(process_single_pdb, args_list)
    else:
        # 单进程处理
        results = []
        iterator = tqdm(args_list, desc="提取特征 V3") if TQDM_AVAILABLE else args_list
        for args_tuple in iterator:
            results.append(process_single_pdb(args_tuple))

    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count

    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")

    if failed_count > 0:
        print(f"\n失败的文件:")
        for pdb_id, success, error_msg in results:
            if not success:
                print(f"  {pdb_id}: {error_msg}")

        # 导出对齐失败清单 CSV
        align_error_keywords = [
            '残基对齐失败',
            '残基名不匹配',
            '链不存在',
            'residue_not_found'
        ]
        alignment_failures = []
        for idx, (pdb_id, success, error_msg) in enumerate(results):
            if success or not error_msg:
                continue
            if any(keyword in str(error_msg) for keyword in align_error_keywords):
                pdb_path = args_list[idx][0]
                alignment_failures.append((pdb_id, str(pdb_path), str(error_msg)))

        if alignment_failures:
            failure_csv = output_dir / 'alignment_failures.csv'
            with open(failure_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['pdb_id', 'pdb_path', 'error'])
                writer.writerows(alignment_failures)
            print(f"\n对齐失败 CSV 已导出: {failure_csv}")
            print(f"  对齐失败数: {len(alignment_failures)}")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='提取蛋白质完整特征 V3 (62维物理化学 + 72维空间标量 + 6×3维空间矢量)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='输入 PDB 文件或目录'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_protein.pdb',
        help='文件匹配模式（默认: *_protein.pdb）'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='complex',
        choices=['chain', 'complex'],
        help='处理模式：chain=单链，complex=多链（默认: complex）'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='最大处理文件数（可选）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的文件'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并行 worker 数量（默认: min(cpu_count, 8)）'
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # 单文件处理
        print(f"处理单个文件: {input_path}")

        try:
            features = extract_complete_features_v3(input_path, mode=args.mode)

            # 保存
            pdb_id = input_path.stem[:-8] if input_path.stem.endswith('_protein') else input_path.stem
            out_file = output_path / f"{pdb_id}_protein.pt"
            torch.save(features, out_file)

            print(f"保存: {out_file}")
            print(f"特征维度:")
            print(f"  物理化学: {features['physchem_features'].shape}")
            print(f"  空间标量: {features['spatial_scalar_features'].shape}")
            print(f"  空间矢量: {features['spatial_vector_features'].shape}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    else:
        # 目录批处理
        print(f"处理目录: {input_path}")
        batch_extract_complete_features_v3(
            pdb_root=input_path,
            output_dir=output_path,
            pattern=args.pattern,
            mode=args.mode,
            max_files=args.max,
            overwrite=args.overwrite,
            n_workers=args.workers
        )


if __name__ == '__main__':
    main()
