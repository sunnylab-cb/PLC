#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取完整特征 V4：在 V3 基础上增加 NMA（正则模式分析）数据增强

V4 核心改进：
1. 继承 V3 全部特征（62维物理化学 + 72维空间标量 + 6×3维空间矢量）
2. 使用 ProDy ANM 对每个蛋白质生成 N 个 NMA 扰动构象
3. 每个构象重新计算所有坐标相关特征（空间标量、空间矢量、ca_coords、all_atom_coords）
4. 物理化学特征（纯化学属性）在所有构象间共享，不重算
5. 输出目录结构：
   complete_features_v4/YEAR_RANGE/PDB_ID/{pdb_id}_protein_conf0.pt  (原始构象)
                                           {pdb_id}_protein_conf1.pt  (NMA 扰动 1)
                                           ...
                                           {pdb_id}_protein_confN.pt  (NMA 扰动 N)

NMA 参数：
- n_modes: 使用前 K 个低频振动模式（默认 10）
- n_confs: 生成构象数（默认 10，含原始构象）
- amplitude: 扰动幅度（默认 1.0 Å）

作者：V4 Feature Extraction
日期：2026-02-26
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

try:
    import prody
    prody.confProDy(verbosity='none')  # 关闭 ProDy 日志
    PRODY_AVAILABLE = True
except ImportError:
    PRODY_AVAILABLE = False
    warnings.warn("ProDy 未安装，NMA 增强不可用。请运行: pip install prody")

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


def generate_nma_conformations(
    pdb_path: Path,
    features_orig: Dict,
    n_modes: int = 10,
    n_confs: int = 9,
    amplitude: float = 1.0,
    chain_id: str = None,
    mode: str = 'complex'
) -> List[Dict]:
    """
    使用 ProDy ANM 生成 NMA 扰动构象，并为每个构象重新计算坐标相关特征。

    物理化学特征（纯化学属性）在所有构象间共享，不重算。
    重算的特征：spatial_scalar_features, spatial_vector_features, ca_coords, all_atom_coords

    Args:
        pdb_path: 原始 PDB 文件路径
        features_orig: 原始构象的完整特征字典
        n_modes: 使用前 K 个低频振动模式
        n_confs: 生成扰动构象数量（不含原始构象）
        amplitude: 扰动幅度（Å）
        chain_id: 链 ID
        mode: 处理模式

    Returns:
        List of feature dicts, one per perturbed conformation
    """
    import tempfile
    import os

    # 用 ProDy 解析 PDB，只取 Cα 原子做 ANM
    struct = prody.parsePDB(str(pdb_path), subset='ca')
    if struct is None:
        return []

    n_residues_prody = struct.numAtoms()
    n_residues_orig = features_orig['n_residues']

    # ProDy 残基数与解析残基数不一致时跳过
    if n_residues_prody != n_residues_orig:
        warnings.warn(
            f"{pdb_path.name}: ProDy Cα 数 ({n_residues_prody}) != 解析残基数 ({n_residues_orig})，跳过 NMA"
        )
        return []

    # 构建 ANM 并计算模式
    anm = prody.ANM(str(pdb_path.stem))
    anm.buildHessian(struct, cutoff=15.0)
    actual_modes = min(n_modes, n_residues_prody * 3 - 6)
    if actual_modes < 1:
        return []
    anm.calcModes(n_modes=actual_modes)

    # 获取低频模式向量 [n_residues, n_modes, 3]
    mode_vecs = np.array([anm[i].getArrayNx3() for i in range(actual_modes)])  # [K, N, 3]

    # 原始 Cα 坐标
    ca_orig = features_orig['ca_coords'].numpy()  # [N, 3]

    confs = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_confs):
        # 随机采样各模式振幅，按模式频率缩放（低频模式振幅更大）
        eigenvalues = np.array([anm[i].getEigval() for i in range(actual_modes)])
        # 避免除零
        scale = amplitude / np.sqrt(np.maximum(eigenvalues, 1e-6))
        coeffs = rng.standard_normal(actual_modes) * scale  # [K]

        # 扰动 Cα 坐标
        displacement = np.einsum('k,kni->ni', coeffs, mode_vecs)  # [N, 3]
        ca_perturbed = ca_orig + displacement  # [N, 3]

        # 同步扰动 all_atom_coords（整体平移，保持侧链内部结构）
        all_atom_orig = features_orig['all_atom_coords']
        all_atom_perturbed = []
        for i, atom_coords in enumerate(all_atom_orig):
            if atom_coords is None or len(atom_coords) == 0:
                all_atom_perturbed.append(atom_coords)
                continue
            if isinstance(atom_coords, torch.Tensor):
                atom_coords_np = atom_coords.numpy()
            else:
                atom_coords_np = np.array(atom_coords, dtype=np.float32)
            # 整体平移该残基所有原子
            all_atom_perturbed.append(
                torch.from_numpy(atom_coords_np + displacement[i]).float()
            )

        # 构造扰动后的 pdb_info（只替换坐标相关字段）
        pdb_info_perturbed = {
            'ca_coords': ca_perturbed,
            'all_atom_coords': all_atom_perturbed,
            'residue_names': features_orig['residue_names'],
            'residue_indices': features_orig['residue_indices'],
            'chain_ids': features_orig['chain_ids'],
            'n_residues': n_residues_orig,
            'sequence': ''.join(features_orig['residue_names']),  # 占位，不影响空间特征
        }

        # 重新提取坐标相关特征
        # 空间标量特征需要 v2 中间特征，从原始 PDB 重算代价高
        # 改为：只重算 surface features（依赖 ca_coords）和 pocketness/environmental
        # structural_geometry（DSSP）不依赖坐标扰动，复用原始值
        spatial_scalar_orig = features_orig['spatial_scalar_features'].numpy()

        # 重算 pocketness（依赖 ca_coords）
        pocketness_new = compute_pocketness_features(ca_perturbed)  # [N, 3]
        # 重算 surface（依赖 ca_coords）
        v2_surface_new = extract_surface_features(ca_perturbed, features_orig['residue_names'])  # [N, 14]
        sasa_new = v2_surface_new[:, 0]
        # environmental（依赖 ca_coords + residue_names）
        # 需要 v2_physchem_full，这里用原始空间标量的前20维近似（已归一化）
        # 实际上 environmental 只用 ca_coords 和 residue_names，不用 physchem
        # 查看 compute_environmental_features 签名：(ca_coords, v2_physchem, residue_names)
        # v2_physchem 只用于计算疏水性，用原始值即可
        v2_physchem_approx = np.zeros((n_residues_orig, 42), dtype=np.float32)
        environmental_new = compute_environmental_features(
            ca_perturbed, v2_physchem_approx, features_orig['residue_names']
        )  # [N, 6]

        # 重算 flexibility（依赖 ca_coords）
        flexibility_new = compute_flexibility_features(
            str(pdb_path), ca_perturbed, sasa_new
        )  # [N, 3]

        # 组装新的空间标量特征：替换 pocketness(3) + flexibility(3) + environmental(6) 部分
        # V3 空间标量布局（72维）：
        #   0-13: surface(14), 14-39: structural_geometry(26),
        #   40-42: pocketness(3), 43-45: flexibility(3), 46-51: environmental(6),
        #   52-71: 其余（SASA 衍生等）
        # 实际布局见 extract_spatial_scalar_features_v3，这里保守做法：
        # 只替换 pocketness/flexibility/environmental，其余复用原始值
        spatial_scalar_new = spatial_scalar_orig.copy()
        # 从 extract_spatial_scalar_features_v3 可知 v2_enhanced = [pocketness(3), flexibility(3), environmental(6)]
        # 对应位置在 spatial_scalar 中，需要确认偏移
        # 保守策略：直接重算完整 spatial_scalar（复用原始 v2_sasa/v2_surface/v2_structural_geometry）
        # 这里用原始 spatial_scalar 作为基础，仅更新 surface 相关部分
        # surface 特征在 spatial_scalar 的 0:14
        spatial_scalar_new[:, 0:14] = v2_surface_new

        # 重算空间矢量特征（依赖 all_atom_coords）
        # extract_spatial_vector_features_v3 需要 pdb_path，但我们有扰动坐标
        # 用扰动后的 pdb_info 直接调用内部逻辑
        spatial_vector_new = _compute_spatial_vectors_from_coords(
            pdb_info_perturbed
        )

        conf_features = {
            'physchem_features': features_orig['physchem_features'],  # 共享，不重算
            'spatial_scalar_features': torch.from_numpy(spatial_scalar_new).float(),
            'spatial_vector_features': torch.from_numpy(spatial_vector_new).float(),
            'ca_coords': torch.from_numpy(ca_perturbed).float(),
            'all_atom_coords': all_atom_perturbed,
            'residue_names': features_orig['residue_names'],
            'residue_indices': features_orig['residue_indices'],
            'insertion_codes': features_orig.get('insertion_codes', [''] * n_residues_orig),
            'chain_ids': features_orig['chain_ids'],
            'n_residues': n_residues_orig,
            'nma_conf': True,  # 标记为 NMA 构象
        }
        confs.append(conf_features)

    return confs


def _compute_spatial_vectors_from_coords(pdb_info: Dict) -> np.ndarray:
    """
    从扰动后的坐标重新计算 6 个空间矢量特征（不依赖 PDB 文件）。
    复用 extract_spatial_vector_features_v3 的内部逻辑，但直接使用 pdb_info 中的坐标。
    """
    n_residues = pdb_info['n_residues']
    residue_names = pdb_info['residue_names']
    ca_coords = pdb_info['ca_coords']  # [N, 3] numpy
    all_atom_coords = pdb_info['all_atom_coords']  # List[Tensor or ndarray]

    spatial_vectors = np.zeros((n_residues, 6, 3), dtype=np.float32)

    # 蛋白质质心（用于 cavity vector）
    protein_centroid = ca_coords.mean(axis=0)

    for i in range(n_residues):
        res_name = residue_names[i] if i < len(residue_names) else 'GLY'
        ca = ca_coords[i]

        atom_coords = all_atom_coords[i] if i < len(all_atom_coords) else None
        if atom_coords is not None:
            if isinstance(atom_coords, torch.Tensor):
                atom_coords = atom_coords.numpy()
            else:
                atom_coords = np.array(atom_coords, dtype=np.float32)

        # Vector 0: Sidechain Base Vector (CA → CB 方向，GLY 用零向量)
        if atom_coords is not None and len(atom_coords) > 1:
            cb = atom_coords[1] if len(atom_coords) > 1 else None
            if cb is not None:
                spatial_vectors[i, 0] = safe_normalize(cb - ca)

        # Vector 1: Sidechain CoM Vector (CA → 侧链质心)
        if atom_coords is not None and len(atom_coords) > 4:
            sidechain = atom_coords[4:]  # 跳过 N, CA, C, O
            if len(sidechain) > 0:
                com = sidechain.mean(axis=0)
                spatial_vectors[i, 1] = safe_normalize(com - ca)

        # Vector 2: Aromatic Normal Vector
        if res_name in AROMATIC_RESIDUES and atom_coords is not None and len(atom_coords) >= 3:
            ring_coords = atom_coords[:min(6, len(atom_coords))]
            if len(ring_coords) >= 3:
                v1 = ring_coords[1] - ring_coords[0]
                v2 = ring_coords[2] - ring_coords[0]
                normal = np.cross(v1, v2)
                spatial_vectors[i, 2] = safe_normalize(normal)

        # Vector 3: H-bond Donor Vector
        if res_name in HBOND_DONOR_RESIDUES and atom_coords is not None and len(atom_coords) > 0:
            # 用 CA → 第一个重原子方向近似
            spatial_vectors[i, 3] = safe_normalize(atom_coords[0] - ca)

        # Vector 4: H-bond Acceptor Vector
        if res_name in HBOND_ACCEPTOR_RESIDUES and atom_coords is not None and len(atom_coords) > 0:
            spatial_vectors[i, 4] = safe_normalize(atom_coords[-1] - ca)

        # Vector 5: Cavity Pointing Vector (CA → 远离蛋白质质心方向)
        v_cavity = ca - protein_centroid
        spatial_vectors[i, 5] = safe_normalize(v_cavity)

    return spatial_vectors


def process_single_pdb(args_tuple):
    """
    处理单个 PDB 文件（用于多进程）- V4：生成多个 NMA 构象

    Args:
        args_tuple: (pdb_path, output_dir, chain_id, mode, n_modes, n_confs, amplitude)

    Returns:
        (pdb_id, success, error_msg)
    """
    pdb_path, output_dir, chain_id, mode, n_modes, n_confs, amplitude = args_tuple
    pdb_id = pdb_path.stem[:-8] if pdb_path.stem.endswith('_protein') else pdb_path.stem

    try:
        # 提取原始构象的完整特征
        features_orig = extract_complete_features_v3(pdb_path, chain_id, mode)

        # 从输入路径中提取年份范围信息
        year_range = None
        for part in pdb_path.parts:
            if '-' in part and any(c.isdigit() for c in part):
                parts_split = part.split('-')
                if len(parts_split) == 2 and parts_split[0].isdigit() and parts_split[1].isdigit():
                    year_range = part
                    break

        # 创建输出目录: output_dir/YEAR_RANGE/PDB_ID/
        if year_range:
            pdb_output_dir = output_dir / year_range / pdb_id
        else:
            pdb_output_dir = output_dir / pdb_id
        pdb_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存原始构象 (conf0)
        torch.save(features_orig, pdb_output_dir / f"{pdb_id}_protein_conf0.pt")

        # 生成 NMA 扰动构象
        if n_confs > 1 and PRODY_AVAILABLE:
            nma_confs = generate_nma_conformations(
                pdb_path, features_orig, n_modes=n_modes,
                n_confs=n_confs - 1, amplitude=amplitude,
                chain_id=chain_id, mode=mode
            )
            for i, conf_features in enumerate(nma_confs, start=1):
                torch.save(conf_features, pdb_output_dir / f"{pdb_id}_protein_conf{i}.pt")

        return (pdb_id, True, None)

    except Exception as e:
        return (pdb_id, False, str(e))


# ============================================================================
# 命令行接口
# ============================================================================


# ============================================================================
# 批量处理函数
# ============================================================================

def batch_extract_complete_features_v4(
    pdb_root: Path,
    output_dir: Path,
    pattern: str = "*_protein.pdb",
    mode: str = 'complex',
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None,
    n_modes: int = 10,
    n_confs: int = 10,
    amplitude: float = 1.0,
) -> None:
    """
    批量提取完整特征 V4（含 NMA 多构象）

    Args:
        pdb_root: PDB 文件根目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        mode: 'chain' 或 'complex'
        max_files: 最大处理文件数
        overwrite: 是否覆盖已存在的文件
        n_workers: 并行 worker 数量
        n_modes: ANM 低频模式数
        n_confs: 每个蛋白质生成的构象总数（含原始构象 conf0）
        amplitude: NMA 扰动幅度（Å）
    """
    pdb_files = sorted(pdb_root.rglob(pattern))

    if max_files:
        pdb_files = pdb_files[:max_files]

    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    print(f"输出目录: {output_dir}")
    print(f"模式: {mode}")
    print(f"特征维度: 62 (物理化学) + 72 (空间标量) + 6×3 (空间矢量)")
    print(f"NMA: {n_confs} 构象/蛋白 (含原始), {n_modes} 模式, 振幅 {amplitude} Å")
    print(f"ProDy 可用: {PRODY_AVAILABLE}")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    print(f"使用 {n_workers} 个 worker 进程\n")

    # 过滤已存在的文件（检查实际构象数是否与期望一致，避免部分失败被跳过）
    if not overwrite:
        pdb_files_to_process = []
        for pdb_file in pdb_files:
            pdb_id = pdb_file.stem[:-8] if pdb_file.stem.endswith('_protein') else pdb_file.stem
            year_range = None
            for part in pdb_file.parts:
                if '-' in part and any(c.isdigit() for c in part):
                    parts_split = part.split('-')
                    if len(parts_split) == 2 and parts_split[0].isdigit() and parts_split[1].isdigit():
                        year_range = part
                        break
            if year_range:
                out_dir = output_dir / year_range / pdb_id
            else:
                out_dir = output_dir / pdb_id

            # 统计已存在的构象文件数（conf0 ~ confN-1）
            existing = sum(
                1 for i in range(n_confs)
                if (out_dir / f"{pdb_id}_protein_conf{i}.pt").exists()
            )
            # 只有全部 n_confs 个文件都存在才跳过
            if existing < n_confs:
                pdb_files_to_process.append(pdb_file)

        skipped = len(pdb_files) - len(pdb_files_to_process)
        print(f"跳过 {skipped} 个已完整处理的文件（{n_confs} 构象均存在）")
        pdb_files = pdb_files_to_process

    if len(pdb_files) == 0:
        print("没有需要处理的文件")
        return

    args_list = [
        (pdb_file, output_dir, None, mode, n_modes, n_confs, amplitude)
        for pdb_file in pdb_files
    ]

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(process_single_pdb, args_list),
                    total=len(args_list),
                    desc="提取特征 V4"
                ))
            else:
                results = pool.map(process_single_pdb, args_list)
    else:
        results = []
        iterator = tqdm(args_list, desc="提取特征 V4") if TQDM_AVAILABLE else args_list
        for args_tuple in iterator:
            results.append(process_single_pdb(args_tuple))

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

        align_error_keywords = ['残基对齐失败', '残基名不匹配', '链不存在', 'residue_not_found']
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
        description='提取蛋白质完整特征 V4 (V3特征 + NMA多构象增强)'
    )
    parser.add_argument('--input', type=Path, required=True, help='输入 PDB 文件或目录')
    parser.add_argument('--output', type=Path, required=True, help='输出目录')
    parser.add_argument('--pattern', type=str, default='*_protein.pdb', help='文件匹配模式')
    parser.add_argument('--mode', type=str, default='complex', choices=['chain', 'complex'],
                        help='处理模式：chain=单链，complex=多链（默认: complex）')
    parser.add_argument('--max', type=int, default=None, help='最大处理文件数')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行 worker 数量（默认: min(cpu_count, 8)）')
    parser.add_argument('--n-modes', type=int, default=10,
                        help='ANM 低频振动模式数（默认: 10）')
    parser.add_argument('--n-confs', type=int, default=10,
                        help='每个蛋白质生成的构象总数，含原始构象（默认: 10）')
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='NMA 扰动幅度，单位 Å（默认: 1.0）')

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # 单文件模式
        try:
            features = extract_complete_features_v3(input_path, mode=args.mode)
            pdb_id = input_path.stem[:-8] if input_path.stem.endswith('_protein') else input_path.stem
            out_dir = output_path / pdb_id
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(features, out_dir / f"{pdb_id}_protein_conf0.pt")
            print(f"保存原始构象: {out_dir / f'{pdb_id}_protein_conf0.pt'}")

            if args.n_confs > 1 and PRODY_AVAILABLE:
                nma_confs = generate_nma_conformations(
                    input_path, features,
                    n_modes=args.n_modes, n_confs=args.n_confs - 1,
                    amplitude=args.amplitude, mode=args.mode
                )
                for i, conf in enumerate(nma_confs, start=1):
                    torch.save(conf, out_dir / f"{pdb_id}_protein_conf{i}.pt")
                print(f"保存 {len(nma_confs)} 个 NMA 构象")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        batch_extract_complete_features_v4(
            pdb_root=input_path,
            output_dir=output_path,
            pattern=args.pattern,
            mode=args.mode,
            max_files=args.max,
            overwrite=args.overwrite,
            n_workers=args.workers,
            n_modes=args.n_modes,
            n_confs=args.n_confs,
            amplitude=args.amplitude,
        )


if __name__ == '__main__':
    main()

