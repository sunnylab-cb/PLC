#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强特征模块 V2 - 新增12维特征

新增特征组成：
1. 显式几何"口袋度"特征 (3维)
   - CX (Convexity Index): 凸度指数
   - Local Packing Density: 局部堆积密度
   - Shell Density Ratio: 壳层密度比

2. 柔性与置信度特征 (3维)
   - Normalized B-factor: 归一化柔性
   - Local Flexibility Variance: 局部柔性方差
   - Relative Surface Exposure (RSE): 相对表面暴露度

3. 平滑化环境物理化学特征 (6维)
   - Env Hydrophobicity: 环境疏水性
   - Hydrophobic Contrast: 疏水对比度
   - Env Positive Charge: 环境正电荷密度
   - Env Negative Charge: 环境负电荷密度
   - Env H-bond Donors: 环境氢键供体密度
   - Env H-bond Acceptors: 环境氢键受体密度

总计：12维
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree
import warnings

try:
    import biotite.structure as bs
    from biotite.structure.io.pdb import PDBFile
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    warnings.warn("Biotite not installed")


# ============================================================================
# 1. 显式几何"口袋度"特征 (3维)
# ============================================================================

def compute_pocketness_features(ca_coords: np.ndarray) -> np.ndarray:
    """
    计算显式几何"口袋度"特征

    Args:
        ca_coords: C-alpha坐标 [N, 3]

    Returns:
        pocketness_features: [N, 3] 包含 [CX, Local_Packing_Density, Shell_Density_Ratio]
    """
    n_residues = len(ca_coords)
    features = np.zeros((n_residues, 3), dtype=np.float32)

    # 构建KD-Tree用于快速邻域查询
    tree = cKDTree(ca_coords)

    # 定义半径
    R_10 = 10.0  # Å
    R_6 = 6.0    # Å
    R_12 = 12.0  # Å

    # 预期体积（用于归一化）
    V_expected_10 = (4/3) * np.pi * (R_10 ** 3) / 100  # 粗略估计每个残基占100 Å³

    for i in range(n_residues):
        # 1. Convexity Index (CX)
        # 查询10Å半径内的邻居数
        neighbors_10 = tree.query_ball_point(ca_coords[i], R_10)
        N_10 = len(neighbors_10) - 1  # 排除自己

        # 归一化到0-1（口袋内残基CX接近1，凸起表面接近0）
        CX = min(N_10 / V_expected_10, 1.0)
        features[i, 0] = CX

        # 2. Local Packing Density
        # 6Å半径内的邻居数
        neighbors_6 = tree.query_ball_point(ca_coords[i], R_6)
        N_6 = len(neighbors_6) - 1

        # 归一化（通常6Å内有5-15个邻居）
        N_6_norm = N_6 / 15.0
        features[i, 1] = min(N_6_norm, 1.0)

        # 3. Shell Density Ratio
        # 12Å半径内的邻居数
        neighbors_12 = tree.query_ball_point(ca_coords[i], R_12)
        N_12 = len(neighbors_12) - 1

        # 计算比率（深口袋：高比率；浅表凹陷：低比率）
        ratio = N_6 / (N_12 + 1e-6)
        features[i, 2] = ratio

    return features


# ============================================================================
# 2. 柔性与置信度特征 (3维)
# ============================================================================

def compute_flexibility_features(
    pdb_path: str,
    ca_coords: np.ndarray,
    sasa_values: np.ndarray
) -> np.ndarray:
    """
    计算柔性与置信度特征

    Args:
        pdb_path: PDB文件路径
        ca_coords: C-alpha坐标 [N, 3]
        sasa_values: SASA值 [N] (来自原有14维特征)

    Returns:
        flexibility_features: [N, 3] 包含 [B_norm, B_local_std, RSE]
    """
    n_residues = len(ca_coords)
    features = np.zeros((n_residues, 3), dtype=np.float32)

    if not BIOTITE_AVAILABLE:
        warnings.warn("Biotite not available, returning zero flexibility features")
        return features

    try:
        # 读取PDB文件获取B-factor
        # 注意：biotite默认不读取B-factor，需要手动解析
        pdb_file = PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1, extra_fields=["b_factor", "occupancy"])

        # 只保留氨基酸
        atom_array = structure[bs.filter_amino_acids(structure) & ~structure.hetero]

        # 提取C-alpha的B-factor
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]

        if len(ca_atoms) != n_residues:
            warnings.warn(f"B-factor数量不匹配: {len(ca_atoms)} vs {n_residues}")
            return features

        # 检查是否有b_factor字段
        if not hasattr(ca_atoms, 'b_factor'):
            warnings.warn("PDB文件中没有B-factor信息，使用默认值")
            b_factors = np.ones(n_residues, dtype=np.float32) * 50.0  # 默认中等柔性
        else:
            b_factors = ca_atoms.b_factor.astype(np.float32)

        # 检查是否是AlphaFold结构（pLDDT）
        # AlphaFold的B-factor列存储pLDDT (0-100)
        if np.all((b_factors >= 0) & (b_factors <= 100)):
            # 反转pLDDT为柔性指标（100-pLDDT）
            b_factors = 100.0 - b_factors

        # 1. Normalized B-factor
        # Min-Max归一化到[0, 1]
        b_min = b_factors.min()
        b_max = b_factors.max()
        if b_max > b_min:
            b_norm = (b_factors - b_min) / (b_max - b_min)
        else:
            b_norm = np.zeros_like(b_factors)

        features[:, 0] = b_norm

        # 2. Local Flexibility Variance
        # 构建KD-Tree
        tree = cKDTree(ca_coords)

        for i in range(n_residues):
            # 找到K=8个最近邻
            distances, indices = tree.query(ca_coords[i], k=9)  # k=9包括自己
            neighbor_indices = indices[1:]  # 排除自己

            # 计算邻居B-factor的标准差
            neighbor_b = b_norm[neighbor_indices]
            b_local_std = np.std(neighbor_b)
            features[i, 1] = b_local_std

        # 3. Relative Surface Exposure (RSE)
        # RSE = (SASA / MaxSASA) * B-factor
        # 用于过滤内部高B-factor和表面低B-factor
        max_sasa = sasa_values.max() if sasa_values.max() > 0 else 1.0
        sasa_norm = sasa_values / max_sasa
        rse = sasa_norm * b_norm
        features[:, 2] = rse

    except Exception as e:
        warnings.warn(f"Failed to compute flexibility features: {e}")

    return features


# ============================================================================
# 3. 平滑化环境物理化学特征 (6维)
# ============================================================================

def compute_environmental_features(
    ca_coords: np.ndarray,
    physchem_features: np.ndarray,
    residue_names: List[str]
) -> np.ndarray:
    """
    计算平滑化环境物理化学特征

    Args:
        ca_coords: C-alpha坐标 [N, 3]
        physchem_features: 原有的42维物理化学特征 [N, 42]
        residue_names: 残基名称列表 (三字母代码)

    Returns:
        env_features: [N, 6] 包含环境特征
    """
    n_residues = len(ca_coords)
    features = np.zeros((n_residues, 6), dtype=np.float32)

    # 构建KD-Tree
    tree = cKDTree(ca_coords)

    # 定义高斯核参数
    sigma = 6.0  # Å
    cutoff = 8.0  # Å

    # 从42维特征中提取关键性质
    # 假设42维特征的组成（需要根据实际情况调整索引）:
    # [0-19]: 基础属性
    # [20-25]: SASA相关
    # [26-33]: 氢键特性
    # [34-37]: 疏水性/极性
    # [38-41]: 电荷/pKa

    # 提取疏水性（假设在索引34）
    if physchem_features.shape[1] >= 35:
        hydrophobicity = physchem_features[:, 34]
    else:
        # 如果没有，使用查找表
        from physicochemical_features import HYDROPHOBICITY
        hydrophobicity = np.array([HYDROPHOBICITY.get(res, 0.0) for res in residue_names])

    # 定义电荷残基
    positive_residues = {'ARG', 'LYS', 'HIS'}
    negative_residues = {'ASP', 'GLU'}

    # 定义氢键供体/受体残基（简化版）
    hbond_donor_residues = {'SER', 'THR', 'TYR', 'ASN', 'GLN', 'ARG', 'LYS', 'HIS', 'TRP'}
    hbond_acceptor_residues = {'ASP', 'GLU', 'ASN', 'GLN', 'SER', 'THR', 'TYR'}

    # 为每个残基计算环境特征
    for i in range(n_residues):
        # 查询cutoff范围内的邻居
        neighbors = tree.query_ball_point(ca_coords[i], cutoff)

        if len(neighbors) <= 1:  # 只有自己
            continue

        # 计算高斯权重
        weights = []
        neighbor_hydro = []
        neighbor_pos_charge = []
        neighbor_neg_charge = []
        neighbor_hbond_donors = []
        neighbor_hbond_acceptors = []

        for j in neighbors:
            if j == i:
                continue

            # 计算距离
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])

            # 高斯权重
            w = np.exp(-(dist ** 2) / (sigma ** 2))
            weights.append(w)

            # 收集邻居的性质
            neighbor_hydro.append(hydrophobicity[j])
            neighbor_pos_charge.append(1.0 if residue_names[j] in positive_residues else 0.0)
            neighbor_neg_charge.append(1.0 if residue_names[j] in negative_residues else 0.0)
            neighbor_hbond_donors.append(1.0 if residue_names[j] in hbond_donor_residues else 0.0)
            neighbor_hbond_acceptors.append(1.0 if residue_names[j] in hbond_acceptor_residues else 0.0)

        if len(weights) == 0:
            continue

        weights = np.array(weights)
        weight_sum = weights.sum()

        # 1. Env Hydrophobicity (加权平均)
        env_hydro = np.sum(weights * neighbor_hydro) / weight_sum
        features[i, 0] = env_hydro

        # 2. Hydrophobic Contrast (环境 - 自身)
        hydro_contrast = env_hydro - hydrophobicity[i]
        features[i, 1] = hydro_contrast

        # 3. Env Positive Charge (加权和)
        env_pos = np.sum(weights * neighbor_pos_charge) / weight_sum
        features[i, 2] = env_pos

        # 4. Env Negative Charge (加权和)
        env_neg = np.sum(weights * neighbor_neg_charge) / weight_sum
        features[i, 3] = env_neg

        # 5. Env H-bond Donors (加权和)
        env_donors = np.sum(weights * neighbor_hbond_donors) / weight_sum
        features[i, 4] = env_donors

        # 6. Env H-bond Acceptors (加权和)
        env_acceptors = np.sum(weights * neighbor_hbond_acceptors) / weight_sum
        features[i, 5] = env_acceptors

    return features


# ============================================================================
# 主函数：提取所有增强特征
# ============================================================================

def extract_enhanced_features_v2(
    pdb_path: str,
    ca_coords: np.ndarray,
    physchem_features: np.ndarray,
    sasa_values: np.ndarray,
    residue_names: List[str]
) -> np.ndarray:
    """
    提取所有V2增强特征（12维）

    Args:
        pdb_path: PDB文件路径
        ca_coords: C-alpha坐标 [N, 3]
        physchem_features: 原有的42维物理化学特征 [N, 42]
        sasa_values: SASA值 [N]
        residue_names: 残基名称列表

    Returns:
        enhanced_features: [N, 12] 增强特征
    """
    print(f"  提取V2增强特征 (12维)...")

    # 1. 显式几何"口袋度"特征 (3维)
    print(f"    - 计算口袋度特征...")
    pocketness_feats = compute_pocketness_features(ca_coords)

    # 2. 柔性与置信度特征 (3维)
    print(f"    - 计算柔性特征...")
    flexibility_feats = compute_flexibility_features(pdb_path, ca_coords, sasa_values)

    # 3. 平滑化环境物理化学特征 (6维)
    print(f"    - 计算环境特征...")
    env_feats = compute_environmental_features(ca_coords, physchem_features, residue_names)

    # 拼接所有特征
    enhanced_features = np.concatenate([
        pocketness_feats,      # 3维
        flexibility_feats,     # 3维
        env_feats              # 6维
    ], axis=1)

    print(f"    ✅ V2增强特征提取完成: {enhanced_features.shape}")

    return enhanced_features
