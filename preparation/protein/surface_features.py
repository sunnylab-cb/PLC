#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表面指纹特征提取模块 (Surface Fingerprints)

基于Cα坐标和氨基酸性质的快速表面特征计算
不依赖MSMS/APBS，使用几何和物理化学近似

输出：14维per-residue表面特征
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install: pip install scipy")

# ============================================================================
# 氨基酸电荷和疏水性数据
# ============================================================================

# 氨基酸电荷（pH 7.0）
AA_CHARGES = {
    'ALA': 0.0, 'CYS': 0.0, 'ASP': -1.0, 'GLU': -1.0,
    'PHE': 0.0, 'GLY': 0.0, 'HIS': 0.5,  'ILE': 0.0,
    'LYS': 1.0, 'LEU': 0.0, 'MET': 0.0,  'ASN': 0.0,
    'PRO': 0.0, 'GLN': 0.0, 'ARG': 1.0,  'SER': 0.0,
    'THR': 0.0, 'VAL': 0.0, 'TRP': 0.0,  'TYR': 0.0
}

# Kyte-Doolittle疏水性指数
AA_HYDROPHOBICITY = {
    'ALA': 1.8,  'CYS': 2.5,  'ASP': -3.5, 'GLU': -3.5,
    'PHE': 2.8,  'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LYS': -3.9, 'LEU': 3.8,  'MET': 1.9,  'ASN': -3.5,
    'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5, 'SER': -0.8,
    'THR': -0.7, 'VAL': 4.2,  'TRP': -0.9, 'TYR': -1.3
}

# 氨基酸极性
AA_POLARITY = {
    'ALA': 0, 'CYS': 1, 'ASP': 1, 'GLU': 1,
    'PHE': 0, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LYS': 1, 'LEU': 0, 'MET': 0, 'ASN': 1,
    'PRO': 0, 'GLN': 1, 'ARG': 1, 'SER': 1,
    'THR': 1, 'VAL': 0, 'TRP': 0, 'TYR': 1
}


# ============================================================================
# 几何特征计算
# ============================================================================

def compute_local_curvature(ca_coords: np.ndarray, window: int = 5) -> np.ndarray:
    """
    计算局部曲率（基于Cα坐标）

    使用滑动窗口拟合圆，计算曲率半径的倒数
    """
    n_residues = len(ca_coords)
    curvature = np.zeros(n_residues, dtype=np.float32)

    for i in range(n_residues):
        start = max(0, i - window // 2)
        end = min(n_residues, i + window // 2 + 1)

        if end - start < 3:
            continue

        # 取局部窗口的坐标
        local_coords = ca_coords[start:end]

        # 计算二阶差分近似曲率
        if i > 0 and i < n_residues - 1:
            v1 = ca_coords[i] - ca_coords[i-1]
            v2 = ca_coords[i+1] - ca_coords[i]

            # 曲率 = |dT/ds| ≈ |v2 - v1| / |v1|
            dv = v2 - v1
            curvature[i] = np.linalg.norm(dv) / (np.linalg.norm(v1) + 1e-6)

    # 归一化到[0, 1]
    if curvature.max() > 0:
        curvature = curvature / (curvature.max() + 1e-6)

    return curvature


def compute_surface_depth(ca_coords: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    """
    计算表面深度（residue depth）

    使用KNN距离的平均值作为深度指标
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(len(ca_coords), dtype=np.float32)

    n_residues = len(ca_coords)
    depth = np.zeros(n_residues, dtype=np.float32)

    # 构建KD树
    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        # 查找k个最近邻（包括自己）
        distances, _ = tree.query(ca_coords[i], k=min(k_neighbors + 1, n_residues))

        # 平均距离作为深度
        depth[i] = np.mean(distances[1:])  # 排除自己（距离=0）

    # 归一化
    if depth.max() > 0:
        depth = depth / depth.max()

    return depth


def compute_local_density(ca_coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """
    计算局部密度（radius范围内的残基数）
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(len(ca_coords), dtype=np.float32)

    n_residues = len(ca_coords)
    density = np.zeros(n_residues, dtype=np.float32)

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        # 查找radius范围内的残基
        neighbors = tree.query_ball_point(ca_coords[i], radius)
        density[i] = len(neighbors) - 1  # 排除自己

    # 归一化
    if density.max() > 0:
        density = density / density.max()

    return density


def compute_surface_roughness(ca_coords: np.ndarray, window: int = 7) -> np.ndarray:
    """
    计算表面粗糙度（局部坐标的标准差）
    """
    n_residues = len(ca_coords)
    roughness = np.zeros(n_residues, dtype=np.float32)

    for i in range(n_residues):
        start = max(0, i - window // 2)
        end = min(n_residues, i + window // 2 + 1)

        local_coords = ca_coords[start:end]

        # 计算局部坐标的标准差
        if len(local_coords) > 1:
            roughness[i] = np.std(local_coords)

    # 归一化
    if roughness.max() > 0:
        roughness = roughness / roughness.max()

    return roughness


def compute_convexity(ca_coords: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    """
    计算凹凸性（convexity）

    正值=凸出，负值=凹陷
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(len(ca_coords), dtype=np.float32)

    n_residues = len(ca_coords)
    convexity = np.zeros(n_residues, dtype=np.float32)

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        # 查找k个最近邻
        distances, indices = tree.query(ca_coords[i], k=min(k_neighbors + 1, n_residues))

        if len(indices) < 4:
            continue

        # 计算邻居的质心
        neighbors = ca_coords[indices[1:]]  # 排除自己
        centroid = np.mean(neighbors, axis=0)

        # 当前点到质心的向量
        vec_to_centroid = centroid - ca_coords[i]

        # 凸度 = 到质心的距离（正=凸出，负=凹陷）
        convexity[i] = np.linalg.norm(vec_to_centroid)

    # 归一化到[-1, 1]
    if convexity.max() > 0:
        convexity = 2 * (convexity / convexity.max()) - 1

    return convexity


# ============================================================================
# 静电特征计算
# ============================================================================

def compute_electrostatic_potential(
    ca_coords: np.ndarray,
    residue_names: List[str],
    radius: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算简化静电势（基于氨基酸电荷）

    使用Coulomb定律的简化版本：V = Σ(q_i / r_i)

    Returns:
        potential: 静电势 [N]
        charge_density: 局部电荷密度 [N]
    """
    if not SCIPY_AVAILABLE:
        n = len(ca_coords)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    n_residues = len(ca_coords)
    potential = np.zeros(n_residues, dtype=np.float32)
    charge_density = np.zeros(n_residues, dtype=np.float32)

    # 获取每个残基的电荷
    charges = np.array([AA_CHARGES.get(res, 0.0) for res in residue_names])

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        # 查找radius范围内的残基
        indices = tree.query_ball_point(ca_coords[i], radius)

        if len(indices) <= 1:
            continue

        # 计算静电势（简化Coulomb定律）
        for j in indices:
            if i == j:
                continue

            distance = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if distance < 0.1:
                continue

            # V = q / r (简化，忽略介电常数)
            potential[i] += charges[j] / distance

        # 局部电荷密度
        local_charges = charges[indices]
        charge_density[i] = np.sum(np.abs(local_charges)) / len(indices)

    # 归一化
    if np.abs(potential).max() > 0:
        potential = potential / (np.abs(potential).max() + 1e-6)

    if charge_density.max() > 0:
        charge_density = charge_density / charge_density.max()

    return potential, charge_density


def compute_polarity_ratio(
    ca_coords: np.ndarray,
    residue_names: List[str],
    radius: float = 8.0
) -> np.ndarray:
    """
    计算局部极性/非极性残基比例
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(len(ca_coords), dtype=np.float32)

    n_residues = len(ca_coords)
    polarity_ratio = np.zeros(n_residues, dtype=np.float32)

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        indices = tree.query_ball_point(ca_coords[i], radius)

        if len(indices) == 0:
            continue

        # 统计极性残基比例
        polar_count = sum(AA_POLARITY.get(residue_names[j], 0) for j in indices)
        polarity_ratio[i] = polar_count / len(indices)

    return polarity_ratio


# ============================================================================
# 疏水性特征计算
# ============================================================================

def compute_hydrophobicity_features(
    ca_coords: np.ndarray,
    residue_names: List[str],
    radius: float = 8.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算疏水性相关特征

    Returns:
        hydrophobicity: 残基疏水性指数 [N]
        local_hydro_ratio: 局部疏水/亲水比例 [N]
        surface_hydrophobicity: 表面疏水性（加权） [N]
    """
    n_residues = len(ca_coords)

    # 1. 残基疏水性指数
    hydrophobicity = np.array([
        AA_HYDROPHOBICITY.get(res, 0.0) for res in residue_names
    ], dtype=np.float32)

    # 归一化到[0, 1]
    hydro_min = min(AA_HYDROPHOBICITY.values())
    hydro_max = max(AA_HYDROPHOBICITY.values())
    hydrophobicity = (hydrophobicity - hydro_min) / (hydro_max - hydro_min + 1e-6)

    if not SCIPY_AVAILABLE:
        return hydrophobicity, np.zeros(n_residues), np.zeros(n_residues)

    # 2. 局部疏水/亲水比例
    local_hydro_ratio = np.zeros(n_residues, dtype=np.float32)
    surface_hydrophobicity = np.zeros(n_residues, dtype=np.float32)

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        indices = tree.query_ball_point(ca_coords[i], radius)

        if len(indices) == 0:
            continue

        # 局部疏水残基比例
        local_hydro = [AA_HYDROPHOBICITY.get(residue_names[j], 0.0) for j in indices]
        hydrophobic_count = sum(1 for h in local_hydro if h > 0)
        local_hydro_ratio[i] = hydrophobic_count / len(indices)

        # 表面疏水性（距离加权）
        for j in indices:
            if i == j:
                weight = 1.0
            else:
                distance = np.linalg.norm(ca_coords[i] - ca_coords[j])
                weight = 1.0 / (distance + 1.0)

            hydro_value = AA_HYDROPHOBICITY.get(residue_names[j], 0.0)
            surface_hydrophobicity[i] += weight * hydro_value

        surface_hydrophobicity[i] /= len(indices)

    # 归一化surface_hydrophobicity
    if np.abs(surface_hydrophobicity).max() > 0:
        surface_hydrophobicity = (surface_hydrophobicity - surface_hydrophobicity.min()) / \
                                 (surface_hydrophobicity.max() - surface_hydrophobicity.min() + 1e-6)

    return hydrophobicity, local_hydro_ratio, surface_hydrophobicity


# ============================================================================
# 映射策略特征
# ============================================================================

def compute_mapping_features(
    ca_coords: np.ndarray,
    residue_names: List[str],
    property_func
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用三种映射策略计算特征

    Args:
        property_func: 函数，输入residue_name，返回property值

    Returns:
        nearest: 最近邻值
        weighted_avg: 加权平均值
        distance_weighted: 距离加权值
    """
    n_residues = len(ca_coords)

    # 获取所有残基的property值
    properties = np.array([property_func(res) for res in residue_names], dtype=np.float32)

    if not SCIPY_AVAILABLE:
        return properties, properties, properties

    nearest = np.zeros(n_residues, dtype=np.float32)
    weighted_avg = np.zeros(n_residues, dtype=np.float32)
    distance_weighted = np.zeros(n_residues, dtype=np.float32)

    tree = cKDTree(ca_coords)

    for i in range(n_residues):
        # 1. 最近邻（自己）
        nearest[i] = properties[i]

        # 2. 加权平均（5个最近邻）
        distances, indices = tree.query(ca_coords[i], k=min(6, n_residues))
        if len(indices) > 1:
            weighted_avg[i] = np.mean(properties[indices[1:]])  # 排除自己
        else:
            weighted_avg[i] = properties[i]

        # 3. 距离加权（10Å范围内）
        indices_radius = tree.query_ball_point(ca_coords[i], 10.0)
        if len(indices_radius) > 1:
            weights = []
            values = []
            for j in indices_radius:
                if i == j:
                    continue
                distance = np.linalg.norm(ca_coords[i] - ca_coords[j])
                weight = 1.0 / (distance + 1.0)
                weights.append(weight)
                values.append(properties[j])

            if weights:
                distance_weighted[i] = np.average(values, weights=weights)
            else:
                distance_weighted[i] = properties[i]
        else:
            distance_weighted[i] = properties[i]

    return nearest, weighted_avg, distance_weighted


# ============================================================================
# 主函数
# ============================================================================

def extract_surface_features(
    ca_coords: np.ndarray,
    residue_names: List[str]
) -> np.ndarray:
    """
    提取完整的14维表面指纹特征

    Args:
        ca_coords: Cα坐标 [N_residues, 3]
        residue_names: 残基名称列表（三字母代码）

    Returns:
        surface_features: [N_residues, 14] 表面特征矩阵
    """
    n_residues = len(ca_coords)
    surface_features = np.zeros((n_residues, 14), dtype=np.float32)

    # 1. 几何特征 (5维)
    surface_features[:, 0] = compute_local_curvature(ca_coords)
    surface_features[:, 1] = compute_convexity(ca_coords)
    surface_features[:, 2] = compute_surface_roughness(ca_coords)
    surface_features[:, 3] = compute_local_density(ca_coords)
    surface_features[:, 4] = compute_surface_depth(ca_coords)

    # 2. 静电特征 (3维)
    potential, charge_density = compute_electrostatic_potential(ca_coords, residue_names)
    surface_features[:, 5] = potential
    surface_features[:, 6] = charge_density
    surface_features[:, 7] = compute_polarity_ratio(ca_coords, residue_names)

    # 3. 疏水性特征 (3维)
    hydro, hydro_ratio, surface_hydro = compute_hydrophobicity_features(ca_coords, residue_names)
    surface_features[:, 8] = hydro
    surface_features[:, 9] = hydro_ratio
    surface_features[:, 10] = surface_hydro

    # 4. 映射策略特征 (3维) - 使用电荷作为示例
    nearest, weighted, dist_weighted = compute_mapping_features(
        ca_coords, residue_names,
        lambda res: AA_CHARGES.get(res, 0.0)
    )
    surface_features[:, 11] = nearest
    surface_features[:, 12] = weighted
    surface_features[:, 13] = dist_weighted

    return surface_features
