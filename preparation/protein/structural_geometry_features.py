#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构几何特征提取模块
包含：
1. 二级结构（DSSP）
2. 微分几何（Frenet标架：曲率、挠率）
"""

from __future__ import annotations

import numpy as np
import torch
import shutil
import math
import warnings
import tempfile
import os
import subprocess
from pathlib import Path

try:
    import pydssp
    PYDSSP_AVAILABLE = True
except ImportError:
    PYDSSP_AVAILABLE = False

try:
    from biotite.structure.io.pdb import PDBFile
    import biotite.structure as bs
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False

try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


# ============================================================================
# 1. 二级结构特征（DSSP）
# ============================================================================

# DSSP 8类二级结构
DSSP_CODES = {
    'H': 0,  # α-helix
    'B': 1,  # β-bridge
    'E': 2,  # β-strand (extended)
    'G': 3,  # 3-10 helix
    'I': 4,  # π-helix
    'T': 5,  # Turn
    'S': 6,  # Bend
    '-': 7,  # Coil/Loop
}

# pyDSSP (通常返回简化版SS，这里做映射)
PYDSSP_MAP = {
    'H': 'H',
    'E': 'E',
    'C': '-',
    '-': '-',
}

def _estimate_ss_from_coords(coords: np.ndarray) -> List[str]:
    """
    备用方案：基于C-alpha坐标估算二级结构
    使用简单的距离和角度规则
    """
    n = len(coords)
    ss = ['-'] * n
    
    if n < 4:
        return ss
        
    # 计算虚拟C-alpha二面角和距离
    for i in range(1, n-2):
        # 简单的距离判据：i和i+3距离
        # Alpha-helix: d(i, i+3) ~ 5.0-6.0 A
        # Beta-sheet: d(i, i+1) ~ 3.8 A (CA-CA distance)
        
        ca1, ca2 = coords[i], coords[i+1]
        dist_1_2 = np.linalg.norm(ca2 - ca1)
        
        if i+3 < n:
            ca4 = coords[i+3]
            dist_1_4 = np.linalg.norm(ca4 - ca1)
            
            if 4.5 < dist_1_4 < 6.5:
                ss[i] = 'H'
                ss[i+1] = 'H'
                continue
                
        if 3.2 < dist_1_2 < 4.0:
             # 这只是非常粗略的估计
             # 更准确的需要计算二面角，但这作为fallback足够了
             pass
             
    return ss

def compute_dssp_features(
    pdb_path: Path,
    residue_indices: List[int],
    chain_ids: List[str]
) -> np.ndarray:
    """
    使用pyDSSP或DSSP二进制文件计算二级结构特征
    """
    n_residues = len(residue_indices)
    dssp_features = np.zeros((n_residues, 8), dtype=np.float32)
    dssp_bin = shutil.which('mkdssp') or shutil.which('dssp')

    # 优先尝试 DSSP 二进制（8类）；无二进制时再尝试 pyDSSP（3类）
    if (not dssp_bin) and PYDSSP_AVAILABLE and BIOTITE_AVAILABLE:
        try:
            pdb_file = PDBFile.read(str(pdb_path))
            structure = pdb_file.get_structure(model=1)
            protein = structure[bs.filter_amino_acids(structure)]
            
            unique_chains = list(dict.fromkeys(chain_ids))
            py_dssp_map = {}
            
            for chain in unique_chains:
                chain_struct = protein[protein.chain_id == chain]
                if len(chain_struct) == 0: continue
                
                # 提取 N, CA, C 原子用于 pyDSSP
                # 注意：有些残基可能缺失原子，我们需要确保对应关系
                res_ids_in_chain = np.unique(chain_struct.res_id)
                coords_list = []
                valid_res_ids = []
                
                for rid in res_ids_in_chain:
                    res_atoms = chain_struct[chain_struct.res_id == rid]
                    atom_names = res_atoms.atom_name

                    # pyDSSP 需要 N, CA, C, O 四个原子
                    if "N" in atom_names and "CA" in atom_names and "C" in atom_names and "O" in atom_names:
                        n_c = res_atoms[res_atoms.atom_name == "N"].coord[0]
                        ca_c = res_atoms[res_atoms.atom_name == "CA"].coord[0]
                        c_c = res_atoms[res_atoms.atom_name == "C"].coord[0]
                        o_c = res_atoms[res_atoms.atom_name == "O"].coord[0]
                        coords_list.append([n_c, ca_c, c_c, o_c])
                        valid_res_ids.append(int(rid))
                
                if len(coords_list) < 3: continue

                try:
                    coords_tensor = torch.tensor(np.array(coords_list), dtype=torch.float32)
                    ss_array = pydssp.assign(coords_tensor)

                    for rid, ss in zip(valid_res_ids, ss_array):
                        py_dssp_map[(str(chain).strip(), rid)] = ss
                except Exception as chain_error:
                    # 单条链失败，跳过该链但继续处理其他链
                    warnings.warn(f"pyDSSP failed for chain {chain} in {pdb_path.name}: {str(chain_error)[:100]}")
                    continue

            # 填充结果
            success_count = 0
            for i, (res_idx, chain_id) in enumerate(zip(residue_indices, chain_ids)):
                key = (str(chain_id).strip(), res_idx)
                ss_char = py_dssp_map.get(key, '-')
                mapped_ss = PYDSSP_MAP.get(ss_char, '-')
                
                if mapped_ss in DSSP_CODES:
                    dssp_features[i, DSSP_CODES[mapped_ss]] = 1.0
                    success_count += 1
                else:
                    dssp_features[i, 7] = 1.0 # Coil
            
            if success_count > 0:
                return dssp_features
                
        except Exception as e:
            # pyDSSP 失败，降级到传统 DSSP
            pass

    # ==========================================
    # 优先方案：调用 DSSP 二进制文件并原生解析
    # ==========================================
    if not dssp_bin:
        dssp_features[:, 7] = 1.0
        return dssp_features

    tmp_file = None
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # mkdssp 4.x 对 PDB 格式要求严格，总是创建临时文件
        # 只保留 ATOM/TER/END 记录，避免 SEQRES 等记录导致解析错误
        fd, tmp_path = tempfile.mkstemp(suffix='.pdb', text=True)
        tmp_file = tmp_path

        with os.fdopen(fd, 'w') as f:
            # 添加必要的 PDB 头
            pdb_name = pdb_path.stem[:4].upper()
            f.write(f"HEADER    PROTEIN                                 01-JAN-00   {pdb_name}\n")

            # 只保留 ATOM/TER/END 记录
            for line in lines:
                if line.startswith(('ATOM', 'TER', 'END')):
                    f.write(line)

        target_pdb_file = tmp_file

        # 新版 mkdssp (4.x) 需要指定输出格式为经典 DSSP 格式
        result = subprocess.run(
            [dssp_bin, "--output-format", "dssp", target_pdb_file],
            capture_output=True, text=True, check=True
        )
        
        output_lines = result.stdout.splitlines()
        data_start = -1
        for i, line in enumerate(output_lines):
            if line.startswith('  #  RESIDUE'):
                data_start = i + 1
                break
        
        if data_start != -1:
            dssp_map = {}
            for line in output_lines[data_start:]:
                if len(line) < 17: continue
                try:
                    res_num_str = line[5:10].strip()
                    chain_id_str = line[11].strip()
                    ss_code = line[16]
                    if ss_code == ' ': ss_code = '-'
                    if res_num_str:
                        dssp_map[(chain_id_str, int(res_num_str))] = ss_code
                except: continue

            for i, (res_idx, chain_id) in enumerate(zip(residue_indices, chain_ids)):
                key = (str(chain_id).strip(), res_idx)
                ss_code = dssp_map.get(key, '-')
                if ss_code in DSSP_CODES:
                    dssp_features[i, DSSP_CODES[ss_code]] = 1.0
                else:
                    dssp_features[i, 7] = 1.0
            return dssp_features

    except subprocess.CalledProcessError as e:
        # mkdssp 执行失败，记录详细错误信息
        error_msg = f"DSSP failed for {pdb_path.name}: {e.stderr if e.stderr else str(e)}"
        warnings.warn(error_msg)
        dssp_features[:, 7] = 1.0
        return dssp_features
    except Exception as e:
        warnings.warn(f"DSSP failed for {pdb_path.name}: {str(e)}")
        dssp_features[:, 7] = 1.0
        return dssp_features
        
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try: os.remove(tmp_file)
            except: pass
            
    return dssp_features



# ============================================================================
# 2. 微分几何特征（Frenet标架）
# ============================================================================

def compute_frenet_features(
    ca_coords: np.ndarray,
    chain_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算Frenet标架的微分几何特征（per-residue）

    基于C-alpha坐标计算：
    - 切向量 (Tangent)
    - 法向量 (Normal)
    - 副法向量 (Binormal)
    - 曲率 (Curvature κ)
    - 挠率 (Torsion τ)

    Args:
        ca_coords: C-alpha坐标 [N_residues, 3]
        chain_ids: 每个残基的链ID列表

    Returns:
        tangent: [N, 3] 切向量
        normal: [N, 3] 法向量
        binormal: [N, 3] 副法向量
        curvature: [N, 1] 曲率
        torsion: [N, 1] 挠率
    """
    n_residues = len(ca_coords)

    # 转换为torch张量以便GPU计算
    coords = torch.from_numpy(ca_coords).float()

    # 初始化输出
    tangent = torch.zeros((n_residues, 3))
    normal = torch.zeros((n_residues, 3))
    binormal = torch.zeros((n_residues, 3))
    curvature = torch.zeros((n_residues, 1))
    torsion = torch.zeros((n_residues, 1))

    # 按链分别处理（避免跨链计算）
    unique_chains = list(dict.fromkeys(chain_ids))  # 保持顺序

    for chain in unique_chains:
        # 获取该链的残基索引
        chain_mask = [i for i, c in enumerate(chain_ids) if c == chain]
        if len(chain_mask) < 4:
            # 链太短，无法计算微分几何
            continue

        chain_coords = coords[chain_mask]  # [N_chain, 3]
        n_chain = len(chain_coords)

        # 1. 计算切向量 (一阶导数)
        # T_i = (r_{i+1} - r_{i-1}) / ||r_{i+1} - r_{i-1}||
        for i in range(n_chain):
            if i == 0:
                # 前向差分
                diff = chain_coords[i+1] - chain_coords[i]
            elif i == n_chain - 1:
                # 后向差分
                diff = chain_coords[i] - chain_coords[i-1]
            else:
                # 中心差分
                diff = chain_coords[i+1] - chain_coords[i-1]

            norm = torch.norm(diff)
            if norm > 1e-6:
                tangent[chain_mask[i]] = diff / norm
            else:
                tangent[chain_mask[i]] = torch.tensor([1.0, 0.0, 0.0])

        # 2. 计算曲率和法向量
        # κ = ||dT/ds||, N = (dT/ds) / κ
        for i in range(1, n_chain - 1):
            T_prev = tangent[chain_mask[i-1]]
            T_curr = tangent[chain_mask[i]]
            T_next = tangent[chain_mask[i+1]]

            # 切向量的导数
            dT = T_next - T_prev
            dT_norm = torch.norm(dT)

            if dT_norm > 1e-6:
                curvature[chain_mask[i]] = dT_norm
                normal[chain_mask[i]] = dT / dT_norm
            else:
                curvature[chain_mask[i]] = 0.0
                # 默认法向量（垂直于切向量）
                T = T_curr
                if abs(T[0]) < 0.9:
                    perp = torch.tensor([1.0, 0.0, 0.0])
                else:
                    perp = torch.tensor([0.0, 1.0, 0.0])
                N = perp - torch.dot(perp, T) * T
                N = N / torch.norm(N)
                normal[chain_mask[i]] = N

        # 3. 计算副法向量
        # B = T × N
        for i in range(n_chain):
            T = tangent[chain_mask[i]]
            N = normal[chain_mask[i]]
            B = torch.linalg.cross(T, N)
            binormal[chain_mask[i]] = B

        # 4. 计算挠率
        # τ = -(dB/ds) · N
        for i in range(1, n_chain - 1):
            B_prev = binormal[chain_mask[i-1]]
            B_next = binormal[chain_mask[i+1]]
            N_curr = normal[chain_mask[i]]

            dB = B_next - B_prev
            tau = -torch.dot(dB, N_curr)
            torsion[chain_mask[i]] = tau.unsqueeze(0)

    return (
        tangent.numpy(),
        normal.numpy(),
        binormal.numpy(),
        curvature.numpy(),
        torsion.numpy()
    )


def compute_geometry_features(
    ca_coords: np.ndarray,
    chain_ids: List[str]
) -> np.ndarray:
    """
    计算完整的几何特征（per-residue）

    包含：
    - 曲率 (1维)
    - 挠率 (1维)
    - 曲率RBF编码 (8维)
    - 挠率RBF编码 (8维)

    Args:
        ca_coords: C-alpha坐标 [N_residues, 3]
        chain_ids: 每个残基的链ID列表

    Returns:
        [N_residues, 18] 几何特征矩阵
    """
    n_residues = len(ca_coords)
    geometry_features = np.zeros((n_residues, 18), dtype=np.float32)

    # 计算Frenet标架
    tangent, normal, binormal, curvature, torsion = compute_frenet_features(
        ca_coords, chain_ids
    )

    # 1. 原始曲率和挠率
    geometry_features[:, 0] = curvature.squeeze()
    geometry_features[:, 1] = torsion.squeeze()

    # 2. 曲率RBF编码 (8个高斯基函数)
    # 典型曲率范围：0-2 (1/Å)
    kappa_centers = np.linspace(0, 2, 8)
    kappa_width = 0.3
    for i, center in enumerate(kappa_centers):
        geometry_features[:, 2+i] = np.exp(-((curvature.squeeze() - center) / kappa_width) ** 2)

    # 3. 挠率RBF编码 (8个高斯基函数)
    # 典型挠率范围：-1 到 1 (1/Å)
    tau_centers = np.linspace(-1, 1, 8)
    tau_width = 0.3
    for i, center in enumerate(tau_centers):
        geometry_features[:, 10+i] = np.exp(-((torsion.squeeze() - center) / tau_width) ** 2)

    return geometry_features


# ============================================================================
# 3. 组合结构几何特征
# ============================================================================

def extract_structural_geometry_features(
    pdb_path: Path,
    ca_coords: np.ndarray,
    residue_indices: List[int],
    chain_ids: List[str]
) -> np.ndarray:
    """
    提取完整的结构几何特征（per-residue）

    包含：
    - DSSP二级结构 (8维 one-hot)
    - 微分几何 (18维)

    Args:
        pdb_path: PDB文件路径
        ca_coords: C-alpha坐标 [N_residues, 3]
        residue_indices: PDB残基编号列表
        chain_ids: 每个残基的链ID列表

    Returns:
        [N_residues, 26] 结构几何特征矩阵
    """
    n_residues = len(residue_indices)
    features = np.zeros((n_residues, 26), dtype=np.float32)

    # 1. DSSP二级结构 (8维)
    dssp_features = compute_dssp_features(pdb_path, residue_indices, chain_ids)
    features[:, :8] = dssp_features

    # 2. 微分几何 (18维)
    geometry_features = compute_geometry_features(ca_coords, chain_ids)
    features[:, 8:26] = geometry_features

    return features


# ============================================================================
# 4. 测试和验证
# ============================================================================

def test_structural_geometry():
    """测试结构几何特征提取"""
    print("=== 测试结构几何特征提取 ===\n")

    # 创建测试数据（α-螺旋）
    # α-螺旋的典型参数：pitch=5.4Å, radius=2.3Å, 3.6残基/圈
    n_residues = 20
    t = np.linspace(0, 4*np.pi, n_residues)
    radius = 2.3
    pitch = 5.4 / (2*np.pi)

    ca_coords = np.zeros((n_residues, 3))
    ca_coords[:, 0] = radius * np.cos(t)
    ca_coords[:, 1] = radius * np.sin(t)
    ca_coords[:, 2] = pitch * t

    chain_ids = ['A'] * n_residues

    print("1. 测试微分几何计算:")
    tangent, normal, binormal, curvature, torsion = compute_frenet_features(
        ca_coords, chain_ids
    )

    print(f"   曲率范围: [{curvature.min():.4f}, {curvature.max():.4f}]")
    print(f"   挠率范围: [{torsion.min():.4f}, {torsion.max():.4f}]")
    print(f"   平均曲率: {curvature.mean():.4f} (理论值: {1/radius:.4f})")
    print()

    print("2. 测试几何特征编码:")
    geometry_features = compute_geometry_features(ca_coords, chain_ids)
    print(f"   特征shape: {geometry_features.shape}")
    print(f"   特征统计: mean={geometry_features.mean():.4f}, std={geometry_features.std():.4f}")
    print()

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_structural_geometry()
