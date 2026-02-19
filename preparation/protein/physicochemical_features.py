#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理化学特征提取模块
为每个残基提供微观化学环境描述（42维特征）

特征组成：
- 基础属性（查表法）：20维
- SASA相关：6维
- 氢键特性：8维
- 疏水性/极性：4维
- 电荷/pKa：4维
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# 尝试导入依赖库
try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    warnings.warn("FreeSASA not installed. SASA features will be zeros.")

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.PDB import PDBParser, SASA as BioPDB_SASA
    import biotite.structure as bs
    from biotite.structure.io.pdb import PDBFile
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("Biopython/Biotite not fully installed. Some features will be limited.")

# 导入氢键网络特征
try:
    from hbond_network_features import compute_hbond_features
    HBOND_AVAILABLE = True
except ImportError:
    HBOND_AVAILABLE = False
    warnings.warn("Hydrogen bond features not available")


# ============================================================================
# 1. 氨基酸静态属性查找表 (Lookup Tables)
# ============================================================================

# 20种标准氨基酸
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
               'THR', 'TRP', 'TYR', 'VAL']

# 单字母到三字母映射
AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

AA_3TO1 = {v: k for k, v in AA_1TO3.items()}


# 疏水性指数 (Kyte-Doolittle scale)
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

# 分子量 (Da)
MOLECULAR_WEIGHT = {
    'ALA': 89.1, 'ARG': 174.2, 'ASN': 132.1, 'ASP': 133.1, 'CYS': 121.2,
    'GLN': 146.2, 'GLU': 147.1, 'GLY': 75.1, 'HIS': 155.2, 'ILE': 131.2,
    'LEU': 131.2, 'LYS': 146.2, 'MET': 149.2, 'PHE': 165.2, 'PRO': 115.1,
    'SER': 105.1, 'THR': 119.1, 'TRP': 204.2, 'TYR': 181.2, 'VAL': 117.1
}

# 侧链体积 (Å³)
SIDE_CHAIN_VOLUME = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
}

# 侧链极性 (0=非极性, 1=极性不带电, 2=正电, 3=负电)
POLARITY = {
    'ALA': 0, 'ARG': 2, 'ASN': 1, 'ASP': 3, 'CYS': 1,
    'GLN': 1, 'GLU': 3, 'GLY': 0, 'HIS': 2, 'ILE': 0,
    'LEU': 0, 'LYS': 2, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 0, 'TYR': 1, 'VAL': 0
}

# 氢键供体数量 (Donor)
H_BOND_DONORS = {
    'ALA': 0, 'ARG': 4, 'ASN': 1, 'ASP': 0, 'CYS': 1,
    'GLN': 1, 'GLU': 0, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 1, 'TYR': 1, 'VAL': 0
}

# 氢键受体数量 (Acceptor)
H_BOND_ACCEPTORS = {
    'ALA': 0, 'ARG': 0, 'ASN': 2, 'ASP': 2, 'CYS': 0,
    'GLN': 2, 'GLU': 2, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LEU': 0, 'LYS': 0, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 0, 'TYR': 1, 'VAL': 0
}

# 芳香性 (0=非芳香, 1=芳香)
AROMATICITY = {
    'ALA': 0, 'ARG': 0, 'ASN': 0, 'ASP': 0, 'CYS': 0,
    'GLN': 0, 'GLU': 0, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LEU': 0, 'LYS': 0, 'MET': 0, 'PHE': 1, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 1, 'TYR': 1, 'VAL': 0
}

# pKa值 (侧链可电离基团的pKa)
PKA_VALUES = {
    'ALA': None, 'ARG': 12.48, 'ASN': None, 'ASP': 3.65, 'CYS': 8.18,
    'GLN': None, 'GLU': 4.25, 'GLY': None, 'HIS': 6.00, 'ILE': None,
    'LEU': None, 'LYS': 10.53, 'MET': None, 'PHE': None, 'PRO': None,
    'SER': None, 'THR': None, 'TRP': None, 'TYR': 10.07, 'VAL': None
}

# 电荷 (pH=7.0时的净电荷)
CHARGE_AT_PH7 = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.1, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}


# ============================================================================
# 2. 特征提取函数
# ============================================================================

def get_basic_features(residue_name: str) -> np.ndarray:
    """
    获取氨基酸的基础静态属性（查表法）

    Args:
        residue_name: 三字母氨基酸代码 (如 'ALA')

    Returns:
        20维特征向量
    """
    if residue_name not in AMINO_ACIDS:
        # 未知残基，返回零向量
        return np.zeros(20, dtype=np.float32)

    features = np.array([
        HYDROPHOBICITY[residue_name] / 4.5,  # 归一化到[-1, 1]
        MOLECULAR_WEIGHT[residue_name] / 204.2,  # 归一化到[0, 1]
        SIDE_CHAIN_VOLUME[residue_name] / 227.8,  # 归一化
        POLARITY[residue_name] / 3.0,  # 归一化到[0, 1]
        H_BOND_DONORS[residue_name] / 4.0,  # 归一化
        H_BOND_ACCEPTORS[residue_name] / 2.0,  # 归一化
        AROMATICITY[residue_name],  # 0或1
        CHARGE_AT_PH7[residue_name],  # -1, 0, 0.1, 1

        # One-hot编码极性类别 (4维)
        1.0 if POLARITY[residue_name] == 0 else 0.0,  # 非极性
        1.0 if POLARITY[residue_name] == 1 else 0.0,  # 极性不带电
        1.0 if POLARITY[residue_name] == 2 else 0.0,  # 正电
        1.0 if POLARITY[residue_name] == 3 else 0.0,  # 负电

        # pKa相关 (4维)
        1.0 if PKA_VALUES[residue_name] is not None else 0.0,  # 是否可电离
        (PKA_VALUES[residue_name] / 14.0) if PKA_VALUES[residue_name] else 0.0,  # pKa归一化
        1.0 if PKA_VALUES[residue_name] and PKA_VALUES[residue_name] < 7.0 else 0.0,  # 酸性
        1.0 if PKA_VALUES[residue_name] and PKA_VALUES[residue_name] > 7.0 else 0.0,  # 碱性

        # 疏水性分类 (4维)
        1.0 if HYDROPHOBICITY[residue_name] > 2.0 else 0.0,  # 强疏水
        1.0 if 0 < HYDROPHOBICITY[residue_name] <= 2.0 else 0.0,  # 弱疏水
        1.0 if -2.0 <= HYDROPHOBICITY[residue_name] <= 0 else 0.0,  # 弱亲水
        1.0 if HYDROPHOBICITY[residue_name] < -2.0 else 0.0,  # 强亲水
    ], dtype=np.float32)

    return features


def compute_sasa_features(pdb_path: Path, residue_names: List[str], residue_indices: List[int], chain_ids: List[str]) -> np.ndarray:
    """
    使用BioPython计算每个残基的溶剂可及表面积特征（真正的per-residue，支持多链）

    Args:
        pdb_path: PDB文件路径
        residue_names: 残基名称列表（三字母代码）
        residue_indices: PDB残基编号列表
        chain_ids: 每个残基的链ID列表

    Returns:
        [N_residues, 6] SASA特征矩阵
    """
    n_residues = len(residue_names)
    sasa_features = np.zeros((n_residues, 6), dtype=np.float32)

    if not BIOPYTHON_AVAILABLE:
        return sasa_features

    try:
        # 使用BioPython计算SASA
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))

        # 使用Shrake-Rupley算法计算per-residue SASA
        sr = BioPDB_SASA.ShrakeRupley()
        sr.compute(structure, level='R')  # R = residue level

        # 标准最大SASA值（Tien et al. 2013）
        max_sasa = {
            'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
            'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
            'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
            'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
        }

        # 创建(chain_id, residue_index)到SASA的映射
        sasa_dict = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.id[1]  # PDB残基编号
                    chain_id = chain.id
                    key = (chain_id, res_id)
                    sasa_dict[key] = residue.sasa

        # 为每个残基填充SASA特征
        for i, (res_name, res_idx, chain_id) in enumerate(zip(residue_names, residue_indices, chain_ids)):
            key = (chain_id, res_idx)
            if key in sasa_dict:
                total_sasa = sasa_dict[key]

                # 计算相对SASA
                max_val = max_sasa.get(res_name, 200.0)
                relative_sasa = total_sasa / max_val if max_val > 0 else 0.0

                # 估计极性/非极性SASA（基于残基类型）
                polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
                if res_name in polar_residues:
                    polar_ratio = 0.7
                else:
                    polar_ratio = 0.3

                polar_sasa = total_sasa * polar_ratio
                apolar_sasa = total_sasa * (1 - polar_ratio)

                # 构建6维SASA特征（真正的per-residue）
                sasa_features[i] = np.array([
                    total_sasa / 300.0,  # 总SASA归一化
                    min(relative_sasa, 1.0),  # 相对SASA（限制在[0,1]）
                    polar_sasa / 200.0,  # 极性SASA归一化
                    apolar_sasa / 200.0,  # 非极性SASA归一化
                    1.0 if relative_sasa > 0.5 else 0.0,  # 是否暴露
                    1.0 if relative_sasa < 0.2 else 0.0,  # 是否埋藏
                ], dtype=np.float32)

        return sasa_features

    except Exception as e:
        warnings.warn(f"SASA calculation failed for {pdb_path}: {e}")
        return sasa_features


def compute_sequence_features(sequence: str, residue_names: List[str]) -> Dict[int, np.ndarray]:
    """
    计算per-residue序列特征（真正的per-residue）

    Args:
        sequence: 氨基酸序列（单字母代码）
        residue_names: 残基名称列表（三字母代码）

    Returns:
        字典: residue_index -> 8维特征向量
    """
    if not BIOPYTHON_AVAILABLE:
        return {}

    try:
        # 使用ProteinAnalysis计算整体序列属性（用于局部窗口）
        analyzer = ProteinAnalysis(sequence)

        # 计算二级结构倾向（整体）
        helix_global, turn_global, sheet_global = analyzer.secondary_structure_fraction()

        # 为每个残基计算per-residue特征
        seq_features = {}
        window_size = 7  # 局部窗口大小

        for i, aa in enumerate(sequence):
            # 1. 局部窗口序列（用于计算局部属性）
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            local_seq = sequence[start:end]

            # 2. 计算局部属性
            try:
                local_analyzer = ProteinAnalysis(local_seq)
                local_aromaticity = local_analyzer.aromaticity()
                local_helix, local_turn, local_sheet = local_analyzer.secondary_structure_fraction()
            except:
                local_aromaticity = 0.0
                local_helix, local_turn, local_sheet = helix_global, turn_global, sheet_global

            # 3. 当前残基的per-residue属性
            is_aromatic = 1.0 if aa in ['F', 'W', 'Y', 'H'] else 0.0
            is_charged = 1.0 if aa in ['R', 'K', 'D', 'E', 'H'] else 0.0
            is_polar = 1.0 if aa in ['S', 'T', 'N', 'Q', 'C', 'Y'] else 0.0

            # 4. 位置特征
            relative_position = (i + 1) / len(sequence)
            is_terminal = 1.0 if i < 10 or i >= len(sequence) - 10 else 0.0

            # 5. 构建8维per-residue特征
            features = np.array([
                is_aromatic,  # 该残基是否芳香
                is_charged,  # 该残基是否带电
                is_polar,  # 该残基是否极性
                local_aromaticity,  # 局部窗口芳香性
                local_helix,  # 局部α-螺旋倾向
                local_turn,  # 局部转角倾向
                local_sheet,  # 局部β-折叠倾向
                relative_position,  # 序列位置归一化
            ], dtype=np.float32)

            seq_features[i] = features

        return seq_features

    except Exception as e:
        warnings.warn(f"Sequence feature calculation failed: {e}")
        return {}


def extract_physicochemical_features(
    pdb_path: Path,
    sequence: str,
    residue_names: List[str],
    residue_indices: List[int],
    chain_ids: List[str]
) -> np.ndarray:
    """
    提取完整的42维物理化学特征（真正的per-residue，支持多链）

    Args:
        pdb_path: PDB文件路径
        sequence: 氨基酸序列（单字母代码）
        residue_names: 残基名称列表（三字母代码）
        residue_indices: PDB残基编号列表
        chain_ids: 每个残基的链ID列表

    Returns:
        [N_residues, 42] 特征矩阵（每个残基一个特征向量）
    """
    n_residues = len(residue_names)
    features = np.zeros((n_residues, 42), dtype=np.float32)

    # 1. 基础属性 (20维) - per-residue
    for i, res_name in enumerate(residue_names):
        features[i, :20] = get_basic_features(res_name)

    # 2. SASA特征 (6维) - per-residue
    sasa_features = compute_sasa_features(pdb_path, residue_names, residue_indices, chain_ids)
    features[:, 20:26] = sasa_features

    # 3. 序列特征 (8维) - per-residue
    seq_dict = compute_sequence_features(sequence, residue_names)
    for i in range(n_residues):
        if i in seq_dict:
            features[i, 26:34] = seq_dict[i]

    # 4. 氢键网络特征 (8维) - per-residue
    if HBOND_AVAILABLE:
        hbond_features = compute_hbond_features(pdb_path, residue_indices, chain_ids)
        features[:, 34:42] = hbond_features

    return features


# ============================================================================
# 3. 批量处理工具
# ============================================================================

def batch_extract_from_esm3_latents(
    latent_dir: Path,
    output_dir: Path,
    max_files: int = None
) -> None:
    """
    从ESM3 latent文件批量提取物理化学特征

    Args:
        latent_dir: ESM3 latent文件目录
        output_dir: 输出目录
        max_files: 最大处理文件数
    """
    import torch

    latent_files = sorted(latent_dir.rglob("*.pt"))
    if max_files:
        latent_files = latent_files[:max_files]

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, latent_path in enumerate(latent_files, 1):
        try:
            # 加载ESM3 latent数据
            data = torch.load(latent_path, weights_only=False)

            sequence = data['sequence']
            residue_names = data['residue_name_3']
            pdb_path = Path(data['pdb_path'])
            chain_id = data['chain_id'][0] if data['chain_id'] else None

            # 提取物理化学特征
            physchem_features = extract_physicochemical_features(
                pdb_path, sequence, residue_names, chain_id
            )

            # 保存特征
            rel_path = latent_path.relative_to(latent_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 将特征添加到原始数据中
            data['physicochemical_features'] = torch.from_numpy(physchem_features)
            torch.save(data, out_path)

            print(f"[{idx}/{len(latent_files)}] Processed {rel_path}")

        except Exception as e:
            print(f"[{idx}/{len(latent_files)}] Failed {latent_path}: {e}")


# ============================================================================
# 4. 测试和验证
# ============================================================================

def test_feature_extraction():
    """测试特征提取功能"""
    print("=== 测试物理化学特征提取 ===\n")

    # 测试基础特征
    print("1. 测试基础特征提取:")
    for aa in ['ALA', 'ARG', 'PHE', 'GLY']:
        features = get_basic_features(aa)
        print(f"  {aa}: shape={features.shape}, mean={features.mean():.3f}, std={features.std():.3f}")

    print("\n2. 测试SASA计算:")
    if FREESASA_AVAILABLE:
        print("  FreeSASA 可用")
    else:
        print("  FreeSASA 不可用，请安装: pip install freesasa")

    print("\n3. 测试序列特征:")
    if BIOPYTHON_AVAILABLE:
        test_seq = "ACDEFGHIKLMNPQRSTVWY"
        seq_features = compute_sequence_features(test_seq)
        print(f"  序列长度: {len(test_seq)}")
        print(f"  特征数量: {len(seq_features)}")
        if seq_features:
            print(f"  特征维度: {seq_features[0].shape}")
    else:
        print("  Biopython 不可用，请安装: pip install biopython")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="物理化学特征提取工具")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--latent-dir", type=str, default="esm3_latents",
                        help="ESM3 latent文件目录")
    parser.add_argument("--output-dir", type=str, default="esm3_latents_with_physchem",
                        help="输出目录")
    parser.add_argument("--max", type=int, default=None, help="最大处理文件数")

    args = parser.parse_args()

    if args.test:
        test_feature_extraction()
    else:
        batch_extract_from_esm3_latents(
            Path(args.latent_dir),
            Path(args.output_dir),
            args.max
        )
