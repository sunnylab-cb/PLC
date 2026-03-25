#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配体特征提取模块 V2 (Ligand Feature Extraction V2)

增强版本 - 修复坐标问题并添加更多特征
- 保留原始实验坐标用于距离计算
- 基于RDKit的化学信息学特征 + 构象柔性分析
- 输出：per-atom特征向量，与蛋白per-residue特征对应

主要改进:
1. ✅ 保存原始实验坐标 (coords_original)
2. ✅ 分离柔性分析和坐标提取
3. ✅ 添加分子级别全局特征
4. ✅ 保存多种坐标版本供不同用途
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem import rdPartialCharges, rdMolAlign
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True

    # 抑制RDKit的警告信息
    RDLogger.DisableLog('rdApp.*')

except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install: conda install -c conda-forge rdkit")


# ============================================================================
# 原子特征编码
# ============================================================================

# 原子类型 (常见药物原子)
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Other']

# 杂化类型 (延迟初始化)
if RDKIT_AVAILABLE:
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
else:
    HYBRIDIZATION_TYPES = []

# 环大小 (药物化学常见)
RING_SIZES = [3, 4, 5, 6, 7, 8]


def one_hot_encode(value, choices):
    """One-hot编码"""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_symbol_features(atom: Chem.Atom) -> List[float]:
    """
    原子符号特征 (11维)
    """
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES[:-1]:
        symbol = 'Other'
    return one_hot_encode(symbol, ATOM_TYPES)


def get_atom_hybridization_features(atom: Chem.Atom) -> List[float]:
    """
    杂化类型特征 (6维)
    """
    hybridization = atom.GetHybridization()
    return one_hot_encode(hybridization, HYBRIDIZATION_TYPES)


def get_atom_aromaticity_features(atom: Chem.Atom) -> List[float]:
    """
    芳香性特征 (1维)
    """
    return [float(atom.GetIsAromatic())]


def get_atom_ring_features(atom: Chem.Atom) -> List[float]:
    """
    环特征 (7维)
    - 是否在环中 (1维)
    - 在特定大小环中 (6维)
    """
    features = [float(atom.IsInRing())]

    for ring_size in RING_SIZES:
        features.append(float(atom.IsInRingSize(ring_size)))

    return features


def get_atom_degree_features(atom: Chem.Atom) -> List[float]:
    """
    度特征 (5维)
    - 总度数 (归一化)
    - 显式价
    - 总价
    - 形式电荷
    - 自由基电子数
    """
    return [
        atom.GetDegree() / 6.0,  # 归一化到[0,1]
        atom.GetExplicitValence() / 6.0,
        atom.GetTotalValence() / 6.0,  # 使用TotalValence代替ImplicitValence
        (atom.GetFormalCharge() + 2) / 4.0,  # 归一化[-2,2]到[0,1]
        atom.GetNumRadicalElectrons() / 2.0
    ]


def get_bond_type_features(bond: Chem.Bond) -> List[float]:
    """
    键特征 (6维)
    - single
    - double
    - triple
    - aromatic
    - conjugated
    - in_ring
    """
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def extract_bond_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于当前工作分子提取 bond graph。

    该函数必须作用在最终用于生成 atom_features/coords 的同一个 mol 上，
    才能保证 edge_index 与 atom_features 的原子顺序严格对齐。
    """
    edge_pairs = []
    edge_attrs = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        attr = np.asarray(get_bond_type_features(bond), dtype=np.float32)

        edge_pairs.append([begin, end])
        edge_pairs.append([end, begin])
        edge_attrs.append(attr)
        edge_attrs.append(attr.copy())

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)

    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    edge_attr = np.asarray(edge_attrs, dtype=np.float32)
    return edge_index, edge_attr


def make_sanitized_descriptor_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    为电子描述符计算创建一个 sanitize 后的副本。

    目的：
    - 不修改主流程里用于坐标、原子顺序和 bond graph 的工作分子
    - 让 Gasteiger charge / Crippen 等依赖 RDKit 原子类型感知的描述符
      在稳定、可比较的分子对象上计算
    """
    mol_copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except Exception as e:
        warnings.warn(f"Sanitize failed for descriptor mol, fallback to raw mol: {e}")
        return mol


# ============================================================================
# 电子效应特征
# ============================================================================

def compute_gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    """
    计算Gasteiger部分电荷

    这是模拟静电相互作用的关键特征
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp('_GasteigerCharge')
            # 处理NaN值
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
        return np.array(charges, dtype=np.float32)
    except Exception as e:
        warnings.warn(f"Gasteiger charge calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


def compute_crippen_logp_contributions(mol: Chem.Mol) -> np.ndarray:
    """
    计算Crippen LogP贡献 (原子级疏水性)
    """
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        logp_contribs = np.array([c[0] for c in contribs], dtype=np.float32)
        return logp_contribs
    except Exception as e:
        warnings.warn(f"Crippen LogP calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


def compute_tpsa_contributions(mol: Chem.Mol) -> np.ndarray:
    """
    计算拓扑极性表面积贡献 (TPSA)
    """
    try:
        contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        return np.array(contribs, dtype=np.float32)
    except Exception as e:
        warnings.warn(f"TPSA calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


# ============================================================================
# 构象柔性分析
# ============================================================================
def generate_conformer_ensemble(
    mol: Chem.Mol,
    num_confs: int = 50,
    max_attempts: int = 1000,
    prune_rms_thresh: float = 0.5,
    strict: bool = True,
    num_threads: int = 0,
) -> Chem.Mol:
    """
    生成构象系综

    使用ETKDGv3算法生成多个低能构象
    """
    if not RDKIT_AVAILABLE:
        return mol

    work_mol = Chem.Mol(mol)
    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in work_mol.GetAtoms())
    if not has_explicit_h:
        work_mol = Chem.AddHs(work_mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = prune_rms_thresh
    params.numThreads = num_threads

    try:
        conf_ids = AllChem.EmbedMultipleConfs(
            work_mol,
            numConfs=num_confs,
            params=params
        )
    except TypeError:
        conf_ids = AllChem.EmbedMultipleConfs(
            work_mol,
            numConfs=num_confs,
            randomSeed=42,
            pruneRmsThresh=prune_rms_thresh,
            numThreads=num_threads,
            maxAttempts=max_attempts
        )
    except Exception as e:
        if strict:
            raise
        warnings.warn(f"Conformer embedding failed in loose mode: {e}")
        if work_mol.GetNumConformers() > 0:
            return work_mol
        AllChem.Compute2DCoords(work_mol)
        return work_mol

    if len(conf_ids) == 0:
        if strict:
            raise ValueError("Failed to generate conformers")
        warnings.warn("No conformers generated in loose mode, fallback to existing coordinates")
        if work_mol.GetNumConformers() > 0:
            return work_mol
        AllChem.Compute2DCoords(work_mol)
        return work_mol

    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(work_mol, numThreads=num_threads)
    except Exception as e:
        if strict:
            raise
        warnings.warn(f"MMFF optimization failed in loose mode: {e}")
        return work_mol

    valid_conf_ids = [conf_ids[i] for i, (converged, energy) in enumerate(results) if converged == 0]
    if len(valid_conf_ids) == 0:
        if strict:
            raise ValueError("All conformer optimizations failed")
        warnings.warn("All conformer optimizations failed in loose mode, keep embedded conformers")
        return work_mol

    return work_mol


def compute_conformer_flexibility(
    mol: Chem.Mol,
    align_to_core: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算构象柔性 (RMSF - Root Mean Square Fluctuation)

    Returns:
        rmsf: [N_atoms] 每个原子的RMSF值
        flexibility_flag: [N_atoms] 柔性标志 (1=柔性, 0=刚性)
    """
    num_atoms = mol.GetNumAtoms()
    num_confs = mol.GetNumConformers()

    if num_confs < 2:
        # 没有足够的构象，返回零
        return np.zeros(num_atoms, dtype=np.float32), np.zeros(num_atoms, dtype=np.float32)

    # 收集所有构象的坐标
    coords_list = []
    for conf in mol.GetConformers():
        coords = conf.GetPositions()
        coords_list.append(coords)

    coords_array = np.array(coords_list)  # [num_confs, num_atoms, 3]

    # 对齐构象 (可选)
    if align_to_core and num_confs > 1:
        # 对齐到第一个构象
        ref_conf_id = 0
        for i in range(1, num_confs):
            try:
                rdMolAlign.AlignMol(mol, mol, i, ref_conf_id)
            except:
                pass

        # 重新收集对齐后的坐标
        coords_list = []
        for conf in mol.GetConformers():
            coords = conf.GetPositions()
            coords_list.append(coords)
        coords_array = np.array(coords_list)

    # 计算RMSF
    mean_coords = coords_array.mean(axis=0)  # [num_atoms, 3]
    squared_deviations = ((coords_array - mean_coords) ** 2).sum(axis=2)  # [num_confs, num_atoms]
    rmsf = np.sqrt(squared_deviations.mean(axis=0))  # [num_atoms]

    # 标记柔性原子 (RMSF > 阈值)
    flexibility_threshold = 1.0  # Å
    flexibility_flag = (rmsf > flexibility_threshold).astype(np.float32)

    return rmsf, flexibility_flag


# ============================================================================
# 完整特征提取
# ============================================================================

def extract_ligand_atom_features(
    mol: Chem.Mol,
    include_conformers: bool = True,
    num_conformers: int = 50,
    use_existing_coords: bool = True,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    """
    提取完整的配体原子特征 (V2 - 修复版)

    Args:
        mol: RDKit分子对象
        include_conformers: 是否包含构象柔性分析
        num_conformers: 生成的构象数量
        use_existing_coords: 是否使用已有的3D坐标（PDBbind数据已有实验结构）

    Returns:
        特征字典，包含：
        - atom_features: [N_atoms, feature_dim] 原子特征矩阵
        - coords: [N_atoms, 3] 当前3D坐标（包含氢原子）
        - coords_original: [N_heavy, 3] ✅ 原始实验坐标（仅重原子，用于距离计算）
        - coords_heavy_current: [N_heavy, 3] ✅ 当前重原子坐标
        - gasteiger_charges: [N_atoms] Gasteiger电荷
        - logp_contribs: [N_atoms] LogP贡献
        - tpsa_contribs: [N_atoms] TPSA贡献
        - rmsf: [N_atoms] 构象柔性
        - flexibility_flag: [N_atoms] 柔性标志
        - num_heavy_atoms: int 重原子数量
        - molecular_descriptors: dict 分子级别描述符
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand feature extraction")

    num_atoms_original = mol.GetNumAtoms()
    num_heavy_atoms = mol.GetNumHeavyAtoms()

    # ========================================================================
    # 步骤1: 保存原始坐标（关键！）
    # ========================================================================
    has_3d_coords = mol.GetNumConformers() > 0
    original_coords_heavy = None
    original_coords_all = None
    original_has_hydrogens = False

    if has_3d_coords and use_existing_coords:
        # 保存原始坐标
        original_conf = mol.GetConformer(0)
        original_coords_all = original_conf.GetPositions().copy()

        # 检查原始文件是否已包含氢原子
        original_has_hydrogens = (num_atoms_original > num_heavy_atoms)

        # 提取重原子坐标（保持原始索引顺序）
        original_coords_heavy = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:  # 不是氢原子
                idx = atom.GetIdx()
                original_coords_heavy.append(original_coords_all[idx])
        original_coords_heavy = np.array(original_coords_heavy, dtype=np.float32)

        if verbose:
            if original_has_hydrogens:
                print(f"  ✅ 保存原始坐标: {num_heavy_atoms}个重原子 + {num_atoms_original - num_heavy_atoms}个氢原子")
            else:
                print(f"  ✅ 保存原始坐标: {num_heavy_atoms}个重原子（无氢）")

    # ========================================================================
    # 步骤2: 处理氢原子（如果原始文件已有氢，则不添加）
    # ========================================================================
    if original_has_hydrogens:
        # 原始文件已包含氢原子，直接使用
        num_atoms_with_h = num_atoms_original
        if verbose:
            print(f"  ✅ 使用原始氢原子: 保持原子顺序不变")
    else:
        # 原始文件没有氢原子，需要添加
        mol = Chem.AddHs(mol, addCoords=has_3d_coords)
        num_atoms_with_h = mol.GetNumAtoms()
        if verbose:
            print(f"  ⚠️  添加氢原子: {num_atoms_original} -> {num_atoms_with_h} 原子")

        # 恢复重原子的原始坐标（因为AddHs可能改变坐标）
        if has_3d_coords and use_existing_coords and original_coords_heavy is not None:
            conf = mol.GetConformer(0)
            heavy_idx = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1:  # 重原子
                    if heavy_idx < len(original_coords_heavy):
                        conf.SetAtomPosition(atom.GetIdx(), original_coords_heavy[heavy_idx].tolist())
                    heavy_idx += 1
            if verbose:
                print(f"  ✅ 恢复原始坐标: 重原子坐标已恢复")

    # ========================================================================
    # 步骤3: 柔性分析（使用深拷贝，完全独立）
    # ========================================================================
    rmsf_full = np.zeros(num_atoms_with_h, dtype=np.float32)
    flexibility_flag_full = np.zeros(num_atoms_with_h, dtype=np.float32)

    if include_conformers and has_3d_coords:
        # 创建深拷贝用于柔性分析（不影响原始分子）
        mol_copy = Chem.Mol(mol.ToBinary())  # 深拷贝
        mol_copy = generate_conformer_ensemble(
            mol_copy,
            num_confs=num_conformers,
            strict=strict,
            num_threads=num_threads,
        )
        rmsf, flexibility_flag = compute_conformer_flexibility(mol_copy)
        num_generated_confs = mol_copy.GetNumConformers()

        if num_generated_confs < 2:
            if strict:
                raise ValueError(f"Insufficient conformers generated: {num_generated_confs}")
            if verbose:
                print(f"  ⚠️  柔性分析降级: 仅{num_generated_confs}个构象")

        # 只保留重原子的柔性信息
        heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        for i, idx in enumerate(heavy_atom_indices):
            if i < len(rmsf):
                rmsf_full[idx] = rmsf[i]
                flexibility_flag_full[idx] = flexibility_flag[i]

        if verbose:
            print(f"  ✅ 柔性分析完成: {num_generated_confs}个构象")
        del mol_copy  # 释放内存
    elif not has_3d_coords:
        # 没有原始坐标，生成新的
        if include_conformers:
            mol = generate_conformer_ensemble(
                mol,
                num_confs=num_conformers,
                strict=strict,
                num_threads=num_threads,
            )
        else:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

        if verbose:
            print(f"  ⚠️  生成新坐标: 无原始实验坐标")

    if mol.GetNumConformers() == 0:
        if strict:
            raise ValueError("No conformer available")
        AllChem.Compute2DCoords(mol)

    # ========================================================================
    # 步骤5: 提取原子特征
    # ========================================================================
    atom_features_list = []

    for atom in mol.GetAtoms():
        features = []

        # 原子身份特征 (11维)
        features.extend(get_atom_symbol_features(atom))

        # 杂化类型 (6维)
        features.extend(get_atom_hybridization_features(atom))

        # 芳香性 (1维)
        features.extend(get_atom_aromaticity_features(atom))

        # 环特征 (7维)
        features.extend(get_atom_ring_features(atom))

        # 度特征 (5维)
        features.extend(get_atom_degree_features(atom))

        atom_features_list.append(features)

    atom_features = np.array(atom_features_list, dtype=np.float32)  # [N_atoms, 30]

    # ========================================================================
    # 步骤6: 电子效应特征
    # ========================================================================
    descriptor_mol = make_sanitized_descriptor_mol(mol)
    gasteiger_charges = compute_gasteiger_charges(descriptor_mol)  # [N_atoms]
    logp_contribs = compute_crippen_logp_contributions(descriptor_mol)  # [N_atoms]
    tpsa_contribs = compute_tpsa_contributions(descriptor_mol)  # [N_atoms]

    # ========================================================================
    # 步骤7: 提取坐标
    # ========================================================================
    conf = mol.GetConformer(0)
    coords_current = conf.GetPositions().astype(np.float32)

    # 提取当前重原子坐标
    coords_heavy_current = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            coords_heavy_current.append(coords_current[atom.GetIdx()])
    coords_heavy_current = np.array(coords_heavy_current, dtype=np.float32)

    # ========================================================================
    # 步骤8: 分子级别描述符
    # ========================================================================
    molecular_descriptors = compute_molecular_descriptors(mol)

    # ========================================================================
    # 步骤8.5: 提取真实化学键图
    # ========================================================================
    edge_index, edge_attr = extract_bond_graph(mol)

    # ========================================================================
    # 步骤9: 拼接所有特征
    # ========================================================================
    # 总维度: 30 (原子特征) + 3 (电子效应) + 2 (柔性) = 35维
    complete_features = np.concatenate([
        atom_features,  # [N, 30]
        gasteiger_charges[:, np.newaxis],  # [N, 1]
        logp_contribs[:, np.newaxis],  # [N, 1]
        tpsa_contribs[:, np.newaxis],  # [N, 1]
        rmsf_full[:, np.newaxis],  # [N, 1]
        flexibility_flag_full[:, np.newaxis]  # [N, 1]
    ], axis=1)  # [N_atoms, 35]

    return {
        # 原子级别特征
        'atom_features': complete_features,  # [N_atoms, 35]
        'edge_index': edge_index,  # [2, E]
        'edge_attr': edge_attr,  # [E, 6]

        # 坐标信息（多个版本）
        'coords': coords_current,  # [N_atoms, 3] 当前坐标（包含氢）
        'coords_original': original_coords_heavy,  # [N_heavy, 3] ✅ 原始实验坐标（仅重原子）
        'coords_original_with_h': original_coords_all,  # [N_original, 3] ✅ 原始坐标（包含原始氢，如果有）
        'coords_heavy_current': coords_heavy_current,  # [N_heavy, 3] 当前重原子坐标

        # 电子效应特征
        'gasteiger_charges': gasteiger_charges,  # [N_atoms]
        'logp_contribs': logp_contribs,  # [N_atoms]
        'tpsa_contribs': tpsa_contribs,  # [N_atoms]

        # 柔性特征
        'rmsf': rmsf_full,  # [N_atoms]
        'flexibility_flag': flexibility_flag_full,  # [N_atoms]

        # 元数据
        'num_atoms': num_atoms_with_h,  # 总原子数（包含氢）
        'num_heavy_atoms': num_heavy_atoms,  # 重原子数
        'has_original_coords': original_coords_heavy is not None,  # 是否有原始坐标

        # 分子级别描述符
        'molecular_descriptors': molecular_descriptors,  # dict
    }


def extract_ligand_features_from_file(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    """
    从文件提取配体特征

    支持格式: .sdf, .mol2, .pdb, .mol
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required")

    # 读取分子 - 仅使用非严格价态加载，不做其他fallback
    if ligand_file.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(ligand_file, sanitize=False, removeHs=False)
        mol = next(suppl, None)
    elif ligand_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(ligand_file, sanitize=False, removeHs=False)
    elif ligand_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(ligand_file, sanitize=False, removeHs=False)
    elif ligand_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(ligand_file, sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {ligand_file}")

    if mol is None:
        raise ValueError(f"Failed to read molecule from {ligand_file}")

    return extract_ligand_atom_features(
        mol,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )


def extract_ligand_features_from_smiles(
    smiles: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    """
    从SMILES字符串提取配体特征
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return extract_ligand_atom_features(
        mol,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )


# ============================================================================
# 分子级别特征 (可选)
# ============================================================================

def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    计算分子级别描述符

    这些可以作为全局特征使用
    """
    if not RDKIT_AVAILABLE:
        return {}

    descriptors = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
        'fraction_csp3': Descriptors.FractionCSP3(mol),
        'num_heteroatoms': Descriptors.NumHeteroatoms(mol)
    }

    return descriptors


# ============================================================================
# 测试函数
# ============================================================================

def test_ligand_features():
    """测试配体特征提取"""
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping test")
        return

    # 测试SMILES: 阿司匹林
    smiles = "CC(=O)Oc1ccccc1C(=O)O"

    print("=== 配体特征提取测试 ===\n")
    print(f"SMILES: {smiles}")

    # 提取特征
    features = extract_ligand_features_from_smiles(
        smiles,
        include_conformers=True,
        num_conformers=10
    )

    print(f"\n原子数: {features['num_atoms']}")
    print(f"特征维度: {features['atom_features'].shape}")
    print(f"坐标维度: {features['coords'].shape}")

    print(f"\nGasteiger电荷范围: [{features['gasteiger_charges'].min():.3f}, {features['gasteiger_charges'].max():.3f}]")
    print(f"LogP贡献范围: [{features['logp_contribs'].min():.3f}, {features['logp_contribs'].max():.3f}]")
    print(f"RMSF范围: [{features['rmsf'].min():.3f}, {features['rmsf'].max():.3f}]")
    print(f"柔性原子数: {features['flexibility_flag'].sum():.0f}/{features['num_atoms']}")

    # 分子描述符
    mol = Chem.MolFromSmiles(smiles)
    descriptors = compute_molecular_descriptors(mol)

    print(f"\n分子描述符:")
    for key, value in descriptors.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    test_ligand_features()
