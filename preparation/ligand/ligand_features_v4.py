#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent ligand feature extraction V4.

This module is self-contained and does not depend on legacy V2/V3 modules.

Feature layout:
- Base atom features: 30
- Electronic features: 3
- Flexibility features: 2
- Advanced atom features: 6
- Formal charge class: 3
- Rotatable-bond context: 3
- Metal-binding flags: 4
- Stereochemistry flags: 3
= 54D local atom features

Then V5 extraction script appends:
- 45D global molecular tag
- 10D shape features
= 109D final per-atom features
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem import Lipinski
    from rdkit.Geometry import Point3D
    RDKIT_AVAILABLE = True
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install: conda install -c conda-forge rdkit")

try:
    from rdkit.Chem import rdFreeSASA
    RDFREESASA_AVAILABLE = True
except Exception:
    rdFreeSASA = None
    RDFREESASA_AVAILABLE = False


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "Other"]
RING_SIZES = [3, 4, 5, 6, 7, 8]

if RDKIT_AVAILABLE:
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
else:
    HYBRIDIZATION_TYPES = []

_HBOND_FEATURE_FACTORY = None


def one_hot_encode(value, choices):
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def get_atom_symbol_features(atom: "Chem.Atom") -> List[float]:
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES[:-1]:
        symbol = "Other"
    return one_hot_encode(symbol, ATOM_TYPES)


def get_atom_hybridization_features(atom: "Chem.Atom") -> List[float]:
    return one_hot_encode(atom.GetHybridization(), HYBRIDIZATION_TYPES)


def get_atom_aromaticity_features(atom: "Chem.Atom") -> List[float]:
    return [float(atom.GetIsAromatic())]


def get_atom_ring_features(atom: "Chem.Atom") -> List[float]:
    feats = [float(atom.IsInRing())]
    for ring_size in RING_SIZES:
        feats.append(float(atom.IsInRingSize(ring_size)))
    return feats


def get_atom_degree_features(atom: "Chem.Atom") -> List[float]:
    return [
        atom.GetDegree() / 6.0,
        atom.GetExplicitValence() / 6.0,
        atom.GetTotalValence() / 6.0,
        (atom.GetFormalCharge() + 2) / 4.0,
        atom.GetNumRadicalElectrons() / 2.0,
    ]


def get_formal_charge_class_features(atom: "Chem.Atom") -> List[float]:
    charge = atom.GetFormalCharge()
    return [
        float(charge < 0),
        float(charge == 0),
        float(charge > 0),
    ]


def get_bond_type_features(bond: "Chem.Bond") -> List[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def extract_bond_graph(mol: "Chem.Mol") -> Tuple[np.ndarray, np.ndarray]:
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

    return np.asarray(edge_pairs, dtype=np.int64).T, np.asarray(edge_attrs, dtype=np.float32)


def make_sanitized_descriptor_mol(mol: "Chem.Mol") -> "Chem.Mol":
    mol_copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except Exception as e:
        warnings.warn(f"Sanitize failed for descriptor mol, fallback to raw mol: {e}")
        return mol


def compute_gasteiger_charges(mol: "Chem.Mol") -> np.ndarray:
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
        return np.array(charges, dtype=np.float32)
    except Exception as e:
        warnings.warn(f"Gasteiger charge calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


def compute_crippen_logp_contributions(mol: "Chem.Mol") -> np.ndarray:
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        return np.array([c[0] for c in contribs], dtype=np.float32)
    except Exception as e:
        warnings.warn(f"Crippen LogP calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


def compute_tpsa_contributions(mol: "Chem.Mol") -> np.ndarray:
    try:
        return np.array(rdMolDescriptors._CalcTPSAContribs(mol), dtype=np.float32)
    except Exception as e:
        warnings.warn(f"TPSA calculation failed: {e}")
        return np.zeros(mol.GetNumAtoms(), dtype=np.float32)


def generate_conformer_ensemble(
    mol: "Chem.Mol",
    num_confs: int = 50,
    max_attempts: int = 1000,
    prune_rms_thresh: float = 0.5,
    strict: bool = False,
    num_threads: int = 0,
) -> "Chem.Mol":
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
        conf_ids = AllChem.EmbedMultipleConfs(work_mol, numConfs=num_confs, params=params)
    except TypeError:
        conf_ids = AllChem.EmbedMultipleConfs(
            work_mol,
            numConfs=num_confs,
            randomSeed=42,
            pruneRmsThresh=prune_rms_thresh,
            numThreads=num_threads,
            maxAttempts=max_attempts,
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
        results = AllChem.MMFFOptimizeMoleculeConfs(work_mol, numThreads=num_threads, maxIters=100000)
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

    invalid_conf_ids = [cid for cid in conf_ids if cid not in valid_conf_ids]
    for cid in reversed(invalid_conf_ids):
        work_mol.RemoveConformer(cid)

    return work_mol


def compute_conformer_flexibility(mol: "Chem.Mol") -> Tuple[np.ndarray, np.ndarray]:
    num_atoms = mol.GetNumAtoms()
    num_confs = mol.GetNumConformers()

    if num_confs < 2:
        return np.zeros(num_atoms, dtype=np.float32), np.zeros(num_atoms, dtype=np.float32)

    try:
        ref_id = 0
        for conf_id in range(1, num_confs):
            rdMolAlign.AlignMol(mol, mol, prbCid=conf_id, refCid=ref_id)
    except Exception as e:
        warnings.warn(f"Conformer alignment failed: {e}")

    coords_list = []
    for conf in mol.GetConformers():
        coords_list.append(conf.GetPositions())
    coords_array = np.array(coords_list)
    mean_coords = coords_array.mean(axis=0)
    squared_deviations = ((coords_array - mean_coords) ** 2).sum(axis=2)
    rmsf = np.sqrt(squared_deviations.mean(axis=0)).astype(np.float32)
    flexibility_flag = (rmsf > 1.0).astype(np.float32)
    return rmsf, flexibility_flag


def _get_hbond_feature_factory() -> Optional["ChemicalFeatures.MolChemicalFeatureFactory"]:
    global _HBOND_FEATURE_FACTORY

    if _HBOND_FEATURE_FACTORY is not None:
        return _HBOND_FEATURE_FACTORY
    if not RDKIT_AVAILABLE:
        return None

    fdef_path = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    if not fdef_path.exists():
        return None

    try:
        _HBOND_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(str(fdef_path))
    except Exception:
        _HBOND_FEATURE_FACTORY = None
    return _HBOND_FEATURE_FACTORY


def compute_atom_sasa(mol: "Chem.Mol") -> np.ndarray:
    """
    计算原子SASA

    处理重合氢原子：使用力场优化修复坐标重合问题
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return np.zeros(0, dtype=np.float32)
    if not RDFREESASA_AVAILABLE or mol.GetNumConformers() == 0:
        return np.zeros(num_atoms, dtype=np.float32)

    # 创建副本用于优化，避免修改原始mol
    mol_copy = Chem.Mol(mol)

    # 力场优化修复重合原子
    try:
        AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=100000)
    except:
        try:
            AllChem.UFFOptimizeMolecule(mol_copy, maxIters=100000)
        except:
            pass

    try:
        radii = rdFreeSASA.classifyAtoms(mol_copy)
        rdFreeSASA.CalcSASA(mol_copy, radii)
        sasa = np.zeros(num_atoms, dtype=np.float32)
        for atom in mol_copy.GetAtoms():
            idx = atom.GetIdx()
            value = 0.0
            if atom.HasProp("SASA"):
                try:
                    value = float(atom.GetProp("SASA"))
                except Exception:
                    value = 0.0
            sasa[idx] = value
        return sasa
    except Exception as exc:
        warnings.warn(f"SASA calculation failed: {exc}")
        return np.zeros(num_atoms, dtype=np.float32)


def compute_hbond_flags(mol: "Chem.Mol") -> Tuple[np.ndarray, np.ndarray]:
    num_atoms = mol.GetNumAtoms()
    donor = np.zeros(num_atoms, dtype=np.float32)
    acceptor = np.zeros(num_atoms, dtype=np.float32)

    feature_factory = _get_hbond_feature_factory()
    if feature_factory is None:
        return donor, acceptor

    try:
        for feature in feature_factory.GetFeaturesForMol(mol):
            atom_ids = feature.GetAtomIds()
            family = feature.GetFamily()
            if family == "Donor":
                for atom_idx in atom_ids:
                    donor[atom_idx] = 1.0
            elif family == "Acceptor":
                for atom_idx in atom_ids:
                    acceptor[atom_idx] = 1.0
    except Exception as exc:
        warnings.warn(f"H-bond feature extraction failed: {exc}")

    return donor, acceptor


def compute_chirality_flags(mol: "Chem.Mol") -> Tuple[np.ndarray, np.ndarray]:
    num_atoms = mol.GetNumAtoms()
    chirality_cw = np.zeros(num_atoms, dtype=np.float32)
    chirality_ccw = np.zeros(num_atoms, dtype=np.float32)

    if not RDKIT_AVAILABLE:
        return chirality_cw, chirality_ccw

    chiral_type = Chem.rdchem.ChiralType
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        tag = atom.GetChiralTag()
        if tag == chiral_type.CHI_TETRAHEDRAL_CW:
            chirality_cw[idx] = 1.0
        elif tag == chiral_type.CHI_TETRAHEDRAL_CCW:
            chirality_ccw[idx] = 1.0
    return chirality_cw, chirality_ccw


def compute_stereochemistry_flags(mol: "Chem.Mol") -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    stereo = np.zeros((num_atoms, 3), dtype=np.float32)

    if not RDKIT_AVAILABLE:
        return stereo

    mol_copy = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistry(mol_copy, force=True, cleanIt=True)
    except Exception as exc:
        warnings.warn(f"Stereochemistry assignment failed: {exc}")
        return stereo

    for atom in mol_copy.GetAtoms():
        idx = atom.GetIdx()
        if atom.HasProp("_CIPCode"):
            stereo[idx, 0] = 1.0
            cip = atom.GetProp("_CIPCode")
            if cip == "R":
                stereo[idx, 1] = 1.0
            elif cip == "S":
                stereo[idx, 2] = 1.0

    return stereo


def compute_rotatable_bond_context(mol: "Chem.Mol") -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    feat = np.zeros((num_atoms, 3), dtype=np.float32)

    if not RDKIT_AVAILABLE:
        return feat

    try:
        matches = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    except Exception as exc:
        warnings.warn(f"Rotatable bond detection failed: {exc}")
        return feat

    adjacency = [set() for _ in range(num_atoms)]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        adjacency[idx] = {nb.GetIdx() for nb in atom.GetNeighbors()}

    for a, b in matches:
        feat[a, 0] = 1.0
        feat[b, 0] = 1.0
        for nb in adjacency[a]:
            feat[nb, 1] = 1.0
        for nb in adjacency[b]:
            feat[nb, 1] = 1.0
        feat[a, 2] += 1.0
        feat[b, 2] += 1.0

    feat[:, 2] = np.clip(feat[:, 2] / 4.0, 0.0, 1.0)
    return feat


def compute_metal_binding_flags(mol: "Chem.Mol") -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    feat = np.zeros((num_atoms, 4), dtype=np.float32)

    if not RDKIT_AVAILABLE:
        return feat

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        aromatic = atom.GetIsAromatic()
        atomic_num = atom.GetAtomicNum()

        is_n = symbol == "N" and not aromatic
        is_o = symbol == "O"
        is_s = symbol == "S"
        is_any = is_n or is_o or is_s

        if atomic_num == 1:
            is_any = False
            is_n = False
            is_o = False
            is_s = False

        if charge > 0 and symbol == "N":
            is_n = False
            is_any = is_o or is_s or is_n

        feat[idx, 0] = float(is_any)
        feat[idx, 1] = float(is_n)
        feat[idx, 2] = float(is_o)
        feat[idx, 3] = float(is_s)

    return feat


def compute_ligand_3d_shape_features(mol: "Chem.Mol") -> list[float]:
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return [0.0] * 5

    try:
        return [
            float(Descriptors3D.PMI1(mol)),
            float(Descriptors3D.PMI2(mol)),
            float(Descriptors3D.PMI3(mol)),
            float(Descriptors3D.SpherocityIndex(mol)),
            float(Descriptors3D.RadiusOfGyration(mol)),
        ]
    except Exception as exc:
        warnings.warn(f"3D shape calculation failed: {exc}")
        return [0.0] * 5


def get_inertial_shape_factor(mol: "Chem.Mol") -> list[float]:
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return [0.0, 0.0]

    try:
        return [float(Descriptors3D.NPR1(mol)), float(Descriptors3D.NPR2(mol))]
    except Exception:
        return [0.0, 0.0]


def compute_principal_axis_spans(mol: "Chem.Mol") -> list[float]:
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return [0.0, 0.0, 0.0]

    try:
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float32)
        if coords.shape[0] < 2:
            return [0.0, 0.0, 0.0]
        centered = coords - coords.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal_axes = vh.T
        projected = centered @ principal_axes
        spans = projected.max(axis=0) - projected.min(axis=0)
        return [float(spans[0]), float(spans[1]), float(spans[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def _robust_load_mol(ligand_file: str) -> "Chem.Mol":
    ligand_path = Path(ligand_file)
    suffix = ligand_path.suffix.lower()
    if suffix == ".sdf":
        mol = next(Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=False), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_path), sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand_path), sanitize=False, removeHs=False)
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {ligand_file}")

    if mol is None:
        raise ValueError(f"Failed to read molecule from {ligand_file}")
    return mol


def _align_atom_count(mol: "Chem.Mol", expected_num_atoms: int) -> "Chem.Mol":
    if mol.GetNumAtoms() == expected_num_atoms:
        return mol

    try:
        mol_with_h = Chem.AddHs(Chem.Mol(mol), addCoords=(mol.GetNumConformers() > 0))
        if mol_with_h.GetNumAtoms() == expected_num_atoms:
            return mol_with_h
    except Exception:
        pass

    return mol


def _set_coords_from_array(mol: "Chem.Mol", coords: np.ndarray) -> None:
    if coords.shape != (mol.GetNumAtoms(), 3):
        return

    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)

    conf = mol.GetConformer(0)
    for atom_idx in range(mol.GetNumAtoms()):
        x, y, z = coords[atom_idx]
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))


def _resize_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == target_size:
        return array

    output = np.zeros(target_size, dtype=np.float32)
    copy_size = min(target_size, array.size)
    output[:copy_size] = array[:copy_size]
    return output


def _load_molecule_from_file(ligand_file: str) -> "Chem.Mol":
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand feature extraction")
    return _robust_load_mol(ligand_file)


def extract_ligand_features_from_file(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    use_existing_coords: bool = True,
    strict: bool = False,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    mol = _load_molecule_from_file(ligand_file)
    num_atoms_original = mol.GetNumAtoms()
    num_heavy_atoms = mol.GetNumHeavyAtoms()

    has_3d_coords = mol.GetNumConformers() > 0
    original_coords_heavy = None
    original_coords_all = None
    original_has_hydrogens = False

    if has_3d_coords and use_existing_coords:
        original_conf = mol.GetConformer(0)
        original_coords_all = original_conf.GetPositions().copy()
        original_has_hydrogens = num_atoms_original > num_heavy_atoms
        original_coords_heavy = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                original_coords_heavy.append(original_coords_all[atom.GetIdx()])
        original_coords_heavy = np.array(original_coords_heavy, dtype=np.float32)
        if verbose:
            if original_has_hydrogens:
                print(f"  ✅ 保存原始坐标: {num_heavy_atoms}个重原子 + {num_atoms_original - num_heavy_atoms}个氢原子")
            else:
                print(f"  ✅ 保存原始坐标: {num_heavy_atoms}个重原子（无氢）")

    if original_has_hydrogens:
        num_atoms_with_h = num_atoms_original
        if verbose:
            print("  ✅ 使用原始氢原子: 保持原子顺序不变")
    else:
        mol = Chem.AddHs(mol, addCoords=has_3d_coords)
        num_atoms_with_h = mol.GetNumAtoms()
        if verbose:
            print(f"  ⚠️  添加氢原子: {num_atoms_original} -> {num_atoms_with_h} 原子")
        if has_3d_coords and use_existing_coords and original_coords_heavy is not None:
            conf = mol.GetConformer(0)
            heavy_idx = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1:
                    if heavy_idx < len(original_coords_heavy):
                        conf.SetAtomPosition(atom.GetIdx(), original_coords_heavy[heavy_idx].tolist())
                    heavy_idx += 1

    rmsf_full = np.zeros(num_atoms_with_h, dtype=np.float32)
    flexibility_flag_full = np.zeros(num_atoms_with_h, dtype=np.float32)

    if include_conformers and has_3d_coords:
        mol_copy = Chem.Mol(mol.ToBinary())
        mol_copy = generate_conformer_ensemble(
            mol_copy,
            num_confs=num_conformers,
            strict=strict,
            num_threads=num_threads,
        )
        rmsf, flexibility_flag = compute_conformer_flexibility(mol_copy)
        heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        for i, idx in enumerate(heavy_atom_indices):
            if i < len(rmsf):
                rmsf_full[idx] = rmsf[i]
                flexibility_flag_full[idx] = flexibility_flag[i]
        if verbose:
            print(f"  ✅ 柔性分析完成: {mol_copy.GetNumConformers()}个构象")
    elif not has_3d_coords:
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

    if mol.GetNumConformers() == 0:
        if strict:
            raise ValueError("No conformer available")
        AllChem.Compute2DCoords(mol)

    atom_feature_rows = []
    for atom in mol.GetAtoms():
        row = []
        row.extend(get_atom_symbol_features(atom))
        row.extend(get_atom_hybridization_features(atom))
        row.extend(get_atom_aromaticity_features(atom))
        row.extend(get_atom_ring_features(atom))
        row.extend(get_atom_degree_features(atom))
        row.extend(get_formal_charge_class_features(atom))
        atom_feature_rows.append(row)
    base_atom_features = np.asarray(atom_feature_rows, dtype=np.float32)  # 33D

    descriptor_mol = make_sanitized_descriptor_mol(mol)
    gasteiger_charges = compute_gasteiger_charges(descriptor_mol)
    logp_contribs = compute_crippen_logp_contributions(descriptor_mol)
    tpsa_contribs = compute_tpsa_contributions(descriptor_mol)

    conf = mol.GetConformer(0)
    coords_current = conf.GetPositions().astype(np.float32)
    coords_heavy_current = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            coords_heavy_current.append(coords_current[atom.GetIdx()])
    coords_heavy_current = np.array(coords_heavy_current, dtype=np.float32)

    edge_index, edge_attr = extract_bond_graph(mol)

    atom_sasa = np.zeros(num_atoms_with_h, dtype=np.float32)
    hbond_donor = np.zeros(num_atoms_with_h, dtype=np.float32)
    hbond_acceptor = np.zeros(num_atoms_with_h, dtype=np.float32)
    chirality_cw = np.zeros(num_atoms_with_h, dtype=np.float32)
    chirality_ccw = np.zeros(num_atoms_with_h, dtype=np.float32)
    stereo_flags = np.zeros((num_atoms_with_h, 3), dtype=np.float32)
    rotatable_context = np.zeros((num_atoms_with_h, 3), dtype=np.float32)
    metal_binding = np.zeros((num_atoms_with_h, 4), dtype=np.float32)
    shape_features = np.zeros(10, dtype=np.float32)

    try:
        advanced_mol = _robust_load_mol(ligand_file)
        advanced_mol = _align_atom_count(advanced_mol, num_atoms_with_h)
        if advanced_mol.GetNumAtoms() == num_atoms_with_h and coords_current.shape == (num_atoms_with_h, 3):
            _set_coords_from_array(advanced_mol, coords_current)

        if advanced_mol.GetNumAtoms() == num_atoms_with_h:
            atom_sasa = compute_atom_sasa(advanced_mol)
            hbond_donor, hbond_acceptor = compute_hbond_flags(advanced_mol)
            chirality_cw, chirality_ccw = compute_chirality_flags(advanced_mol)
            stereo_flags = compute_stereochemistry_flags(advanced_mol)
            rotatable_context = compute_rotatable_bond_context(advanced_mol)
            metal_binding = compute_metal_binding_flags(advanced_mol)
            shape_features = np.asarray(
                compute_ligand_3d_shape_features(advanced_mol)
                + get_inertial_shape_factor(advanced_mol)
                + compute_principal_axis_spans(advanced_mol),
                dtype=np.float32,
            )
        else:
            warnings.warn(
                f"Atom count mismatch for advanced features: {ligand_file} "
                f"(mol={advanced_mol.GetNumAtoms()}, expected={num_atoms_with_h})"
            )
    except Exception as exc:
        warnings.warn(f"Advanced ligand feature extraction failed: {exc}")

    atom_sasa = _resize_vector(atom_sasa, num_atoms_with_h)
    hbond_donor = _resize_vector(hbond_donor, num_atoms_with_h)
    hbond_acceptor = _resize_vector(hbond_acceptor, num_atoms_with_h)
    chirality_cw = _resize_vector(chirality_cw, num_atoms_with_h)
    chirality_ccw = _resize_vector(chirality_ccw, num_atoms_with_h)
    shape_features = _resize_vector(shape_features, 10)

    lipophilic_sasa = atom_sasa * logp_contribs

    local_atom_features = np.concatenate([
        base_atom_features,                              # 33
        gasteiger_charges[:, None],                      # 1
        logp_contribs[:, None],                          # 1
        tpsa_contribs[:, None],                          # 1
        rmsf_full[:, None],                              # 1
        flexibility_flag_full[:, None],                  # 1
        atom_sasa[:, None],                              # 1
        lipophilic_sasa[:, None],                        # 1
        hbond_donor[:, None],                            # 1
        hbond_acceptor[:, None],                         # 1
        chirality_cw[:, None],                           # 1
        chirality_ccw[:, None],                          # 1
        rotatable_context,                               # 3
        metal_binding,                                   # 4
        stereo_flags,                                    # 3
    ], axis=1).astype(np.float32)                        # = 54

    molecular_descriptors = {
        "molecular_weight": Descriptors.MolWt(descriptor_mol),
        "logp": Descriptors.MolLogP(descriptor_mol),
        "tpsa": Descriptors.TPSA(descriptor_mol),
        "num_h_donors": Descriptors.NumHDonors(descriptor_mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(descriptor_mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(descriptor_mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(descriptor_mol),
        "num_aliphatic_rings": Descriptors.NumAliphaticRings(descriptor_mol),
        "fraction_csp3": Descriptors.FractionCSP3(descriptor_mol),
        "num_heteroatoms": Descriptors.NumHeteroatoms(descriptor_mol),
        "shape_pmi1": float(shape_features[0]),
        "shape_pmi2": float(shape_features[1]),
        "shape_pmi3": float(shape_features[2]),
        "shape_spherocity": float(shape_features[3]),
        "shape_radius_of_gyration": float(shape_features[4]),
        "shape_npr1": float(shape_features[5]),
        "shape_npr2": float(shape_features[6]),
        "shape_span_l": float(shape_features[7]),
        "shape_span_w": float(shape_features[8]),
        "shape_span_h": float(shape_features[9]),
    }

    return {
        "atom_features": local_atom_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "coords": coords_current,
        "coords_original": original_coords_heavy,
        "coords_original_with_h": original_coords_all,
        "coords_heavy_current": coords_heavy_current,
        "gasteiger_charges": gasteiger_charges,
        "logp_contribs": logp_contribs,
        "tpsa_contribs": tpsa_contribs,
        "rmsf": rmsf_full,
        "flexibility_flag": flexibility_flag_full,
        "atom_sasa": atom_sasa,
        "lipophilic_sasa": lipophilic_sasa.astype(np.float32),
        "hbond_donor": hbond_donor,
        "hbond_acceptor": hbond_acceptor,
        "chirality_cw": chirality_cw,
        "chirality_ccw": chirality_ccw,
        "rotatable_bond_context": rotatable_context.astype(np.float32),
        "formal_charge_class": np.asarray([get_formal_charge_class_features(atom) for atom in mol.GetAtoms()], dtype=np.float32),
        "metal_binding_flag": metal_binding.astype(np.float32),
        "stereochemistry_flag": stereo_flags.astype(np.float32),
        "shape_features": shape_features.astype(np.float32),
        "num_atoms": num_atoms_with_h,
        "num_heavy_atoms": num_heavy_atoms,
        "has_original_coords": original_coords_heavy is not None,
        "molecular_descriptors": molecular_descriptors,
        "feature_dim_local": local_atom_features.shape[1],
    }
