#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配体特征提取模块 V3

在V2基础上新增：
- 原子级SASA与暴露疏水面积
- 氢键供体/受体标志
- 原子手性标志
- 分子3D形状描述符

输出维度：
- V2: 35维 per-atom
- V3: 41维 per-atom (35 + 6)
- 额外输出10维 shape_features
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit import RDLogger
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem import Descriptors3D
    from rdkit.Geometry import Point3D
    RDKIT_AVAILABLE = True
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install: conda install -c conda-forge rdkit")

try:
    from rdkit.Chem import rdFreeSASA
    RDFREESASA_AVAILABLE = True
except Exception:
    rdFreeSASA = None
    RDFREESASA_AVAILABLE = False

from ligand_features_v2 import extract_ligand_features_from_file as extract_v2_features


_HBOND_FEATURE_FACTORY = None


def _robust_load_mol(ligand_file: str) -> "Chem.Mol":
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required")

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
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return np.zeros(0, dtype=np.float32)
    if not RDFREESASA_AVAILABLE or mol.GetNumConformers() == 0:
        return np.zeros(num_atoms, dtype=np.float32)

    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        rdFreeSASA.CalcSASA(mol, radii)
        sasa = np.zeros(num_atoms, dtype=np.float32)
        for atom in mol.GetAtoms():
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


def compute_ligand_3d_shape_features(mol: "Chem.Mol") -> list[float]:
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return [0.0] * 5

    try:
        pmi1 = float(Descriptors3D.PMI1(mol))
        pmi2 = float(Descriptors3D.PMI2(mol))
        pmi3 = float(Descriptors3D.PMI3(mol))
        spherocity = float(Descriptors3D.SpherocityIndex(mol))
        radius_of_gyration = float(Descriptors3D.RadiusOfGyration(mol))
        return [pmi1, pmi2, pmi3, spherocity, radius_of_gyration]
    except Exception as exc:
        warnings.warn(f"3D shape calculation failed: {exc}")
        return [0.0] * 5


def get_inertial_shape_factor(mol: "Chem.Mol") -> list[float]:
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return [0.0, 0.0]

    try:
        npr1 = float(Descriptors3D.NPR1(mol))
        npr2 = float(Descriptors3D.NPR2(mol))
        return [npr1, npr2]
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


def extract_ligand_features_from_file(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    base_features = extract_v2_features(
        ligand_file,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )

    base_atom_features = np.asarray(base_features['atom_features'], dtype=np.float32)
    num_atoms = int(base_atom_features.shape[0])

    atom_sasa = np.zeros(num_atoms, dtype=np.float32)
    hbond_donor = np.zeros(num_atoms, dtype=np.float32)
    hbond_acceptor = np.zeros(num_atoms, dtype=np.float32)
    chirality_cw = np.zeros(num_atoms, dtype=np.float32)
    chirality_ccw = np.zeros(num_atoms, dtype=np.float32)
    shape_features = np.zeros(10, dtype=np.float32)

    try:
        mol = _robust_load_mol(ligand_file)
        mol = _align_atom_count(mol, num_atoms)

        coords = np.asarray(base_features.get('coords'), dtype=np.float32)
        if mol.GetNumAtoms() == num_atoms and coords.shape == (num_atoms, 3):
            _set_coords_from_array(mol, coords)

        if mol.GetNumAtoms() == num_atoms:
            atom_sasa = compute_atom_sasa(mol)
            hbond_donor, hbond_acceptor = compute_hbond_flags(mol)
            chirality_cw, chirality_ccw = compute_chirality_flags(mol)

            shape_values = (
                compute_ligand_3d_shape_features(mol)
                + get_inertial_shape_factor(mol)
                + compute_principal_axis_spans(mol)
            )
            shape_features = np.asarray(shape_values, dtype=np.float32)
        else:
            warnings.warn(
                f"Atom count mismatch for advanced features: {ligand_file} "
                f"(mol={mol.GetNumAtoms()}, expected={num_atoms})"
            )
    except Exception as exc:
        warnings.warn(f"Advanced ligand feature extraction failed: {exc}")

    atom_sasa = _resize_vector(atom_sasa, num_atoms)
    hbond_donor = _resize_vector(hbond_donor, num_atoms)
    hbond_acceptor = _resize_vector(hbond_acceptor, num_atoms)
    chirality_cw = _resize_vector(chirality_cw, num_atoms)
    chirality_ccw = _resize_vector(chirality_ccw, num_atoms)
    logp_contribs = _resize_vector(base_features['logp_contribs'], num_atoms)

    lipophilic_sasa = atom_sasa * logp_contribs
    atom_extensions = np.stack(
        [atom_sasa, lipophilic_sasa, hbond_donor, hbond_acceptor, chirality_cw, chirality_ccw],
        axis=1
    ).astype(np.float32)

    atom_features_v3 = np.concatenate([base_atom_features, atom_extensions], axis=1).astype(np.float32)

    shape_features = _resize_vector(shape_features, 10)

    molecular_descriptors = dict(base_features.get('molecular_descriptors', {}))
    molecular_descriptors.update({
        'shape_pmi1': float(shape_features[0]),
        'shape_pmi2': float(shape_features[1]),
        'shape_pmi3': float(shape_features[2]),
        'shape_spherocity': float(shape_features[3]),
        'shape_radius_of_gyration': float(shape_features[4]),
        'shape_npr1': float(shape_features[5]),
        'shape_npr2': float(shape_features[6]),
        'shape_span_l': float(shape_features[7]),
        'shape_span_w': float(shape_features[8]),
        'shape_span_h': float(shape_features[9]),
    })

    result = base_features.copy()
    result['atom_features'] = atom_features_v3
    result['atom_sasa'] = atom_sasa
    result['lipophilic_sasa'] = lipophilic_sasa.astype(np.float32)
    result['hbond_donor'] = hbond_donor
    result['hbond_acceptor'] = hbond_acceptor
    result['chirality_cw'] = chirality_cw
    result['chirality_ccw'] = chirality_ccw
    result['shape_features'] = shape_features.astype(np.float32)
    result['molecular_descriptors'] = molecular_descriptors
    return result


def extract_ligand_features_v3(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> Dict[str, np.ndarray]:
    return extract_ligand_features_from_file(
        ligand_file,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )
