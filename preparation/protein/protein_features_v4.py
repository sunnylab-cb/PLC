#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent protein feature extraction V4.

Design goals:
- Protein-only features, no ligand-dependent information
- Preserve output field names used by the current training pipeline
- Preserve coordinate fields used for distance calculations
- Add stronger backbone/local-environment features
- Support train-set-level normalization through an external stats JSON
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preparation.protein.extract_physchem_from_pdb import parse_pdb_protein
from preparation.protein.physicochemical_features import extract_physicochemical_features
from preparation.protein.surface_features import extract_surface_features
from preparation.protein.structural_geometry_features import extract_structural_geometry_features
from preparation.protein.enhanced_features_v2 import (
    compute_pocketness_features,
    compute_flexibility_features,
    compute_environmental_features,
)

try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("Biopython not available. Some geometry features may be limited.")


AMINO_ACIDS = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
    'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG',
    'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

NONPOLAR_RESIDUES = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'GLY']
POLAR_RESIDUES = ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN']
CHARGED_RESIDUES = ['LYS', 'ARG', 'HIS', 'ASP', 'GLU']
AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}
HBOND_DONOR_RESIDUES = {'SER', 'THR', 'TYR', 'CYS', 'ASN', 'GLN', 'ARG', 'LYS', 'HIS', 'TRP'}
HBOND_ACCEPTOR_RESIDUES = {'ASP', 'GLU', 'SER', 'THR', 'TYR', 'ASN', 'GLN', 'HIS'}
METAL_BINDER_RESIDUES = {'HIS', 'ASP', 'GLU', 'CYS', 'MET', 'TYR', 'SER', 'THR', 'ASN', 'GLN'}

AROMATIC_RESIDUE_ATOMS = {
    'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2'],
}


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    mask = (norm > eps).astype(np.float32)
    return np.where(mask, v / (norm + eps), 0.0)


def angle_to_sin_cos(angle: float) -> Tuple[float, float]:
    return float(np.sin(angle)), float(np.cos(angle))


def dihedral_angle(p0, p1, p2, p3) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def get_atom_coord_by_name(atom_names: List[str], atom_coords: np.ndarray, atom_name: str) -> Optional[np.ndarray]:
    for name, coord in zip(atom_names, atom_coords):
        if name == atom_name:
            return np.asarray(coord, dtype=np.float32)
    return None


def get_sidechain_heavy_atoms(atom_names: List[str], atom_coords: np.ndarray) -> np.ndarray:
    keep = []
    for name, coord in zip(atom_names, atom_coords):
        if name not in {'N', 'CA', 'C', 'O'}:
            keep.append(coord)
    return np.asarray(keep, dtype=np.float32) if keep else np.zeros((0, 3), dtype=np.float32)


def extract_physchem_features_v4(
    pdb_info: Dict,
    v2_physchem: np.ndarray,
    v2_sequence: np.ndarray,
    v2_hbond: np.ndarray,
    v2_structural_geometry: Optional[np.ndarray] = None,
) -> np.ndarray:
    residue_names = pdb_info['residue_names']
    n_residues = len(residue_names)
    physchem_features = np.zeros((n_residues, 62), dtype=np.float32)

    physchem_features[:, 0:20] = v2_physchem[:, 0:20]
    physchem_features[:, 20:28] = v2_sequence
    physchem_features[:, 28:36] = v2_hbond

    for i, res_name in enumerate(residue_names):
        if res_name in AMINO_ACIDS:
            idx = AMINO_ACIDS.index(res_name)
            physchem_features[i, 36 + idx] = 1.0

    if v2_structural_geometry is not None and v2_structural_geometry.shape[1] >= 8:
        dssp8 = v2_structural_geometry[:, :8]
        ss3_scores = np.stack([
            dssp8[:, [0, 3, 4]].sum(axis=1),
            dssp8[:, [1, 2]].sum(axis=1),
            dssp8[:, [5, 6, 7]].sum(axis=1),
        ], axis=1)
        valid_mask = ss3_scores.sum(axis=1) > 0
        if np.any(valid_mask):
            ss3_idx = np.argmax(ss3_scores[valid_mask], axis=1)
            physchem_features[valid_mask, 56:59] = 0.0
            physchem_features[valid_mask, 56 + ss3_idx] = 1.0
        if np.any(~valid_mask):
            physchem_features[~valid_mask, 56:59] = np.array([0, 0, 1], dtype=np.float32)
    else:
        physchem_features[:, 56:59] = np.array([0, 0, 1], dtype=np.float32)

    for i, res_name in enumerate(residue_names):
        if res_name in NONPOLAR_RESIDUES:
            physchem_features[i, 59:62] = [1, 0, 0]
        elif res_name in POLAR_RESIDUES:
            physchem_features[i, 59:62] = [0, 1, 0]
        elif res_name in CHARGED_RESIDUES:
            physchem_features[i, 59:62] = [0, 0, 1]

    return physchem_features


def compute_local_density_raw(ca_coords: np.ndarray, radii: List[float] = [5.0, 10.0, 15.0]) -> np.ndarray:
    n_residues = len(ca_coords)
    density_features = np.zeros((n_residues, 4), dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i, radius in enumerate(radii):
        neighbors_count = np.sum((dist_matrix > 0) & (dist_matrix < radius), axis=1)
        volume = (4.0 / 3.0) * np.pi * (radius ** 3)
        density_features[:, i] = neighbors_count / volume

    density_features[:, 3] = density_features[:, 1] - density_features[:, 0]
    return density_features


def compute_distance_to_center_raw(ca_coords: np.ndarray) -> np.ndarray:
    center = ca_coords.mean(axis=0)
    return np.linalg.norm(ca_coords - center, axis=1).astype(np.float32)


def compute_surface_normal_magnitude(ca_coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    n_residues = len(ca_coords)
    normal_magnitude = np.zeros(n_residues, dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
        neighbors_coords = ca_coords[neighbors_mask]
        if len(neighbors_coords) < 3:
            continue
        coords_centered = neighbors_coords - ca_coords[i]
        cov_matrix = coords_centered.T @ coords_centered
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        if eigenvalues.max() > 1e-8:
            normal_magnitude[i] = 1.0 - (eigenvalues.min() / eigenvalues.max())

    return normal_magnitude


def compute_local_charge_density_raw(ca_coords: np.ndarray, residue_names: List[str], radii: List[float] = [5.0, 10.0]) -> np.ndarray:
    n_residues = len(ca_coords)
    charge_density = np.zeros((n_residues, 4), dtype=np.float32)
    positive_residues = {'LYS', 'ARG', 'HIS'}
    negative_residues = {'ASP', 'GLU'}
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors = [residue_names[k] for k in range(n_residues) if neighbors_mask[k]]
            volume = (4.0 / 3.0) * np.pi * (radius ** 3)
            charge_density[i, j] = sum(1 for res in neighbors if res in positive_residues) / volume
            charge_density[i, j + 2] = sum(1 for res in neighbors if res in negative_residues) / volume

    return charge_density


def compute_local_hydrophobicity(ca_coords: np.ndarray, residue_names: List[str], radii: List[float] = [5.0, 10.0]) -> np.ndarray:
    n_residues = len(ca_coords)
    hydrophobicity = np.zeros((n_residues, 4), dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors = [residue_names[k] for k in range(n_residues) if neighbors_mask[k]]
            if not neighbors:
                continue
            hydrophobicity[i, j] = sum(1 for res in neighbors if res in NONPOLAR_RESIDUES) / len(neighbors)
            hydrophobicity[i, j + 2] = sum(1 for res in neighbors if res in POLAR_RESIDUES or res in CHARGED_RESIDUES) / len(neighbors)

    return hydrophobicity


def compute_backbone_torsion_features(pdb_info: Dict) -> np.ndarray:
    n_residues = pdb_info['n_residues']
    features = np.zeros((n_residues, 6), dtype=np.float32)
    atom_names = pdb_info['all_atom_names']
    atom_coords = pdb_info['all_atom_coords']
    chain_ids = pdb_info['chain_ids']

    N = [get_atom_coord_by_name(n, c, 'N') for n, c in zip(atom_names, atom_coords)]
    CA = [get_atom_coord_by_name(n, c, 'CA') for n, c in zip(atom_names, atom_coords)]
    C = [get_atom_coord_by_name(n, c, 'C') for n, c in zip(atom_names, atom_coords)]

    for i in range(n_residues):
        phi = psi = omega = None

        if i > 0 and chain_ids[i - 1] == chain_ids[i]:
            if C[i - 1] is not None and N[i] is not None and CA[i] is not None and C[i] is not None:
                phi = dihedral_angle(C[i - 1], N[i], CA[i], C[i])
            if CA[i - 1] is not None and C[i - 1] is not None and N[i] is not None and CA[i] is not None:
                omega = dihedral_angle(CA[i - 1], C[i - 1], N[i], CA[i])

        if i < n_residues - 1 and chain_ids[i + 1] == chain_ids[i]:
            if N[i] is not None and CA[i] is not None and C[i] is not None and N[i + 1] is not None:
                psi = dihedral_angle(N[i], CA[i], C[i], N[i + 1])

        if phi is not None:
            features[i, 0], features[i, 1] = angle_to_sin_cos(phi)
        if psi is not None:
            features[i, 2], features[i, 3] = angle_to_sin_cos(psi)
        if omega is not None:
            features[i, 4], features[i, 5] = angle_to_sin_cos(omega)

    return features


def compute_sidechain_centroids(pdb_info: Dict) -> np.ndarray:
    centroids = np.zeros((pdb_info['n_residues'], 3), dtype=np.float32)
    ca_coords = pdb_info['ca_coords']
    for i, (names, coords) in enumerate(zip(pdb_info['all_atom_names'], pdb_info['all_atom_coords'])):
        sidechain = get_sidechain_heavy_atoms(names, coords)
        if len(sidechain) > 0:
            centroids[i] = sidechain.mean(axis=0)
        else:
            centroids[i] = ca_coords[i]
    return centroids


def compute_sidechain_packing_features(pdb_info: Dict, radii: Tuple[float, float] = (6.0, 10.0)) -> np.ndarray:
    ca_coords = pdb_info['ca_coords']
    centroids = compute_sidechain_centroids(pdb_info)
    n_residues = len(ca_coords)
    out = np.zeros((n_residues, 3), dtype=np.float32)
    dist_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)

    for i in range(n_residues):
        same_chain_neighbors = (dist_matrix[i] > 0)
        dists = dist_matrix[i][same_chain_neighbors]
        out[i, 0] = float(dists.min()) if len(dists) else 0.0  # nearest non-self sidechain centroid distance

        close6 = np.sum((dist_matrix[i] > 0) & (dist_matrix[i] < radii[0]))
        close10 = np.sum((dist_matrix[i] > 0) & (dist_matrix[i] < radii[1]))
        out[i, 1] = close6 / 20.0
        out[i, 2] = 1.0 - (close6 / (close10 + 1e-6))  # local void ratio proxy

    return out


def compute_environment_composition_histograms(ca_coords: np.ndarray, residue_names: List[str], radius: float = 8.0) -> np.ndarray:
    n_residues = len(ca_coords)
    out = np.zeros((n_residues, 4), dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
        neighbors = [residue_names[k] for k in range(n_residues) if neighbors_mask[k]]
        if not neighbors:
            continue
        out[i, 0] = sum(1 for res in neighbors if res in AROMATIC_RESIDUES) / len(neighbors)
        out[i, 1] = sum(1 for res in neighbors if res in HBOND_DONOR_RESIDUES) / len(neighbors)
        out[i, 2] = sum(1 for res in neighbors if res in HBOND_ACCEPTOR_RESIDUES) / len(neighbors)
        out[i, 3] = sum(1 for res in neighbors if res in {'GLY', 'PRO'}) / len(neighbors)

    return out


def compute_metal_binding_potential(ca_coords: np.ndarray, residue_names: List[str], radii: Tuple[float, float] = (6.0, 10.0)) -> np.ndarray:
    n_residues = len(ca_coords)
    out = np.zeros((n_residues, 3), dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    intrinsic = np.array([1.0 if res in METAL_BINDER_RESIDUES else 0.0 for res in residue_names], dtype=np.float32)
    out[:, 0] = intrinsic

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors = intrinsic[neighbors_mask]
            out[i, j + 1] = float(neighbors.mean()) if neighbors.size else 0.0

    return out


def compute_local_anisotropy(ca_coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    n_residues = len(ca_coords)
    anis = np.zeros(n_residues, dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
        neighbors = ca_coords[neighbors_mask]
        if len(neighbors) < 3:
            continue
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[0] > 1e-8:
            anis[i] = (eigvals[0] - eigvals[-1]) / (eigvals[0] + 1e-8)
    return anis


def compute_cavity_pointing_vectors_multi_scale(ca_coords: np.ndarray, radii: Tuple[float, float, float] = (6.0, 10.0, 14.0)) -> np.ndarray:
    n_residues = len(ca_coords)
    out = np.zeros((n_residues, len(radii), 3), dtype=np.float32)
    dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    for i in range(n_residues):
        for j, radius in enumerate(radii):
            neighbors_mask = (dist_matrix[i] > 0) & (dist_matrix[i] < radius)
            neighbors = ca_coords[neighbors_mask]
            if len(neighbors) == 0:
                continue
            com_neighbors = neighbors.mean(axis=0)
            out[i, j] = safe_normalize(ca_coords[i] - com_neighbors)
    return out


def extract_spatial_vector_features_v4(pdb_path: Path, pdb_info: Dict, chain_id: str = None) -> np.ndarray:
    ca_coords = pdb_info['ca_coords']
    residue_names = pdb_info['residue_names']
    residue_indices = pdb_info['residue_indices']
    insertion_codes = pdb_info.get('insertion_codes', [''] * len(residue_names))
    chain_ids = pdb_info['chain_ids']
    n_residues = len(residue_names)

    spatial_vectors = np.zeros((n_residues, 8, 3), dtype=np.float32)

    if not BIOPYTHON_AVAILABLE:
        warnings.warn("Biopython unavailable; returning zero spatial vectors")
        return spatial_vectors

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
    except Exception as e:
        warnings.warn(f"无法解析 PDB 文件 {pdb_path}: {e}")
        return spatial_vectors

    multi_scale_cavity = compute_cavity_pointing_vectors_multi_scale(ca_coords)

    for i, (res_name, res_idx, chain_id_i, ins_code) in enumerate(
        zip(residue_names, residue_indices, chain_ids, insertion_codes)
    ):
        ca_coord = ca_coords[i]
        chain_key = str(chain_id_i).strip()
        if chain_key not in structure[0]:
            continue

        chain = structure[0][chain_key]
        icode = ins_code if ins_code else ' '
        try:
            residue = chain[(' ', int(res_idx), icode)]
        except Exception:
            continue

        if residue.get_resname().strip() != str(res_name).strip():
            continue

        n_coord = residue['N'].get_coord() if 'N' in residue else None
        c_coord = residue['C'].get_coord() if 'C' in residue else None
        o_coord = residue['O'].get_coord() if 'O' in residue else None

        spatial_vectors[i, 0] = compute_sidechain_base_vector(residue, res_name, ca_coord, n_coord, c_coord)
        spatial_vectors[i, 1] = compute_sidechain_com_vector(residue, res_name, ca_coord, spatial_vectors[i, 0])
        spatial_vectors[i, 2] = compute_aromatic_normal_vector(residue, res_name, ca_coord)
        spatial_vectors[i, 3] = compute_hbond_donor_vector(residue, res_name, spatial_vectors[i, 1])
        spatial_vectors[i, 4] = compute_hbond_acceptor_vector(residue, res_name, c_coord, o_coord, spatial_vectors[i, 1])
        spatial_vectors[i, 5] = multi_scale_cavity[i, 0]
        spatial_vectors[i, 6] = multi_scale_cavity[i, 1]
        spatial_vectors[i, 7] = multi_scale_cavity[i, 2]

    return spatial_vectors


def compute_sidechain_base_vector(residue, res_name: str, ca_coord: np.ndarray, n_coord: Optional[np.ndarray], c_coord: Optional[np.ndarray]) -> np.ndarray:
    if res_name != 'GLY' and 'CB' in residue:
        return safe_normalize(residue['CB'].get_coord() - ca_coord)

    if n_coord is not None and c_coord is not None:
        v1 = safe_normalize(ca_coord - n_coord)
        v2 = safe_normalize(ca_coord - c_coord)
        bisector = safe_normalize(v1 + v2)
        cb_virtual = ca_coord - bisector * 1.5
        return safe_normalize(cb_virtual - ca_coord)

    return np.zeros(3, dtype=np.float32)


def compute_sidechain_com_vector(residue, res_name: str, ca_coord: np.ndarray, v_base: np.ndarray) -> np.ndarray:
    sidechain_atoms = []
    for atom in residue.get_atoms():
        atom_name = atom.get_name()
        element = atom.element
        if atom_name not in ['N', 'CA', 'C', 'O'] and element != 'H':
            sidechain_atoms.append(atom.get_coord())

    if sidechain_atoms:
        return safe_normalize(np.mean(sidechain_atoms, axis=0) - ca_coord)
    return v_base


def compute_aromatic_normal_vector(residue, res_name: str, ca_coord: np.ndarray) -> np.ndarray:
    if res_name not in AROMATIC_RESIDUE_ATOMS:
        return np.zeros(3, dtype=np.float32)

    ring_coords = [residue[a].get_coord() for a in AROMATIC_RESIDUE_ATOMS[res_name] if a in residue]
    if len(ring_coords) < 3:
        return np.zeros(3, dtype=np.float32)

    ring_coords = np.array(ring_coords)
    ring_center = ring_coords.mean(axis=0)
    coords_centered = ring_coords - ring_center
    cov_matrix = coords_centered.T @ coords_centered

    try:
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        min_idx = np.argmin(eigvals)
        normal = eigvecs[:, min_idx].real
        if np.dot(normal, ca_coord - ring_center) < 0:
            normal = -normal
        return safe_normalize(normal)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32)


def compute_hbond_donor_vector(residue, res_name: str, v_com: np.ndarray) -> np.ndarray:
    return v_com if res_name in HBOND_DONOR_RESIDUES else np.zeros(3, dtype=np.float32)


def compute_hbond_acceptor_vector(residue, res_name: str, c_coord: Optional[np.ndarray], o_coord: Optional[np.ndarray], v_com: np.ndarray) -> np.ndarray:
    if res_name in HBOND_ACCEPTOR_RESIDUES:
        return -v_com
    if c_coord is not None and o_coord is not None:
        return safe_normalize(o_coord - c_coord)
    return np.zeros(3, dtype=np.float32)


def get_default_normalization_stats() -> Dict[str, Dict]:
    return {
        "scalar_means": {},
        "scalar_stds": {},
        "version": "protein_features_v4_unset",
        "notes": "Run fit_normalization_stats() and save to JSON for train-set-level normalization.",
    }


def load_normalization_stats(stats_path: Optional[Path]) -> Dict[str, Dict]:
    if stats_path is None or not stats_path.exists():
        return get_default_normalization_stats()
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_normalization_stats(stats: Dict, stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def fit_normalization_stats(raw_spatial_scalar_list: Iterable[np.ndarray], feature_names: List[str]) -> Dict:
    all_features = np.concatenate(list(raw_spatial_scalar_list), axis=0)
    means = all_features.mean(axis=0)
    stds = all_features.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return {
        "version": "protein_features_v4_stats_v1",
        "feature_names": feature_names,
        "scalar_means": means.tolist(),
        "scalar_stds": stds.tolist(),
    }


def normalize_spatial_scalar_features(raw_features: np.ndarray, stats: Optional[Dict]) -> np.ndarray:
    if stats is None or "scalar_means" not in stats or "scalar_stds" not in stats:
        return raw_features.astype(np.float32)

    means = np.asarray(stats["scalar_means"], dtype=np.float32)
    stds = np.asarray(stats["scalar_stds"], dtype=np.float32)
    if means.shape[0] != raw_features.shape[1] or stds.shape[0] != raw_features.shape[1]:
        warnings.warn("Normalization stats shape mismatch, fallback to raw features")
        return raw_features.astype(np.float32)

    return ((raw_features - means) / stds).astype(np.float32)


def get_spatial_scalar_feature_names_v4() -> List[str]:
    names = []
    names += [f"v2_sasa_{i}" for i in range(6)]
    names += [f"v2_surface_{i}" for i in range(14)]
    names += [f"v2_structural_geometry_{i}" for i in range(26)]
    names += [f"v2_enhanced_{i}" for i in range(12)]
    names += [f"local_density_raw_{i}" for i in range(4)]
    names += ["distance_to_center_raw"]
    names += ["surface_normal_magnitude"]
    names += [f"charge_density_raw_{i}" for i in range(4)]
    names += [f"hydrophobicity_{i}" for i in range(4)]
    names += ["phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos"]
    names += ["nearest_sidechain_dist", "packing_score", "void_ratio"]
    names += ["env_aromatic_ratio", "env_hbond_donor_ratio", "env_hbond_acceptor_ratio", "env_gly_pro_ratio"]
    names += ["self_metal_binder", "neighbor_metal_binder_6A", "neighbor_metal_binder_10A"]
    names += ["local_anisotropy"]
    return names


def extract_spatial_scalar_features_v4(
    pdb_info: Dict,
    v2_sasa: np.ndarray,
    v2_surface: np.ndarray,
    v2_structural_geometry: np.ndarray,
    v2_enhanced: np.ndarray,
    normalization_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ca_coords = pdb_info['ca_coords']
    residue_names = pdb_info['residue_names']
    n_residues = len(residue_names)

    raw = np.zeros((n_residues, 89), dtype=np.float32)
    raw[:, 0:6] = v2_sasa
    raw[:, 6:20] = v2_surface
    raw[:, 20:46] = v2_structural_geometry
    raw[:, 46:58] = v2_enhanced

    raw[:, 58:62] = compute_local_density_raw(ca_coords)
    raw[:, 62] = compute_distance_to_center_raw(ca_coords)
    raw[:, 63] = compute_surface_normal_magnitude(ca_coords)
    raw[:, 64:68] = compute_local_charge_density_raw(ca_coords, residue_names)
    raw[:, 68:72] = compute_local_hydrophobicity(ca_coords, residue_names)
    raw[:, 72:78] = compute_backbone_torsion_features(pdb_info)
    raw[:, 78:81] = compute_sidechain_packing_features(pdb_info)
    raw[:, 81:85] = compute_environment_composition_histograms(ca_coords, residue_names)
    raw[:, 85:88] = compute_metal_binding_potential(ca_coords, residue_names)
    raw[:, 88] = compute_local_anisotropy(ca_coords)

    normalized = normalize_spatial_scalar_features(raw, normalization_stats)
    return raw, normalized


def extract_complete_features_v4(
    pdb_path: Path,
    chain_id: str = None,
    mode: str = 'complex',
    normalization_stats: Optional[Dict] = None,
) -> Dict:
    pdb_info = parse_pdb_protein(pdb_path, chain_id, mode)
    n_residues = pdb_info['n_residues']

    v2_physchem_full = extract_physicochemical_features(
        pdb_path,
        pdb_info['sequence'],
        pdb_info['residue_names'],
        pdb_info['residue_indices'],
        pdb_info['chain_ids'],
    )
    v2_surface = extract_surface_features(pdb_info['ca_coords'], pdb_info['residue_names'])
    v2_structural_geometry = extract_structural_geometry_features(
        pdb_path,
        pdb_info['ca_coords'],
        pdb_info['residue_indices'],
        pdb_info['chain_ids'],
    )
    pocketness = compute_pocketness_features(pdb_info['ca_coords'])
    sasa_values = v2_surface[:, 0]
    flexibility = compute_flexibility_features(str(pdb_path), pdb_info['ca_coords'], sasa_values)
    environmental = compute_environmental_features(
        pdb_info['ca_coords'], v2_physchem_full, pdb_info['residue_names']
    )
    v2_enhanced = np.concatenate([pocketness, flexibility, environmental], axis=1)

    v2_basic = v2_physchem_full[:, 0:20]
    v2_sasa = v2_physchem_full[:, 20:26]
    v2_sequence = v2_physchem_full[:, 26:34]
    v2_hbond = v2_physchem_full[:, 34:42]

    physchem_features = extract_physchem_features_v4(
        pdb_info, v2_basic, v2_sequence, v2_hbond, v2_structural_geometry
    )
    spatial_scalar_raw, spatial_scalar_features = extract_spatial_scalar_features_v4(
        pdb_info, v2_sasa, v2_surface, v2_structural_geometry, v2_enhanced, normalization_stats
    )
    spatial_vector_features = extract_spatial_vector_features_v4(pdb_path, pdb_info, chain_id)

    if np.isnan(physchem_features).any():
        warnings.warn("物理化学特征包含 NaN")
    if np.isnan(spatial_scalar_features).any():
        warnings.warn("空间标量特征包含 NaN")
    if np.isnan(spatial_vector_features).any():
        warnings.warn("空间矢量特征包含 NaN")

    return {
        'physchem_features': torch.from_numpy(physchem_features).float(),
        'spatial_scalar_features': torch.from_numpy(spatial_scalar_features).float(),
        'spatial_scalar_raw_features': torch.from_numpy(spatial_scalar_raw).float(),
        'spatial_vector_features': torch.from_numpy(spatial_vector_features).float(),
        'ca_coords': torch.from_numpy(pdb_info['ca_coords']).float(),
        'all_atom_coords': pdb_info['all_atom_coords'],
        'all_atom_names': pdb_info['all_atom_names'],
        'residue_names': pdb_info['residue_names'],
        'residue_indices': pdb_info['residue_indices'],
        'insertion_codes': pdb_info.get('insertion_codes', [''] * n_residues),
        'chain_ids': pdb_info['chain_ids'],
        'n_residues': n_residues,
        'spatial_scalar_feature_names': get_spatial_scalar_feature_names_v4(),
    }
