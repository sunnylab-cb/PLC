#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip ligand feature extraction — V6-Min pipeline.

Extends the V5 pipeline (109D per-atom features) with classifier-first routing
and explicit feature validity tracking.  V5 x[109] layout is UNCHANGED.

New per-.pt fields (V6):
  x_group_mask   [N, 14] float32  — which feature groups are valid (0/1)
  x_group_source [N, 14] int8     — FeatureSource enum per group
  x_atom_extra   [N, 14] float32  — periodic-table element features
  x_atom_extra_mask [N, 14] float32
  x_atom_extra_source [N, 14] int8 — FeatureSource enum per extra field
  operation_status [12]  int8     — OperationStatus enum per pipeline op
  ligand_type    [14]    float32  — multi-hot type flags
  ligand_route           int      — LigandRoute enum (primary path)
  conformer_policy       int      — ConformerPolicy enum
  hydrogen_policy        int      — HydrogenPolicy enum
  fragment_count         int
  fragment_id            [N] int64 — connected-component id per current atom
  largest_fragment_atom_fraction  float
  fragment_pruned        bool
  feature_completeness_ratio  float
  feature_schema_version      str
  ligand_type_from_id_table   bool
  ccd_lookup_used             bool

Routing (5 paths):
  PURE_METAL_ION       → bypass V5, only base topology group is valid
  INORGANIC_POLYATOMIC → bypass V5, only base topology group is valid
  METAL_COMPLEX        → V5 strict=False, no conformers
  NO_CONFORMER_BIOL    → V5 strict=False, no conformers
  REGULAR_ORGANIC      → full V5 pipeline
  MULTIFRAGMENT        → raw-graph fallback, keep all fragments

Directory layout
────────────────
  {biolip_root}/shared_receptors2/{pdb_id}/pairs/
    {pdb_id}_{chain}/
      {bs_id}_{lig_id}_{chain}_{resseq}/
        pair_meta.json
        *_ligand_ori.sdf   ← input (Kabsch-reconstructed)

Output
──────
  {output_root}/{pdb_id}/{pdb_id}_{chain}_{bs_id}_{lig_id}.pt

Usage
─────
conda run -n AIAA python3 preparation/ligand/biolip_ligand_extract.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/ligand_features_v6 \\
    --max 5 --workers 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pickle
import shutil
import signal
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    from tqdm import tqdm as _tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'preparation' / 'ligand'))

# ── V5 ligand feature pipeline ────────────────────────────────────────────────
try:
    from extract_ligand_features_v5 import extract_ligand_features_v5
    V5_AVAILABLE = True
except ImportError as _e:
    V5_AVAILABLE = False
    _V5_IMPORT_ERR = str(_e)

# ── V6 ligand types contract ──────────────────────────────────────────────────
try:
    from ligand_biolip_types import (
        FEATURE_SCHEMA_VERSION,
        FeatureSource, OperationStatus, LigandRoute, HydrogenPolicy, ConformerPolicy,
        N_LIGAND_TYPES, N_X_GROUPS, N_OPERATIONS, X_ATOM_EXTRA_DIM,
        OP_IDX,
        classify_ligand_type, get_conformer_policy, get_hydrogen_policy,
        raw_mol_summary_from_file, element_feature_vector,
        feature_completeness_ratio,
        empty_operation_status, empty_group_mask, empty_group_source,
    )
    V6_TYPES_AVAILABLE = True
except ImportError as _ve:
    V6_TYPES_AVAILABLE = False
    _V6_TYPES_IMPORT_ERR = str(_ve)


# ── pair discovery ────────────────────────────────────────────────────────────

def _parse_meta_file(meta_path: Path) -> Optional[dict]:
    # meta_path: .../shared_receptors2/{pdb_id}/pairs/{pdb_id}_{chain}/{bs_dir}/pair_meta.json
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    bs_dir    = meta_path.parent
    chain_dir = bs_dir.parent
    pdb_dir   = chain_dir.parent.parent  # .../shared_receptors/{pdb_id}
    pdb_id    = pdb_dir.name
    parts     = chain_dir.name.split('_', 1)
    chain     = parts[1] if len(parts) == 2 else chain_dir.name
    return {
        'pdb_id':   pdb_id,
        'chain':    meta.get('protein_chain', chain),
        'bs_id':    meta.get('binding_site_id', bs_dir.name.split('_')[0]),
        'lig_id':   meta.get('ligand_id', ''),
        'resseq':   str(meta.get('ligand_resseq', '')),
        'pair_dir': bs_dir,
        'meta':     meta,
    }


def _discover_pairs(biolip_root: Path, n_threads: int = 32) -> List[dict]:
    import subprocess
    receptors_root = biolip_root / 'shared_receptors2'
    if not receptors_root.is_dir():
        raise FileNotFoundError(f'{receptors_root} not found')

    result = subprocess.run(
        ['find', str(receptors_root), '-mindepth', '5', '-maxdepth', '5',
         '-name', 'pair_meta.json'],
        stdout=subprocess.PIPE, text=True, check=True,
    )
    meta_paths = [Path(p) for p in result.stdout.splitlines() if p.strip()]
    print(f'  Found {len(meta_paths):,} pair_meta.json files, reading…', flush=True)

    pairs: List[dict] = []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for item in ex.map(_parse_meta_file, meta_paths):
            if item is not None:
                pairs.append(item)

    pairs.sort(key=lambda p: (p['pdb_id'], p['chain'], p['bs_id'], p['lig_id']))
    return pairs


def _output_stem(pair: dict) -> str:
    return f"{pair['pdb_id']}_{pair['chain']}_{pair['bs_id']}_{pair['lig_id']}"


def _output_path(pair: dict, output_root: Path) -> Path:
    """Return {output_root}/{pdb_id}/{stem}.pt"""
    return output_root / pair['pdb_id'] / f"{_output_stem(pair)}.pt"


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = max(0, int(seconds))

    def __enter__(self):
        if self.seconds > 0:
            def _handler(signum, frame):
                raise TimeoutError(f'timeout after {self.seconds}s')
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
        return False


# ── V6 helpers ────────────────────────────────────────────────────────────────

def _load_mol_raw(mol_file: Path):
    """Load mol with sanitize=False, removeHs=False (lightweight, no chemistry)."""
    try:
        from rdkit import Chem
        suffix = mol_file.suffix.lower()
        if suffix in ('.sdf', '.mol'):
            return next(Chem.SDMolSupplier(str(mol_file), sanitize=False, removeHs=False), None)
        elif suffix == '.mol2':
            return Chem.MolFromMol2File(str(mol_file), sanitize=False, removeHs=False)
        elif suffix == '.pdb':
            return Chem.MolFromPDBFile(str(mol_file), sanitize=False, removeHs=False)
    except Exception:
        pass
    return None


def _atom_extra_from_mol(mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build x_atom_extra [N, 14] and mask [N, 14] for all atoms in mol.
    Uses only element symbol → periodic table (works on unsanitized mol).
    """
    atoms = list(mol.GetAtoms())
    n = len(atoms)
    x_extra = np.zeros((n, X_ATOM_EXTRA_DIM), dtype=np.float32)
    x_extra_mask = np.zeros((n, X_ATOM_EXTRA_DIM), dtype=np.float32)
    for i, atom in enumerate(atoms):
        feat, msk = element_feature_vector(atom.GetSymbol())
        x_extra[i] = feat
        x_extra_mask[i] = msk
    return x_extra, x_extra_mask


def _atom_extra_with_addh(mol_file: Path, n_expected: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mol, apply AddHs to match V5 atom count, build x_atom_extra [N, 14].
    Pads with zeros if count mismatches (graceful fallback).
    """
    x_extra = np.zeros((n_expected, X_ATOM_EXTRA_DIM), dtype=np.float32)
    x_extra_mask = np.zeros((n_expected, X_ATOM_EXTRA_DIM), dtype=np.float32)

    mol = _load_mol_raw(mol_file)
    if mol is None:
        return x_extra, x_extra_mask
    try:
        from rdkit import Chem
        mol = Chem.AddHs(mol, addCoords=(mol.GetNumConformers() > 0))
    except Exception:
        pass

    atoms = list(mol.GetAtoms())
    for i in range(min(len(atoms), n_expected)):
        feat, msk = element_feature_vector(atoms[i].GetSymbol())
        x_extra[i] = feat
        x_extra_mask[i] = msk
    return x_extra, x_extra_mask


def _coords_from_mol(mol) -> Optional[np.ndarray]:
    """Extract [N, 3] float32 coords from mol's first conformer."""
    try:
        if mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer(0)
        n = mol.GetNumAtoms()
        return np.array([[*conf.GetAtomPosition(i)] for i in range(n)], dtype=np.float32)
    except Exception:
        return None


def _extract_bond_graph_raw(mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build [2, 2E] edge_index (bidirectional) and [2E, 6] edge_attr from unsanitized mol.
    Bond type one-hot: [SINGLE, DOUBLE, TRIPLE, AROMATIC, conjugated, inRing].
    """
    bonds = list(mol.GetBonds())
    if not bonds:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)

    try:
        from rdkit.Chem import BondType
        n_bonds = len(bonds)
        edge_index = np.zeros((2, 2 * n_bonds), dtype=np.int64)
        edge_attr  = np.zeros((2 * n_bonds, 6), dtype=np.float32)
        for i, bond in enumerate(bonds):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0, 2*i] = u;   edge_index[1, 2*i] = v
            edge_index[0, 2*i+1] = v; edge_index[1, 2*i+1] = u
            btype = bond.GetBondType()
            attr = [
                float(btype == BondType.SINGLE),
                float(btype == BondType.DOUBLE),
                float(btype == BondType.TRIPLE),
                float(btype == BondType.AROMATIC),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]
            edge_attr[2*i] = attr
            edge_attr[2*i+1] = attr
        return edge_index, edge_attr
    except Exception:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)


def _safe_atom_scalar(fn, default: float = 0.0) -> float:
    try:
        val = fn()
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _base_atom_features_from_raw_mol(mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the safe, non-descriptor part of V5 x[109] for bypass routes.

    We intentionally fill only fields that are direct atom/topology facts from
    the raw SDF graph: atom type, UNSPECIFIED hybridization, aromatic/ring
    flags when available, degree/charge class, and simple N/O/S metal-donor
    flags. Descriptor/conformer/global/shape slots remain zero and masked out.
    """
    atoms = list(mol.GetAtoms())
    n_atoms = len(atoms)
    x = np.zeros((n_atoms, 109), dtype=np.float32)
    formal_charge_class = np.zeros((n_atoms, 3), dtype=np.float32)
    metal_binding = np.zeros((n_atoms, 4), dtype=np.float32)

    atom_type_idx = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'P': 5,
        'Cl': 6, 'BR': 7, 'I': 8, 'H': 9,
    }

    for i, atom in enumerate(atoms):
        symbol = atom.GetSymbol()
        idx = atom_type_idx.get(symbol, atom_type_idx.get(symbol.upper(), 10))
        x[i, idx] = 1.0

        # Hybridization is not reliable for unsanitized inorganic/metal graphs.
        x[i, 16] = 1.0  # UNSPECIFIED

        try:
            x[i, 17] = float(atom.GetIsAromatic())
        except Exception:
            pass

        try:
            x[i, 18] = float(atom.IsInRing())
            for offset, ring_size in enumerate([3, 4, 5, 6, 7, 8], start=19):
                x[i, offset] = float(atom.IsInRingSize(ring_size))
        except Exception:
            pass

        degree = _safe_atom_scalar(atom.GetDegree)
        explicit_valence = _safe_atom_scalar(atom.GetExplicitValence, degree)
        total_valence = _safe_atom_scalar(atom.GetTotalValence, explicit_valence)
        charge = _safe_atom_scalar(atom.GetFormalCharge)
        radicals = _safe_atom_scalar(atom.GetNumRadicalElectrons)
        x[i, 25:30] = [
            degree / 6.0,
            explicit_valence / 6.0,
            total_valence / 6.0,
            (charge + 2.0) / 4.0,
            radicals / 2.0,
        ]

        formal_charge_class[i] = [
            float(charge < 0),
            float(charge == 0),
            float(charge > 0),
        ]
        x[i, 30:33] = formal_charge_class[i]

        is_n = symbol == 'N' and x[i, 17] == 0 and charge <= 0
        is_o = symbol == 'O'
        is_s = symbol == 'S'
        if atom.GetAtomicNum() == 1:
            is_n = is_o = is_s = False
        metal_binding[i] = [
            float(is_n or is_o or is_s),
            float(is_n),
            float(is_o),
            float(is_s),
        ]
        x[i, 47:51] = metal_binding[i]

    return x, formal_charge_class, metal_binding


def _valid_rmsf(features: dict, requested_conformers: bool) -> bool:
    """True only when V5 returned a non-trivial conformer-ensemble RMSF."""
    if not requested_conformers:
        return False
    rmsf = np.asarray(features.get('rmsf', []), dtype=np.float32).reshape(-1)
    if rmsf.size == 0 or not np.isfinite(rmsf).all():
        return False
    heavy_n = int(features.get('num_heavy_atoms', 0) or 0)
    if heavy_n <= 1:
        return False
    # V4 uses all-zero RMSF for skipped/failed conformer analysis.
    return bool(np.nanmax(np.abs(rmsf)) > 1e-6)


def _normalized_zero_shape() -> np.ndarray:
    """V5-normalized value produced when raw V4 shape_features were all zero."""
    norm_params = {
        0: (3879.901123, 6650.009766),
        1: (13954.526367, 28178.578125),
        2: (15758.538086, 30137.193359),
        4: (4.633801, 1.638986),
        7: (14.585329, 5.682965),
        8: (8.701250, 3.143703),
        9: (5.480688, 2.575974),
    }
    out = np.zeros(10, dtype=np.float32)
    for dim, (mean, std) in norm_params.items():
        out[dim] = (0.0 - mean) / std
    return out


_V5_NORMALIZED_ZERO_SHAPE = _normalized_zero_shape()


def _valid_shape(features: dict) -> bool:
    """Reject NaN and the known V5 normalized-zero sentinel shape."""
    shape = np.asarray(features.get('shape_features', []), dtype=np.float32).reshape(-1)
    if shape.size != 10 or not np.isfinite(shape).all():
        return False
    if int(features.get('num_heavy_atoms', 0) or 0) <= 1:
        return False
    return not np.allclose(shape, _V5_NORMALIZED_ZERO_SHAPE, atol=1e-5, rtol=1e-5)


def _build_group_mask_source(
    n_atoms: int,
    route: 'LigandRoute',
    v5_ok: bool,
    rmsf_valid: bool,
    shape_valid: bool,
    has_exp_coords: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (group_mask [N, G=14] float32, group_source [N, G=14] int8).

    Group layout (matches X_GROUP_FIELDS in ligand_biolip_types.py):
      0  local_gasteiger_charge     3  rmsf_and_flexibility  ← conformer-dep
      1  local_crippen_logp         4  sasa_and_lipophilic_sasa
      2  local_tpsa_contrib         5  hbond_donor_acceptor
      6  chirality                  7  stereochemistry
      8  rot_bond_context           9  degree_and_formal_charge
      10 global_morgan_43           11 global_mollogp
      12 global_tpsa                13 shape_3d

    Source semantics by route:
      REGULAR_ORGANIC  + v5_ok → RDKIT_FULL_SANITIZE  (full sanitize guaranteed)
      METAL_COMPLEX / NO_CONFORMER_BIOLOGICAL + v5_ok → FALLBACK_HEURISTIC
        (V5 internally uses try/except; some descriptors may silently be zero)

    shape_3d (group 13) logic:
      - rmsf_valid=True → valid from conformer ensemble
      - rmsf_valid=False + shape_valid + has_exp_coords + n_atoms > 1 → EXPERIMENTAL_COORDS
        (3D bound pose shape is meaningful even without ETKDG)
      - otherwise → NOT_APPLICABLE

    rmsf_and_flexibility (group 3): ONLY valid if actual RMSF is non-trivial.
    """
    mask = empty_group_mask(n_atoms)
    src  = empty_group_source(n_atoms)

    if route in (LigandRoute.PURE_METAL_ION, LigandRoute.INORGANIC_POLYATOMIC):
        src[:, :] = int(FeatureSource.NOT_APPLICABLE)
        mask[:, 9] = 1.0
        src[:, 9] = int(FeatureSource.FALLBACK_HEURISTIC)
        return mask, src

    if not v5_ok:
        return mask, src  # all UNAVAILABLE (default from empty_group_source)

    # Source for chemistry-derived groups depends on route
    if route == LigandRoute.REGULAR_ORGANIC:
        chem_src = int(FeatureSource.RDKIT_FULL_SANITIZE)
    else:
        # METAL_COMPLEX / NO_CONFORMER_BIOLOGICAL: V5 may have used internal fallbacks
        chem_src = int(FeatureSource.FALLBACK_HEURISTIC)

    na_src  = int(FeatureSource.NOT_APPLICABLE)
    exp_src = int(FeatureSource.EXPERIMENTAL_COORDS)

    # Non-conformer-dependent groups: 0,1,2,4,5,6,7,8,9,10,11,12
    for g in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        mask[:, g] = 1.0
        src[:, g]  = chem_src

    # RMSF + flexibility (group 3): needs conformer ensemble
    if rmsf_valid:
        mask[:, 3] = 1.0; src[:, 3] = chem_src
    else:
        src[:, 3] = na_src  # RMSF silently zero without conformers

    # shape_3d (group 13): valid from conformers OR experimental bound pose
    if shape_valid and rmsf_valid:
        mask[:, 13] = 1.0; src[:, 13] = chem_src
    elif shape_valid and has_exp_coords and n_atoms > 1:
        # 3D shape from bound experimental coordinates is meaningful
        mask[:, 13] = 1.0; src[:, 13] = exp_src
    elif shape_valid and n_atoms > 1:
        mask[:, 13] = 1.0; src[:, 13] = int(FeatureSource.FALLBACK_HEURISTIC)
    else:
        src[:, 13] = na_src

    return mask, src


def _finite_vector(features: dict, key: str, expected_first_dim: int) -> bool:
    arr = np.asarray(features.get(key, []), dtype=np.float32)
    return bool(arr.shape[0] == expected_first_dim and np.isfinite(arr).all())


def _nonzero_vector(features: dict, key: str, expected_first_dim: int) -> bool:
    if not _finite_vector(features, key, expected_first_dim):
        return False
    arr = np.asarray(features.get(key, []), dtype=np.float32)
    return bool(np.nanmax(np.abs(arr)) > 1e-8)


def _apply_v5_group_quality(
    group_mask: np.ndarray,
    group_src: np.ndarray,
    features: dict,
) -> None:
    """
    Downgrade groups that V5 returned as zero/invalid sentinel values.

    V4/V5 do not expose per-substep status.  This cannot prove every descriptor
    is chemically correct, but it prevents the most common silent-zero failures
    from being advertised as valid to the model.
    """
    n_atoms = int(features.get('num_atoms', 0) or 0)
    n_heavy = int(features.get('num_heavy_atoms', 0) or 0)

    def invalidate(g: int):
        group_mask[:, g] = 0.0
        group_src[:, g] = int(FeatureSource.UNAVAILABLE)

    if n_atoms <= 0:
        group_mask[:, :] = 0.0
        group_src[:, :] = int(FeatureSource.UNAVAILABLE)
        return

    # Descriptor groups where an all-zero vector is usually V4 fallback, not
    # true chemistry, once there is more than one heavy atom.
    suspicious_nonzero = [
        (0, 'gasteiger_charges'),
        (1, 'logp_contribs'),
        (4, 'atom_sasa'),
    ]
    for g, key in suspicious_nonzero:
        if n_heavy > 1 and not _nonzero_vector(features, key, n_atoms):
            invalidate(g)

    # TPSA can be truly zero for hydrocarbons, so only reject NaN/shape mismatch.
    for g, key in [(2, 'tpsa_contribs')]:
        if not _finite_vector(features, key, n_atoms):
            invalidate(g)

    global_features = np.asarray(features.get('global_features', []), dtype=np.float32).reshape(-1)
    if global_features.size != 45 or not np.isfinite(global_features).all():
        for g in [10, 11, 12]:
            invalidate(g)
    elif not bool(np.nanmax(np.abs(global_features[:43])) > 1e-8):
        invalidate(10)


def _fragment_info_from_mol(mol, n_atoms: int):
    """Returns (fragment_count, largest_fragment_atom_fraction)."""
    try:
        from rdkit.Chem import rdmolops
        frags = rdmolops.GetMolFrags(mol)
        n_frags = len(frags)
        if n_frags <= 1 or n_atoms == 0:
            return n_frags, 1.0
        atoms_per_frag = [len(frag_atoms) for frag_atoms in frags]
        largest_frac = max(atoms_per_frag) / n_atoms
        return n_frags, float(largest_frac)
    except Exception:
        return 1, 1.0


def _atom_fragment_ids_and_fraction(mol_file: Path, n_expected: int):
    """
    Fragment ids aligned to current atom order, plus largest atom-fragment fraction.
    This does not prune anything; ids are only provenance for multifragment inputs.
    """
    fragment_id = np.zeros(n_expected, dtype=np.int64)
    mol = _load_mol_raw(mol_file)
    if mol is None:
        return fragment_id, 1.0

    try:
        from rdkit.Chem import rdmolops
        frags = rdmolops.GetMolFrags(mol)
        atom_to_frag = {idx: fi for fi, frag in enumerate(frags) for idx in frag}
        for atom_idx in range(min(mol.GetNumAtoms(), n_expected)):
            fragment_id[atom_idx] = atom_to_frag.get(atom_idx, 0)

        if n_expected == 0 or len(frags) <= 1:
            return fragment_id, 1.0
        counts = [len(frag) for frag in frags]
        largest_frac = max(counts) / max(1, sum(counts))
        return fragment_id, float(largest_frac)
    except Exception:
        return fragment_id, 1.0


def _build_minimal_data_from_mol(
    mol,
    pair: dict,
    stem: str,
    ori_sdf: Path,
    ligand_type_vec: np.ndarray,
    route: 'LigandRoute',
    type_from_id_table: bool,
    conf_policy: 'ConformerPolicy',
    h_policy: 'HydrogenPolicy',
    op_status: np.ndarray,
    mol_summary,
) -> dict:
    """
    Build output data dict for ligands that bypass V5 or fall back from V5.
    Only safe base atom/topology fields are filled; descriptor/conformer groups
    remain zero and NOT_APPLICABLE.
    Per-atom scalars use the actual all-atom count (no AddHs called).
    """
    n_total = mol.GetNumAtoms()
    heavy_mask = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()], dtype=bool)
    n_heavy = int(heavy_mask.sum())

    coords_all = _coords_from_mol(mol)
    has_coords = coords_all is not None
    if coords_all is None:
        coords_all = np.zeros((n_total, 3), dtype=np.float32)
    coords_heavy = coords_all[heavy_mask]

    edge_index, edge_attr = _extract_bond_graph_raw(mol)
    x_atom_extra, x_atom_extra_mask = _atom_extra_from_mol(mol)
    x_atom_extra_source = np.where(
        x_atom_extra_mask > 0,
        int(FeatureSource.PERIODIC_TABLE_LOOKUP),
        int(FeatureSource.UNAVAILABLE),
    ).astype(np.int8)

    group_mask = empty_group_mask(n_total)
    group_src  = empty_group_source(n_total)
    group_src[:, :] = int(FeatureSource.NOT_APPLICABLE)
    group_mask[:, 9] = 1.0
    group_src[:, 9] = int(FeatureSource.FALLBACK_HEURISTIC)

    n_frags, largest_frac = _fragment_info_from_mol(mol, n_total)
    atom_features, formal_charge_class, metal_binding = _base_atom_features_from_raw_mol(mol)

    # fragment_id [N]: which fragment each current atom belongs to (0-indexed)
    fragment_id = np.zeros(n_total, dtype=np.int64)
    try:
        from rdkit.Chem import rdmolops
        frags = rdmolops.GetMolFrags(mol)
        atom_to_frag = {idx: fi for fi, frag in enumerate(frags) for idx in frag}
        for atom_idx in range(n_total):
            fragment_id[atom_idx] = atom_to_frag.get(atom_idx, 0)
    except Exception:
        pass

    zeros_n  = np.zeros(n_total, dtype=np.float32)
    zeros_n3 = np.zeros((n_total, 3), dtype=np.float32)
    zeros_n4 = np.zeros((n_total, 4), dtype=np.float32)

    return {
        # ── V5-compatible keys ───────────────────────────────────────────────
        'atom_features':          torch.from_numpy(atom_features).float(),
        'edge_index':             torch.from_numpy(edge_index),
        'edge_attr':              torch.from_numpy(edge_attr),
        'coords':                 torch.from_numpy(coords_all),
        'coords_heavy_current':   torch.from_numpy(coords_heavy),
        'coords_original':        torch.from_numpy(coords_heavy),
        'gasteiger_charges':      torch.from_numpy(zeros_n),
        'logp_contribs':          torch.from_numpy(zeros_n),
        'tpsa_contribs':          torch.from_numpy(zeros_n),
        'rmsf':                   torch.from_numpy(zeros_n),
        'flexibility_flag':       torch.from_numpy(zeros_n),
        'atom_sasa':              torch.from_numpy(zeros_n),
        'lipophilic_sasa':        torch.from_numpy(zeros_n),
        'hbond_donor':            torch.from_numpy(zeros_n),
        'hbond_acceptor':         torch.from_numpy(zeros_n),
        'chirality_cw':           torch.from_numpy(zeros_n),
        'chirality_ccw':          torch.from_numpy(zeros_n),
        'rotatable_bond_context': torch.from_numpy(zeros_n3),
        'formal_charge_class':    torch.from_numpy(formal_charge_class),
        'metal_binding_flag':     torch.from_numpy(metal_binding),
        'stereochemistry_flag':   torch.from_numpy(zeros_n3),
        'shape_features':         torch.zeros(10),
        'global_features':        torch.zeros(45),
        'num_atoms':              n_total,
        'num_heavy_atoms':        n_heavy,
        'has_original_coords':    has_coords,
        'feature_dim_total':      109,
        'feature_dim_local':      54,
        'ligand_path':            str(ori_sdf),
        'pdb_id':                 pair['pdb_id'],
        'protein_chain':          pair['chain'],
        'binding_site_id':        pair['bs_id'],
        'ligand_id':              pair['lig_id'],
        'ligand_resseq':          pair['resseq'],
        'molecular_descriptors':  {},
        # ── V6 keys ──────────────────────────────────────────────────────────
        'feature_schema_version':         FEATURE_SCHEMA_VERSION,
        'ligand_type':                    torch.from_numpy(ligand_type_vec),
        'ligand_route':                   int(route),
        'ligand_type_from_id_table':      bool(type_from_id_table),
        'ccd_lookup_used':                False,
        'conformer_policy':               int(conf_policy),
        'hydrogen_policy':                int(h_policy),
        'operation_status':               torch.from_numpy(op_status),
        'x_group_mask':                   torch.from_numpy(group_mask),
        'x_group_source':                 torch.from_numpy(group_src),
        'x_atom_extra':                   torch.from_numpy(x_atom_extra),
        'x_atom_extra_mask':              torch.from_numpy(x_atom_extra_mask),
        'x_atom_extra_source':            torch.from_numpy(x_atom_extra_source),
        'fragment_count':                 n_frags,
        'fragment_id':                    torch.from_numpy(fragment_id),
        'fragment_pruned':                False,
        'largest_fragment_atom_fraction': largest_frac,
        'feature_completeness_ratio':     float(feature_completeness_ratio(group_mask)),
    }


def _atomic_save_and_meta(
    data: dict,
    out_path: Path,
    stem: str,
    pair: dict,
    overwrite: bool,
) -> Tuple[bool, str]:
    """Atomic write to tmp then rename; copy pair_meta.json alongside."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f'{stem}: write error: {e}'

    meta_src = pair['pair_dir'] / 'pair_meta.json'
    meta_dst = out_path.with_name(f'{stem}_meta.json')
    if meta_src.exists() and (not meta_dst.exists() or overwrite):
        shutil.copy2(meta_src, meta_dst)

    n_heavy   = data.get('num_heavy_atoms', 0)
    route_val = data.get('ligand_route', -1)
    route_str = ''
    if V6_TYPES_AVAILABLE and isinstance(route_val, int) and route_val >= 0:
        try:
            route_str = f' [{LigandRoute(route_val).name}]'
        except Exception:
            pass
    return True, f'{stem}: OK  ({n_heavy} heavy){route_str}'


# ── single-pair extraction ────────────────────────────────────────────────────

def extract_pair(
    pair             : dict,
    output_root      : Path,
    overwrite        : bool = False,
    num_conformers   : int  = 50,
    rdkit_threads    : int  = 1,
    per_pair_timeout : int  = 0,
    fallback_no_conformers: bool = True,
) -> Tuple[bool, str]:
    stem     = _output_stem(pair)
    out_path = _output_path(pair, output_root)

    if out_path.exists() and not overwrite:
        return True, f'{stem}: SKIP'

    # ── Locate _ligand_ori.sdf ────────────────────────────────────────────────
    meta     = pair['meta']
    pair_dir = pair['pair_dir']
    ori_sdf_name = meta.get('ligand_reference_sdf_file') or meta.get('ligand_reference_mol2_file')
    if ori_sdf_name is None:
        return False, f'{stem}: pair_meta.json missing ligand_reference_sdf_file'

    ori_sdf = pair_dir / ori_sdf_name
    if not ori_sdf.exists():
        candidates = list(pair_dir.glob('*_ligand_ori.sdf'))
        if candidates:
            ori_sdf = candidates[0]
        else:
            return False, f'{stem}: {ori_sdf_name} not found'

    # ── V6: classify BEFORE heavy chemistry ──────────────────────────────────
    use_v6 = V6_TYPES_AVAILABLE
    if use_v6:
        mol_summary = raw_mol_summary_from_file(ori_sdf)
        ligand_type_vec, route, type_from_id_table = classify_ligand_type(meta, mol_summary)
        conf_policy = get_conformer_policy(route, mol_summary)
        h_policy    = get_hydrogen_policy(route)
        op_status   = empty_operation_status()
        op_status[OP_IDX['rdkit_parse']] = int(
            OperationStatus.SUCCESS if mol_summary.load_ok else OperationStatus.FAILED
        )
    else:
        route = None
        mol_summary = None
        ligand_type_vec = None
        type_from_id_table = False
        conf_policy = None
        h_policy    = None
        op_status   = None

    # ── Pure metal ion: bypass V5 entirely ───────────────────────────────────
    if use_v6 and route == LigandRoute.PURE_METAL_ION:
        mol = _load_mol_raw(ori_sdf)
        if mol is None:
            op_status[OP_IDX['rdkit_parse']] = int(OperationStatus.FAILED)
            return False, f'{stem}: pure metal mol load failed'
        for op in ['rdkit_full_sanitize', 'rdkit_partial_sanitize', 'rdkit_add_h',
                   'gasteiger', 'crippen_logp', 'tpsa', 'morgan',
                   'etkdg', 'mmff', 'sasa', 'shape']:
            op_status[OP_IDX[op]] = int(OperationStatus.NOT_APPLICABLE)
        data = _build_minimal_data_from_mol(
            mol, pair, stem, ori_sdf, ligand_type_vec, route, type_from_id_table,
            conf_policy, h_policy, op_status, mol_summary,
        )
        return _atomic_save_and_meta(data, out_path, stem, pair, overwrite)

    # ── Inorganic polyatomic: bypass V5 ──────────────────────────────────────
    if use_v6 and route == LigandRoute.INORGANIC_POLYATOMIC:
        mol = _load_mol_raw(ori_sdf)
        if mol is None:
            op_status[OP_IDX['rdkit_parse']] = int(OperationStatus.FAILED)
            return False, f'{stem}: inorganic mol load failed'
        for op in ['rdkit_full_sanitize', 'rdkit_partial_sanitize', 'rdkit_add_h',
                   'gasteiger', 'crippen_logp', 'tpsa', 'morgan',
                   'etkdg', 'mmff', 'sasa', 'shape']:
            op_status[OP_IDX[op]] = int(OperationStatus.NOT_APPLICABLE)
        data = _build_minimal_data_from_mol(
            mol, pair, stem, ori_sdf, ligand_type_vec, route, type_from_id_table,
            conf_policy, h_policy, op_status, mol_summary,
        )
        return _atomic_save_and_meta(data, out_path, stem, pair, overwrite)

    # ── Multifragment: legal disconnected graph, do not enter V5 ─────────────
    if use_v6 and route == LigandRoute.MULTIFRAGMENT_NO_CONFORMER:
        mol = _load_mol_raw(ori_sdf)
        if mol is None:
            op_status[OP_IDX['rdkit_parse']] = int(OperationStatus.FAILED)
            return False, f'{stem}: multifragment mol load failed'
        for op in ['rdkit_full_sanitize', 'rdkit_partial_sanitize', 'rdkit_add_h',
                   'gasteiger', 'crippen_logp', 'tpsa', 'morgan',
                   'etkdg', 'mmff', 'sasa', 'shape']:
            op_status[OP_IDX[op]] = int(OperationStatus.SKIPPED_BY_POLICY)
        data = _build_minimal_data_from_mol(
            mol, pair, stem, ori_sdf, ligand_type_vec, route, type_from_id_table,
            conf_policy, h_policy, op_status, mol_summary,
        )
        return _atomic_save_and_meta(data, out_path, stem, pair, overwrite)

    # ── V5 pipeline for METAL_COMPLEX / NO_CONFORMER_BIOL / REGULAR_ORGANIC ──
    if use_v6:
        # Conformers only for REGULAR_ORGANIC; other routes use experimental coords
        use_conformers = (conf_policy == ConformerPolicy.FULL_ETKDG_MMFF and num_conformers > 0)
        # Only REGULAR_ORGANIC uses strict V5 parsing
        v5_strict = (route == LigandRoute.REGULAR_ORGANIC)
    else:
        use_conformers = (num_conformers > 0)
        v5_strict = False

    v5_ok = False
    v5_conf_ok = False
    v5_shape_ok = False
    features = None
    fallback_tag = ''

    try:
        with _Timeout(per_pair_timeout):
            features = extract_ligand_features_v5(
                str(ori_sdf),
                include_conformers=use_conformers,
                num_conformers=num_conformers if use_conformers else 0,
                strict=v5_strict,
                verbose=False,
                num_threads=rdkit_threads,
        )
        v5_ok = True
        v5_conf_ok = _valid_rmsf(features, use_conformers)
        v5_shape_ok = _valid_shape(features)
    except Exception as exc_main:
        if fallback_no_conformers and use_conformers:
            try:
                features = extract_ligand_features_v5(
                    str(ori_sdf),
                    include_conformers=False,
                    num_conformers=0,
                    strict=False,
                    verbose=False,
                    num_threads=rdkit_threads,
                )
                v5_ok = True
                v5_conf_ok = False
                v5_shape_ok = _valid_shape(features)
                fallback_tag = ' (fallback:no_conformers)'
                if use_v6:
                    op_status[OP_IDX['etkdg']] = int(OperationStatus.FAILED)
                    op_status[OP_IDX['mmff']]  = int(OperationStatus.FAILED)
            except Exception as exc_fb:
                if use_v6 and mol_summary is not None and mol_summary.is_disconnected:
                    mol = _load_mol_raw(ori_sdf)
                    if mol is not None:
                        for op in ['rdkit_full_sanitize', 'rdkit_partial_sanitize',
                                   'gasteiger', 'crippen_logp', 'tpsa', 'morgan',
                                   'sasa', 'shape']:
                            op_status[OP_IDX[op]] = int(OperationStatus.FAILED)
                        op_status[OP_IDX['rdkit_add_h']] = int(OperationStatus.FAILED)
                        op_status[OP_IDX['etkdg']] = int(OperationStatus.SKIPPED_BY_POLICY)
                        op_status[OP_IDX['mmff']] = int(OperationStatus.SKIPPED_BY_POLICY)
                        data = _build_minimal_data_from_mol(
                            mol, pair, stem, ori_sdf, ligand_type_vec, route, type_from_id_table,
                            conf_policy, HydrogenPolicy.ADD_H_FAILED, op_status, mol_summary,
                        )
                        return _atomic_save_and_meta(data, out_path, stem, pair, overwrite)
                return False, (
                    f'{stem}: V5 failed: {exc_fb}\n'
                    f'{traceback.format_exc()[:400]}'
                )
        else:
            if use_v6 and mol_summary is not None and mol_summary.is_disconnected:
                mol = _load_mol_raw(ori_sdf)
                if mol is not None:
                    for op in ['rdkit_full_sanitize', 'rdkit_partial_sanitize',
                               'gasteiger', 'crippen_logp', 'tpsa', 'morgan',
                               'sasa', 'shape']:
                        op_status[OP_IDX[op]] = int(OperationStatus.FAILED)
                    op_status[OP_IDX['rdkit_add_h']] = int(OperationStatus.FAILED)
                    op_status[OP_IDX['etkdg']] = int(OperationStatus.SKIPPED_BY_POLICY)
                    op_status[OP_IDX['mmff']] = int(OperationStatus.SKIPPED_BY_POLICY)
                    data = _build_minimal_data_from_mol(
                        mol, pair, stem, ori_sdf, ligand_type_vec, route, type_from_id_table,
                        conf_policy, HydrogenPolicy.ADD_H_FAILED, op_status, mol_summary,
                    )
                    return _atomic_save_and_meta(data, out_path, stem, pair, overwrite)
            return False, (
                f'{stem}: V5 failed: {exc_main}\n'
                f'{traceback.format_exc()[:400]}'
            )

    # ── Annotate operation_status from V5 outcome ─────────────────────────────
    if use_v6 and v5_ok:
        for op in ['rdkit_full_sanitize', 'rdkit_add_h', 'gasteiger',
                   'crippen_logp', 'tpsa', 'morgan', 'sasa']:
            op_status[OP_IDX[op]] = int(OperationStatus.UNVERIFIED)
        op_status[OP_IDX['rdkit_partial_sanitize']] = int(OperationStatus.SKIPPED_BY_POLICY)

        if v5_conf_ok:
            op_status[OP_IDX['etkdg']]  = int(OperationStatus.SUCCESS)
            op_status[OP_IDX['mmff']]   = int(OperationStatus.SUCCESS)
            op_status[OP_IDX['shape']]  = int(
                OperationStatus.SUCCESS if v5_shape_ok else OperationStatus.FAILED
            )
        else:
            for op in ['etkdg', 'mmff']:
                if use_conformers:
                    op_status[OP_IDX[op]] = int(OperationStatus.FAILED)
                elif op_status[OP_IDX[op]] == int(OperationStatus.NOT_ATTEMPTED):
                    op_status[OP_IDX[op]] = int(OperationStatus.SKIPPED_BY_POLICY)
            op_status[OP_IDX['shape']] = int(
                OperationStatus.SUCCESS if v5_shape_ok else OperationStatus.FAILED
            )

    # ── Build V6 arrays ───────────────────────────────────────────────────────
    n_atoms = features['num_atoms']   # total incl. H (after AddHs in V5)

    if use_v6:
        has_exp = bool(features.get('has_original_coords', False))
        group_mask, group_src = _build_group_mask_source(
            n_atoms, route, v5_ok, v5_conf_ok, v5_shape_ok, has_exp_coords=has_exp
        )
        _apply_v5_group_quality(group_mask, group_src, features)
        x_atom_extra, x_atom_extra_mask = _atom_extra_with_addh(ori_sdf, n_atoms)
        x_atom_extra_source = np.where(
            x_atom_extra_mask > 0,
            int(FeatureSource.PERIODIC_TABLE_LOOKUP),
            int(FeatureSource.UNAVAILABLE),
        ).astype(np.int8)
        if group_mask.shape[1] > 13 and group_mask[0, 13] > 0:
            op_status[OP_IDX['shape']] = int(OperationStatus.SUCCESS)
        if (not use_conformers) and (not has_exp) and v5_shape_ok:
            op_status[OP_IDX['etkdg']] = int(OperationStatus.UNVERIFIED)
            op_status[OP_IDX['mmff']] = int(OperationStatus.UNVERIFIED)
        for op, group_idx in [
            ('gasteiger', 0),
            ('crippen_logp', 1),
            ('tpsa', 2),
            ('morgan', 10),
            ('sasa', 4),
        ]:
            op_status[OP_IDX[op]] = int(
                OperationStatus.SUCCESS
                if group_mask[:, group_idx].sum() > 0
                else OperationStatus.FAILED
            )
        comp_ratio  = float(feature_completeness_ratio(group_mask))
        n_frags     = mol_summary.fragment_count
        fragment_id, largest_frac = _atom_fragment_ids_and_fraction(
            ori_sdf, n_atoms
        )
    else:
        group_mask = np.zeros((n_atoms, 14), dtype=np.float32)
        group_src  = np.zeros((n_atoms, 14), dtype=np.int8)
        x_atom_extra = np.zeros((n_atoms, 14), dtype=np.float32)
        x_atom_extra_mask = np.zeros((n_atoms, 14), dtype=np.float32)
        x_atom_extra_source = np.zeros((n_atoms, 14), dtype=np.int8)
        comp_ratio  = 0.0
        n_frags     = 1
        largest_frac = 1.0
        fragment_id = np.zeros(n_atoms, dtype=np.int64)

    # ── Assemble output dict ──────────────────────────────────────────────────
    data: dict = {
        # ── V5-compatible keys ───────────────────────────────────────────────
        'atom_features':          torch.from_numpy(features['atom_features']).float(),
        'edge_index':             torch.from_numpy(features['edge_index']).long(),
        'edge_attr':              torch.from_numpy(features['edge_attr']).float(),
        'coords':                 torch.from_numpy(features['coords']).float(),
        'coords_heavy_current':   torch.from_numpy(features['coords_heavy_current']).float(),
        'gasteiger_charges':      torch.from_numpy(features['gasteiger_charges']).float(),
        'logp_contribs':          torch.from_numpy(features['logp_contribs']).float(),
        'tpsa_contribs':          torch.from_numpy(features['tpsa_contribs']).float(),
        'rmsf':                   torch.from_numpy(features['rmsf']).float(),
        'flexibility_flag':       torch.from_numpy(features['flexibility_flag']).float(),
        'atom_sasa':              torch.from_numpy(features['atom_sasa']).float(),
        'lipophilic_sasa':        torch.from_numpy(features['lipophilic_sasa']).float(),
        'hbond_donor':            torch.from_numpy(features['hbond_donor']).float(),
        'hbond_acceptor':         torch.from_numpy(features['hbond_acceptor']).float(),
        'chirality_cw':           torch.from_numpy(features['chirality_cw']).float(),
        'chirality_ccw':          torch.from_numpy(features['chirality_ccw']).float(),
        'rotatable_bond_context': torch.from_numpy(features['rotatable_bond_context']).float(),
        'formal_charge_class':    torch.from_numpy(features['formal_charge_class']).float(),
        'metal_binding_flag':     torch.from_numpy(features['metal_binding_flag']).float(),
        'stereochemistry_flag':   torch.from_numpy(features['stereochemistry_flag']).float(),
        'shape_features':         torch.from_numpy(features['shape_features']).float(),
        'global_features':        torch.from_numpy(features['global_features']).float(),
        'num_atoms':              features['num_atoms'],
        'num_heavy_atoms':        features['num_heavy_atoms'],
        'has_original_coords':    features['has_original_coords'],
        'feature_dim_total':      features['feature_dim_total'],
        'feature_dim_local':      features.get('feature_dim_local', 54),
        'ligand_path':            str(ori_sdf),
        'pdb_id':                 pair['pdb_id'],
        'protein_chain':          pair['chain'],
        'binding_site_id':        pair['bs_id'],
        'ligand_id':              pair['lig_id'],
        'ligand_resseq':          pair['resseq'],
        'molecular_descriptors':  features.get('molecular_descriptors', {}),
        # ── V6 keys ──────────────────────────────────────────────────────────
        'feature_schema_version':         FEATURE_SCHEMA_VERSION if use_v6 else 'biolip_ligand_v5',
        'ligand_type':                    torch.from_numpy(ligand_type_vec) if use_v6 else torch.zeros(14),
        'ligand_route':                   int(route) if use_v6 else -1,
        'ligand_type_from_id_table':      bool(type_from_id_table) if use_v6 else False,
        'ccd_lookup_used':                False,
        'conformer_policy':               int(conf_policy) if use_v6 else -1,
        'hydrogen_policy':                int(h_policy) if use_v6 else -1,
        'operation_status':               torch.from_numpy(op_status) if use_v6 else torch.zeros(12, dtype=torch.int8),
        'x_group_mask':                   torch.from_numpy(group_mask),
        'x_group_source':                 torch.from_numpy(group_src),
        'x_atom_extra':                   torch.from_numpy(x_atom_extra),
        'x_atom_extra_mask':              torch.from_numpy(x_atom_extra_mask),
        'x_atom_extra_source':            torch.from_numpy(x_atom_extra_source),
        'fragment_count':                 n_frags,
        'fragment_id':                    torch.from_numpy(fragment_id).long(),
        'fragment_pruned':                False,
        'largest_fragment_atom_fraction': largest_frac,
        'feature_completeness_ratio':     comp_ratio,
    }

    if features.get('coords_original') is not None:
        data['coords_original'] = torch.from_numpy(features['coords_original']).float()
    if features.get('coords_original_with_h') is not None:
        data['coords_original_with_h'] = torch.from_numpy(features['coords_original_with_h']).float()

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f'{stem}: write error: {e}'

    # Copy pair_meta.json
    meta_src = pair['pair_dir'] / 'pair_meta.json'
    meta_dst = out_path.with_name(f'{stem}_meta.json')
    if meta_src.exists() and (not meta_dst.exists() or overwrite):
        shutil.copy2(meta_src, meta_dst)

    n_heavy_out = features['num_heavy_atoms']
    dim_out     = features['feature_dim_total']
    route_str   = f' [{route.name}]' if (use_v6 and route is not None) else ''
    return True, f'{stem}: OK  ({n_heavy_out} heavy, {dim_out}D){route_str}{fallback_tag}'


# ── pool worker (top-level for pickle) ───────────────────────────────────────

def _pool_task(args: tuple) -> Tuple[bool, str]:
    pair, output_root, overwrite, num_conformers, rdkit_threads, per_pair_timeout, fallback_no_conformers = args
    return extract_pair(
        pair,
        output_root,
        overwrite,
        num_conformers,
        rdkit_threads,
        per_pair_timeout,
        fallback_no_conformers,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='BioLip ligand feature extraction (V6-Min pipeline).'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip dataset root (contains shared_receptors/)')
    parser.add_argument('--output', required=True,
                        help='Output root for .pt files')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num_conformers', type=int, default=50,
                        help='Conformers for RMSF/flexibility (0 = skip, default 50)')
    parser.add_argument('--rdkit-threads', type=int, default=1,
                        help='RDKit threads per task (default 1)')
    parser.add_argument('--per-pair-timeout', type=int, default=0,
                        help='Per-pair timeout in seconds (0 = disable)')
    parser.add_argument('--no-fallback-no-conformers', action='store_true',
                        help='Disable fallback to num_conformers=0 on failure')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pairs (for testing)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pairs-cache', default=None,
                        help='Path to cache file for discovered pairs (pickle).')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Force re-discovery even if cache exists.')
    args = parser.parse_args()

    if not V5_AVAILABLE:
        print(f'ERROR: V5 ligand pipeline not available: {_V5_IMPORT_ERR}')
        return 1
    if not V6_TYPES_AVAILABLE:
        print(f'WARNING: V6 types not available ({_V6_TYPES_IMPORT_ERR}); '
              f'falling back to V5 output (no routing, no masks)')

    biolip_root = Path(args.biolip_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    import time as _time
    cache_path = Path(args.pairs_cache) if args.pairs_cache else biolip_root / 'pairs_cache_v6.pkl'

    if cache_path and cache_path.exists() and not args.refresh_cache:
        print(f'Loading pairs from cache: {cache_path}', flush=True)
        _t0 = _time.time()
        with open(cache_path, 'rb') as f:
            pairs = pickle.load(f)
        print(f'Cache loaded in {_time.time()-_t0:.1f}s  ({len(pairs):,} pairs)', flush=True)
    else:
        print('Discovering binding pairs…', flush=True)
        _t0 = _time.time()
        pairs = _discover_pairs(biolip_root, n_threads=64)
        print(f'Discovery done in {_time.time()-_t0:.1f}s  ({len(pairs):,} pairs)', flush=True)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Pairs cached to: {cache_path}', flush=True)

    if args.max:
        pairs = pairs[:args.max]

    import os as _os
    n_workers_eff = max(1, args.workers)
    if args.rdkit_threads == 0:
        rdkit_threads = max(1, (_os.cpu_count() or 1) // n_workers_eff)
    else:
        rdkit_threads = max(1, args.rdkit_threads)

    total = len(pairs)
    print(f'Total pairs   : {total:,}')
    print(f'Output root   : {output_root}')
    print(f'Workers       : {args.workers}')
    print(f'Conformers    : {args.num_conformers}')
    print(f'RDKit thr/task: {rdkit_threads}')
    print(f'Pair timeout  : {args.per_pair_timeout}s (0=off)')
    print(f'Fallback      : {not args.no_fallback_no_conformers}')
    print(f'V6 routing    : {V6_TYPES_AVAILABLE}')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(
        output_root=output_root,
        overwrite=args.overwrite,
        num_conformers=args.num_conformers,
        rdkit_threads=rdkit_threads,
        per_pair_timeout=args.per_pair_timeout,
        fallback_no_conformers=(not args.no_fallback_no_conformers),
    )

    # Single-worker shortcut
    if args.workers <= 1:
        failures: List[Tuple[str, str]] = []
        bar = _tqdm(pairs, total=total, desc='extract', dynamic_ncols=True) if TQDM_OK else pairs
        for idx, pair in enumerate(bar, 1):
            ok, msg = extract_pair(pair, **kw)
            if TQDM_OK:
                bar.set_postfix_str(msg[-60:])
            else:
                print(f'[{idx}/{total}] {msg}', flush=True)
            if not ok:
                failures.append((_output_stem(pair), msg))

        if failures:
            fail_path = output_root / 'failures.tsv'
            with open(fail_path, 'w') as f:
                f.write('pair_id\terror\n')
                for pid, err in failures:
                    f.write(f'{pid}\t{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
            print(f'{len(failures)} failures → {fail_path}')
            return 1
        print(f'\nDone. {total}/{total} pairs processed.')
        return 0

    # Multi-worker pool
    n_workers = min(args.workers, total)
    tasks = [
        (
            p,
            output_root,
            args.overwrite,
            args.num_conformers,
            rdkit_threads,
            args.per_pair_timeout,
            (not args.no_fallback_no_conformers),
        )
        for p in pairs
    ]

    ctx = mp.get_context('fork')
    failures: List[Tuple[str, str]] = []
    ok_count = 0

    with ctx.Pool(processes=n_workers) as pool:
        itr = pool.imap_unordered(_pool_task, tasks, chunksize=4)
        bar = _tqdm(itr, total=total, desc='extract', dynamic_ncols=True) if TQDM_OK else itr
        for ok, msg in bar:
            if ok:
                ok_count += 1
            else:
                failures.append(msg)
            if TQDM_OK:
                bar.set_postfix(ok=ok_count, fail=len(failures))
            else:
                print(msg, flush=True)

    if failures:
        fail_path = output_root / 'failures.tsv'
        with open(fail_path, 'w') as f:
            f.write('pair_id\terror\n')
            for err in failures:
                f.write(f'{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
        print(f'{len(failures)} failures → {fail_path}')
        return 1
    print(f'\nDone. {ok_count}/{total} pairs processed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
