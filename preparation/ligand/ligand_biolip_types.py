#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ligand_biolip_types.py — BioLiP V6-Min ligand classification contract
======================================================================
Enums, element tables, CCD code sets, feature schema, and the
classifier that must run BEFORE any RDKit chemistry pipeline.

Design red lines
  1. V5 x[109] order / semantics never change.
  2. No silent zero: every fallback needs mask + source.
  3. Classifier (classify_ligand_type) runs before RDKit.
  4. ligand_route (primary path) is separate from ligand_type_multihot (attributes).
  5. operation_status uses an enum, not bool (distinguishes failed vs skipped).
  6. hydrogen_policy is explicit; edge_index_coord is ligand-internal only.

Used by: biolip_ligand_extract.py (V6 pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Feature schema version  (bump when any dim / semantic changes)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA_VERSION = "biolip_ligand_v6_min_multifragment_route_2026_05"

FEATURE_SCHEMA: Dict[str, object] = {
    "schema_version":        "2026_05_multifragment_route",
    "name":                  "biolip_ligand_v6_min",
    "x_dim":                 109,      # V5 backbone, never changes
    "edge_attr_dim":         6,        # V5 bond edge, never changes
    "x_atom_extra_dim":      14,
    "x_group_mask_dim":      14,       # G = 14 derived/high-risk V5 groups
    "ligand_type_dim":       14,       # T = 14 type bits (multi-hot)
    "n_operations":          12,
    "source_enum_version":   1,
    "operation_enum_version":1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core enums
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureSource(IntEnum):
    """Where a feature value came from."""
    UNAVAILABLE            = 0
    RDKIT_FULL_SANITIZE    = 1
    RDKIT_PARTIAL_SANITIZE = 2
    EXPERIMENTAL_COORDS    = 3
    PERIODIC_TABLE_LOOKUP  = 4
    CCD_LOOKUP             = 5   # Phase 2: real CCD dict fetch
    CODE_TABLE             = 6   # Phase 1: offline curated ligand-ID list
    FALLBACK_HEURISTIC     = 7
    NOT_APPLICABLE         = 8   # semantically undefined for this ligand type


class OperationStatus(IntEnum):
    """
    Result of one pipeline operation.
    Distinguishes 'failed' from 'skipped by policy' from 'not tried'.
    """
    NOT_ATTEMPTED      = 0   # code path never reached
    SKIPPED_BY_POLICY  = 1   # intentionally bypassed (e.g. metal → no ETKDG)
    FAILED             = 2   # attempted but raised exception / returned invalid
    SUCCESS            = 3
    NOT_APPLICABLE     = 4   # semantically meaningless here (e.g. RMSF on metal)
    UNVERIFIED         = 5   # V5 returned, but sub-operation has no explicit status


class LigandRoute(IntEnum):
    """
    Primary processing path.  Mutually exclusive; controls the feature
    extraction strategy in biolip_ligand_extract.py.
    """
    UNKNOWN_OR_FAILED        = 0
    PURE_METAL_ION           = 1   # single-atom metal; skip all RDKit chemistry
    INORGANIC_POLYATOMIC     = 2   # SO4 / PO4 / NO3 etc.; no-conformer, no AddHs
    METAL_COMPLEX            = 3   # HEM / CLA / Pt drugs; partial sanitize
    NO_CONFORMER_BIOLOGICAL  = 4   # nucleotide / peptide / glycan / lipid / large
    REGULAR_ORGANIC          = 5   # full V5 pipeline
    MULTIFRAGMENT_NO_CONFORMER = 6 # disconnected graph; raw-graph fallback


class HydrogenPolicy(IntEnum):
    """
    How hydrogens were / will be handled for this ligand.
    Saves as graph-level scalar.
    """
    EXPLICIT_ONLY       = 0   # keep only explicit H from SDF/mol2
    V5_ADD_H            = 1   # V5 backbone used; it preserves explicit H or calls AddHs
    ADD_H_FAILED        = 2   # tried AddHs but failed; fell back to explicit
    SKIPPED_BY_POLICY   = 3   # bypass routes: do not add H
    UNKNOWN             = 4


class ConformerPolicy(IntEnum):
    """Conformer generation strategy."""
    FULL_ETKDG_MMFF     = 0   # 50 conformers, ETKDGv3 + MMFF (V5 default)
    RAW_COORDS_ONLY     = 1   # use input coordinates; skip ETKDG/MMFF
    EXPERIMENTAL_ONLY   = 1   # backward-compatible alias
    FAILED_FALLBACK     = 2   # tried full, failed; used experimental or 2D
    SKIPPED_METAL       = 3   # not applicable for metal / inorganic


# ═══════════════════════════════════════════════════════════════════════════════
# Ligand type multi-hot (14 bits)
# ═══════════════════════════════════════════════════════════════════════════════

LIGAND_TYPES: List[str] = [
    'pure_metal_ion',           # 0   primary
    'inorganic_polyatomic',     # 1   primary
    'metal_complex',            # 2   primary  (any organic+metal)
    'metalloporphyrin',         # 3   additive  (subtype of metal_complex)
    'nucleotide_cofactor',      # 4   primary
    'peptide_ligand',           # 5   primary
    'nucleic_acid_fragment',    # 6   primary
    'carbohydrate_glycan',      # 7   primary
    'lipid_long_chain',         # 8   primary
    'unusual_element',          # 9   additive  (B/Se/As/Si/Te/Ge present)
    'multifragment',            # 10  additive  (disconnected mol graph)
    'large_flexible',           # 11  additive  (heavy>50 or rot_bonds>20)
    'regular_organic',          # 12  primary   (full V5 pipeline)
    'unknown_or_unclassified',  # 13  catch-all (parse failure / no match)
]
N_LIGAND_TYPES = len(LIGAND_TYPES)           # 14
LIGAND_TYPE_IDX: Dict[str, int] = {t: i for i, t in enumerate(LIGAND_TYPES)}


# ═══════════════════════════════════════════════════════════════════════════════
# V5 x[109] group slice map  (which dims each logical group occupies)
# ═══════════════════════════════════════════════════════════════════════════════
# Must stay in sync with ligand_features_v4.py concatenation order.

X109_GROUP_SLICES: Dict[str, List[int]] = {
    # ── Local atom features (54D, dims 0–53) ─────────────────────────────
    'local_atom_type':           list(range(0,  11)),   # 11D  one-hot ATOM_TYPES
    'local_hybridization':       list(range(11, 17)),   # 6D   one-hot HYBRIDIZATION
    'local_aromaticity':         [17],                  # 1D
    'local_ring':                list(range(18, 25)),   # 7D
    'local_degree':              list(range(25, 30)),   # 5D
    'local_formal_charge_class': list(range(30, 33)),   # 3D   neg/neutral/pos
    'local_gasteiger_charge':    [33],                  # 1D   ← RDKit Gasteiger
    'local_crippen_logp':        [34],                  # 1D   ← RDKit Crippen
    'local_tpsa_contrib':        [35],                  # 1D   ← RDKit TPSA
    'local_rmsf':                [36],                  # 1D   ← ETKDG/MMFF
    'local_flexibility_flag':    [37],                  # 1D   ← derived from RMSF
    'local_atom_sasa':           [38],                  # 1D   ← rdFreeSASA
    'local_lipophilic_sasa':     [39],                  # 1D   ← sasa × logp
    'local_hbond_donor':         [40],                  # 1D
    'local_hbond_acceptor':      [41],                  # 1D
    'local_chirality':           list(range(42, 44)),   # 2D   cw + ccw
    'local_rot_bond_context':    list(range(44, 47)),   # 3D
    'local_metal_binding':       list(range(47, 51)),   # 4D   N/O/S donor flags
    'local_stereochemistry':     list(range(51, 54)),   # 3D
    # ── Global molecular tag (45D, dims 54–98) ────────────────────────────
    # Source: pypropel.mol.get_global_tag(mol, output_dim=45)
    # Layout: [Morgan43D | MolLogP | TPSA/100]  (see V5 doc)
    'global_morgan_43':          list(range(54, 97)),   # 43D
    'global_mollogp':            [97],                  # 1D
    'global_tpsa':               [98],                  # 1D
    # ── Normalized shape features (10D, dims 99–108) ──────────────────────
    # PMI1/2/3, SpherocityIndex, RadiusOfGyration, NPR1, NPR2, spans L/W/H
    'shape_3d':                  list(range(99, 109)),  # 10D
}

# Verify total coverage = 109
_all_dims = sorted({d for dims in X109_GROUP_SLICES.values() for d in dims})
assert _all_dims == list(range(109)), "X109_GROUP_SLICES does not cover all 109 dims"


# ═══════════════════════════════════════════════════════════════════════════════
# x_group_mask / x_group_source  (14 derived/high-risk groups → G=14)
# ═══════════════════════════════════════════════════════════════════════════════
# These groups track fields that can silently degrade under RDKit/conformer
# fallback. Stable base atom identity/topology fields are not masked here.
# Saved as [N, G] arrays:  mask=float32 {0,1},  source=int8 FeatureSource.

X_GROUP_FIELDS: List[str] = [
    'local_gasteiger_charge',    # 0  → 'local_gasteiger_charge'
    'local_crippen_logp',        # 1  → 'local_crippen_logp'
    'local_tpsa_contrib',        # 2  → 'local_tpsa_contrib'
    'rmsf_and_flexibility',      # 3  → 'local_rmsf' + 'local_flexibility_flag'
    'sasa_and_lipophilic_sasa',  # 4  → 'local_atom_sasa' + 'local_lipophilic_sasa'
    'hbond_donor_acceptor',      # 5  → 'local_hbond_donor' + 'local_hbond_acceptor'
    'chirality',                 # 6  → 'local_chirality'
    'stereochemistry',           # 7  → 'local_stereochemistry'
    'rot_bond_context',          # 8  → 'local_rot_bond_context'
    'degree_and_formal_charge',  # 9  → 'local_degree' + 'local_formal_charge_class'
                                 #        NOTE: this group is per-atom (not broadcast)
    'global_morgan_43',          # 10 → 'global_morgan_43'  (molecule-broadcast)
    'global_mollogp',            # 11 → 'global_mollogp'    (molecule-broadcast)
    'global_tpsa',               # 12 → 'global_tpsa'       (molecule-broadcast)
    'shape_3d',                  # 13 → 'shape_3d'          (molecule-broadcast)
]
N_X_GROUPS = len(X_GROUP_FIELDS)    # 14


def empty_group_mask(n_atoms: int = 1) -> np.ndarray:
    """All groups default to invalid (mask=0)."""
    return np.zeros((n_atoms, N_X_GROUPS), dtype=np.float32)


def empty_group_source(n_atoms: int = 1) -> np.ndarray:
    """All groups default to UNAVAILABLE."""
    return np.full((n_atoms, N_X_GROUPS), FeatureSource.UNAVAILABLE, dtype=np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# operation_status  (K=12 operations × OperationStatus enum)
# ═══════════════════════════════════════════════════════════════════════════════

OPERATIONS: List[str] = [
    'rdkit_parse',            # 0  mol loaded from SDF / mol2
    'rdkit_full_sanitize',    # 1  Chem.SanitizeMol full
    'rdkit_partial_sanitize', # 2  SANITIZE_ALL ^ SANITIZE_PROPERTIES
    'rdkit_add_h',            # 3  Chem.AddHs
    'gasteiger',              # 4  ComputeGasteigerCharges
    'crippen_logp',           # 5  _CalcCrippenContribs
    'tpsa',                   # 6  _CalcTPSAContribs
    'morgan',                 # 7  global tag (Morgan FP + LogP + TPSA)
    'etkdg',                  # 8  ETKDGv3 embed
    'mmff',                   # 9  MMFF optimization
    'sasa',                   # 10 rdFreeSASA
    'shape',                  # 11 3D shape descriptors
]
N_OPERATIONS = len(OPERATIONS)           # 12
OP_IDX: Dict[str, int] = {op: i for i, op in enumerate(OPERATIONS)}


def empty_operation_status() -> np.ndarray:
    """All operations start as NOT_ATTEMPTED."""
    return np.full(N_OPERATIONS, OperationStatus.NOT_ATTEMPTED, dtype=np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# Element sets
# ═══════════════════════════════════════════════════════════════════════════════

METAL_ELEMENTS: Set[str] = {
    'LI', 'NA', 'K',  'RB', 'CS', 'FR',
    'BE', 'MG', 'CA', 'SR', 'BA', 'RA',
    'SC', 'TI', 'V',  'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'Y',  'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
    'HF', 'TA', 'W',  'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
    'AL', 'GA', 'IN', 'SN', 'TL', 'PB', 'BI',
    'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU',
    'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU',
    'TH', 'U',
}

TRANSITION_METALS: Set[str] = {
    'SC', 'TI', 'V',  'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'Y',  'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
    'HF', 'TA', 'W',  'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
}

ALKALI_OR_ALKALINE: Set[str] = {
    'LI', 'NA', 'K', 'RB', 'CS', 'FR',
    'BE', 'MG', 'CA', 'SR', 'BA', 'RA',
}

HALOGENS: Set[str] = {'F', 'CL', 'BR', 'I', 'AT'}

METALLOIDS: Set[str] = {'B', 'SI', 'GE', 'AS', 'SB', 'TE'}

# RDKit valence / descriptor handling is unreliable for these
UNUSUAL_ELEMENTS: Set[str] = METALLOIDS | {'SE', 'PT', 'RU', 'OS', 'IR', 'AU', 'AG', 'HG'}


# ═══════════════════════════════════════════════════════════════════════════════
# CCD code lookup tables  (Phase-1 offline lists, not full CCD dict)
# ═══════════════════════════════════════════════════════════════════════════════

METALLOPORPHYRIN_CODES: Set[str] = {
    'HEM', 'HEC', 'HEB', 'HEA', 'HEV', 'HME', 'HDD', 'SRM', 'FEC', 'FDE', 'F43',
    'CLA', 'CLB', 'CHL', 'BCL', 'BCB', 'BPH', 'BPB', 'BCQ',
    'COB', 'B12', 'CNC', 'CO8', 'C2C',
    'MG1', 'MG2',
}

METAL_DRUG_CODES: Set[str] = {'CIS', 'CPT', 'OXP'}

METAL_COMPLEX_CODES: Set[str] = METALLOPORPHYRIN_CODES | METAL_DRUG_CODES

NUCLEOTIDE_COFACTOR_CODES: Set[str] = {
    'ATP', 'ADP', 'AMP', 'ADN', 'ANP', 'ACP', 'APC', 'APP', 'ATD',
    'A2P', 'A3P', 'PAD', 'ADX', '5AD',
    'GTP', 'GDP', 'GMP', 'GMX', 'GNP', 'GSP',
    'CTP', 'CDP', 'CMP',
    'UTP', 'UDP', 'UMP', 'URI',
    'TTP', 'TDP', 'TMP',
    'ITP', 'IDP', 'IMP',
    'NAD', 'NAP', 'NDP', 'NMN', 'NAI',
    'FAD', 'FMN', 'FDA', 'F42',
    'COA', 'CAA', 'ACO',
    'SAM', 'SAH', 'MTA',
    '5GP', '3GP', '2GP', 'CAP',
}

NUCLEIC_ACID_FRAGMENT_CODES: Set[str] = {
    'A', 'C', 'G', 'U', 'I',
    'DA', 'DC', 'DG', 'DT', 'DU', 'DI',
    'ADE', 'CYT', 'GUA', 'URA', 'THY',
    'PSU', '5MC', 'OMC', 'OMG', '1MA', '2MG', 'M2G', '7MG',
}

PEPTIDE_CODES: Set[str] = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL', 'HYP', 'SEP', 'TPO', 'PTR', 'CSO', 'CME', 'MLY',
}

INORGANIC_POLYATOMIC_CODES: Set[str] = {
    'SO4', 'SO3', 'SUL', 'MSU',
    'PO4', 'PO3', 'HPO', 'PHO', 'H2P',
    'NO3', 'NO2',
    'CO3', 'HCO',
    'VO4', 'VO3', 'WO4',
    'CLO', 'PER',
    'F', 'CL', 'BR', 'IOD',
    'CN', 'N3', 'NCO',
    'SUI',
}

CARBOHYDRATE_CODES: Set[str] = {
    'NAG', 'NDG', 'BGC', 'GLC', 'GAL', 'MAN', 'FUC', 'XYS', 'XYP',
    'RIB', 'AFL', 'SIA', 'NGA', 'FCA', 'GLA', 'GXL', 'BMA', 'AGL',
    'RAM', 'GCU', 'IDR', 'RIP', 'GLB', 'G6P', 'F6P', 'G1P',
    'FRU', 'TAG', 'SOR', 'MTT', 'FUL', 'INS', 'LFR', 'SG7',
}

LIPID_CODES: Set[str] = {
    'PLM', 'OLA', 'STE', 'ACD', 'LAC', 'MYR', 'PAM',
    'LNO', 'LNL', 'ARD', 'PLO', 'OLC', 'OLB',
    'LPL', 'OPC', 'LPS', 'LPA', 'PC1', 'PE1',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Periodic table element properties  (14D x_atom_extra)
# ═══════════════════════════════════════════════════════════════════════════════
# Sources:
#   Covalent radii: Alvarez (2008) Dalton Trans. 2832
#   vdW radii: Bondi (1964) + Alvarez (2013) Dalton Trans. 8617
#   Electronegativity: Pauling scale
# -1 = unknown / not established

@dataclass
class ElementProps:
    symbol: str
    atomic_number: int
    period: int
    group: int          # 0 for lanthanides / actinides
    block: int          # 0=s, 1=p, 2=d, 3=f
    pauling_en: float   # -1 = unknown
    cov_radius_pm: float
    vdw_radius_pm: float


_ELEMENT_TABLE_RAW = [
    # sym   Z   per grp blk   en    cov   vdw
    ('H',   1,  1,  1,  0,  2.20,   31,  120),
    ('HE',  2,  1, 18,  0,  0.00,   28,  140),
    ('LI',  3,  2,  1,  0,  0.98,  128,  182),
    ('BE',  4,  2,  2,  0,  1.57,   96,  153),
    ('B',   5,  2, 13,  1,  2.04,   84,  192),
    ('C',   6,  2, 14,  1,  2.55,   77,  170),
    ('N',   7,  2, 15,  1,  3.04,   75,  155),
    ('O',   8,  2, 16,  1,  3.44,   73,  152),
    ('F',   9,  2, 17,  1,  3.98,   71,  147),
    ('NE', 10,  2, 18,  1,  0.00,   69,  154),
    ('NA', 11,  3,  1,  0,  0.93,  166,  227),
    ('MG', 12,  3,  2,  0,  1.31,  141,  173),
    ('AL', 13,  3, 13,  1,  1.61,  121,  184),
    ('SI', 14,  3, 14,  1,  1.90,  111,  210),
    ('P',  15,  3, 15,  1,  2.19,  107,  180),
    ('S',  16,  3, 16,  1,  2.58,  105,  180),
    ('CL', 17,  3, 17,  1,  3.16,  102,  175),
    ('AR', 18,  3, 18,  1,  0.00,  106,  188),
    ('K',  19,  4,  1,  0,  0.82,  203,  275),
    ('CA', 20,  4,  2,  0,  1.00,  176,  231),
    ('SC', 21,  4,  3,  2,  1.36,  170,  211),
    ('TI', 22,  4,  4,  2,  1.54,  160,  200),
    ('V',  23,  4,  5,  2,  1.63,  153,  200),
    ('CR', 24,  4,  6,  2,  1.66,  139,  200),
    ('MN', 25,  4,  7,  2,  1.55,  161,  200),
    ('FE', 26,  4,  8,  2,  1.83,  152,  200),
    ('CO', 27,  4,  9,  2,  1.88,  150,  200),
    ('NI', 28,  4, 10,  2,  1.91,  149,  163),
    ('CU', 29,  4, 11,  2,  1.90,  138,  140),
    ('ZN', 30,  4, 12,  2,  1.65,  131,  139),
    ('GA', 31,  4, 13,  1,  1.81,  126,  187),
    ('GE', 32,  4, 14,  1,  2.01,  122,  211),
    ('AS', 33,  4, 15,  1,  2.18,  119,  185),
    ('SE', 34,  4, 16,  1,  2.55,  116,  190),
    ('BR', 35,  4, 17,  1,  2.96,  114,  185),
    ('KR', 36,  4, 18,  1,  3.00,  110,  202),
    ('RB', 37,  5,  1,  0,  0.82,  220,  303),
    ('SR', 38,  5,  2,  0,  0.95,  195,  249),
    ('Y',  39,  5,  3,  2,  1.22,  190,  200),
    ('ZR', 40,  5,  4,  2,  1.33,  175,  200),
    ('NB', 41,  5,  5,  2,  1.60,  164,  200),
    ('MO', 42,  5,  6,  2,  2.16,  154,  200),
    ('TC', 43,  5,  7,  2,  1.90,  147,  200),
    ('RU', 44,  5,  8,  2,  2.20,  146,  200),
    ('RH', 45,  5,  9,  2,  2.28,  142,  200),
    ('PD', 46,  5, 10,  2,  2.20,  139,  163),
    ('AG', 47,  5, 11,  2,  1.93,  145,  172),
    ('CD', 48,  5, 12,  2,  1.69,  144,  158),
    ('IN', 49,  5, 13,  1,  1.78,  142,  193),
    ('SN', 50,  5, 14,  1,  1.96,  139,  217),
    ('SB', 51,  5, 15,  1,  2.05,  139,  206),
    ('TE', 52,  5, 16,  1,  2.10,  138,  206),
    ('I',  53,  5, 17,  1,  2.66,  133,  198),
    ('XE', 54,  5, 18,  1,  2.60,  131,  216),
    ('CS', 55,  6,  1,  0,  0.79,  244,  343),
    ('BA', 56,  6,  2,  0,  0.89,  215,  268),
    # Lanthanides — group=0, block=f
    ('LA', 57,  6,  0,  3,  1.10,  207,  240),
    ('CE', 58,  6,  0,  3,  1.12,  204,  240),
    ('PR', 59,  6,  0,  3,  1.13,  203,  240),
    ('ND', 60,  6,  0,  3,  1.14,  201,  240),
    ('PM', 61,  6,  0,  3,  1.13,  199,  240),
    ('SM', 62,  6,  0,  3,  1.17,  198,  240),
    ('EU', 63,  6,  0,  3,  1.20,  198,  240),
    ('GD', 64,  6,  0,  3,  1.20,  196,  240),
    ('TB', 65,  6,  0,  3,  1.10,  194,  240),
    ('DY', 66,  6,  0,  3,  1.22,  192,  240),
    ('HO', 67,  6,  0,  3,  1.23,  192,  240),
    ('ER', 68,  6,  0,  3,  1.24,  189,  240),
    ('TM', 69,  6,  0,  3,  1.25,  190,  240),
    ('YB', 70,  6,  0,  3,  1.10,  187,  240),
    ('LU', 71,  6,  0,  3,  1.27,  187,  240),
    ('HF', 72,  6,  4,  2,  1.30,  175,  200),
    ('TA', 73,  6,  5,  2,  1.50,  170,  200),
    ('W',  74,  6,  6,  2,  2.36,  162,  200),
    ('RE', 75,  6,  7,  2,  1.90,  151,  200),
    ('OS', 76,  6,  8,  2,  2.20,  144,  200),
    ('IR', 77,  6,  9,  2,  2.20,  141,  200),
    ('PT', 78,  6, 10,  2,  2.28,  136,  175),
    ('AU', 79,  6, 11,  2,  2.54,  136,  166),
    ('HG', 80,  6, 12,  2,  2.00,  132,  155),
    ('TL', 81,  6, 13,  1,  1.62,  145,  196),
    ('PB', 82,  6, 14,  1,  2.33,  146,  202),
    ('BI', 83,  6, 15,  1,  2.02,  148,  207),
    # Actinides — group=0, block=f
    ('TH', 90,  7,  0,  3,  1.30,  206,  240),
    ('U',  92,  7,  0,  3,  1.38,  196,  186),
]

ELEMENT_PROPS: Dict[str, ElementProps] = {}
for _row in _ELEMENT_TABLE_RAW:
    _sym, _Z, _per, _grp, _blk, _en, _cov, _vdw = _row
    ELEMENT_PROPS[_sym] = ElementProps(
        symbol=_sym, atomic_number=_Z, period=_per, group=_grp, block=_blk,
        pauling_en=_en, cov_radius_pm=_cov, vdw_radius_pm=_vdw,
    )

_UNKNOWN_PROPS = ElementProps(
    symbol='?', atomic_number=0, period=0, group=0, block=1,
    pauling_en=-1.0, cov_radius_pm=-1.0, vdw_radius_pm=-1.0,
)


def get_element_props(symbol: str) -> ElementProps:
    return ELEMENT_PROPS.get(symbol.strip().upper(), _UNKNOWN_PROPS)


# x_atom_extra: 14D feature + 14D mask per atom
# Dims:
#   0  atomic_number / 100
#   1  period / 7
#   2  group / 18
#   3-6 block one-hot [s, p, d, f]
#   7  pauling_en / 4
#   8  covalent_radius / 300 pm
#   9  vdw_radius / 400 pm
#   10 is_metal
#   11 is_transition_metal
#   12 is_halogen
#   13 is_metalloid

X_ATOM_EXTRA_DIM = 14


def element_feature_vector(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (feat [14], mask [14]).
    All dims come from PERIODIC_TABLE_LOOKUP — no guessed oxidation state,
    no coordination number.  mask=1 iff value is known from the table.
    """
    p   = get_element_props(symbol)
    sym = symbol.strip().upper()
    feat = np.zeros(X_ATOM_EXTRA_DIM, dtype=np.float32)
    mask = np.zeros(X_ATOM_EXTRA_DIM, dtype=np.float32)

    if p.atomic_number > 0:
        feat[0] = p.atomic_number / 100.0
        feat[1] = p.period / 7.0
        feat[2] = p.group / 18.0
        feat[3 + p.block] = 1.0
        mask[0:7] = 1.0                              # Z, period, group, block: always reliable

        feat[10] = float(sym in METAL_ELEMENTS)
        feat[11] = float(sym in TRANSITION_METALS)
        feat[12] = float(sym in HALOGENS)
        feat[13] = float(sym in METALLOIDS)
        mask[10:14] = 1.0                            # classification flags: always reliable

        if p.pauling_en >= 0:
            feat[7] = p.pauling_en / 4.0
            mask[7] = 1.0
        if p.cov_radius_pm > 0:
            feat[8] = p.cov_radius_pm / 300.0
            mask[8] = 1.0
        if p.vdw_radius_pm > 0:
            feat[9] = p.vdw_radius_pm / 400.0
            mask[9] = 1.0

    return feat, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Raw mol summary  (lightweight, no heavy RDKit chemistry)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawMolSummary:
    heavy_count: int = 0
    fragment_count: int = 1
    element_symbols: Set[str] = field(default_factory=set)
    has_metal: bool = False
    has_unusual_element: bool = False
    is_disconnected: bool = False
    rotatable_bond_count: int = 0
    has_nonzero_formal_charge: bool = False    # current raw mol has any formal charge ≠ 0
    load_ok: bool = False


def raw_mol_summary_from_file(mol_file: Path) -> RawMolSummary:
    """
    Lightweight mol read (sanitize=False) for basic topology — no MMFF,
    no Gasteiger, no conformers.  Used by the classifier before RDKit pipeline.
    """
    s = RawMolSummary()
    if not Path(mol_file).exists():
        return s
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, rdmolops

        suffix = str(mol_file).rsplit('.', 1)[-1].lower()
        mol = None
        if suffix in ('sdf', 'mol'):
            mol = next(Chem.SDMolSupplier(str(mol_file), sanitize=False, removeHs=False), None)
        elif suffix == 'mol2':
            mol = Chem.MolFromMol2File(str(mol_file), sanitize=False, removeHs=False)
        elif suffix == 'pdb':
            mol = Chem.MolFromPDBFile(str(mol_file), sanitize=False, removeHs=False)
        if mol is None:
            return s

        s.load_ok = True
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        s.heavy_count = len(heavy)
        s.element_symbols = {a.GetSymbol().upper() for a in heavy}
        s.has_metal = bool(s.element_symbols & METAL_ELEMENTS)
        s.has_unusual_element = bool(s.element_symbols & UNUSUAL_ELEMENTS)
        s.has_nonzero_formal_charge = any(a.GetFormalCharge() != 0 for a in heavy)

        try:
            frags = rdmolops.GetMolFrags(mol)
            s.fragment_count = len(frags)
            s.is_disconnected = len(frags) > 1
        except Exception:
            pass
        try:
            s.rotatable_bond_count = rdMolDescriptors.CalcNumRotatableBonds(mol)
        except Exception:
            pass
    except Exception:
        pass
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Ligand classifier
# ═══════════════════════════════════════════════════════════════════════════════

def classify_ligand_type(
    pair_meta: dict,
    mol_summary: Optional[RawMolSummary] = None,
) -> tuple[np.ndarray, LigandRoute, bool]:
    """
    Classify a BioLiP ligand.

    Parameters
    ----------
    pair_meta   : dict from pair_meta.json
    mol_summary : RawMolSummary (if None, uses pair_meta fields only — less reliable)

    Returns
    -------
    ligand_type_multihot [N_LIGAND_TYPES]
    ligand_route         LigandRoute  (primary processing path)
    type_from_id_table   bool         (True = code table, False = heuristic)
    """
    if mol_summary is None:
        mol_summary = RawMolSummary()

    lig_id = (pair_meta.get('ligand_id') or '').strip().upper()
    heavy  = mol_summary.heavy_count or int(pair_meta.get('ligand_heavy_atom_count') or 0)

    vec = np.zeros(N_LIGAND_TYPES, dtype=np.float32)
    route = LigandRoute.UNKNOWN_OR_FAILED
    from_id_table = False
    primary_set = False

    def _set_primary(name: str, from_table: bool, r: LigandRoute):
        nonlocal primary_set, from_id_table, route
        if not primary_set:
            vec[LIGAND_TYPE_IDX[name]] = 1.0
            primary_set = True
            from_id_table = from_table
            route = r

    # ── Primary categories (priority order, mutually exclusive) ──────────

    # 1. Pure metal ion
    if heavy == 1 and mol_summary.element_symbols & METAL_ELEMENTS:
        _set_primary('pure_metal_ion', True, LigandRoute.PURE_METAL_ION)
    # Fallback: use pair_meta if mol couldn't be loaded (mol_summary.load_ok=False)
    elif heavy == 1 and lig_id in METAL_ELEMENTS:
        _set_primary('pure_metal_ion', False, LigandRoute.PURE_METAL_ION)

    # 2. Inorganic polyatomic  (code table first, then heuristic)
    # Note: allow metals in heuristic — metallooxyanions (VO4/WO4/MnO4) have no C
    # and are correctly classified as inorganic even if the metal triggers has_metal.
    elif lig_id in INORGANIC_POLYATOMIC_CODES:
        _set_primary('inorganic_polyatomic', True, LigandRoute.INORGANIC_POLYATOMIC)
    elif (heavy <= 8
          and 'C' not in mol_summary.element_symbols
          and bool(mol_summary.element_symbols)):
        _set_primary('inorganic_polyatomic', False, LigandRoute.INORGANIC_POLYATOMIC)

    # 3. Metal complex  (code table first, then "has metal + organic")
    elif lig_id in METAL_COMPLEX_CODES:
        _set_primary('metal_complex', True, LigandRoute.METAL_COMPLEX)
        if lig_id in METALLOPORPHYRIN_CODES:
            vec[LIGAND_TYPE_IDX['metalloporphyrin']] = 1.0
    elif mol_summary.has_metal and heavy > 1:
        _set_primary('metal_complex', False, LigandRoute.METAL_COMPLEX)

    # 4. Nucleotide / cofactor
    elif lig_id in NUCLEOTIDE_COFACTOR_CODES:
        _set_primary('nucleotide_cofactor', True, LigandRoute.NO_CONFORMER_BIOLOGICAL)

    # 5. Nucleic-acid fragment
    elif lig_id in NUCLEIC_ACID_FRAGMENT_CODES:
        _set_primary('nucleic_acid_fragment', True, LigandRoute.NO_CONFORMER_BIOLOGICAL)

    # 6. Peptide / amino-acid ligand
    elif lig_id in PEPTIDE_CODES:
        _set_primary('peptide_ligand', True, LigandRoute.NO_CONFORMER_BIOLOGICAL)

    # 7. Carbohydrate / glycan
    elif lig_id in CARBOHYDRATE_CODES:
        _set_primary('carbohydrate_glycan', True, LigandRoute.NO_CONFORMER_BIOLOGICAL)

    # 8. Lipid / long-chain
    elif lig_id in LIPID_CODES:
        _set_primary('lipid_long_chain', True, LigandRoute.NO_CONFORMER_BIOLOGICAL)

    # ── Additive flags (independent of primary) ───────────────────────────

    if mol_summary.has_unusual_element:
        vec[LIGAND_TYPE_IDX['unusual_element']] = 1.0

    if mol_summary.is_disconnected:
        vec[LIGAND_TYPE_IDX['multifragment']] = 1.0

    large = heavy > 50 or mol_summary.rotatable_bond_count > 20
    if large:
        vec[LIGAND_TYPE_IDX['large_flexible']] = 1.0

    if mol_summary.is_disconnected:
        # Disconnected graphs are a first-class route: keep all fragments and
        # avoid V5 AddHs/ETKDG/MMFF semantics entirely.
        if not primary_set:
            from_id_table = False
        return vec, LigandRoute.MULTIFRAGMENT_NO_CONFORMER, from_id_table

    # ── Catch-all ─────────────────────────────────────────────────────────
    if not primary_set:
        if mol_summary.load_ok or heavy > 0:
            vec[LIGAND_TYPE_IDX['regular_organic']] = 1.0
            route = LigandRoute.REGULAR_ORGANIC
            # Large flexible: upgrade to no-conformer to avoid invalid ETKDG
            # semantics on highly flexible graphs.
            if large:
                route = LigandRoute.NO_CONFORMER_BIOLOGICAL
        else:
            vec[LIGAND_TYPE_IDX['unknown_or_unclassified']] = 1.0
            route = LigandRoute.UNKNOWN_OR_FAILED
        from_id_table = False
    else:
        # Upgrade route for large flexible regular organic (ID-table-classified)
        if route == LigandRoute.REGULAR_ORGANIC and large:
            route = LigandRoute.NO_CONFORMER_BIOLOGICAL

    return vec, route, from_id_table


# ═══════════════════════════════════════════════════════════════════════════════
# Conformer & hydrogen policy helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_conformer_policy(
    route: LigandRoute,
    mol_summary: RawMolSummary,
) -> ConformerPolicy:
    """
    Separate from ligand type — a small nucleotide (ATP, 31 heavy atoms)
    is not "large", but still skips ETKDG/MMFF.
    """
    if route in (LigandRoute.PURE_METAL_ION, LigandRoute.INORGANIC_POLYATOMIC):
        return ConformerPolicy.SKIPPED_METAL
    if route == LigandRoute.MULTIFRAGMENT_NO_CONFORMER:
        return ConformerPolicy.RAW_COORDS_ONLY
    if route in (LigandRoute.METAL_COMPLEX, LigandRoute.NO_CONFORMER_BIOLOGICAL):
        return ConformerPolicy.RAW_COORDS_ONLY
    return ConformerPolicy.FULL_ETKDG_MMFF


def get_hydrogen_policy(route: LigandRoute) -> HydrogenPolicy:
    """
    Explicit mapping from route to hydrogen handling strategy.
    Saved as scalar in .pt so dataloader knows the node set.

    Only PURE_METAL_ION and INORGANIC_POLYATOMIC bypass V5 (no AddHs).
    All other routes go through V5 which calls AddHs internally, so the
    actual node set always includes H — report V5_ADD_H for honesty.
    """
    if route in (LigandRoute.PURE_METAL_ION, LigandRoute.INORGANIC_POLYATOMIC):
        return HydrogenPolicy.SKIPPED_BY_POLICY
    if route == LigandRoute.MULTIFRAGMENT_NO_CONFORMER:
        return HydrogenPolicy.EXPLICIT_ONLY
    if route in (LigandRoute.METAL_COMPLEX, LigandRoute.NO_CONFORMER_BIOLOGICAL,
                 LigandRoute.REGULAR_ORGANIC):
        return HydrogenPolicy.V5_ADD_H
    return HydrogenPolicy.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# Quick helpers
# ═══════════════════════════════════════════════════════════════════════════════

def is_pure_metal_pair(pair_meta: dict) -> bool:
    """Fast check from pair_meta only (no mol file read needed)."""
    lig = (pair_meta.get('ligand_id') or '').strip().upper()
    heavy = int(pair_meta.get('ligand_heavy_atom_count') or 0)
    return heavy == 1 and lig in METAL_ELEMENTS


def feature_completeness_ratio(group_mask: np.ndarray) -> float:
    """Fraction of x_group_mask groups that are valid (averaged across atoms)."""
    if group_mask.size == 0:
        return 0.0
    return float(group_mask.mean())
