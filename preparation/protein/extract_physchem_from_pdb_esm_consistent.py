#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESM 对齐版 PDB 解析器。

目标：让 feature 侧残基定义尽量与 `esm2_extract_latents_loose.py` 一致。
核心策略：
- 以 `ProteinComplex.from_pdb()` 生成残基主索引（chain/resseq/icode/sequence）
- 用 Bio.PDB 按同样 key 回填 residue_name / all_atom_coords / all_atom_names
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Bio.PDB import PDBParser
from esm.utils import residue_constants
from esm.utils.structure.protein_complex import ProteinComplex


def _normalize_icode(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _aa1_from_complex_token(token) -> str:
    if isinstance(token, int):
        if 0 <= token < len(residue_constants.restypes):
            return residue_constants.restypes[token]
        return "X"
    token_str = str(token)
    return token_str if token_str else "X"


def _aa3_from_aa1(aa_1: str) -> str:
    return residue_constants.restype_1to3.get(aa_1, "UNK")


def _build_biopython_residue_map(pdb_path: Path) -> Dict[Tuple[str, int, str], object]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    residue_map: Dict[Tuple[str, int, str], object] = {}
    if len(structure) == 0:
        return residue_map

    model = next(structure.get_models())
    for chain in model:
        chain_id = str(chain.get_id())
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            if str(hetflag).strip():
                continue
            key = (chain_id, int(resseq), _normalize_icode(icode))
            residue_map[key] = residue

    return residue_map


def _parse_pdb_biopython_fallback(pdb_path: Path, chain_id: str = None, mode: str = "complex") -> Dict:
    """
    纯 BioPython 解析路径，用于 ProteinComplex.from_pdb 因列宽问题失败时的 fallback。
    输出 dict 格式与正常路径完全一致。
    """
    from Bio.PDB import is_aa

    AA_1TO3 = {v: k for k, v in residue_constants.restype_1to3.items()}
    AA_3TO1 = residue_constants.restype_1to3  # 3->1 via inverse
    _3to1 = {v: k for k, v in residue_constants.restype_1to3.items()}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("_", str(pdb_path))
    model = next(structure.get_models())

    chains_to_use = [c for c in model] if mode == "complex" else None
    if mode == "chain":
        if chain_id is None:
            chain_id = next(model).get_id()
        chains_to_use = [model[chain_id]]

    residue_names, residue_indices, insertion_codes, chain_ids = [], [], [], []
    aa_1_list, ca_coords_out, all_atom_coords, all_atom_names = [], [], [], []

    for chain in chains_to_use:
        for residue in chain:
            het, resseq, icode = residue.get_id()
            if het.strip():
                continue
            if not is_aa(residue, standard=False):
                continue
            res_name = residue.get_resname().strip()
            aa_1 = _3to1.get(res_name, "X")

            residue_names.append(res_name)
            residue_indices.append(int(resseq))
            insertion_codes.append(_normalize_icode(icode))
            chain_ids.append(chain.get_id())
            aa_1_list.append(aa_1)

            ca_coord = np.zeros(3, dtype=np.float32)
            if "CA" in residue:
                ca_coord = np.asarray(residue["CA"].get_coord(), dtype=np.float32)
            ca_coords_out.append(ca_coord)

            coords, names = [], []
            for atom in residue.get_atoms():
                if (atom.element or "").strip().upper() == "H":
                    continue
                names.append(atom.get_name().strip())
                coords.append(np.asarray(atom.get_coord(), dtype=np.float32))
            all_atom_coords.append(np.asarray(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32))
            all_atom_names.append(names)

    return {
        "sequence": "".join(aa_1_list),
        "residue_names": residue_names,
        "residue_indices": residue_indices,
        "insertion_codes": insertion_codes,
        "chain_ids": chain_ids,
        "ca_coords": np.asarray(ca_coords_out, dtype=np.float32),
        "all_atom_coords": all_atom_coords,
        "all_atom_names": all_atom_names,
        "pdb_path": str(pdb_path),
        "n_residues": len(residue_names),
        "mode": mode,
    }


def parse_pdb_protein(pdb_path: Path, chain_id: str = None, mode: str = "complex") -> Dict:
    """
    ESM 对齐版 parse。

    Args:
        pdb_path: PDB 文件路径
        chain_id: 当 mode='chain' 时指定链
        mode: 'complex' 或 'chain'
    """
    try:
        complex_obj = ProteinComplex.from_pdb(pdb_path)
    except Exception as e:
        if "pre-decimal columns" in str(e) or "column" in str(e).lower():
            import warnings
            warnings.warn(f"{pdb_path.name}: biotite column width error, falling back to BioPython: {e}")
            return _parse_pdb_biopython_fallback(pdb_path, chain_id, mode)
        raise

    raw_mask = complex_obj.chain_id != -1
    chain_lookup = complex_obj.metadata.chain_lookup

    chain_ids_all = np.empty(len(complex_obj.chain_id), dtype=object)
    chain_ids_all[:] = ""
    for idx, chain_numeric in enumerate(complex_obj.chain_id):
        if int(chain_numeric) == -1:
            continue
        chain_ids_all[idx] = chain_lookup[int(chain_numeric)]

    if mode == "chain":
        if chain_id is None:
            unique_chains = [c for c in sorted(set(chain_ids_all[raw_mask].tolist())) if c]
            if not unique_chains:
                raise ValueError(f"No protein chains found in {pdb_path}")
            chain_id = unique_chains[0]
        mask = raw_mask & (chain_ids_all == chain_id)
    else:
        mask = raw_mask

    seq_tokens = complex_obj.sequence
    aa_1_list: List[str] = []
    for keep, token in zip(mask, seq_tokens):
        if keep:
            aa_1_list.append(_aa1_from_complex_token(token))

    chain_ids = [chain_lookup[int(x)] for x in complex_obj.chain_id[mask]]
    residue_indices = [int(x) for x in complex_obj.residue_index[mask]]
    insertion_codes = [_normalize_icode(x) for x in complex_obj.insertion_code[mask].tolist()]

    atom37_positions = np.asarray(complex_obj.atom37_positions, dtype=np.float32)[mask]
    atom37_mask = np.asarray(complex_obj.atom37_mask)[mask]
    ca_index = residue_constants.atom_order["CA"]
    ca_coords_from_atom37 = atom37_positions[:, ca_index, :]
    has_ca = atom37_mask[:, ca_index].astype(bool)

    residue_map = _build_biopython_residue_map(pdb_path)

    residue_names: List[str] = []
    all_atom_coords: List[np.ndarray] = []
    all_atom_names: List[List[str]] = []

    ca_coords_out: List[np.ndarray] = []

    for i, (chain, resseq, icode, aa_1) in enumerate(zip(chain_ids, residue_indices, insertion_codes, aa_1_list)):
        key = (str(chain), int(resseq), _normalize_icode(icode))
        residue = residue_map.get(key)

        if residue is None:
            residue_names.append(_aa3_from_aa1(aa_1))
            ca_coord = ca_coords_from_atom37[i] if has_ca[i] else np.zeros(3, dtype=np.float32)
            if not np.isfinite(ca_coord).all():
                ca_coord = np.zeros(3, dtype=np.float32)
            ca_coords_out.append(np.asarray(ca_coord, dtype=np.float32))

            if has_ca[i]:
                all_atom_coords.append(np.asarray([ca_coord], dtype=np.float32))
                all_atom_names.append(["CA"])
            else:
                all_atom_coords.append(np.zeros((0, 3), dtype=np.float32))
                all_atom_names.append([])
            continue

        residue_names.append(str(residue.get_resname()).strip())

        coords = []
        names = []
        for atom in residue.get_atoms():
            element = (atom.element or "").strip().upper()
            if element == "H":
                continue
            names.append(atom.get_name().strip())
            coords.append(np.asarray(atom.get_coord(), dtype=np.float32))

        ca_coord = None
        if "CA" in residue:
            ca_coord = np.asarray(residue["CA"].get_coord(), dtype=np.float32)
        elif has_ca[i]:
            ca_coord = np.asarray(ca_coords_from_atom37[i], dtype=np.float32)

        if len(coords) == 0 and has_ca[i]:
            coords = [np.asarray(ca_coords_from_atom37[i], dtype=np.float32)]
            names = ["CA"]

        if ca_coord is None:
            if len(coords) > 0:
                ca_coord = np.asarray(coords[0], dtype=np.float32)
            else:
                ca_coord = np.zeros(3, dtype=np.float32)
        if not np.isfinite(ca_coord).all():
            if len(coords) > 0 and np.isfinite(np.asarray(coords)).all():
                ca_coord = np.asarray(coords[0], dtype=np.float32)
            else:
                ca_coord = np.zeros(3, dtype=np.float32)
        ca_coords_out.append(ca_coord)

        all_atom_coords.append(np.asarray(coords, dtype=np.float32))
        all_atom_names.append(names)

    return {
        "sequence": "".join(aa_1_list),
        "residue_names": residue_names,
        "residue_indices": residue_indices,
        "insertion_codes": insertion_codes,
        "chain_ids": chain_ids,
        "ca_coords": np.asarray(ca_coords_out, dtype=np.float32),
        "all_atom_coords": all_atom_coords,
        "all_atom_names": all_atom_names,
        "pdb_path": str(pdb_path),
        "n_residues": len(residue_names),
        "mode": mode,
    }
