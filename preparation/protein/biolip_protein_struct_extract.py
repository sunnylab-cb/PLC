#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip protein structural feature extraction.

Reads receptor_fixed.cif (or receptor_full.cif as fallback) and extracts
the same V4 feature set used for PDBbind, adapted for CIF input via gemmi.

Feature dimensions (identical to V4)
──────────────────────────────────────
  physchem_features      [N, 62]
  spatial_scalar_features [N, 89]   (normalised if stats_json provided)
  spatial_vector_features [N, 8, 3]
  ca_coords              [N, 3]
  + residue metadata (names, indices, insertion_codes, chain_ids)

Key adaptation: parse_cif_protein() replaces parse_pdb_protein().
All downstream feature modules (surface, geometry, enhanced, v4 scalars)
are reused unchanged — they only need the pdb_info dict + a temp PDB for
DSSP/B-factor steps.

Usage
─────
# smoke-test
conda run -n AIAA python3 preparation/protein/biolip_protein_struct_extract.py \\
    --biolip_root    /home/husrcf/Code/ProtBind/BioLip \\
    --fixed_root     /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --output         /home/husrcf/Code/ProtBind/BioLip/protein_features \\
    --max 5 --workers 1

# full run with normalisation stats
conda run -n AIAA python3 preparation/protein/biolip_protein_struct_extract.py \\
    --biolip_root    /home/husrcf/Code/ProtBind/BioLip \\
    --fixed_root     /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --output         /home/husrcf/Code/ProtBind/BioLip/protein_features \\
    --stats_json     /home/husrcf/Code/ProtBind/PLC/data/protein_norm_stats.json \\
    --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── gemmi ─────────────────────────────────────────────────────────────────────
try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

# ── V4 feature modules (reused unchanged) ────────────────────────────────────
try:
    from preparation.protein.physicochemical_features import (
        extract_physicochemical_features,
        AA_3TO1,
    )
    from preparation.protein.surface_features import extract_surface_features
    from preparation.protein.structural_geometry_features import (
        extract_structural_geometry_features,
    )
    from preparation.protein.enhanced_features_v2 import (
        compute_pocketness_features,
        compute_flexibility_features,
        compute_environmental_features,
    )
    from preparation.protein.protein_features_v4 import (
        extract_physchem_features_v4,
        extract_spatial_scalar_features_v4,
        extract_spatial_vector_features_v4,
        get_spatial_scalar_feature_names_v4,
        load_normalization_stats,
    )
    V4_AVAILABLE = True
except ImportError as _e:
    V4_AVAILABLE = False
    _V4_IMPORT_ERR = str(_e)

# ── standard amino acids (gemmi uses 3-letter codes) ─────────────────────────
_STD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # protonation variants that OpenMM may write
    'HID', 'HIE', 'HIP', 'ASH', 'GLH', 'LYN',
}
# map protonation variants back to canonical 3-letter
_CANONICAL = {
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'ASH': 'ASP', 'GLH': 'GLU', 'LYN': 'LYS',
    'MSE': 'MET', 'SEC': 'CYS',
}


# ── CIF parser ────────────────────────────────────────────────────────────────

def parse_cif_protein(cif_path: Path) -> Dict:
    """
    Parse a receptor CIF file with gemmi and return a pdb_info dict
    compatible with the V4 feature pipeline.

    Handles:
    - Multi-chain complexes (all chains included)
    - Insertion codes (seqid.icode)
    - Protonation variants from OpenMM (HID/HIE/HIP/ASH/GLH/LYN)
    - Non-standard residues mapped to canonical names
    - Hydrogen atoms excluded from all_atom_coords/names

    Returns dict with keys:
      sequence, residue_names, residue_indices, insertion_codes,
      chain_ids, ca_coords [N,3], all_atom_coords List[ndarray],
      all_atom_names List[List[str]], n_residues, pdb_path, mode
    """
    if not GEMMI_AVAILABLE:
        raise ImportError('gemmi is required: conda install -c conda-forge gemmi')

    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError(f'No models in {cif_path}')
    model = st[0]

    residue_names: List[str]  = []
    residue_indices: List[int] = []
    insertion_codes: List[str] = []
    chain_ids: List[str]       = []
    sequence: List[str]        = []
    ca_coords_list: List[np.ndarray] = []
    all_atom_coords: List[np.ndarray] = []
    all_atom_names: List[List[str]]   = []

    for chain in model:
        for res in chain:
            # Skip non-amino-acid residues (HETATM, water, etc.)
            raw_name = res.name.strip()
            canonical = _CANONICAL.get(raw_name, raw_name)
            if canonical not in _STD_AA and raw_name not in _STD_AA:
                continue

            # Find CA atom
            ca_atom = res.find_atom('CA', '*')
            if ca_atom is None:
                continue

            res_seq  = res.seqid.num
            icode    = res.seqid.icode.strip()   # '' or 'A'/'B'/…
            chain_id = chain.name.strip()

            residue_names.append(canonical)
            residue_indices.append(int(res_seq))
            insertion_codes.append(icode)
            chain_ids.append(chain_id)
            sequence.append(AA_3TO1.get(canonical, 'X'))
            ca_coords_list.append(np.array(ca_atom.pos.tolist(), dtype=np.float32))

            # All heavy atoms for this residue
            heavy_coords: List[np.ndarray] = []
            heavy_names:  List[str]        = []
            for atom in res:
                if atom.element == gemmi.Element('H'):
                    continue
                heavy_coords.append(np.array(atom.pos.tolist(), dtype=np.float32))
                heavy_names.append(atom.name.strip())
            all_atom_coords.append(
                np.stack(heavy_coords, axis=0) if heavy_coords
                else np.zeros((0, 3), dtype=np.float32)
            )
            all_atom_names.append(heavy_names)

    if not residue_names:
        raise ValueError(f'No standard amino acid residues found in {cif_path}')

    return {
        'sequence':        ''.join(sequence),
        'residue_names':   residue_names,
        'residue_indices': residue_indices,
        'insertion_codes': insertion_codes,
        'chain_ids':       chain_ids,
        'ca_coords':       np.stack(ca_coords_list, axis=0),   # [N, 3]
        'all_atom_coords': all_atom_coords,
        'all_atom_names':  all_atom_names,
        'n_residues':      len(residue_names),
        'pdb_path':        str(cif_path),
        'mode':            'complex',
    }


def _write_temp_pdb_from_cif(cif_path: Path) -> str:
    """
    Write a temporary PDB converted by gemmi from CIF.
    Used by DSSP / B-factor modules that require a PDB file path.
    """
    fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
    os.close(fd)
    st = gemmi.read_structure(str(cif_path))
    st.write_pdb(tmp_path)
    return tmp_path


# ── per-structure extraction ──────────────────────────────────────────────────

def extract_protein_features(
    pdb_id       : str,
    fixed_root   : Path,
    biolip_root  : Path,
    output_root  : Path,
    overwrite    : bool = False,
    norm_stats   : Optional[Dict] = None,
) -> Tuple[bool, str]:
    """
    Extract V4-compatible protein features from receptor_fixed.cif.
    Falls back to receptor_full.cif if fixed is not available.

    Output: {output_root}/{pdb_id}_protein.pt
    """
    out_path = output_root / f'{pdb_id}_protein.pt'
    if out_path.exists() and not overwrite:
        return True, f'{pdb_id}: SKIP'

    # Locate CIF — prefer fixed (minimized), fall back to original
    cif_fixed = fixed_root / pdb_id / 'receptor_fixed.cif'
    cif_orig  = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
    if cif_fixed.exists():
        cif_path = cif_fixed
        src_tag  = 'fixed'
    elif cif_orig.exists():
        cif_path = cif_orig
        src_tag  = 'original'
    else:
        return False, f'{pdb_id}: no CIF found'

    tmp_pdb: Optional[str] = None
    try:
        pdb_info = parse_cif_protein(cif_path)
        n = pdb_info['n_residues']

        # Write temp PDB for modules that need a file path (DSSP, B-factors)
        tmp_pdb = _write_temp_pdb_from_cif(cif_path)
        tmp_pdb_path = Path(tmp_pdb)

        # ── V4 feature computation (reuse existing modules) ───────────────
        v2_physchem_full = extract_physicochemical_features(
            tmp_pdb_path,
            pdb_info['sequence'],
            pdb_info['residue_names'],
            pdb_info['residue_indices'],
            pdb_info['chain_ids'],
        )
        v2_surface = extract_surface_features(
            pdb_info['ca_coords'], pdb_info['residue_names']
        )
        v2_structural_geometry = extract_structural_geometry_features(
            tmp_pdb_path,
            pdb_info['ca_coords'],
            pdb_info['residue_indices'],
            pdb_info['chain_ids'],
        )
        pocketness  = compute_pocketness_features(pdb_info['ca_coords'])
        sasa_values = v2_physchem_full[:, 0]
        flexibility = compute_flexibility_features(
            tmp_pdb, pdb_info['ca_coords'], sasa_values
        )
        environmental = compute_environmental_features(
            pdb_info['ca_coords'], v2_physchem_full, pdb_info['residue_names']
        )
        v2_enhanced = np.concatenate([pocketness, flexibility, environmental], axis=1)

        v2_basic    = v2_physchem_full[:, 0:20]
        v2_sasa     = v2_physchem_full[:, 20:26]
        v2_sequence = v2_physchem_full[:, 26:34]
        v2_hbond    = v2_physchem_full[:, 34:42]

        physchem_features = extract_physchem_features_v4(
            pdb_info, v2_basic, v2_sequence, v2_hbond, v2_structural_geometry
        )
        spatial_scalar_raw, spatial_scalar_features = extract_spatial_scalar_features_v4(
            pdb_info, v2_sasa, v2_surface, v2_structural_geometry,
            v2_enhanced, norm_stats,
        )
        spatial_vector_features = extract_spatial_vector_features_v4(
            tmp_pdb_path, pdb_info
        )

    except Exception as e:
        return False, f'{pdb_id}: {e}\n{traceback.format_exc()[:400]}'
    finally:
        if tmp_pdb and os.path.exists(tmp_pdb):
            try:
                os.unlink(tmp_pdb)
            except OSError:
                pass

    data = {
        'physchem_features':          torch.from_numpy(physchem_features).float(),
        'spatial_scalar_features':    torch.from_numpy(spatial_scalar_features).float(),
        'spatial_scalar_raw_features': torch.from_numpy(spatial_scalar_raw).float(),
        'spatial_vector_features':    torch.from_numpy(spatial_vector_features).float(),
        'ca_coords':                  torch.from_numpy(pdb_info['ca_coords']).float(),
        'residue_names':              pdb_info['residue_names'],
        'residue_indices':            pdb_info['residue_indices'],
        'insertion_codes':            pdb_info['insertion_codes'],
        'chain_ids':                  pdb_info['chain_ids'],
        'n_residues':                 n,
        'pdb_id':                     pdb_id,
        'cif_source':                 src_tag,
        'spatial_scalar_feature_names': get_spatial_scalar_feature_names_v4(),
    }

    # Atomic write
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_out = output_root / f'.{pdb_id}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp_out)
        os.replace(tmp_out, out_path)
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        return False, f'{pdb_id}: write error: {e}'

    n_chains = len(set(pdb_info['chain_ids']))
    return True, f'{pdb_id}: OK  ({n_chains} chains, {n} residues, src={src_tag})'


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_entry(
    pdb_ids     : List[str],
    fixed_root  : Path,
    biolip_root : Path,
    output_root : Path,
    overwrite   : bool,
    norm_stats  : Optional[Dict],
    worker_id   : int,
    failure_tsv : Path,
) -> None:
    total    = len(pdb_ids)
    failures : List[Tuple[str, str]] = []

    for idx, pdb_id in enumerate(pdb_ids, 1):
        ok, msg = extract_protein_features(
            pdb_id, fixed_root, biolip_root, output_root, overwrite, norm_stats
        )
        print(f'[W{worker_id} {idx}/{total}] {msg}', flush=True)
        if not ok:
            failures.append((pdb_id, msg))

    if failures:
        failure_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_tsv, 'w') as f:
            f.write('pdb_id\terror\n')
            for pid, err in failures:
                f.write(f'{pid}\t{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
        print(f'[W{worker_id}] {len(failures)} failures → {failure_tsv}')

    raise SystemExit(1 if failures else 0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='BioLip protein structural feature extraction (V4-compatible, CIF input).'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip root (contains shared_receptors/)')
    parser.add_argument('--fixed_root', required=True,
                        help='PDBFixer output root (contains {pdb_id}/receptor_fixed.cif)')
    parser.add_argument('--output', required=True,
                        help='Output root for .pt files')
    parser.add_argument('--stats_json', default=None,
                        help='Normalisation stats JSON (from fit-stats mode); '
                             'omit to save raw scalar features')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pdb_ids (for testing)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pdbid_list_file', default=None,
                        help='Text file of pdb_ids (one per line); '
                             'default: auto-discover from fixed_root')
    args = parser.parse_args()

    if not GEMMI_AVAILABLE:
        print('ERROR: gemmi not installed.  conda install -c conda-forge gemmi')
        return 1
    if not V4_AVAILABLE:
        print(f'ERROR: V4 feature modules not available: {_V4_IMPORT_ERR}')
        return 1

    biolip_root = Path(args.biolip_root).resolve()
    fixed_root  = Path(args.fixed_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    norm_stats = None
    if args.stats_json:
        norm_stats = load_normalization_stats(Path(args.stats_json))
        print(f'Normalisation stats: {args.stats_json}')

    # Collect pdb_ids
    if args.pdbid_list_file:
        pdb_ids = [
            l.strip() for l in Path(args.pdbid_list_file).read_text().splitlines()
            if l.strip()
        ]
        pdb_ids = list(dict.fromkeys(pdb_ids))
        print(f'PDB-ID list from file: {len(pdb_ids):,}')
    else:
        # Auto-discover from fixed_root (prefer fixed) then fall back to biolip_root
        pdb_ids = sorted(
            p.name for p in fixed_root.iterdir()
            if p.is_dir() and (p / 'receptor_fixed.cif').exists()
        )
        # Also include pdb_ids that only have receptor_full.cif
        orig_ids = {
            p.name for p in (biolip_root / 'shared_receptors').iterdir()
            if p.is_dir() and (p / 'receptor_full.cif').exists()
        }
        all_ids = sorted(set(pdb_ids) | orig_ids)
        pdb_ids = all_ids
        print(f'Auto-discovered: {len(pdb_ids):,} pdb_ids '
              f'({len([p for p in fixed_root.iterdir() if p.is_dir()]):,} fixed, '
              f'{len(orig_ids):,} original)')

    if args.max:
        pdb_ids = pdb_ids[:args.max]
        print(f'Limiting to first {len(pdb_ids)} pdb_ids.')

    total = len(pdb_ids)
    print(f'Total      : {total:,}')
    print(f'Output     : {output_root}')
    print(f'Workers    : {args.workers}')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(fixed_root=fixed_root, biolip_root=biolip_root,
               output_root=output_root, overwrite=args.overwrite,
               norm_stats=norm_stats)

    if args.workers <= 1:
        failures: List[Tuple[str, str]] = []
        for idx, pdb_id in enumerate(pdb_ids, 1):
            ok, msg = extract_protein_features(pdb_id, **kw)
            print(f'[{idx}/{total}] {msg}', flush=True)
            if not ok:
                failures.append((pdb_id, msg))

        if failures:
            fail_path = output_root / 'failures.tsv'
            with open(fail_path, 'w') as f:
                f.write('pdb_id\terror\n')
                for pid, err in failures:
                    f.write(f'{pid}\t{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
            print(f'{len(failures)} failures → {fail_path}')
            return 1
        print(f'\nDone. {total}/{total} receptors processed.')
        return 0

    n_workers = min(args.workers, total)
    splits    = [[] for _ in range(n_workers)]
    for i, pid in enumerate(pdb_ids):
        splits[i % n_workers].append(pid)

    ctx   = mp.get_context('fork')
    procs = []
    for wid, chunk in enumerate(splits):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker_entry,
            args=(chunk, fixed_root, biolip_root, output_root,
                  args.overwrite, norm_stats, wid,
                  output_root / f'failures_w{wid}.tsv'),
        )
        p.start()
        procs.append(p)

    rc = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
