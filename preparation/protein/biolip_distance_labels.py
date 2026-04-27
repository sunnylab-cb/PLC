#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip residue–ligand distance labels (per binding pair).

For each binding pair:
  - protein: receptor_fixed.cif (fallback receptor_full.cif)
  - ligand : *_ligand_ori.sdf (reconstructed complete ligand)

Computes minimum heavy-atom distance from each protein residue to any ligand
heavy atom, then derives binary labels using a configurable threshold.

Default threshold: 5.0 Å (user-specified for this project)

Output
──────
{output_root}/{pdb_id}_{chain}_{bs_id}_{lig_id}_labels.pt

Fields:
  min_distance      : FloatTensor [N_res]
  binding_label     : BoolTensor  [N_res]   (min_distance <= threshold)
  threshold_angstrom: float
  residue_chain_id  : List[str]
  residue_index     : List[int]
  insertion_code    : List[str]
  residue_name      : List[str]
  pdb_id, protein_chain, binding_site_id, ligand_id, ligand_resseq, ligand_path

Usage
─────
# smoke-test
conda run -n AIAA python3 preparation/protein/biolip_distance_labels.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --fixed_root  /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --output      /home/husrcf/Code/ProtBind/BioLip/distance_labels \\
    --threshold 5.0 --max 10 --workers 1

# full run
conda run -n AIAA python3 preparation/protein/biolip_distance_labels.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --fixed_root  /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --output      /home/husrcf/Code/ProtBind/BioLip/distance_labels \\
    --threshold 5.0 --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ── gemmi / rdkit ─────────────────────────────────────────────────────────────
try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ── pair discovery ────────────────────────────────────────────────────────────

def _discover_pairs(biolip_root: Path) -> List[dict]:
    pairs = []
    receptors_root = biolip_root / 'shared_receptors'
    if not receptors_root.is_dir():
        raise FileNotFoundError(f'{receptors_root} not found')

    for pdb_dir in sorted(receptors_root.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdb_id = pdb_dir.name
        pairs_root = pdb_dir / 'pairs'
        if not pairs_root.is_dir():
            continue

        for chain_dir in sorted(pairs_root.iterdir()):
            if not chain_dir.is_dir():
                continue
            parts = chain_dir.name.split('_', 1)
            chain = parts[1] if len(parts) == 2 else chain_dir.name

            for bs_dir in sorted(chain_dir.iterdir()):
                if not bs_dir.is_dir():
                    continue
                meta_path = bs_dir / 'pair_meta.json'
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue

                pairs.append({
                    'pdb_id':   pdb_id,
                    'chain':    meta.get('protein_chain', chain),
                    'bs_id':    meta.get('binding_site_id', bs_dir.name.split('_')[0]),
                    'lig_id':   meta.get('ligand_id', ''),
                    'resseq':   str(meta.get('ligand_resseq', '')),
                    'pair_dir': bs_dir,
                    'meta':     meta,
                })

    return pairs


def _output_stem(pair: dict) -> str:
    return f"{pair['pdb_id']}_{pair['chain']}_{pair['bs_id']}_{pair['lig_id']}_labels"


# ── structure parsing ─────────────────────────────────────────────────────────

def _load_ligand_heavy_coords(sdf_path: Path) -> np.ndarray:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    mol = next(suppl, None)
    if mol is None:
        raise ValueError(f'Failed to read ligand SDF: {sdf_path}')
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(p.x), float(p.y), float(p.z)])
    if not coords:
        raise ValueError(f'No heavy atoms in ligand: {sdf_path}')
    return np.asarray(coords, dtype=np.float32)


def _load_residue_heavy_coords(cif_path: Path, target_chain: Optional[str]) -> Tuple[
    List[str], List[int], List[str], List[str], List[np.ndarray]
]:
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError(f'No models in {cif_path}')
    model = st[0]

    chain_ids: List[str] = []
    res_indices: List[int] = []
    icodes: List[str] = []
    res_names: List[str] = []
    res_heavy_coords: List[np.ndarray] = []

    for chain in model:
        cid = chain.name.strip()
        if target_chain is not None and cid != target_chain:
            continue

        for res in chain:
            # amino-acid residues only (gemmi Python binding has no is_amino_acid())
            if getattr(res, 'het_flag', None) != 'A':
                continue

            heavy = []
            for atom in res:
                if atom.element == gemmi.Element('H'):
                    continue
                heavy.append([atom.pos.x, atom.pos.y, atom.pos.z])
            if not heavy:
                continue

            chain_ids.append(cid)
            res_indices.append(int(res.seqid.num))
            icodes.append(res.seqid.icode.strip())
            res_names.append(res.name.strip())
            res_heavy_coords.append(np.asarray(heavy, dtype=np.float32))

    if not res_heavy_coords:
        raise ValueError(f'No amino-acid heavy atoms found in {cif_path} (chain={target_chain})')

    return chain_ids, res_indices, icodes, res_names, res_heavy_coords


# ── distance computation ──────────────────────────────────────────────────────

def _min_dist_per_residue(residue_heavy_coords: List[np.ndarray], ligand_heavy_coords: np.ndarray) -> np.ndarray:
    """
    For each residue (all heavy atoms), compute min distance to any ligand heavy atom.
    """
    dmin = np.zeros(len(residue_heavy_coords), dtype=np.float32)
    for i, rxyz in enumerate(residue_heavy_coords):
        # rxyz: [Nr, 3], ligand: [Nl, 3]
        diff = rxyz[:, None, :] - ligand_heavy_coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)   # [Nr, Nl]
        dmin[i] = float(dist.min())
    return dmin


# ── per-pair extraction ───────────────────────────────────────────────────────

def extract_pair_labels(
    pair         : dict,
    fixed_root   : Path,
    biolip_root  : Path,
    output_root  : Path,
    threshold    : float,
    overwrite    : bool = False,
) -> Tuple[bool, str]:
    stem     = _output_stem(pair)
    out_path = output_root / f'{stem}.pt'
    if out_path.exists() and not overwrite:
        return True, f'{stem}: SKIP'

    pdb_id = pair['pdb_id']
    chain  = pair['chain']

    cif_fixed = fixed_root / pdb_id / 'receptor_fixed.cif'
    cif_orig  = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
    if cif_fixed.exists():
        cif_path = cif_fixed
        src_tag = 'fixed'
    elif cif_orig.exists():
        cif_path = cif_orig
        src_tag = 'original'
    else:
        return False, f'{stem}: no receptor CIF found'

    meta = pair['meta']
    pair_dir = pair['pair_dir']
    ori_name = meta.get('ligand_reference_sdf_file')
    if ori_name is None:
        cands = list(pair_dir.glob('*_ligand_ori.sdf'))
        if not cands:
            return False, f'{stem}: ligand_reference_sdf_file missing and *_ligand_ori.sdf not found'
        ori_sdf = cands[0]
    else:
        ori_sdf = pair_dir / ori_name
        if not ori_sdf.exists():
            cands = list(pair_dir.glob('*_ligand_ori.sdf'))
            if cands:
                ori_sdf = cands[0]
            else:
                return False, f'{stem}: {ori_name} not found'

    try:
        lig_xyz = _load_ligand_heavy_coords(ori_sdf)
        chain_ids, res_indices, icodes, res_names, res_heavy_xyz = _load_residue_heavy_coords(
            cif_path, target_chain=chain
        )
        min_dist = _min_dist_per_residue(res_heavy_xyz, lig_xyz)
        label = min_dist <= float(threshold)

        data = {
            'min_distance':        torch.from_numpy(min_dist).float(),
            'binding_label':       torch.from_numpy(label.astype(np.bool_)),
            'threshold_angstrom':  float(threshold),
            'residue_chain_id':    chain_ids,
            'residue_index':       res_indices,
            'insertion_code':      icodes,
            'residue_name':        res_names,
            'pdb_id':              pdb_id,
            'protein_chain':       chain,
            'binding_site_id':     pair['bs_id'],
            'ligand_id':           pair['lig_id'],
            'ligand_resseq':       pair['resseq'],
            'ligand_path':         str(ori_sdf),
            'receptor_source':     src_tag,
        }

    except Exception as e:
        return False, f'{stem}: {e}\n{traceback.format_exc()[:400]}'

    output_root.mkdir(parents=True, exist_ok=True)
    tmp = output_root / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f'{stem}: write error: {e}'

    n_res = len(res_indices)
    n_pos = int(label.sum())
    return True, f'{stem}: OK  ({n_res} residues, positives={n_pos}, threshold={threshold:.1f}Å, src={src_tag})'


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_entry(
    pairs        : List[dict],
    fixed_root   : Path,
    biolip_root  : Path,
    output_root  : Path,
    threshold    : float,
    overwrite    : bool,
    worker_id    : int,
    failure_tsv  : Path,
) -> None:
    total    = len(pairs)
    failures : List[Tuple[str, str]] = []

    for idx, pair in enumerate(pairs, 1):
        ok, msg = extract_pair_labels(
            pair, fixed_root, biolip_root, output_root, threshold, overwrite
        )
        print(f'[W{worker_id} {idx}/{total}] {msg}', flush=True)
        if not ok:
            failures.append((_output_stem(pair), msg))

    if failures:
        failure_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_tsv, 'w') as f:
            f.write('pair_id\terror\n')
            for pid, err in failures:
                f.write(f'{pid}\t{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
        print(f'[W{worker_id}] {len(failures)} failures → {failure_tsv}')

    raise SystemExit(1 if failures else 0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='BioLip residue–ligand distance labels from receptor CIF + _ligand_ori.sdf'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip root (contains shared_receptors/)')
    parser.add_argument('--fixed_root', required=True,
                        help='PDBFixer output root (contains {pdb_id}/receptor_fixed.cif)')
    parser.add_argument('--output', required=True,
                        help='Output root for label .pt files')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Binding threshold in Å (default: 5.0)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pairs (testing)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not GEMMI_AVAILABLE:
        print('ERROR: gemmi not installed.  conda install -c conda-forge gemmi')
        return 1
    if not RDKIT_AVAILABLE:
        print('ERROR: RDKit not installed.  conda install -c conda-forge rdkit')
        return 1

    biolip_root = Path(args.biolip_root).resolve()
    fixed_root  = Path(args.fixed_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print('Discovering binding pairs…', flush=True)
    pairs = _discover_pairs(biolip_root)
    if args.max:
        pairs = pairs[:args.max]

    total = len(pairs)
    print(f'Total pairs  : {total:,}')
    print(f'Output root  : {output_root}')
    print(f'Workers      : {args.workers}')
    print(f'Threshold    : {args.threshold:.2f} Å')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(
        fixed_root=fixed_root,
        biolip_root=biolip_root,
        output_root=output_root,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )

    if args.workers <= 1:
        failures: List[Tuple[str, str]] = []
        for idx, pair in enumerate(pairs, 1):
            ok, msg = extract_pair_labels(pair, **kw)
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

    n_workers = min(args.workers, total)
    splits    = [[] for _ in range(n_workers)]
    for i, p in enumerate(pairs):
        splits[i % n_workers].append(p)

    ctx   = mp.get_context('fork')
    procs = []
    for wid, chunk in enumerate(splits):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker_entry,
            args=(
                chunk, fixed_root, biolip_root, output_root,
                args.threshold, args.overwrite,
                wid, output_root / f'failures_w{wid}.tsv',
            ),
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
