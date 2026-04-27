#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip ligand feature extraction — wraps the existing V5 pipeline.

For each binding pair, reads pair_meta.json to locate the reconstructed
reference ligand (_ligand_ori.sdf) and extracts 109D per-atom features.

Directory layout
────────────────
  {biolip_root}/shared_receptors/{pdb_id}/pairs/
    {pdb_id}_{chain}/
      {bs_id}_{lig_id}_{chain}_{resseq}/
        pair_meta.json
        *_ligand_ori.sdf          ← input (Kabsch-reconstructed, all atoms)

Output
──────
  {output_root}/{pdb_id}_{chain}_{bs_id}_{lig_id}.pt

Feature dimensions (V5)
───────────────────────
  atom_features  [N_atoms, 109]  (54 local + 45 global + 10 shape)
  edge_index     [2, E]
  edge_attr      [E, 6]
  coords         [N_atoms, 3]    (all atoms incl. H, best conformer)
  coords_heavy_current [N_heavy, 3]
  coords_original      [N_heavy, 3]  (experimental pose coords)
  + per-atom chemical properties (gasteiger, logp, sasa, hbond, …)

Usage
─────
# smoke-test (first 5 pairs)
conda run -n AIAA python3 preparation/ligand/biolip_ligand_extract.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/ligand_features \\
    --max 5 --workers 1

# full run
conda run -n AIAA python3 preparation/ligand/biolip_ligand_extract.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/ligand_features \\
    --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── V5 ligand feature pipeline ────────────────────────────────────────────────
try:
    sys.path.insert(0, str(_PROJECT_ROOT / 'preparation' / 'ligand'))
    from extract_ligand_features_v5 import extract_ligand_features_v5
    V5_AVAILABLE = True
except ImportError as _e:
    V5_AVAILABLE = False
    _V5_IMPORT_ERR = str(_e)


# ── pair discovery ────────────────────────────────────────────────────────────

def _discover_pairs(biolip_root: Path) -> List[dict]:
    """
    Walk shared_receptors/ and collect all binding pairs.

    Returns a list of dicts with keys:
      pdb_id, chain, bs_id, lig_id, resseq, pair_dir, meta
    """
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
            # chain_dir.name == "{pdb_id}_{chain}"
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
    return f"{pair['pdb_id']}_{pair['chain']}_{pair['bs_id']}_{pair['lig_id']}"


# ── single-pair extraction ────────────────────────────────────────────────────

def extract_pair(
    pair      : dict,
    output_root: Path,
    overwrite  : bool = False,
    num_conformers: int = 50,
) -> Tuple[bool, str]:
    stem     = _output_stem(pair)
    out_path = output_root / f'{stem}.pt'

    if out_path.exists() and not overwrite:
        return True, f'{stem}: SKIP'

    # Locate _ligand_ori.sdf
    meta     = pair['meta']
    pair_dir = pair['pair_dir']
    ori_sdf_name = meta.get('ligand_reference_sdf_file') or meta.get('ligand_reference_mol2_file')
    if ori_sdf_name is None:
        return False, f'{stem}: pair_meta.json missing ligand_reference_sdf_file'

    ori_sdf = pair_dir / ori_sdf_name
    if not ori_sdf.exists():
        # fallback: try _ori.sdf by glob
        candidates = list(pair_dir.glob('*_ligand_ori.sdf'))
        if candidates:
            ori_sdf = candidates[0]
        else:
            return False, f'{stem}: {ori_sdf_name} not found'

    try:
        features = extract_ligand_features_v5(
            str(ori_sdf),
            include_conformers=(num_conformers > 0),
            num_conformers=num_conformers,
            strict=False,   # don't crash on unusual ligands
            verbose=False,
        )
    except Exception as e:
        return False, f'{stem}: feature extraction failed: {e}\n{traceback.format_exc()[:400]}'

    # Build output dict
    data: dict = {
        'atom_features':        torch.from_numpy(features['atom_features']).float(),
        'edge_index':           torch.from_numpy(features['edge_index']).long(),
        'edge_attr':            torch.from_numpy(features['edge_attr']).float(),
        'coords':               torch.from_numpy(features['coords']).float(),
        'coords_heavy_current': torch.from_numpy(features['coords_heavy_current']).float(),
        'gasteiger_charges':    torch.from_numpy(features['gasteiger_charges']).float(),
        'logp_contribs':        torch.from_numpy(features['logp_contribs']).float(),
        'tpsa_contribs':        torch.from_numpy(features['tpsa_contribs']).float(),
        'rmsf':                 torch.from_numpy(features['rmsf']).float(),
        'flexibility_flag':     torch.from_numpy(features['flexibility_flag']).float(),
        'atom_sasa':            torch.from_numpy(features['atom_sasa']).float(),
        'lipophilic_sasa':      torch.from_numpy(features['lipophilic_sasa']).float(),
        'hbond_donor':          torch.from_numpy(features['hbond_donor']).float(),
        'hbond_acceptor':       torch.from_numpy(features['hbond_acceptor']).float(),
        'chirality_cw':         torch.from_numpy(features['chirality_cw']).float(),
        'chirality_ccw':        torch.from_numpy(features['chirality_ccw']).float(),
        'rotatable_bond_context': torch.from_numpy(features['rotatable_bond_context']).float(),
        'formal_charge_class':  torch.from_numpy(features['formal_charge_class']).float(),
        'metal_binding_flag':   torch.from_numpy(features['metal_binding_flag']).float(),
        'stereochemistry_flag': torch.from_numpy(features['stereochemistry_flag']).float(),
        'shape_features':       torch.from_numpy(features['shape_features']).float(),
        'global_features':      torch.from_numpy(features['global_features']).float(),
        # metadata
        'num_atoms':            features['num_atoms'],
        'num_heavy_atoms':      features['num_heavy_atoms'],
        'has_original_coords':  features['has_original_coords'],
        'feature_dim_total':    features['feature_dim_total'],
        'ligand_path':          str(ori_sdf),
        'pdb_id':               pair['pdb_id'],
        'protein_chain':        pair['chain'],
        'binding_site_id':      pair['bs_id'],
        'ligand_id':            pair['lig_id'],
        'ligand_resseq':        pair['resseq'],
    }
    if features.get('coords_original') is not None:
        data['coords_original'] = torch.from_numpy(features['coords_original']).float()
    if features.get('coords_original_with_h') is not None:
        data['coords_original_with_h'] = torch.from_numpy(features['coords_original_with_h']).float()

    # Atomic write
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = output_root / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f'{stem}: write error: {e}'

    n_atoms = features['num_heavy_atoms']
    dim     = features['feature_dim_total']
    return True, f'{stem}: OK  ({n_atoms} heavy atoms, {dim}D)'


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_entry(
    pairs        : List[dict],
    output_root  : Path,
    overwrite    : bool,
    num_conformers: int,
    worker_id    : int,
    failure_tsv  : Path,
) -> None:
    total    = len(pairs)
    failures : List[Tuple[str, str]] = []

    for idx, pair in enumerate(pairs, 1):
        ok, msg = extract_pair(pair, output_root, overwrite, num_conformers)
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
        description='BioLip ligand feature extraction (V5 pipeline, _ligand_ori.sdf input).'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip dataset root (contains shared_receptors/)')
    parser.add_argument('--output', required=True,
                        help='Output root for .pt files')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num_conformers', type=int, default=50,
                        help='Conformers for RMSF/flexibility (0 = skip, default 50)')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pairs (for testing)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not V5_AVAILABLE:
        print(f'ERROR: V5 ligand pipeline not available: {_V5_IMPORT_ERR}')
        return 1

    biolip_root = Path(args.biolip_root).resolve()
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
    print(f'Conformers   : {args.num_conformers}')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(output_root=output_root, overwrite=args.overwrite,
               num_conformers=args.num_conformers)

    # single-worker shortcut
    if args.workers <= 1:
        failures: List[Tuple[str, str]] = []
        for idx, pair in enumerate(pairs, 1):
            ok, msg = extract_pair(pair, **kw)
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

    # multi-worker round-robin
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
            args=(chunk, output_root, args.overwrite, args.num_conformers,
                  wid, output_root / f'failures_w{wid}.tsv'),
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
