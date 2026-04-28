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
import pickle
import shutil
import signal
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── V5 ligand feature pipeline ────────────────────────────────────────────────
try:
    sys.path.insert(0, str(_PROJECT_ROOT / 'preparation' / 'ligand'))
    from extract_ligand_features_v5 import extract_ligand_features_v5
    V5_AVAILABLE = True
except ImportError as _e:
    V5_AVAILABLE = False
    _V5_IMPORT_ERR = str(_e)


# ── pair discovery ────────────────────────────────────────────────────────────

def _parse_meta_file(meta_path: Path) -> Optional[dict]:
    # meta_path: .../shared_receptors/{pdb_id}/pairs/{pdb_id}_{chain}/{bs_dir}/pair_meta.json
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
    receptors_root = biolip_root / 'shared_receptors'
    if not receptors_root.is_dir():
        raise FileNotFoundError(f'{receptors_root} not found')

    # Use find to collect all pair_meta.json paths at OS level (much faster than Python walk)
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


# ── single-pair extraction ────────────────────────────────────────────────────

def extract_pair(
    pair      : dict,
    output_root: Path,
    overwrite  : bool = False,
    num_conformers: int = 50,
    rdkit_threads: int = 1,
    per_pair_timeout: int = 0,
    fallback_no_conformers: bool = True,
) -> Tuple[bool, str]:
    stem     = _output_stem(pair)
    out_path = _output_path(pair, output_root)

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
        with _Timeout(per_pair_timeout):
            features = extract_ligand_features_v5(
                str(ori_sdf),
                include_conformers=(num_conformers > 0),
                num_conformers=num_conformers,
                strict=False,   # don't crash on unusual ligands
                verbose=False,
                num_threads=rdkit_threads,
            )
    except Exception as e:
        if fallback_no_conformers and num_conformers > 0:
            try:
                features = extract_ligand_features_v5(
                    str(ori_sdf),
                    include_conformers=False,
                    num_conformers=0,
                    strict=False,
                    verbose=False,
                    num_threads=rdkit_threads,
                )
            except Exception as e2:
                return False, f'{stem}: feature extraction failed: {e2}\n{traceback.format_exc()[:400]}'
            else:
                fallback_tag = ' (fallback:no_conformers)'
        else:
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

    # Atomic write into {output_root}/{pdb_id}/
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f'{stem}: write error: {e}'

    # Copy pair_meta.json alongside the .pt file
    meta_src = pair['pair_dir'] / 'pair_meta.json'
    meta_dst = out_path.with_name(f'{stem}_meta.json')
    if meta_src.exists() and (not meta_dst.exists() or overwrite):
        shutil.copy2(meta_src, meta_dst)

    n_atoms = features['num_heavy_atoms']
    dim     = features['feature_dim_total']
    tag = locals().get('fallback_tag', '')
    return True, f'{stem}: OK  ({n_atoms} heavy atoms, {dim}D){tag}'


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
        description='BioLip ligand feature extraction (V5 pipeline, _ligand_ori.sdf input).'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip dataset root (contains shared_receptors/)')
    parser.add_argument('--output', required=True,
                        help='Output root for .pt files')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num_conformers', type=int, default=50,
                        help='Conformers for RMSF/flexibility (0 = skip, default 50)')
    parser.add_argument('--rdkit-threads', type=int, default=1,
                        help='RDKit threads per task (default 1; avoid CPU oversubscription)')
    parser.add_argument('--per-pair-timeout', type=int, default=0,
                        help='Per-pair timeout in seconds (0 = disable)')
    parser.add_argument('--no-fallback-no-conformers', action='store_true',
                        help='Disable fallback to num_conformers=0 when conformer path fails/timeouts')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pairs (for testing)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pairs-cache', default=None,
                        help='Path to cache file for discovered pairs (pickle). '
                             'Saves on first run, loads on subsequent runs.')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Force re-discovery even if cache exists.')
    args = parser.parse_args()

    if not V5_AVAILABLE:
        print(f'ERROR: V5 ligand pipeline not available: {_V5_IMPORT_ERR}')
        return 1

    biolip_root = Path(args.biolip_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    import time as _time
    cache_path = Path(args.pairs_cache) if args.pairs_cache else None

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
    print(f'Total pairs  : {total:,}')
    print(f'Output root  : {output_root}')
    print(f'Workers      : {args.workers}')
    print(f'Conformers   : {args.num_conformers}')
    print(f'RDKit thr/task: {args.rdkit_threads}')
    print(f'Pair timeout : {args.per_pair_timeout}s (0=off)')
    print(f'Fallback     : {not args.no_fallback_no_conformers}')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(
        output_root=output_root,
        overwrite=args.overwrite,
        num_conformers=args.num_conformers,
        rdkit_threads=args.rdkit_threads,
        per_pair_timeout=args.per_pair_timeout,
        fallback_no_conformers=(not args.no_fallback_no_conformers),
    )

    # single-worker shortcut
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

    # multi-worker pool (dynamic load balancing)
    n_workers = min(args.workers, total)
    tasks = [
        (
            p,
            output_root,
            args.overwrite,
            args.num_conformers,
            args.rdkit_threads,
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
