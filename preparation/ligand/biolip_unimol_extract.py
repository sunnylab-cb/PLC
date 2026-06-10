#!/usr/bin/env python3
"""
BioLip Uni-Mol ligand embedding extractor.

Scans {ligand_features_root}/{pdb_id}/*.pt, reads the 'ligand_path' field
from each file to locate the source SDF, runs Uni-Mol inference, and saves
per-pair embeddings to {output_root}/{pdb_id}/{stem}_unimol.pt.

Output schema per file:
    {
        'embedding':        tensor [512]   float32   — CLS-token repr
        'pdb_id':           str
        'protein_chain':    str
        'binding_site_id':  str
        'ligand_id':        str
        'ligand_path':      str            — source SDF path
        'n_atoms':          int            — heavy + H atoms used
    }

Usage
─────
# smoke-test (first 10 pairs, single GPU)
python3 biolip_unimol_extract.py \\
    --ligand_features /hpc2ssd/JH_DATA/spooler/shuang886/biolip/ligand_features \\
    --output          /hpc2hdd/home/shuang886/jhspoolers/biolip/unimol_features \\
    --device cuda:0 --max 10

# full run multi-GPU
python3 biolip_unimol_extract.py \\
    --ligand_features /hpc2ssd/JH_DATA/spooler/shuang886/biolip/ligand_features \\
    --output          /hpc2hdd/home/shuang886/jhspoolers/biolip/unimol_features \\
    --devices 0,1,2,3 --batch_size 64
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── SDF loader (runs in worker processes) ─────────────────────────────────────

def _load_sdf(sdf_path: str) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """Load SDF and return (stem, atom_symbols, coords). None on failure."""
    try:
        from rdkit import Chem
        path = Path(sdf_path)
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        conf   = mol.GetConformer()
        atoms  = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        return path.stem, atoms, coords
    except Exception:
        return None


# ── task discovery ─────────────────────────────────────────────────────────────

def discover_tasks(
    ligand_features_root: Path,
    output_root: Path,
    overwrite: bool = False,
    excluded_ids: Optional[set] = None,
) -> List[dict]:
    """
    Scan ligand_features_root for .pt files and build task list.
    Each task dict: {stem, sdf_path, out_path, meta}.
    """
    tasks = []
    excluded_ids = excluded_ids or set()

    for pdb_dir in sorted(ligand_features_root.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdb_id = pdb_dir.name
        if pdb_id in excluded_ids:
            continue

        for pt_file in sorted(pdb_dir.glob('*.pt')):
            stem = pt_file.stem  # e.g. 101m_A_BS01_HEM
            out_path = output_root / pdb_id / f'{stem}_unimol.pt'
            if out_path.exists() and not overwrite:
                continue

            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
            except Exception:
                continue

            sdf_path = data.get('ligand_path')
            if not sdf_path or not Path(sdf_path).exists():
                continue

            tasks.append({
                'stem':             stem,
                'sdf_path':         sdf_path,
                'out_path':         str(out_path),
                'pdb_id':           data.get('pdb_id', pdb_id),
                'protein_chain':    data.get('protein_chain', ''),
                'binding_site_id':  data.get('binding_site_id', ''),
                'ligand_id':        data.get('ligand_id', ''),
            })

    return tasks


# ── Uni-Mol batch inference ────────────────────────────────────────────────────

def _run_unimol_batch(
    mol_data: List[Tuple[str, List[str], np.ndarray]],
    model,
) -> Dict[str, np.ndarray]:
    """Run Uni-Mol on a batch. Returns {stem: embedding [512]}."""
    batch_input = {
        'atoms':       [atoms  for _, atoms,  _ in mol_data],
        'coordinates': [coords for _, _,  coords in mol_data],
    }
    try:
        reprs = model.get_repr(
            batch_input,
            return_atomic_reprs=False,
            return_tensor=True,
        )  # [N, 512]
        reprs = reprs.detach().cpu().numpy()
    except Exception as e:
        log.warning(f'Batch inference failed ({e}), falling back to per-sample')
        reprs_list = []
        for _, atoms, coords in mol_data:
            r = model.get_repr(
                {'atoms': atoms, 'coordinates': coords},
                return_atomic_reprs=False,
                return_tensor=True,
            )
            reprs_list.append(r.detach().cpu().numpy()[0])
        reprs = np.stack(reprs_list, axis=0)

    out = {}
    for i, (stem, _, _) in enumerate(mol_data):
        emb = np.array(reprs[i], dtype=np.float32)
        if emb.shape != (512,) or not np.isfinite(emb).all() or np.abs(emb).sum() <= 1e-6:
            raise RuntimeError(
                f'Invalid embedding for {stem}: shape={emb.shape}, '
                f'finite={np.isfinite(emb).all()}, abs_sum={np.abs(emb).sum():.3e}'
            )
        out[stem] = emb
    return out


# ── worker (one per device) ────────────────────────────────────────────────────

def _worker(
    tasks       : List[dict],
    output_root : Path,
    device_str  : str,
    batch_size  : int,
    tag         : str,
    failure_tsv : Path,
    cpu_threads : int = 0,
):
    if device_str == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
    else:
        idx = device_str.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = idx

    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        print(f'[{tag}] ERROR: unimol_tools not installed', flush=True)
        raise SystemExit(1)

    model = UniMolRepr(data_type='molecule', remove_hs=False)
    log.info(f'[{tag}] Uni-Mol model loaded on {device_str}')

    total    = len(tasks)
    failures : List[Tuple[str, str]] = []
    n_done   = 0
    n_skip   = 0

    # Process in batches, load SDF just-in-time
    for b_start in range(0, total, batch_size):
        batch_tasks = tasks[b_start: b_start + batch_size]

        # Load SDFs
        mol_data   : List[Tuple[str, List[str], np.ndarray, dict]] = []
        load_failed: List[dict] = []
        for task in batch_tasks:
            result = _load_sdf(task['sdf_path'])
            if result is None:
                load_failed.append(task)
                n_skip += 1
            else:
                stem, atoms, coords = result
                mol_data.append((stem, atoms, coords, task))

        for task in load_failed:
            failures.append((task['stem'], f"SDF load failed: {task['sdf_path']}"))

        if not mol_data:
            continue

        # Uni-Mol inference
        try:
            embeddings = _run_unimol_batch(
                [(stem, atoms, coords) for stem, atoms, coords, _ in mol_data],
                model,
            )
        except Exception as e:
            for stem, _, _, task in mol_data:
                failures.append((task['stem'], f'Uni-Mol failed: {e}'))
            continue

        # Save
        for stem, atoms, coords, task in mol_data:
            emb = embeddings.get(stem)
            if emb is None:
                failures.append((task['stem'], 'Missing embedding after inference'))
                continue
            out_path = Path(task['out_path'])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.parent / f'.{stem}_{uuid.uuid4().hex}.tmp.pt'
            try:
                torch.save({
                    'embedding':       torch.from_numpy(emb),
                    'pdb_id':          task['pdb_id'],
                    'protein_chain':   task['protein_chain'],
                    'binding_site_id': task['binding_site_id'],
                    'ligand_id':       task['ligand_id'],
                    'ligand_path':     task['sdf_path'],
                    'n_atoms':         len(atoms),
                }, tmp)
                os.replace(tmp, out_path)
                n_done += 1
            except Exception as e:
                tmp.unlink(missing_ok=True)
                failures.append((task['stem'], f'Save error: {e}'))

        idx_done = min(b_start + batch_size, total)
        print(f'[{tag} {idx_done}/{total}] done={n_done} fail={len(failures)} skip={n_skip}',
              flush=True)

    if failures:
        failure_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_tsv, 'w') as f:
            f.write('stem\terror\n')
            for stem, err in failures:
                f.write(f'{stem}\t{err.replace(chr(9)," ").replace(chr(10)," ")}\n')
        print(f'[{tag}] {len(failures)} failures → {failure_tsv}', flush=True)

    raise SystemExit(1 if failures else 0)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_devices(s: str) -> List[str]:
    out = []
    for p in s.split(','):
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            out.append(f'cuda:{p}')
        elif p == 'cuda':
            out.append('cuda:0')
        else:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description='BioLip Uni-Mol ligand embedding extractor.'
    )
    parser.add_argument('--ligand_features', required=True,
                        help='Root of ligand_features dir ({pdb_id}/*.pt)')
    parser.add_argument('--output', required=True,
                        help='Output root for _unimol.pt files')
    parser.add_argument('--device', default='cuda',
                        help='Single-device run: cuda, cuda:0, cpu')
    parser.add_argument('--devices', default=None,
                        help='Multi-device: "0,1,2,3" or "0,1,cpu"')
    parser.add_argument('--cpu_threads', type=int, default=32,
                        help='Torch CPU threads for CPU worker (default 32)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Uni-Mol inference batch size (default 64)')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N tasks (testing)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--excluded_ids', default=None,
                        help='Text file of pdb_ids to skip (one per line)')
    args = parser.parse_args()

    ligand_root = Path(args.ligand_features).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    excluded_ids = set()
    if args.excluded_ids:
        excluded_ids = {
            l.strip() for l in Path(args.excluded_ids).read_text().splitlines()
            if l.strip()
        }
        log.info(f'Excluded IDs loaded: {len(excluded_ids):,}')

    log.info('Discovering tasks…')
    t0 = time.time()
    tasks = discover_tasks(ligand_root, output_root, args.overwrite, excluded_ids)
    log.info(f'  {len(tasks):,} tasks found in {time.time()-t0:.1f}s')

    if args.max:
        tasks = tasks[:args.max]
        log.info(f'  Limiting to first {len(tasks)} tasks')

    if not tasks:
        log.info('Nothing to do.')
        return 0

    log.info(f'Output root: {output_root}')

    devices = _parse_devices(args.devices) if args.devices else [args.device]
    log.info(f'Devices: {devices}')

    # Split tasks round-robin across devices
    splits: List[List[dict]] = [[] for _ in devices]
    for i, task in enumerate(tasks):
        splits[i % len(devices)].append(task)

    if len(devices) == 1:
        # Run in-process (easier debugging)
        _worker(
            splits[0], output_root, devices[0],
            args.batch_size, devices[0],
            output_root / f'failures_{devices[0].replace(":","_")}.tsv',
            args.cpu_threads if devices[0] == 'cpu' else 0,
        )
        return 0

    ctx   = mp.get_context('spawn')
    procs = []
    for dev, chunk in zip(devices, splits):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker,
            args=(
                chunk, output_root, dev, args.batch_size, dev,
                output_root / f'failures_{dev.replace(":","_")}.tsv',
                args.cpu_threads if dev == 'cpu' else 0,
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
