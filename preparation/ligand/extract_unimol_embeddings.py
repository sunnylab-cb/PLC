#!/usr/bin/env python3
"""
離線提取 Uni-Mol 分子嵌入

從源頭 SDF 文件（/home/husrcf/Code/ProtBind/New/）直接讀取，
避免與 ligand_features_v5 中的多種座標欄位混淆。

輸入目錄結構:
    {source_root}/{year_range}/{pdb_id}/{pdb_id}_ligand.sdf

輸出目錄結構:
    {output_root}/{year_range}/{pdb_id}/{pdb_id}_unimol.pt
    內容: {'embedding': tensor[512], 'pdb_id': str, 'n_atoms': int}

安裝依賴:
    conda run -n AIAA pip install unimol-tools

使用方法:
    # 提取全部訓練資料
    conda run -n AIAA python preparation/ligand/extract_unimol_embeddings.py \\
        --source_root /home/husrcf/Code/ProtBind/New \\
        --output_root data/unimol_embeddings \\
        --batch_size 64 --workers 4

    # 只跑某個 year_range（測試用）
    conda run -n AIAA python preparation/ligand/extract_unimol_embeddings.py \\
        --source_root /home/husrcf/Code/ProtBind/New \\
        --output_root data/unimol_embeddings \\
        --year_ranges 1981-2000 --batch_size 32
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

YEAR_RANGES = ['1981-2000', '2001-2010', '2011-2019']


# ---------------------------------------------------------------------------
# SDF loading (runs in worker processes)
# ---------------------------------------------------------------------------

def _load_one_sdf(sdf_path: str) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """
    Load one SDF file and return (pdb_id, atom_symbols, coords).
    Returns None on failure.

    使用 removeHs=False：Uni-Mol 在包含 H 原子的情況下預訓練，
    移除 H 會降低嵌入品質。
    """
    try:
        from rdkit import Chem
        path = Path(sdf_path)
        pdb_id = path.parent.name

        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if mol is None:
            # Try without sanitize as fallback
            mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        if mol is None or mol.GetNumConformers() == 0:
            return None

        conf   = mol.GetConformer()
        atoms  = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        return pdb_id, atoms, coords

    except Exception as e:
        log.debug(f"Failed to load {sdf_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Scan source directory
# ---------------------------------------------------------------------------

def scan_sdf_files(
    source_root: Path,
    output_root: Path,
    year_ranges: List[str],
    overwrite: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Scan source directory for SDF files.

    Returns list of (sdf_path, output_path, year_range) for files
    that need processing (not yet done, unless overwrite=True).
    """
    tasks = []
    for yr in year_ranges:
        yr_dir = source_root / yr
        if not yr_dir.exists():
            log.warning(f"Year range dir not found: {yr_dir}")
            continue
        for pdb_dir in sorted(yr_dir.iterdir()):
            if not pdb_dir.is_dir():
                continue
            pdb_id  = pdb_dir.name
            sdf_path = pdb_dir / f'{pdb_id}_ligand.sdf'
            if not sdf_path.exists():
                continue
            out_dir  = output_root / yr / pdb_id
            out_path = out_dir / f'{pdb_id}_unimol.pt'
            if not overwrite and out_path.exists():
                continue
            tasks.append((str(sdf_path), str(out_path), yr))

    return tasks


# ---------------------------------------------------------------------------
# Uni-Mol batch inference
# ---------------------------------------------------------------------------

def run_unimol_batch(
    mol_data: List[Tuple[str, List[str], np.ndarray]],
    model,
) -> Dict[str, np.ndarray]:
    """
    Run Uni-Mol on a batch of molecules.

    mol_data: list of (pdb_id, atom_symbols, coords_np)
    Returns: {pdb_id: embedding_np [512]}
    """
    # unimol-tools UniMolRepr.get_repr expects either:
    #   - a list of SMILES strings, or
    #   - a single dict with batched custom conformers:
    #       {'atoms': [[...], ...], 'coordinates': [coords_np, ...]}
    # Passing a list of dicts is rejected by current unimol-tools and was the
    # cause of silently written all-zero embeddings in the previous extractor.
    batch_input = {
        'atoms':       [atoms for _, atoms, _ in mol_data],
        'coordinates': [coords for _, _, coords in mol_data],
    }
    try:
        cls_reprs = model.get_repr(
            batch_input,
            return_atomic_reprs=False,
            return_tensor=True,
        )  # torch.Tensor [N, 512]
        cls_reprs = cls_reprs.detach().cpu().numpy()
    except Exception as e:
        log.warning(f"Uni-Mol batch failed: {e}, falling back to per-sample")
        cls_reprs = []
        for _, atoms, coords in mol_data:
            try:
                r = model.get_repr(
                    {'atoms': atoms, 'coordinates': coords},
                    return_atomic_reprs=False,
                    return_tensor=True,
                )
                cls_reprs.append(r.detach().cpu().numpy()[0])
            except Exception as e2:
                raise RuntimeError(f"Uni-Mol per-sample inference failed: {e2}") from e2

    embeddings = {}
    for i, (pdb_id, _, _) in enumerate(mol_data):
        emb = cls_reprs[i]
        if emb is None:
            raise RuntimeError(f"Uni-Mol returned None embedding for {pdb_id}")
        emb = np.array(emb, dtype=np.float32)
        if emb.shape != (512,) or not np.isfinite(emb).all() or np.abs(emb).sum() <= 1e-6:
            raise RuntimeError(
                f"Invalid Uni-Mol embedding for {pdb_id}: "
                f"shape={emb.shape}, finite={np.isfinite(emb).all()}, "
                f"abs_sum={np.abs(emb).sum():.3e}"
            )
        embeddings[pdb_id] = emb
    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='離線提取 Uni-Mol 分子嵌入（從源頭 SDF）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--source_root', default='/home/husrcf/Code/ProtBind/New',
                        help='源頭 SDF 根目錄')
    parser.add_argument('--output_root', default='data/unimol_embeddings',
                        help='輸出根目錄（相對於 PLC/）')
    parser.add_argument('--year_ranges', nargs='+', default=YEAR_RANGES,
                        help='要處理的 year_range 列表')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Uni-Mol 推理 batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='SDF 讀取的並行進程數')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆蓋已存在的輸出文件')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Uni-Mol 推理設備')
    args = parser.parse_args()

    # Resolve paths relative to script location (PLC root)
    plc_root    = Path(__file__).resolve().parents[2]
    source_root = Path(args.source_root)
    output_root = plc_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Source root : {source_root}")
    log.info(f"Output root : {output_root}")
    log.info(f"Year ranges : {args.year_ranges}")

    # ── Scan tasks ────────────────────────────────────────────────────────────
    log.info("Scanning SDF files...")
    tasks = scan_sdf_files(
        source_root, output_root, args.year_ranges, args.overwrite
    )
    log.info(f"  {len(tasks)} SDF files to process")
    if not tasks:
        log.info("Nothing to do. Use --overwrite to re-extract.")
        return

    # ── Load Uni-Mol model ────────────────────────────────────────────────────
    log.info("Loading Uni-Mol model...")
    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        log.error(
            "unimol_tools not installed.\n"
            "Run: conda run -n AIAA pip install unimol-tools"
        )
        sys.exit(1)

    # UniMolRepr handles device selection internally via environment or config
    # We set CUDA_VISIBLE_DEVICES or force CPU if needed
    if args.device == 'cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model = UniMolRepr(data_type='molecule', remove_hs=False)
    log.info("Uni-Mol model loaded.")

    # ── Load SDF files in parallel, infer in batches ─────────────────────────
    sdf_paths   = [t[0] for t in tasks]
    out_paths   = {t[0]: t[1] for t in tasks}   # sdf_path → out_path
    year_of     = {t[0]: t[2] for t in tasks}    # sdf_path → yr

    loaded_mols: List[Tuple[str, List[str], np.ndarray]] = []
    failed      = []
    n_skip      = 0

    log.info(f"Loading SDF files with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_load_one_sdf, p): p for p in sdf_paths}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Loading SDF', unit='mol'):
            sdf_path = futures[fut]
            result   = fut.result()
            if result is None:
                failed.append(sdf_path)
                n_skip += 1
            else:
                loaded_mols.append((result[0], result[1], result[2],
                                    out_paths[sdf_path], year_of[sdf_path]))

    log.info(f"Loaded {len(loaded_mols)} / {len(tasks)}  (failed: {n_skip})")
    if failed:
        fail_log = output_root / 'failed_sdf.txt'
        fail_log.write_text('\n'.join(failed))
        log.warning(f"Failed SDF list saved to: {fail_log}")

    # ── Uni-Mol batch inference ───────────────────────────────────────────────
    t0       = time.time()
    n_done   = 0
    n_batches = (len(loaded_mols) + args.batch_size - 1) // args.batch_size

    for b_start in tqdm(range(0, len(loaded_mols), args.batch_size),
                        total=n_batches, desc='Uni-Mol inference', unit='batch'):
        batch = loaded_mols[b_start: b_start + args.batch_size]

        # pdb_id, atoms, coords for model; out_path and yr for saving
        mol_input = [(pdb_id, atoms, coords)
                     for pdb_id, atoms, coords, _, _ in batch]
        embeddings = run_unimol_batch(mol_input, model)

        # Save each embedding
        for pdb_id, atoms, coords, out_path, yr in batch:
            emb_np = embeddings.get(pdb_id)
            if emb_np is None:
                raise RuntimeError(f"Missing Uni-Mol embedding after inference for {pdb_id}")
            out_file = Path(out_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'embedding': torch.from_numpy(emb_np),  # [512] float32
                    'pdb_id':    pdb_id,
                    'n_atoms':   len(atoms),
                },
                out_file,
            )
            n_done += 1

    elapsed = time.time() - t0
    log.info(
        f"Done. {n_done} embeddings saved to {output_root}  "
        f"({elapsed:.0f}s, {n_done/max(elapsed, 1e-6):.1f} mol/s)"
    )


if __name__ == '__main__':
    main()
