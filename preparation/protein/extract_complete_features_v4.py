#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent extraction entrypoint for complete protein features V4.

Modes:
1. fit-stats  : compute train-set-level normalization stats from raw spatial scalar features
2. extract    : extract normalized V4 features using a fixed stats JSON
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Optional

import torch
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from preparation.protein.protein_features_v4 import (
    extract_complete_features_v4,
    fit_normalization_stats,
    get_spatial_scalar_feature_names_v4,
    load_normalization_stats,
    save_normalization_stats,
)


def _collect_input_files(pdb_root: Path, pattern: str = "*_protein*.pdb", max_files: Optional[int] = None) -> list[Path]:
    files = sorted(pdb_root.rglob(pattern))
    if max_files:
        files = files[:max_files]
    return files


def process_single_pdb_extract(args_tuple):
    pdb_path, output_dir, chain_id, mode, stats_json, overwrite = args_tuple
    stem = pdb_path.stem
    import re
    m = re.match(r'^(.+?)(_protein)?(_conf\d+)?$', stem)
    pdb_id = m.group(1) if m else stem
    conf_suffix = m.group(3) if (m and m.group(3)) else ''
    out_stem = f"{pdb_id}_protein{conf_suffix}"

    year_range = None
    for part in pdb_path.parts:
        if '-' in part and any(c.isdigit() for c in part):
            split = part.split('-')
            if len(split) == 2 and split[0].isdigit() and split[1].isdigit():
                year_range = part
                break

    if year_range:
        out_dir = output_dir / year_range / pdb_id
    else:
        out_dir = output_dir / pdb_id
    out_file = out_dir / f"{out_stem}.pt"

    # Fast path: skip expensive feature extraction if output already exists.
    if out_file.exists() and not overwrite:
        return True, f"{pdb_id}: SKIP (already exists)"

    try:
        stats = load_normalization_stats(Path(stats_json)) if stats_json else None
        features = extract_complete_features_v4(pdb_path, chain_id=chain_id, mode=mode, normalization_stats=stats)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(features, out_file)
        return True, f"{pdb_id}: OK ({features['n_residues']} residues)"
    except Exception as e:
        import traceback
        return False, f"{pdb_path.name}: {str(e)[:120]}\n{traceback.format_exc()[:300]}"


def process_single_pdb_raw(args_tuple):
    pdb_path, chain_id, mode = args_tuple
    try:
        features = extract_complete_features_v4(pdb_path, chain_id=chain_id, mode=mode, normalization_stats=None)
        raw = features['spatial_scalar_raw_features'].numpy()
        return True, raw
    except Exception:
        return False, None


def fit_stats_mode(
    pdb_root: Path,
    output_json: Path,
    mode: str = 'complex',
    chain_id: Optional[str] = None,
    pattern: str = "*_protein*.pdb",
    max_files: Optional[int] = None,
    workers: Optional[int] = None,
) -> None:
    files = _collect_input_files(pdb_root, pattern=pattern, max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No protein files found in {pdb_root} with pattern {pattern}")

    if workers is None:
        workers = min(mp.cpu_count(), len(files))

    tasks = [(p, chain_id, mode) for p in files]

    raw_features = []
    if workers == 1:
        iterator = tqdm(tasks, desc="Fitting stats", unit="file") if TQDM_AVAILABLE else tasks
        for task in iterator:
            ok, raw = process_single_pdb_raw(task)
            if ok and raw is not None:
                raw_features.append(raw)
    else:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(process_single_pdb_raw, tasks)
        iterator = tqdm(results, desc="Fitting stats", unit="file") if TQDM_AVAILABLE else results
        for ok, raw in iterator:
            if ok and raw is not None:
                raw_features.append(raw)

    if not raw_features:
        raise RuntimeError("No valid raw features collected for stats fitting")

    stats = fit_normalization_stats(raw_features, get_spatial_scalar_feature_names_v4())
    save_normalization_stats(stats, output_json)
    print(f"Saved stats to: {output_json}")
    print(f"Feature count: {len(stats['feature_names'])}")


def extract_mode(
    pdb_root: Path,
    output_dir: Path,
    stats_json: Optional[Path],
    mode: str = 'complex',
    chain_id: Optional[str] = None,
    pattern: str = "*_protein*.pdb",
    max_files: Optional[int] = None,
    workers: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    files = _collect_input_files(pdb_root, pattern=pattern, max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No protein files found in {pdb_root} with pattern {pattern}")

    if workers is None:
        workers = min(mp.cpu_count(), len(files))

    print(f"Processing proteins from: {pdb_root}")
    print(f"Output directory: {output_dir}")
    print("Feature dimensions: physchem=62, spatial_scalar=89, spatial_vector=8x3")
    print(f"Using stats JSON: {stats_json if stats_json else 'NONE (raw scalar features)'}")
    print(f"Using {workers} worker processes")

    tasks = [(p, output_dir, chain_id, mode, str(stats_json) if stats_json else None, overwrite) for p in files]

    success_count = 0
    fail_count = 0
    skip_count = 0
    if workers == 1:
        iterator = tqdm(tasks, desc="Extracting", unit="file") if TQDM_AVAILABLE else tasks
        for task in iterator:
            success, message = process_single_pdb_extract(task)
            if "SKIP" in message:
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1
            if TQDM_AVAILABLE:
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"{status} {message}")
    else:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(process_single_pdb_extract, tasks)
        iterator = tqdm(results, desc="Extracting", unit="file") if TQDM_AVAILABLE else results
        for success, message in iterator:
            if "SKIP" in message:
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1
            if TQDM_AVAILABLE:
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"{status} {message}")

    print("\n=== Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total:   {len(files)}")


def main():
    parser = argparse.ArgumentParser(description="Independent protein complete features V4")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_fit = subparsers.add_parser("fit-stats", help="fit train-set-level normalization stats")
    p_fit.add_argument("--input", type=str, required=True, help="protein PDB root")
    p_fit.add_argument("--output-json", type=str, required=True, help="stats JSON output path")
    p_fit.add_argument("--mode", type=str, default="complex", choices=["complex", "chain"])
    p_fit.add_argument("--chain-id", type=str, default=None)
    p_fit.add_argument("--pattern", type=str, default="*_protein*.pdb")
    p_fit.add_argument("--max", type=int, default=None)
    p_fit.add_argument("--workers", type=int, default=None)

    p_ext = subparsers.add_parser("extract", help="extract normalized V4 features")
    p_ext.add_argument("--input", type=str, required=True, help="protein PDB root")
    p_ext.add_argument("--output", type=str, required=True, help="feature output root")
    p_ext.add_argument("--stats-json", type=str, default=None, help="normalization stats JSON")
    p_ext.add_argument("--mode", type=str, default="complex", choices=["complex", "chain"])
    p_ext.add_argument("--chain-id", type=str, default=None)
    p_ext.add_argument("--pattern", type=str, default="*_protein*.pdb")
    p_ext.add_argument("--max", type=int, default=None)
    p_ext.add_argument("--workers", type=int, default=None)
    p_ext.add_argument("--overwrite", action="store_true", help="overwrite existing feature files")

    args = parser.parse_args()

    if args.command == "fit-stats":
        fit_stats_mode(
            pdb_root=Path(args.input),
            output_json=Path(args.output_json),
            mode=args.mode,
            chain_id=args.chain_id,
            pattern=args.pattern,
            max_files=args.max,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    elif args.command == "extract":
        extract_mode(
            pdb_root=Path(args.input),
            output_dir=Path(args.output),
            stats_json=Path(args.stats_json) if args.stats_json else None,
            mode=args.mode,
            chain_id=args.chain_id,
            pattern=args.pattern,
            max_files=args.max,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
