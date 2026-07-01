#!/usr/bin/env python3
"""
Offline feature normalization / clamp utility.

This script rewrites selected feature files into a new output root while keeping
the original directory structure and all non-feature keys intact.  It never
modifies source files in-place.

Default dry-run style usage (sample a few files into tmp/):
    python scripts/apply_feature_normalization.py \
      --data-root data \
      --stats data/stats/stats_output.pt \
      --output-root tmp/feature_norm_check \
      --features ligand unimol spatial \
      --max-files 5

Full export for V11m3/V12 configs:
    python scripts/apply_feature_normalization.py \
      --data-root data \
      --stats data/stats/stats_output.pt \
      --output-root data \
      --features ligand unimol spatial \
      --all \
      --ligand-out-dir ligand_features_v5_norm \
      --unimol-out-dir unimol_embeddings_norm \
      --protein-out-dirs complete_features_v4_nma_norm complete_features_v4_norm

Normalization policy from full-dataset stats:
  - ligand atom_features[109]: z-score CONTINUOUS_NORM dims only.
  - Uni-Mol embedding[512]: z-score all dims.
  - protein spatial_scalar_features[89]: keep existing z-score, clamp four
    extreme dims to [-5, 5].
  - ESM2 and physchem are intentionally not processed here.
  - ESM3 embedding[1536]: z-score all dims when requested.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


DEFAULT_SPATIAL_CLAMP_DIMS = [62, 76, 78, 79]


def _as_tensor(x, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def _continuous_norm_dims(stats: dict) -> List[int]:
    """Match scripts/compute_feature_stats.py CONTINUOUS_NORM decision."""
    std = np.asarray(stats["std"])
    mn = np.asarray(stats["min"])
    mx = np.asarray(stats["max"])
    all_int = np.asarray(stats["all_int"])
    rng = mx - mn
    dims = np.where((~all_int) & ((rng > 5.0) | (std > 2.0)))[0]
    return [int(i) for i in dims.tolist()]


def _zscore_dims(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    if not dims:
        return x
    out = x.clone().float()
    idx = torch.tensor(list(dims), dtype=torch.long)
    denom = std[idx].clamp_min(1e-6)
    out[..., idx] = (out[..., idx] - mean[idx]) / denom
    return out


def _zscore_all(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.float() - mean) / std.clamp_min(1e-6)


def _summarize_delta(before: torch.Tensor, after: torch.Tensor, dims: Sequence[int]) -> str:
    if not dims:
        return "no dims"
    idx = torch.tensor(list(dims), dtype=torch.long)
    b = before.float()[..., idx].reshape(-1, len(dims))
    a = after.float()[..., idx].reshape(-1, len(dims))
    return (
        f"dims={len(dims)} "
        f"before_absmax={b.abs().max().item():.4g} before_std_mean={b.std(dim=0, unbiased=False).mean().item():.4g} "
        f"after_absmax={a.abs().max().item():.4g} after_std_mean={a.std(dim=0, unbiased=False).mean().item():.4g}"
    )


def _relative_destination(src: Path, in_root: Path, out_root: Path, out_dir_name: str) -> Path:
    rel = src.relative_to(in_root)
    return out_root / out_dir_name / rel


def _save_pt(obj, dst: Path, *, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        print(f"[skip exists] {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, dst)
    return True


def _sample_paths(paths: Sequence[Path], max_files: int | None) -> List[Path]:
    paths = sorted(paths)
    if max_files is None:
        return list(paths)
    return list(paths[:max_files])


def _run_parallel(paths: Sequence[Path], worker_fn, *, workers: int, log_every: int, prefix: str) -> None:
    if workers <= 1:
        for i, src in enumerate(paths, 1):
            msg = worker_fn(src)
            if msg and (i <= 5 or i == len(paths) or (log_every > 0 and i % log_every == 0)):
                print(f"{msg} ({i}/{len(paths)})")
        return

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(worker_fn, src): src for src in paths}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as exc:
                raise RuntimeError(f"{prefix} failed for {futs[fut]}: {exc}") from exc
            if msg and (done <= 5 or done == len(paths) or (log_every > 0 and done % log_every == 0)):
                print(f"{msg} ({done}/{len(paths)})")


def process_ligands(args, stats: dict) -> None:
    in_root = Path(args.data_root) / args.ligand_in_dir
    paths = _sample_paths(list(in_root.rglob("*_ligand.pt")), args.max_files)
    if not paths:
        print(f"[ligand] no files found under {in_root}")
        return

    ligand_stats = stats["ligand"]
    dims = _continuous_norm_dims(ligand_stats)
    mean = _as_tensor(ligand_stats["mean"])
    std = _as_tensor(ligand_stats["std"])

    print(
        f"[ligand] files={len(paths)} source={in_root} out_dir={args.ligand_out_dir} "
        f"norm_dims={len(dims)} workers={args.workers}"
    )

    def _worker(src: Path) -> str:
        obj = torch.load(src, map_location="cpu", weights_only=False)
        if "atom_features" not in obj:
            raise KeyError(f"{src} missing atom_features")
        before = obj["atom_features"].float()
        if before.shape[-1] != mean.numel():
            raise ValueError(f"{src} atom_features dim={before.shape[-1]} expected={mean.numel()}")
        new_obj = copy.deepcopy(obj)
        after = _zscore_dims(before, mean, std, dims)
        new_obj["atom_features"] = after
        new_obj["feature_normalization"] = {
            "ligand_atom_features": "zscore_continuous_norm_dims",
            "norm_dims": dims,
            "stats_source": str(args.stats),
        }
        dst = _relative_destination(src, in_root, Path(args.output_root), args.ligand_out_dir)
        wrote = _save_pt(new_obj, dst, overwrite=args.overwrite)
        return f"[ligand] {'wrote' if wrote else 'kept '} {dst} | {_summarize_delta(before, after, dims)}"

    _run_parallel(paths, _worker, workers=args.workers, log_every=args.log_every, prefix="[ligand]")


def process_unimol(args, stats: dict) -> None:
    in_root = Path(args.data_root) / args.unimol_in_dir
    paths = _sample_paths(list(in_root.rglob("*_unimol.pt")), args.max_files)
    if not paths:
        print(f"[unimol] no files found under {in_root}")
        return

    unimol_stats = stats["unimol"]
    mean = _as_tensor(unimol_stats["mean"])
    std = _as_tensor(unimol_stats["std"])
    dims = list(range(mean.numel()))

    print(
        f"[unimol] files={len(paths)} source={in_root} out_dir={args.unimol_out_dir} "
        f"norm_dims=all/{len(dims)} workers={args.workers}"
    )

    def _worker(src: Path) -> str:
        obj = torch.load(src, map_location="cpu", weights_only=True)
        is_dict = isinstance(obj, dict)
        embed = obj.get("embedding") if is_dict else obj
        if not isinstance(embed, torch.Tensor):
            raise TypeError(f"{src} embedding is {type(embed).__name__}")
        before = embed.float()
        if before.shape != mean.shape:
            raise ValueError(f"{src} embedding shape={tuple(before.shape)} expected={tuple(mean.shape)}")
        after = _zscore_all(before, mean, std)
        if is_dict:
            new_obj = copy.deepcopy(obj)
            new_obj["embedding"] = after
            new_obj["feature_normalization"] = {
                "unimol_embedding": "zscore_all_dims",
                "stats_source": str(args.stats),
            }
        else:
            new_obj = after
        dst = _relative_destination(src, in_root, Path(args.output_root), args.unimol_out_dir)
        wrote = _save_pt(new_obj, dst, overwrite=args.overwrite)
        return f"[unimol] {'wrote' if wrote else 'kept '} {dst} | {_summarize_delta(before, after, dims)}"

    _run_parallel(paths, _worker, workers=args.workers, log_every=args.log_every, prefix="[unimol]")


def process_esm3(args, stats: dict) -> None:
    in_root = Path(args.data_root) / args.esm3_in_dir
    paths = _sample_paths(list(in_root.rglob("*_protein.pt")), args.max_files)
    if not paths:
        print(f"[esm3] no files found under {in_root}")
        return

    esm3_stats = stats["esm3"]
    mean = _as_tensor(esm3_stats["mean"])
    std = _as_tensor(esm3_stats["std"])
    dims = list(range(mean.numel()))

    print(
        f"[esm3] files={len(paths)} source={in_root} out_dir={args.esm3_out_dir} "
        f"norm_dims=all/{len(dims)} workers={args.workers}"
    )

    def _worker(src: Path) -> str:
        obj = torch.load(src, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "embeddings" not in obj:
            raise KeyError(f"{src} missing embeddings")
        before = obj["embeddings"].float()
        if before.shape[-1] != mean.numel():
            raise ValueError(f"{src} embeddings dim={before.shape[-1]} expected={mean.numel()}")
        new_obj = copy.deepcopy(obj)
        after = _zscore_all(before, mean, std)
        new_obj["embeddings"] = after
        new_obj["feature_normalization"] = {
            "esm3_embeddings": "zscore_all_dims",
            "stats_source": str(args.stats),
        }
        dst = _relative_destination(src, in_root, Path(args.output_root), args.esm3_out_dir)
        wrote = _save_pt(new_obj, dst, overwrite=args.overwrite)
        return f"[esm3] {'wrote' if wrote else 'kept '} {dst} | {_summarize_delta(before, after, dims)}"

    _run_parallel(paths, _worker, workers=args.workers, log_every=args.log_every, prefix="[esm3]")


def _protein_patterns(root: Path) -> List[Path]:
    files = list(root.rglob("*_protein.pt"))
    files.extend(root.rglob("*_protein_conf*.pt"))
    return sorted(set(files))


def process_spatial(args, stats: dict) -> None:
    out_dirs = list(args.protein_out_dirs)
    if len(out_dirs) != len(args.protein_in_dirs):
        raise ValueError("--protein-out-dirs must have the same length as --protein-in-dirs")
    dims = [int(d) for d in args.spatial_clamp_dims]

    for in_dir, out_dir in zip(args.protein_in_dirs, out_dirs):
        in_root = Path(args.data_root) / in_dir
        paths = _sample_paths(_protein_patterns(in_root), args.max_files)
        if not paths:
            print(f"[spatial] no files found under {in_root}")
            continue
        print(f"[spatial] files={len(paths)} source={in_root} out_dir={out_dir} clamp_dims={dims}")
        for src in paths:
            obj = torch.load(src, map_location="cpu", weights_only=False)
            if "spatial_scalar_features" not in obj:
                raise KeyError(f"{src} missing spatial_scalar_features")
            before = obj["spatial_scalar_features"].float()
            if before.shape[-1] <= max(dims):
                raise ValueError(f"{src} spatial dim={before.shape[-1]} cannot clamp dims={dims}")
            new_obj = copy.deepcopy(obj)
            after = before.clone()
            after[..., dims] = after[..., dims].clamp(args.spatial_clamp_min, args.spatial_clamp_max)
            new_obj["spatial_scalar_features"] = after
            new_obj["feature_normalization"] = {
                "spatial_scalar_features": "clamp_extreme_dims",
                "clamp_dims": dims,
                "clamp_range": [args.spatial_clamp_min, args.spatial_clamp_max],
                "stats_source": str(args.stats),
            }
            dst = _relative_destination(src, in_root, Path(args.output_root), out_dir)
            wrote = _save_pt(new_obj, dst, overwrite=args.overwrite)
            changed = (before[..., dims] != after[..., dims]).sum().item()
            print(
                f"[spatial] {'wrote' if wrote else 'kept '} {dst} | "
                f"dims={len(dims)} changed_values={changed} "
                f"before_absmax={before[..., dims].abs().max().item():.4g} "
                f"after_absmax={after[..., dims].abs().max().item():.4g}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--stats", default="data/stats/stats_output.pt")
    parser.add_argument("--output-root", default="tmp/feature_norm_check")
    parser.add_argument("--features", nargs="+", default=["ligand", "unimol", "spatial"],
                        choices=["ligand", "unimol", "spatial", "esm3"])
    parser.add_argument("--max-files", type=int, default=5,
                        help="Files per feature type for dry-run. Ignored when --all is set.")
    parser.add_argument("--all", action="store_true", help="Process all matching files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in output root if present.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel file workers for ligand/UniMol export.")
    parser.add_argument("--log-every", type=int, default=500, help="Progress print interval for full export.")

    parser.add_argument("--ligand-in-dir", default="ligand_features_v5")
    parser.add_argument("--ligand-out-dir", default="ligand_features_v5_norm")
    parser.add_argument("--unimol-in-dir", default="unimol_embeddings")
    parser.add_argument("--unimol-out-dir", default="unimol_embeddings_norm")
    parser.add_argument("--esm3-in-dir", default="esm3_latents")
    parser.add_argument("--esm3-out-dir", default="esm3_latents_norm")
    parser.add_argument("--protein-in-dirs", nargs="+", default=["complete_features_v4_nma", "complete_features_v4"])
    parser.add_argument("--protein-out-dirs", nargs="+", default=["complete_features_v4_nma_norm", "complete_features_v4_norm"])
    parser.add_argument("--spatial-clamp-dims", nargs="+", type=int, default=DEFAULT_SPATIAL_CLAMP_DIMS)
    parser.add_argument("--spatial-clamp-min", type=float, default=-5.0)
    parser.add_argument("--spatial-clamp-max", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        args.max_files = None

    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise FileNotFoundError(f"stats file not found: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    print(f"[stats] {stats_path}")
    print(f"[output_root] {Path(args.output_root).resolve()}")
    print(f"[mode] {'ALL FILES' if args.max_files is None else f'first {args.max_files} files per feature'}")

    if "ligand" in args.features:
        process_ligands(args, stats)
    if "unimol" in args.features:
        process_unimol(args, stats)
    if "esm3" in args.features:
        process_esm3(args, stats)
    if "spatial" in args.features:
        process_spatial(args, stats)


if __name__ == "__main__":
    main()
