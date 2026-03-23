#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMA 数据增强脚本（向量化重写版）

从已提取的 V3 特征 .pt 文件出发，用向量化 numpy ANM 生成扰动构象特征，
只更新 ca_coords 和 spatial_vector_features（cavity pointing vector），
其余特征（physchem / spatial_scalar / 侧链向量）保持不变。

不再依赖 ProDy，不再写扰动 PDB，速度提升 ~10x。

输入目录结构（--input）：
    FEAT_ROOT/YEAR_RANGE/PDB_ID/{pdb_id}_protein.pt

输出目录结构（--output）：
    NMA_ROOT/YEAR_RANGE/PDB_ID/{pdb_id}_protein_conf0.pt  (原始构象，直接复制)
                                {pdb_id}_protein_conf1.pt  (NMA 扰动 1)
                                ...

用法：
    python scripts/apply_nma_augmentation.py \\
        --input  /home/husrcf/Code/ProtBind/PLC/data/complete_features_v3 \\
        --output /home/husrcf/Code/ProtBind/NMA \\
        --n-confs 10 --n-modes 10 --amplitude 1.0 --workers 32
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# 向量化 ANM
# ============================================================================

def build_anm(ca_coords: np.ndarray, cutoff: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化构建 ANM Hessian 并返回前 n_modes 个非零模式。

    Returns:
        eigenvalues: [K]
        eigenvectors: [3N, K]  (每列一个模式)
    """
    N = len(ca_coords)
    diff = ca_coords[:, None] - ca_coords[None, :]   # [N,N,3]
    dist2 = (diff ** 2).sum(-1)                       # [N,N]
    mask = (dist2 < cutoff ** 2) & (dist2 > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        outer = (
            np.einsum('ijk,ijl->ijkl', diff, diff)
            / np.where(mask, dist2, 1.0)[:, :, None, None]
        )
    outer *= mask[:, :, None, None]   # [N,N,3,3]

    H = np.zeros((3 * N, 3 * N), dtype=np.float64)

    # 对角块
    diag = outer.sum(axis=1)          # [N,3,3]
    for a in range(3):
        for b in range(3):
            H[a::3, b::3] += np.diag(diag[:, a, b])

    # 非对角块
    ii, jj = np.where(mask)
    for a in range(3):
        for b in range(3):
            H[ii * 3 + a, jj * 3 + b] -= outer[ii, jj, a, b]

    return H


def calc_modes(H: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Hessian 提取前 n_modes 个非零模式（跳过 6 个刚体模式）。

    Returns:
        eigenvalues [K], eigenvectors [3N, K]
    """
    n_request = n_modes + 6 + 2   # 多请求几个以防数值零模式
    n_request = min(n_request, H.shape[0] - 1)

    if SCIPY_AVAILABLE:
        vals, vecs = eigsh(H, k=n_request, sigma=1e-6, which='LM')
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    else:
        vals, vecs = np.linalg.eigh(H)

    # 跳过接近零的刚体模式
    nz = vals > 1e-4
    vals, vecs = vals[nz], vecs[:, nz]
    return vals[:n_modes], vecs[:, :n_modes]


# ============================================================================
# 特征扰动
# ============================================================================

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > eps, v / (norm + eps), 0.0)


def perturb_features(
    feat: dict,
    displacement: np.ndarray,   # [N, 3]
) -> dict:
    """
    给定 Cα 位移，返回更新后的特征字典（浅拷贝，只替换受影响的 tensor）。

    更新项：
      - ca_coords
      - spatial_vector_features[:, 5, :]  (cavity pointing vector)
    """
    out = {k: v for k, v in feat.items()}   # 浅拷贝，共享未修改的 tensor

    ca_orig = feat['ca_coords'].numpy()     # [N, 3]
    ca_new = ca_orig + displacement         # [N, 3]
    out['ca_coords'] = torch.from_numpy(ca_new.astype(np.float32))

    # 重算 cavity pointing vector（idx=5）：normalize(Cα - CoM_neighbors)，向量化
    N = len(ca_new)
    diff_ca = ca_new[:, None] - ca_new[None, :]      # [N,N,3]
    dist2 = (diff_ca ** 2).sum(-1)                   # [N,N]
    nb_mask = (dist2 > 0) & (dist2 < 10.0 ** 2)     # [N,N]
    nb_count = nb_mask.sum(axis=1, keepdims=True).clip(min=1)          # [N,1]
    com = (ca_new[None, :] * nb_mask[:, :, None]).sum(axis=1) / nb_count  # [N,3]
    cavity_vecs = _safe_normalize(ca_new - com).astype(np.float32)
    # 孤立残基（无邻居）保持零向量
    cavity_vecs[nb_mask.sum(axis=1) == 0] = 0.0

    sv = feat['spatial_vector_features'].clone()   # [N, 6, 3]
    sv[:, 5, :] = torch.from_numpy(cavity_vecs)
    out['spatial_vector_features'] = sv

    return out


# ============================================================================
# 单个 .pt 处理
# ============================================================================

def _year_range(path: Path) -> Optional[str]:
    for part in path.parts:
        if '-' in part:
            sp = part.split('-')
            if len(sp) == 2 and sp[0].isdigit() and sp[1].isdigit():
                return part
    return None


def process_single(args_tuple) -> Tuple[str, bool, str]:
    pt_path, output_root, n_modes, n_confs, amplitude, overwrite = args_tuple
    pdb_id = pt_path.stem[:-8] if pt_path.stem.endswith('_protein') else pt_path.stem

    try:
        yr = _year_range(pt_path)
        out_dir = (output_root / yr / pdb_id) if yr else (output_root / pdb_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # conf0：直接复制原始特征
        conf0_pt = out_dir / f"{pdb_id}_protein_conf0.pt"
        if overwrite or not conf0_pt.exists():
            feat0 = torch.load(pt_path, map_location='cpu', weights_only=False)
            torch.save(feat0, conf0_pt)
        else:
            feat0 = torch.load(conf0_pt, map_location='cpu', weights_only=False)

        if n_confs <= 1:
            return (pdb_id, True, None)

        # 构建 ANM
        ca = feat0['ca_coords'].numpy().astype(np.float64)   # [N, 3]
        N = len(ca)
        H = build_anm(ca, cutoff=15.0)
        actual_modes = min(n_modes, N * 3 - 6)
        if actual_modes < 1:
            return (pdb_id, True, "残基数不足，仅保存 conf0")

        vals, vecs = calc_modes(H, n_modes=actual_modes)
        # vecs: [3N, K]，reshape 为 [K, N, 3]
        mode_vecs = vecs.T.reshape(len(vals), N, 3)
        scale = amplitude / np.sqrt(np.maximum(vals, 1e-6))   # [K]

        rng = np.random.default_rng(seed=42)

        for conf_idx in range(1, n_confs):
            conf_pt = out_dir / f"{pdb_id}_protein_conf{conf_idx}.pt"
            if not overwrite and conf_pt.exists():
                continue

            coeffs = rng.standard_normal(len(vals)) * scale   # [K]
            disp = np.einsum('k,kni->ni', coeffs, mode_vecs)  # [N, 3]

            feat_new = perturb_features(feat0, disp.astype(np.float32))
            torch.save(feat_new, conf_pt)

        return (pdb_id, True, None)

    except Exception as e:
        return (pdb_id, False, str(e))


# ============================================================================
# 批量处理
# ============================================================================

def batch_apply_nma(
    input_root: Path,
    output_root: Path,
    pattern: str = "*_protein.pt",
    n_modes: int = 10,
    n_confs: int = 10,
    amplitude: float = 1.0,
    max_files: Optional[int] = None,
    overwrite: bool = False,
    n_workers: Optional[int] = None,
) -> None:
    pt_files = sorted(input_root.rglob(pattern))
    if max_files:
        pt_files = pt_files[:max_files]

    print(f"找到 {len(pt_files)} 个 .pt 文件")
    print(f"输入: {input_root}")
    print(f"输出: {output_root}")
    print(f"NMA: {n_confs} 构象/蛋白, {n_modes} 模式, 振幅 {amplitude} Å")
    print(f"scipy 可用: {SCIPY_AVAILABLE}")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    print(f"Workers: {n_workers}\n")

    if not overwrite:
        to_process = []
        for pt_file in pt_files:
            pdb_id = pt_file.stem[:-8] if pt_file.stem.endswith('_protein') else pt_file.stem
            yr = _year_range(pt_file)
            out_dir = (output_root / yr / pdb_id) if yr else (output_root / pdb_id)
            existing = sum(
                1 for i in range(n_confs)
                if (out_dir / f"{pdb_id}_protein_conf{i}.pt").exists()
            )
            if existing < n_confs:
                to_process.append(pt_file)
        print(f"跳过 {len(pt_files) - len(to_process)} 个已完整处理的蛋白质")
        pt_files = to_process

    if not pt_files:
        print("没有需要处理的文件")
        return

    args_list = [
        (pt_file, output_root, n_modes, n_confs, amplitude, overwrite)
        for pt_file in pt_files
    ]

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(process_single, args_list),
                    total=len(args_list),
                    desc="NMA 增强"
                ))
            else:
                results = pool.map(process_single, args_list)
    else:
        results = []
        it = tqdm(args_list, desc="NMA 增强") if TQDM_AVAILABLE else args_list
        for a in it:
            results.append(process_single(a))

    success = sum(1 for _, ok, _ in results if ok)
    failed  = len(results) - success
    print(f"\n完成: 成功 {success}, 失败 {failed}")
    if failed:
        for pdb_id, ok, msg in results:
            if not ok:
                print(f"  FAIL {pdb_id}: {msg}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NMA 数据增强：从 V3 特征 .pt 生成扰动构象，只更新 ca_coords 和 cavity vector'
    )
    parser.add_argument('--input',  type=Path,
                        default=Path('/home/husrcf/Code/ProtBind/PLC/data/complete_features_v3'),
                        help='输入特征根目录')
    parser.add_argument('--output', type=Path,
                        default=Path('/home/husrcf/Code/ProtBind/NMA'),
                        help='输出根目录')
    parser.add_argument('--pattern',   type=str,   default='*_protein.pt')
    parser.add_argument('--n-modes',   type=int,   default=10,  help='ANM 低频模式数')
    parser.add_argument('--n-confs',   type=int,   default=10,  help='构象总数（含原始）')
    parser.add_argument('--amplitude', type=float, default=1.0, help='扰动幅度 Å')
    parser.add_argument('--max',       type=int,   default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers',   type=int,   default=None)

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 输入目录不存在: {args.input}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    batch_apply_nma(
        input_root=args.input.resolve(),
        output_root=args.output.resolve(),
        pattern=args.pattern,
        n_modes=args.n_modes,
        n_confs=args.n_confs,
        amplitude=args.amplitude,
        max_files=args.max,
        overwrite=args.overwrite,
        n_workers=args.workers,
    )


if __name__ == '__main__':
    main()
