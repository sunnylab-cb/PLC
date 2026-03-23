#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMA 蛋白质构象生成脚本 (仅 PDB 版)

功能：
1. 纯净生成：移除了所有与 PyTorch 和特征提取相关的逻辑，仅进行结构扰动。
2. 目录镜像：从输入目录（如 New/）读取原始 PDB 文件，完美保留其内部的任意嵌套目录结构，
   并镜像保存到输出目录（如 NMA/）。
3. 高效 I/O：优化了 PDB 文件的读写机制，使用内存列表拼接和批量写入加速。

用法：
    python apply_nma_only_pdb.py --input ./New --output ./NMA --n-confs 10 --workers 32
"""

from __future__ import annotations

import os

# ============================================================================
# 【核心修复】：防止底层线程爆炸 (Thread Contention)
# 必须在 import numpy 和 prody 之前设置环境变量！
# 强制底层 BLAS/LAPACK 库单线程运行，防止 128 个 worker 产生上万个线程导致死锁或极慢。
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import argparse
import multiprocessing as mp
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import prody
    prody.confProDy(verbosity='none')
    PRODY_AVAILABLE = True
except ImportError:
    PRODY_AVAILABLE = False


# ============================================================================
# PDB 写出核心工具 (加速版)
# ============================================================================

def write_perturbed_pdb(
    orig_pdb_path: Path,
    ca_displacement: np.ndarray,
    out_path: Path
) -> bool:
    """
    根据 Cα 的位移，将残基内所有原子同步平移并保存为新 PDB。
    【加速点】：一次性读取到内存，使用列表推导和批量写入，避免频繁的磁盘 I/O。
    """
    try:
        with open(orig_pdb_path, 'r') as f:
            lines = f.readlines()

        # ProDy 识别为氨基酸的残基名集合（含修饰残基如 CSO/MSE，排除离子如 CA2+）
        _PRODY_AA = {
            'ALA','ARG','ARN','ASH','ASN','ASP','ASX','CME','CSB','CSO','CYS','CYX',
            'GLH','GLN','GLU','GLX','GLY','HID','HIE','HIP','HIS','HISD','HISE','HISP',
            'HSD','HSE','HSP','ILE','LEU','LYN','LYS','MEN','MET','MSE','PHD','PHE',
            'PRO','PTR','SEC','SEP','SER','THR','TPO','TRP','TYM','TYR','VAL','XAA','XLE',
        }

        # 第一步：按行顺序收集所有符合条件的 CA 行索引
        # 不做 key 去重——NCS 对称体中同链同编号的残基是不同拷贝，ProDy 全部纳入
        # 只跳过 altloc 非空格/A（同一残基的备用构象）和非氨基酸（离子等）
        ca_line_indices = []
        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')) and len(line) > 26 and line[12:16].strip() == 'CA':
                if line[16] not in (' ', 'A'):
                    continue
                if line[17:20].strip() not in _PRODY_AA:
                    continue
                ca_line_indices.append(i)

        if len(ca_line_indices) != len(ca_displacement):
            return False

        # 第二步：建立行索引 → 位移的映射
        disp_by_line = {ca_line_indices[i]: ca_displacement[i] for i in range(len(ca_line_indices))}

        # 第三步：遍历所有行，对每个残基的所有原子施加对应 CA 的位移
        # 用当前残基的 (chain, res_seq, icode, copy_idx) 追踪所属 CA
        # 策略：遇到 CA 行时记录其位移，同一残基的其他原子复用该位移
        out_lines = []
        current_disp = None
        current_res_key = None
        current_copy = -1  # 同 key 的第几个拷贝

        # 预建 CA 行集合用于快速判断
        ca_line_set = set(ca_line_indices)
        # 记录每个 key 已出现的次数，用于区分 NCS 拷贝
        key_count: dict = {}

        def fmt_c(c):
            s = f"{c:8.3f}"
            if len(s) > 8: s = f"{c:8.2f}"
            if len(s) > 8: s = f"{c:8.1f}"
            return s[-8:]

        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')) and len(line) > 26:
                try:
                    res_key = (line[21], line[22:26].strip(), line[26].strip())
                    atom_name = line[12:16].strip()

                    # 遇到新残基（key 变化或 NCS 新拷贝）时重置位移
                    if i in ca_line_set:
                        # 这是一个 CA 行，更新当前位移
                        current_disp = disp_by_line[i]
                        current_res_key = res_key
                    elif res_key != current_res_key:
                        # 非 CA 行但残基 key 变了，说明这个残基没有 CA（缺失），清空位移
                        current_disp = None
                        current_res_key = res_key

                    if current_disp is not None:
                        x = float(line[30:38]) + current_disp[0]
                        y = float(line[38:46]) + current_disp[1]
                        z = float(line[46:54]) + current_disp[2]
                        line = line[:30] + fmt_c(x) + fmt_c(y) + fmt_c(z) + line[54:]
                except (ValueError, IndexError):
                    pass
            out_lines.append(line)

        # 批量写出
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as fout:
            fout.writelines(out_lines)

        return True

    except Exception as e:
        warnings.warn(f"write_perturbed_pdb failed for {orig_pdb_path.name}: {e}")
        return False


# ============================================================================
# 单个 PDB 处理
# ============================================================================

def process_single(args_tuple) -> Tuple[str, bool, str]:
    """
    处理单个 PDB：生成 NMA 构象 PDB。
    """
    pdb_path, input_root, output_root, n_modes, n_confs, amplitude = args_tuple
    
    # 提取蛋白ID (兼容 1abc_protein.pdb 或 1abc.pdb)
    pdb_id = pdb_path.stem[:-8] if pdb_path.stem.endswith('_protein') else pdb_path.stem

    try:
        # 【核心逻辑】：保留内部目录结构
        # 计算相对路径，例如 input_root/2021/1abc/1abc_protein.pdb
        # rel_dir 就是 2021/1abc/
        rel_dir = pdb_path.parent.relative_to(input_root)
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否所有构象都已经生成（用于断点续传/跳过已处理）
        existing_confs = sum(
            1 for i in range(n_confs) 
            if (out_dir / f"{pdb_id}_protein_conf{i}.pdb").exists()
        )
        if existing_confs == n_confs:
            return (pdb_id, True, None) # 全部存在，直接跳过

        # ── conf0：原始构象 ──────────────────────────────────────────────────
        conf0_pdb = out_dir / f"{pdb_id}_protein_conf0.pdb"
        if not conf0_pdb.exists():
            shutil.copy2(pdb_path, conf0_pdb)

        # ── NMA 扰动构象 ─────────────────────────────────────────────────────
        if n_confs <= 1 or not PRODY_AVAILABLE:
            return (pdb_id, True, None)

        # 用 ProDy 解析，【加速点】：只解析 CA 原子，极大减少内存和解析时间
        struct = prody.parsePDB(str(pdb_path), subset='ca')
        if struct is None:
            return (pdb_id, True, "ProDy 无法解析，仅保存 conf0")

        n_ca = struct.numAtoms()

        # 构建 ANM (各向异性网络模型)
        anm = prody.ANM(pdb_id)
        anm.buildHessian(struct, cutoff=15.0)
        actual_modes = min(n_modes, n_ca * 3 - 6)
        if actual_modes < 1:
            return (pdb_id, True, "有效模式数不足，仅保存 conf0")
        
        anm.calcModes(n_modes=actual_modes)

        # 预先计算模式向量和缩放系数 [K, N, 3]
        mode_vecs = np.array([anm[i].getArrayNx3() for i in range(actual_modes)])
        eigenvalues = np.array([anm[i].getEigval() for i in range(actual_modes)])
        # 防止特征值过小导致数值不稳定
        scale = amplitude / np.sqrt(np.maximum(eigenvalues, 1e-6))

        rng = np.random.default_rng(seed=42) # 固定随机种子以保证结果可复现

        for conf_idx in range(1, n_confs):
            conf_pdb = out_dir / f"{pdb_id}_protein_conf{conf_idx}.pdb"

            if conf_pdb.exists():
                continue

            # 使用高斯分布随机线性组合低频模式
            coeffs = rng.standard_normal(actual_modes) * scale
            displacement = np.einsum('k,kni->ni', coeffs, mode_vecs)  # [N, 3] 快速张量乘法

            # 写出扰动 PDB
            ok = write_perturbed_pdb(pdb_path, displacement, conf_pdb)
            if not ok:
                warnings.warn(f"Failed to write perturbed PDB for {pdb_id} conf {conf_idx}")

        return (pdb_id, True, None)

    except Exception as e:
        return (pdb_id, False, str(e))


# ============================================================================
# 批量处理与主程序
# ============================================================================

def batch_apply_nma(
    input_root: Path,
    output_root: Path,
    pattern: str = "*_protein.pdb",
    n_modes: int = 10,
    n_confs: int = 10,
    amplitude: float = 1.0,
    n_workers: Optional[int] = None,
) -> None:
    
    pdb_files = sorted(input_root.rglob(pattern))
    
    print(f"============== NMA PDB 增强工具 ==============")
    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    print(f"输入: {input_root}")
    print(f"输出: {output_root}")
    print(f"NMA: {n_confs} 构象/蛋白, {n_modes} 模式, 振幅 {amplitude} Å")
    
    if not PRODY_AVAILABLE:
        print("警告: ProDy 未安装，将仅复制原文件 (conf0)！")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 16) # 防止线程过多导致 I/O 阻塞
    print(f"并行进程数 (Workers): {n_workers}\n")

    if not pdb_files:
        print("没有需要处理的文件，程序退出。")
        return

    # 跳过已完整处理的文件
    to_process = []
    for pdb_file in pdb_files:
        pdb_id = pdb_file.stem[:-8] if pdb_file.stem.endswith('_protein') else pdb_file.stem
        rel_dir = pdb_file.parent.relative_to(input_root)
        out_dir = output_root / rel_dir
        existing = sum(
            1 for i in range(n_confs)
            if (out_dir / f"{pdb_id}_protein_conf{i}.pdb").exists()
        )
        if existing < n_confs:
            to_process.append(pdb_file)
    print(f"跳过 {len(pdb_files) - len(to_process)} 个已完整处理的蛋白质")
    pdb_files = to_process

    if not pdb_files:
        print("没有需要处理的文件，程序退出。")
        return

    # 准备参数列表
    args_list = [
        (pdb_file, input_root, output_root, n_modes, n_confs, amplitude)
        for pdb_file in pdb_files
    ]

    success = 0
    failed = 0
    
    if n_workers > 1:
        # 使用 spawn 避免 ProDy/LAPACK 在 fork 后死锁
        # chunksize=1 保证 tqdm 实时更新
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap_unordered(process_single, args_list, chunksize=1),
                    total=len(args_list),
                    desc="NMA 生成进度"
                ))
            else:
                results = list(pool.imap_unordered(process_single, args_list, chunksize=1))
    else:
        results = []
        it = tqdm(args_list, desc="NMA 生成进度") if TQDM_AVAILABLE else args_list
        for a in it:
            results.append(process_single(a))

    for pdb_id, ok, msg in results:
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  [失败] {pdb_id}: {msg}")

    print(f"\n================ 完成 ================")
    print(f"成功: {success}")
    print(f"失败: {failed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NMA 数据增强 (仅 PDB)：从指定目录读取并生成扰动 PDB，保持内部结构'
    )
    parser.add_argument('--input',  type=Path, required=True,
                        help='输入 PDB 根目录 (例如: /home/husrcf/Code/ProtBind/New)')
    parser.add_argument('--output', type=Path, required=True,
                        help='输出根目录 (例如: /home/husrcf/Code/ProtBind/NMA)')
    parser.add_argument('--pattern',   type=str,   default='*_protein.pdb', help='匹配的PDB后缀名')
    parser.add_argument('--n-modes',   type=int,   default=10,  help='ANM 低频模式数')
    parser.add_argument('--n-confs',   type=int,   default=10,  help='每个蛋白生成的构象总数（含原始）')
    parser.add_argument('--amplitude', type=float, default=1.0, help='扰动幅度 (Å)')
    parser.add_argument('--workers',   type=int,   default=None, help='多进程数')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    batch_apply_nma(
        input_root=args.input.resolve(),
        output_root=args.output.resolve(),
        pattern=args.pattern,
        n_modes=args.n_modes,
        n_confs=args.n_confs,
        amplitude=args.amplitude,
        n_workers=args.workers,
    )