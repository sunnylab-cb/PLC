#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本机 DSSP 可用性自检脚本（单个 PDB）

功能：
1) 检查 mkdssp/dssp 二进制是否存在
2) 用 mkdssp 对一个 *_protein.pdb 直接执行并解析 SS 码分布
3) 调用项目内 compute_dssp_features()，查看实际提取到的 8 维 one-hot 分布

用法：
  python scripts/test_dssp_local.py
  python scripts/test_dssp_local.py --pdb database/pdbbind/New/1981-2000/1jmi/1jmi_protein.pdb
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
import sys

import numpy as np

# 让脚本可从项目根导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'preparation' / 'protein'))

from preparation.protein.extract_physchem_from_pdb import parse_pdb_protein
from preparation.protein.structural_geometry_features import compute_dssp_features

DSSP_INDEX_TO_CODE = {
    0: 'H',
    1: 'B',
    2: 'E',
    3: 'G',
    4: 'I',
    5: 'T',
    6: 'S',
    7: '-',
}


def auto_pick_pdb() -> Path:
    candidates = [
        ROOT / 'database' / 'pdbbind' / 'New',
        ROOT / 'predictions' / 'pdb',
        ROOT,
    ]
    for base in candidates:
        if not base.exists():
            continue
        found = sorted(base.rglob('*_protein.pdb'))
        if found:
            return found[0]
    raise FileNotFoundError('未找到 *_protein.pdb，请通过 --pdb 指定')


def sanitize_pdb_for_dssp(pdb_path: Path) -> str:
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        pdb_name = pdb_path.stem[:4].upper()
        f.write(f"HEADER    PROTEIN                                 01-JAN-00   {pdb_name}\n")
        for line in lines:
            if line.startswith(('ATOM', 'TER', 'END')):
                f.write(line)
    return tmp_path


def run_mkdssp_and_parse(pdb_path: Path) -> tuple[str, Counter]:
    dssp_bin = shutil.which('mkdssp') or shutil.which('dssp')
    if not dssp_bin:
        raise RuntimeError('未找到 mkdssp/dssp 可执行文件')

    tmp_file = sanitize_pdb_for_dssp(pdb_path)
    try:
        cmd = [dssp_bin, '--output-format', 'dssp', tmp_file]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or '').strip()
            stdout = (e.stdout or '').strip()
            raise RuntimeError(f'DSSP command failed. stderr={stderr[:300]} stdout={stdout[:300]}')
        lines = result.stdout.splitlines()

        start = -1
        for i, line in enumerate(lines):
            if line.startswith('  #  RESIDUE'):
                start = i + 1
                break
        if start == -1:
            raise RuntimeError('DSSP 输出未找到数据区')

        ss_codes = []
        for line in lines[start:]:
            if len(line) < 17:
                continue
            ss_code = line[16]
            if ss_code == ' ':
                ss_code = '-'
            ss_codes.append(ss_code)

        return dssp_bin, Counter(ss_codes)
    finally:
        try:
            os.remove(tmp_file)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description='本机 DSSP 可用性测试')
    parser.add_argument('--pdb', type=Path, default=None, help='测试用 *_protein.pdb 文件')
    args = parser.parse_args()

    pdb_path = args.pdb.resolve() if args.pdb else auto_pick_pdb().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f'PDB 不存在: {pdb_path}')

    print(f'测试 PDB: {pdb_path}')

    dssp_bin = shutil.which('mkdssp') or shutil.which('dssp')
    print(f'mkdssp/dssp: {dssp_bin if dssp_bin else "NOT FOUND"}')

    # 1) 直接跑 mkdssp 并统计 SS 码
    try:
        used_bin, ss_counter = run_mkdssp_and_parse(pdb_path)
        print(f'\n[OK] 直接 DSSP 执行成功: {used_bin}')
        print('DSSP 原始 SS 码分布:')
        for k in sorted(ss_counter):
            print(f'  {k}: {ss_counter[k]}')
    except Exception as e:
        print(f'\n[FAIL] 直接 DSSP 执行失败: {e}')

    # 2) 走项目提取函数（返回 [N,8] one-hot）
    try:
        pdb_info = parse_pdb_protein(pdb_path, chain_id=None, mode='complex')
        dssp_feat = compute_dssp_features(
            pdb_path,
            pdb_info['residue_indices'],
            pdb_info['chain_ids'],
        )

        print('\n项目 compute_dssp_features 输出:')
        print(f'  shape: {dssp_feat.shape}')
        nonzero_rows = int((dssp_feat.sum(axis=1) > 0).sum())
        print(f'  非零行: {nonzero_rows}/{dssp_feat.shape[0]}')

        counts = dssp_feat.sum(axis=0)
        print('  8维通道计数 (H,B,E,G,I,T,S,-):')
        for idx, val in enumerate(counts.tolist()):
            print(f"    {idx}({DSSP_INDEX_TO_CODE[idx]}): {int(val)}")

        active = [idx for idx, val in enumerate(counts.tolist()) if val > 0]
        active_codes = [DSSP_INDEX_TO_CODE[i] for i in active]
        print(f'  激活通道索引: {active}')
        print(f'  激活SS码: {active_codes}')

        if np.isnan(dssp_feat).any():
            print('  [WARN] 检测到 NaN')
        else:
            print('  [OK] 无 NaN')

    except Exception as e:
        print(f'\n[FAIL] compute_dssp_features 失败: {e}')


if __name__ == '__main__':
    main()
