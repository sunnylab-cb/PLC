#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 PDBFixer 批量补全 PDB 文件中缺失的原子（如无 CA 的残基）。
修复后的文件覆盖原文件，原文件备份为 *_protein.pdb.bak。

用法：
  python scripts/fix_pdb_missing_atoms.py \
      --data-root /home/husrcf/Code/ProtBind/New \
      --workers 16 \
      [--overwrite]   # 跳过已有 .bak 的文件
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
from pathlib import Path
from typing import Tuple


def fix_single(args: Tuple[Path, bool]) -> Tuple[str, bool, str]:
    """
    处理单个 PDB 文件。
    Returns: (pdb_path_str, success, message)
    """
    pdb_path, overwrite = args
    bak_path = pdb_path.with_suffix('.pdb.bak')

    try:
        import warnings
        warnings.filterwarnings('ignore')

        # 已修复过则跳过（除非 overwrite）
        if bak_path.exists() and not overwrite:
            return (str(pdb_path), True, 'skipped')

        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()

        # 检查是否有缺失原子需要补全
        n_missing = sum(len(v) for v in fixer.missingAtoms.values())
        if n_missing == 0 and not fixer.missingResidues:
            return (str(pdb_path), True, 'no_missing')

        fixer.addMissingAtoms()

        # 备份原文件
        pdb_path.rename(bak_path)

        # 写修复后的文件
        out = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, out)
        pdb_path.write_text(out.getvalue())

        return (str(pdb_path), True, f'fixed missing_atoms={n_missing}')

    except Exception as e:
        return (str(pdb_path), False, str(e))


def main():
    parser = argparse.ArgumentParser(description='PDBFixer 批量补全缺失原子')
    parser.add_argument('--data-root', default='/home/husrcf/Code/ProtBind/New',
                        help='PDB 数据根目录')
    parser.add_argument('--workers', type=int, default=16, help='并行进程数')
    parser.add_argument('--overwrite', action='store_true', help='重新处理已有 .bak 的文件')
    parser.add_argument('--pattern', default='*_protein.pdb', help='文件匹配模式')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pdb_files = sorted(data_root.rglob(args.pattern))
    total = len(pdb_files)
    print(f'找到 {total} 个文件，使用 {args.workers} 个进程')

    tasks = [(p, args.overwrite) for p in pdb_files]

    fixed = skipped = no_missing = failed = 0
    failures = []

    from tqdm import tqdm
    with mp.Pool(processes=args.workers) as pool:
        pbar = tqdm(pool.imap_unordered(fix_single, tasks), total=total,
                    desc='fixing', dynamic_ncols=True)
        for path, success, msg in pbar:
            if not success:
                failed += 1
                failures.append((path, msg))
            elif msg == 'skipped':
                skipped += 1
            elif msg == 'no_missing':
                no_missing += 1
            else:
                fixed += 1
            pbar.set_postfix(fixed=fixed, no_missing=no_missing,
                             skipped=skipped, failed=failed)

    print(f'\n完成:')
    print(f'  修复:      {fixed}')
    print(f'  无缺失:    {no_missing}')
    print(f'  跳过:      {skipped}')
    print(f'  失败:      {failed}')

    if failures:
        print(f'\n失败列表（前20）:')
        for path, msg in failures[:20]:
            print(f'  {path}: {msg}')

        # 写失败日志
        log = data_root.parent / 'fix_pdb_failures.txt'
        with open(log, 'w') as f:
            for path, msg in failures:
                f.write(f'{path}\t{msg}\n')
        print(f'\n失败日志: {log}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
