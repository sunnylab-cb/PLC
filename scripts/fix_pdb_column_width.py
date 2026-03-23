#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 PDB 文件坐标列宽问题：用 BioPython 读取后重新写入，自动规范化格式。
原文件备份为 *.pdb.bak
"""

from __future__ import annotations
import argparse
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO


def fix_pdb_column_width(pdb_path: Path, backup: bool = True) -> bool:
    """
    用 BioPython 重写 PDB 文件，修复坐标列宽问题。

    Returns:
        True if successful, False otherwise
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("_", str(pdb_path))

        # 备份原文件
        if backup:
            bak_path = pdb_path.with_suffix('.pdb.bak')
            if not bak_path.exists():
                pdb_path.rename(bak_path)
            else:
                # 如果备份已存在，从备份读取
                structure = parser.get_structure("_", str(bak_path))

        # 重写文件
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))

        return True
    except Exception as e:
        print(f"Error fixing {pdb_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='修复 PDB 坐标列宽问题')
    parser.add_argument('--input', required=True, help='PDB 文件或包含失败列表的 TSV')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.suffix == '.tsv':
        # 从 TSV 读取失败列表
        with open(input_path) as f:
            lines = f.readlines()[1:]  # skip header
        pdb_files = [Path(line.split('\t')[0]) for line in lines if line.strip()]
    elif input_path.is_file():
        pdb_files = [input_path]
    else:
        print(f"Invalid input: {input_path}")
        return 1

    print(f"Processing {len(pdb_files)} files...")

    success = 0
    failed = 0
    for pdb_path in pdb_files:
        if not pdb_path.exists():
            print(f"Not found: {pdb_path}")
            failed += 1
            continue

        if fix_pdb_column_width(pdb_path, backup=not args.no_backup):
            print(f"✓ {pdb_path.name}")
            success += 1
        else:
            print(f"✗ {pdb_path.name}")
            failed += 1

    print(f"\nDone: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
