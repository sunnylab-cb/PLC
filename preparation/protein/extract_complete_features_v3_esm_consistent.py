#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_complete_features_v3 的 ESM 对齐副本。

说明：
- 不改原脚本 `extract_complete_features_v3.py`
- 仅在此副本里替换 `parse_pdb_protein` 为 ESM 对齐版本
"""

from __future__ import annotations

from pathlib import Path
import sys

# 保持与原脚本同样的导入路径行为
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preparation.protein import extract_complete_features_v3 as base
from preparation.protein.extract_physchem_from_pdb_esm_consistent import parse_pdb_protein as parse_pdb_protein_esm


# 关键：覆盖原模块全局 parse 函数
base.parse_pdb_protein = parse_pdb_protein_esm


if __name__ == "__main__":
    base.main()

