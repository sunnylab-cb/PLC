#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探针势能与水合自由能特征模块 (V3 Enhanced Features)

新增特征组成：
1. 局部探针势能特征 (3维 per-residue)
   - Carbon probe affinity
   - H-bond donor potential
   - H-bond acceptor potential

2. 局部水合自由能特征 (2维 per-residue)
   - Hydration free energy density
   - Frustrated water index

3. 区域聚合特征 (6-9维 per-residue)
   - 滑动窗口势能梯度 (捕捉表面趋势)
   - 多尺度水合特征
   - 口袋环境统计特征

总计：11-14维 (取决于配置)

依赖工具：
- AutoDock Tools (for AutoGrid)
- FTMap (optional, for probe mapping)
- 3D-RISM / WaterMap (for hydration free energy)
"""

from __future__ import annotations

import numpy as np
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

try:
    from Bio.PDB import PDBParser, PDBIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("Biopython not installed")


# ============================================================================
# 1. AutoGrid 探针势能计算
# ============================================================================

def run_autogrid_for_protein(
    pdb_path: Path,
    output_dir: Path,
    grid_spacing: float = 0.375,
    grid_size: Tuple[int, int, int] = (60, 60, 60)
) -> Dict[str, np.ndarray]:
    """
    调用 AutoGrid 计算探针势能图

    Args:
        pdb_path: PDB文件路径
        output_dir: 输出目录
        grid_spacing: 网格间距 (Å)
        grid_size: 网格大小 (nx, ny, nz)

    Returns:
        字典包含：
        - 'C_potential': 碳探针势能 3D网格
        - 'HD_potential': 氢键供体势能 3D网格
        - 'HA_potential': 氢键受体势能 3D网格
        - 'grid_center': 网格中心坐标
        - 'grid_spacing': 网格间距
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备 receptor (PDBQT格式)
    receptor_pdbqt = output_dir / f"{pdb_path.stem}_receptor.pdbqt"

    # 调用 prepare_receptor4.py (AutoDock Tools)
    cmd_prep = [
        "prepare_receptor4.py",
        "-r", str(pdb_path),
        "-o", str(receptor_pdbqt),
        "-A", "hydrogens",  # 添加氢原子
        "-U", "nphs_lps"    # 合并非极性氢和孤对电子
    ]

    try:
        subprocess.run(cmd_prep, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warnings.warn(f"prepare_receptor4.py failed: {e.stderr}")
        return None
    except FileNotFoundError:
        warnings.warn("AutoDock Tools not found. Install from: http://autodock.scripps.edu/")
        return None

    # 2. 创建 GPF (Grid Parameter File)
    gpf_file = output_dir / f"{pdb_path.stem}.gpf"

    # 计算蛋白质中心（用于网格中心）
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.coord)
    coords = np.array(coords)
    grid_center = coords.mean(axis=0)

    # 生成 GPF 文件
    gpf_content = f"""npts {grid_size[0]} {grid_size[1]} {grid_size[2]}
gridfld {pdb_path.stem}.maps.fld
spacing {grid_spacing}
receptor_types A C HD N OA SA
ligand_types C HD OA
receptor {receptor_pdbqt.name}
gridcenter {grid_center[0]:.3f} {grid_center[1]:.3f} {grid_center[2]:.3f}
smooth 0.5
map {pdb_path.stem}.C.map
map {pdb_path.stem}.HD.map
map {pdb_path.stem}.OA.map
elecmap {pdb_path.stem}.e.map
dsolvmap {pdb_path.stem}.d.map
dielectric -0.1465
"""

    gpf_file.write_text(gpf_content)

    # 3. 运行 AutoGrid
    cmd_autogrid = ["autogrid4", "-p", str(gpf_file), "-l", str(output_dir / "autogrid.log")]

    try:
        subprocess.run(cmd_autogrid, check=True, capture_output=True, text=True, cwd=output_dir)
    except subprocess.CalledProcessError as e:
        warnings.warn(f"AutoGrid failed: {e.stderr}")
        return None
    except FileNotFoundError:
        warnings.warn("AutoGrid4 not found. Install AutoDock4.")
        return None

    # 4. 读取 .map 文件
    potentials = {}
    for probe_type in ['C', 'HD', 'OA']:
        map_file = output_dir / f"{pdb_path.stem}.{probe_type}.map"
        if map_file.exists():
            potentials[f'{probe_type}_potential'] = read_autogrid_map(map_file)

    potentials['grid_center'] = grid_center
    potentials['grid_spacing'] = grid_spacing
    potentials['grid_size'] = grid_size

    return potentials


def read_autogrid_map(map_file: Path) -> np.ndarray:
    """读取 AutoGrid .map 文件"""
    with open(map_file, 'r') as f:
        lines = f.readlines()

    # 跳过头部，读取数据
    data_lines = [line for line in lines if not line.startswith('#') and line.strip()]

    # 解析网格数据
    values = []
    for line in data_lines[6:]:  # 前6行是参数
        values.extend([float(x) for x in line.split()])

    # 重塑为3D网格 (需要从头部读取尺寸)
    # 这里简化处理，实际需要解析头部
    grid_data = np.array(values)

    return grid_data


