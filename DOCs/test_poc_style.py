#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试：验证PoC风格距离计算是否正常工作
"""

import sys
from pathlib import Path

# 添加pypropel路径
pypropel_path = Path(__file__).parent.parent / 'pypropel_unibio'
if pypropel_path.exists():
    sys.path.insert(0, str(pypropel_path))

print("="*70)
print("测试PoC风格距离计算")
print("="*70)

# 1. 检查依赖
print("\n【1. 检查依赖】")

try:
    import pypropel.str as ppstr
    import pypropel.mol as ppmol
    import pypropel.dist as ppdist
    print("  ✅ pypropel可用")
except ImportError as e:
    print(f"  ❌ pypropel不可用: {e}")
    sys.exit(1)

try:
    from rdkit import Chem
    print("  ✅ RDKit可用")
except ImportError:
    print("  ❌ RDKit不可用")
    sys.exit(1)

try:
    from Bio.PDB import PDBParser
    print("  ✅ BioPython可用")
except ImportError:
    print("  ❌ BioPython不可用")
    sys.exit(1)

import torch
import numpy as np
print("  ✅ PyTorch可用")
print("  ✅ NumPy可用")

# 2. 测试单个复合物
print("\n【2. 测试单个复合物】")

test_protein = '/home/husrcf/Code/ProtBind/P-L/1981-2000/1jmi/1jmi_protein.pdb'
test_ligand = '/home/husrcf/Code/ProtBind/New/1981-2000/1jmi/1jmi_ligand.sdf'

if not Path(test_protein).exists():
    print(f"  ❌ 测试蛋白不存在: {test_protein}")
    sys.exit(1)

if not Path(test_ligand).exists():
    print(f"  ❌ 测试配体不存在: {test_ligand}")
    sys.exit(1)

print(f"  蛋白: {test_protein}")
print(f"  配体: {test_ligand}")

# 3. 加载结构
print("\n【3. 加载结构】")

try:
    structure = ppstr.load_pdb(test_protein)
    print(f"  ✅ 蛋白加载成功")
except Exception as e:
    print(f"  ❌ 蛋白加载失败: {e}")
    sys.exit(1)

try:
    mol = ppmol.load_sdf(test_ligand)
    print(f"  ✅ 配体加载成功")
except Exception as e:
    print(f"  ❌ 配体加载失败: {e}")
    sys.exit(1)

# 4. 提取坐标
print("\n【4. 提取配体坐标】")

try:
    ligand_coords = ppmol.ligand_coords(mol)
    print(f"  ✅ 配体坐标: {ligand_coords.shape}")
    print(f"     中心: {ligand_coords.mean(axis=0)}")
except Exception as e:
    print(f"  ❌ 坐标提取失败: {e}")
    sys.exit(1)

# 5. 计算距离
print("\n【5. 计算距离（PoC方法）】")

try:
    distances_df = ppdist.protein_ligand_distances(structure, ligand_coords)
    print(f"  ✅ 距离计算成功")
    print(f"     残基数: {len(distances_df)}")
    print(f"     距离范围: {distances_df['distance'].min():.2f} - {distances_df['distance'].max():.2f} Å")
    print(f"     平均距离: {distances_df['distance'].mean():.2f} Å")
except Exception as e:
    print(f"  ❌ 距离计算失败: {e}")
    sys.exit(1)

# 6. 检查氢原子
print("\n【6. 验证氢原子处理】")

# 检查蛋白质第一个残基
model = structure[0]
chain = list(model)[0]
residue = list(chain)[0]

all_atoms = list(residue.get_atoms())
heavy_atoms = [a for a in all_atoms if a.element != 'H']

print(f"  第一个残基: {residue.get_resname()} {residue.get_id()[1]}")
print(f"    总原子数: {len(all_atoms)}")
print(f"    重原子数: {len(heavy_atoms)}")
print(f"    氢原子数: {len(all_atoms) - len(heavy_atoms)}")

if len(all_atoms) > len(heavy_atoms):
    print(f"  ✅ 蛋白质包含氢原子（PoC方法会使用它们）")
else:
    print(f"  ⚠️  蛋白质不包含氢原子")

# 7. 对比PLC方法
print("\n【7. 对比PLC方法】")

# 使用PLC方法计算（仅重原子）
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure_plc = parser.get_structure('protein', test_protein)

distances_plc = []
for model in structure_plc:
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue

            # 仅重原子
            residue_coords = np.array([
                atom.get_coord() for atom in residue.get_atoms()
                if atom.element != 'H'
            ])

            if len(residue_coords) == 0:
                continue

            # 计算最小距离
            diff = residue_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dist = dists.min()
            distances_plc.append(min_dist)

distances_plc = np.array(distances_plc)
distances_poc = distances_df['distance'].values

print(f"  PoC方法平均距离: {distances_poc.mean():.2f} Å")
print(f"  PLC方法平均距离: {distances_plc.mean():.2f} Å")
print(f"  差异: {distances_plc.mean() - distances_poc.mean():.4f} Å")

diff = distances_plc - distances_poc
print(f"  平均绝对差异: {np.abs(diff).mean():.4f} Å")
print(f"  最大差异: {np.abs(diff).max():.4f} Å")

if np.abs(diff).mean() > 0.1:
    print(f"  ✅ 检测到显著差异（符合预期）")
else:
    print(f"  ⚠️  差异较小（可能氢原子较少）")

# 8. 总结
print("\n" + "="*70)
print("✅ 所有测试通过！")
print("="*70)
print("\n可以开始运行批量计算：")
print("  python calculate_distances_poc_style.py \\")
print("      --data_root /home/husrcf/Code/ProtBind/New \\")
print("      --output data/residue_ligand_distances_poc_style.pt \\")
print("      --workers 64")
