#!/usr/bin/env python3
"""
调试 1ibg 的 VDW 计算问题
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from calculate_distances_vdw import (
    get_vdw_radius,
    extract_atom_symbols_from_ligand,
    calculate_vdw_based_distances,
    TOLERANCE
)

# 加载数据
protein_file = Path('data/complete_features/1981-2000/1ibg/1ibg_protein.pt')
ligand_file = Path('data/ligand_features_v3/1981-2000/1ibg/1ibg_ligand.pt')

print("=== 加载数据 ===")
protein_data = torch.load(protein_file, weights_only=False)
ligand_data = torch.load(ligand_file, weights_only=False)

print(f"蛋白质残基数: {len(protein_data['all_atom_coords'])}")
print(f"配体原子数: {ligand_data['num_atoms']}")
print(f"配体重原子数: {ligand_data['num_heavy_atoms']}")

# 提取坐标
protein_all_atom_coords = protein_data['all_atom_coords']
protein_all_atom_names = protein_data.get('all_atom_names', [])
ligand_coords = ligand_data['coords'].numpy()

print(f"\n=== 坐标信息 ===")
print(f"蛋白质 all_atom_coords 长度: {len(protein_all_atom_coords)}")
print(f"蛋白质 all_atom_names 长度: {len(protein_all_atom_names)}")
print(f"配体 coords shape: {ligand_coords.shape}")

# 检查第一个残基
if len(protein_all_atom_coords) > 0:
    first_res_coords = protein_all_atom_coords[0]
    first_res_names = protein_all_atom_names[0] if len(protein_all_atom_names) > 0 else []
    print(f"\n第一个残基:")
    print(f"  原子数: {len(first_res_coords) if first_res_coords is not None else 0}")
    print(f"  原子名称数: {len(first_res_names)}")
    if len(first_res_names) > 0:
        print(f"  原子名称示例: {first_res_names[:5]}")

# 提取配体原子符号
print(f"\n=== 提取配体原子符号 ===")
ligand_symbols = extract_atom_symbols_from_ligand(ligand_file)
print(f"提取的符号数: {len(ligand_symbols)}")
print(f"符号示例: {ligand_symbols[:10]}")

# 统计原子类型
from collections import Counter
symbol_counts = Counter(ligand_symbols)
print(f"原子类型分布: {dict(symbol_counts)}")

# 确保符号列表长度匹配
if len(ligand_symbols) < len(ligand_coords):
    print(f"\n⚠️ 符号数({len(ligand_symbols)}) < 坐标数({len(ligand_coords)})")
    ligand_symbols.extend(['C'] * (len(ligand_coords) - len(ligand_symbols)))
    print(f"扩展后符号数: {len(ligand_symbols)}")

# 计算 VDW 距离
print(f"\n=== 计算 VDW 距离 ===")
min_distances, is_binding = calculate_vdw_based_distances(
    protein_all_atom_coords,
    protein_all_atom_names,
    ligand_coords,
    ligand_symbols,
    TOLERANCE
)

n_binding = is_binding.sum()
n_total = len(is_binding)
binding_ratio = n_binding / n_total if n_total > 0 else 0.0

print(f"结合残基数: {n_binding}/{n_total} ({binding_ratio:.1%})")
print(f"最小距离范围: [{min_distances.min():.2f}, {min_distances.max():.2f}] Å")

# 如果结合位点为 0，打印详细信息
if n_binding == 0:
    print(f"\n⚠️ 没有结合位点！")
    print(f"最小距离的前10个值: {np.sort(min_distances)[:10]}")

    # 检查第一个残基的详细计算
    print(f"\n=== 检查第一个残基的详细计算 ===")
    res_coords = protein_all_atom_coords[0]
    res_names = protein_all_atom_names[0] if len(protein_all_atom_names) > 0 else []

    if res_coords is not None and len(res_coords) > 0:
        if isinstance(res_coords, torch.Tensor):
            res_coords = res_coords.numpy()

        print(f"残基原子数: {len(res_coords)}")
        print(f"残基原子名称: {res_names[:5] if len(res_names) > 0 else '无'}")

        # 计算第一个残基到所有配体原子的距离
        min_dist_res0 = np.inf
        for atom_idx, p_coord in enumerate(res_coords):
            if atom_idx < len(res_names):
                atom_name = res_names[atom_idx]
                p_symbol = ''.join(c for c in atom_name if c.isalpha())[:2]
            else:
                p_symbol = 'C'

            p_vdw = get_vdw_radius(p_symbol)

            for j in range(len(ligand_coords)):
                l_coord = ligand_coords[j]
                l_symbol = ligand_symbols[j]
                l_vdw = get_vdw_radius(l_symbol)

                dist = np.linalg.norm(p_coord - l_coord)
                vdw_threshold = p_vdw + l_vdw + TOLERANCE

                if dist < min_dist_res0:
                    min_dist_res0 = dist
                    closest_info = (atom_name, p_symbol, p_vdw, l_symbol, l_vdw, vdw_threshold, dist)

        print(f"\n第一个残基的最近接触:")
        atom_name, p_symbol, p_vdw, l_symbol, l_vdw, vdw_threshold, dist = closest_info
        print(f"  蛋白原子: {atom_name} ({p_symbol}, VDW={p_vdw:.2f}Å)")
        print(f"  配体原子: {l_symbol} (VDW={l_vdw:.2f}Å)")
        print(f"  距离: {dist:.2f}Å")
        print(f"  VDW阈值: {vdw_threshold:.2f}Å")
        print(f"  是否结合: {dist < vdw_threshold}")
else:
    print(f"\n✓ 有 {n_binding} 个结合残基")
    binding_indices = np.where(is_binding)[0]
    print(f"结合残基索引示例: {binding_indices[:10]}")
