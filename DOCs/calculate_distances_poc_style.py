#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PoC方法计算残基-配体距离（包含氢原子）

这个脚本完全复制PoC的距离计算pipeline，但输出格式与PLC的residue_ligand_distances_v2.pt一致。
用于消融实验：对比氢原子处理对模型性能的影响。

关键差异：
- PoC方法：使用所有原子（包括氢原子）计算距离
- PLC方法：仅使用重原子计算距离

输出格式：
{
    'pdb_id': {
        'distances': torch.Tensor,  # (N,) 每个残基到配体的最小距离
        'n_residues': int,
        'min_distance': float,
        'max_distance': float,
        'mean_distance': float
    }
}
"""

import sys
from pathlib import Path
import torch
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加pypropel路径
pypropel_path = Path(__file__).parent.parent / 'pypropel_unibio'
if pypropel_path.exists():
    sys.path.insert(0, str(pypropel_path))

try:
    import pypropel.str as ppstr
    import pypropel.mol as ppmol
    import pypropel.dist as ppdist
    PYPROPEL_AVAILABLE = True
except ImportError:
    print("❌ pypropel not found. Please ensure pypropel_unibio is in the same directory.")
    PYPROPEL_AVAILABLE = False


def calculate_distances_poc_style(
    protein_pdb: str,
    ligand_file: str
) -> Optional[np.ndarray]:
    """
    使用PoC方法计算距离（包含氢原子）

    完全复制pypropel的距离计算pipeline：
    1. 使用所有原子（包括氢原子）
    2. 计算残基所有原子到配体所有原子的最小距离

    Parameters
    ----------
    protein_pdb : str
        蛋白质PDB文件路径
    ligand_file : str
        配体文件路径（.sdf或.mol2）

    Returns
    -------
    np.ndarray or None
        距离数组 (N,)，按 (chain, res_id) 排序
    """
    if not PYPROPEL_AVAILABLE:
        return None

    try:
        # 1. 加载蛋白质结构
        structure = ppstr.load_pdb(protein_pdb)

        # 2. 加载配体（尝试多种方式以处理特殊原子）
        mol = None
        if ligand_file.endswith('.sdf'):
            # 先尝试默认加载
            mol = ppmol.load_sdf(ligand_file)

            # 如果失败，尝试不sanitize加载（处理特殊原子如硼）
            if mol is None:
                from rdkit import Chem
                suppl = Chem.SDMolSupplier(ligand_file, sanitize=False, removeHs=False)
                mol = next(suppl, None)
        elif ligand_file.endswith('.mol2'):
            mol = ppmol.load_mol2(ligand_file)
        else:
            return None

        if mol is None:
            return None

        # 3. 提取配体坐标（PoC方式：所有原子）
        ligand_coords = ppmol.ligand_coords(mol)  # 包含所有原子

        # 4. 计算距离（PoC方式：使用pypropel）
        distances_df = ppdist.protein_ligand_distances(structure, ligand_coords)

        # 5. 排序：按 (chain, res_id) 确保与PLC一致
        distances_df = distances_df.sort_values(['chain', 'res_id'])

        # 6. 提取距离数组
        distances = distances_df['distance'].values.astype(np.float32)

        return distances

    except Exception as e:
        print(f"  Error: {e}")
        return None


def process_single_complex(args) -> Tuple[Optional[str], Optional[Dict]]:
    """处理单个复合物"""
    complex_dir, year_dir = args
    complex_id = complex_dir.name

    try:
        # 查找蛋白和配体文件
        protein_pdb = complex_dir / f'{complex_id}_protein.pdb'

        # 优先级：.sdf > .mol2
        ligand_file = None
        for ext in ['.sdf', '.mol2']:
            ligand_path = complex_dir / f'{complex_id}_ligand{ext}'
            if ligand_path.exists():
                ligand_file = ligand_path
                break

        if not protein_pdb.exists() or ligand_file is None:
            return None, None

        # 使用PoC方法计算距离
        distances = calculate_distances_poc_style(
            str(protein_pdb),
            str(ligand_file)
        )

        if distances is None or len(distances) == 0:
            return None, None

        # 构建结果（格式与PLC一致）
        result_data = {
            'distances': torch.from_numpy(distances).float(),
            'n_residues': len(distances),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'mean_distance': float(distances.mean())
        }

        return complex_id, result_data

    except Exception as e:
        # print(f"错误: {complex_id} - {e}")
        return None, None


def batch_calculate_distances(
    data_root: str,
    output_file: str,
    num_workers: int = 64
):
    """
    批量计算所有复合物的残基-配体距离（PoC方法）
    """
    from tqdm import tqdm

    data_root_path = Path(data_root)

    distances_dict = {}
    tasks = []

    # 1. 收集所有任务
    print("收集任务...")
    for year_dir in ['1981-2000', '2001-2010', '2011-2019']:
        year_path = data_root_path / year_dir
        if not year_path.exists():
            continue

        for complex_dir in year_path.iterdir():
            if not complex_dir.is_dir():
                continue
            tasks.append((complex_dir, year_dir))

    print(f"总共有 {len(tasks)} 个复合物待处理")
    print(f"使用 {num_workers} 个进程进行计算...")
    print(f"方法: PoC风格（包含氢原子）")

    # 2. 并行处理
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_complex, tasks),
            total=len(tasks),
            desc="计算距离 (PoC方法)"
        ))

    # 3. 汇总结果
    success_count = 0
    for complex_id, data in results:
        if complex_id is not None and data is not None:
            distances_dict[complex_id] = data
            success_count += 1

    # 保存
    torch.save(distances_dict, output_file)
    print(f"\n保存到: {output_file}")
    print(f"成功处理: {success_count} / {len(tasks)} 个复合物")

    # 统计信息
    if success_count > 0:
        all_distances = []
        for data in distances_dict.values():
            all_distances.extend(data['distances'].numpy())
        all_distances = np.array(all_distances)

        print(f"\n=== 统计信息 ===")
        print(f"总残基数: {len(all_distances)}")
        print(f"距离范围: {all_distances.min():.2f} - {all_distances.max():.2f} Å")
        print(f"平均距离: {all_distances.mean():.2f} ± {all_distances.std():.2f} Å")
        print(f"中位数: {np.median(all_distances):.2f} Å")


def compare_with_plc(poc_file: str, plc_file: str, num_samples: int = 10):
    """
    对比PoC方法和PLC方法的距离
    """
    print("\n" + "="*70)
    print("对比 PoC方法 vs PLC方法")
    print("="*70)

    # 加载数据
    poc_data = torch.load(poc_file, weights_only=False)
    plc_data = torch.load(plc_file, weights_only=False)

    print(f"\nPoC数据: {len(poc_data)} 个复合物")
    print(f"PLC数据: {len(plc_data)} 个复合物")

    # 找到共同的复合物
    common_ids = set(poc_data.keys()) & set(plc_data.keys())
    print(f"共同复合物: {len(common_ids)} 个")

    if len(common_ids) == 0:
        print("❌ 没有共同的复合物！")
        return

    # 随机选择样本
    import random
    sample_ids = random.sample(list(common_ids), min(num_samples, len(common_ids)))

    print(f"\n分析 {len(sample_ids)} 个样本...")

    all_diffs = []

    for pdb_id in sample_ids:
        poc_dist = poc_data[pdb_id]['distances'].numpy()
        plc_dist = plc_data[pdb_id]['distances'].numpy()

        if len(poc_dist) != len(plc_dist):
            print(f"  ⚠️  {pdb_id}: 残基数不匹配")
            continue

        diff = plc_dist - poc_dist  # PLC - PoC
        abs_diff = np.abs(diff)
        all_diffs.extend(abs_diff)

        print(f"\n{pdb_id}:")
        print(f"  PoC平均: {poc_dist.mean():.2f} Å")
        print(f"  PLC平均: {plc_dist.mean():.2f} Å")
        print(f"  差异: {diff.mean():.4f} Å (PLC - PoC)")
        print(f"  绝对差异: {abs_diff.mean():.4f} Å")

    # 汇总统计
    all_diffs = np.array(all_diffs)

    print(f"\n{'='*70}")
    print(f"汇总统计 ({len(sample_ids)} 个复合物)")
    print(f"{'='*70}")
    print(f"平均绝对差异: {all_diffs.mean():.4f} Å")
    print(f"中位数差异: {np.median(all_diffs):.4f} Å")
    print(f"最大差异: {all_diffs.max():.4f} Å")
    print(f"\n差异分布:")
    print(f"  < 0.1Å:  {(all_diffs < 0.1).sum()} ({(all_diffs < 0.1).sum()/len(all_diffs)*100:.1f}%)")
    print(f"  < 0.5Å:  {(all_diffs < 0.5).sum()} ({(all_diffs < 0.5).sum()/len(all_diffs)*100:.1f}%)")
    print(f"  < 1.0Å:  {(all_diffs < 1.0).sum()} ({(all_diffs < 1.0).sum()/len(all_diffs)*100:.1f}%)")
    print(f"  >= 1.0Å: {(all_diffs >= 1.0).sum()} ({(all_diffs >= 1.0).sum()/len(all_diffs)*100:.1f}%)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='使用PoC方法计算距离（包含氢原子）- 用于消融实验'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/husrcf/Code/ProtBind/New',
        help='数据根目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/husrcf/Code/ProtBind/PLC/data/residue_ligand_distances_poc_style.pt',
        help='输出文件路径'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=64,
        help='并行进程数'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='对比PoC方法和PLC方法'
    )
    parser.add_argument(
        '--plc_file',
        type=str,
        default='/home/husrcf/Code/ProtBind/PLC/data/residue_ligand_distances_v2.pt',
        help='PLC距离文件（用于对比）'
    )

    args = parser.parse_args()

    if not PYPROPEL_AVAILABLE:
        print("❌ pypropel未安装或未找到")
        print("请确保pypropel_unibio在当前目录下")
        sys.exit(1)

    print("="*70)
    print("使用PoC方法计算残基-配体距离")
    print("="*70)
    print(f"\n方法: 包含氢原子（完全复制PoC pipeline）")
    print(f"数据: {args.data_root}")
    print(f"输出: {args.output}")

    # 批量计算
    batch_calculate_distances(
        data_root=args.data_root,
        output_file=args.output,
        num_workers=args.workers
    )

    # 对比（如果指定）
    if args.compare and Path(args.plc_file).exists():
        compare_with_plc(args.output, args.plc_file)
