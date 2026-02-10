#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给现有的 ligand_features_v3 文件添加氢原子坐标

由于全局特征计算使用了氢原子，我们需要确保坐标也包含氢原子。
这个脚本会重新加载原始配体文件，提取包含氢的原始坐标，并更新 pt 文件。
"""

import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# 添加pypropel路径
pypropel_path = Path(__file__).parent.parent.parent / 'pypropel_unibio'
if pypropel_path.exists():
    sys.path.insert(0, str(pypropel_path))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pypropel.mol as ppmol
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"❌ RDKit或pypropel不可用: {e}")
    RDKIT_AVAILABLE = False


def extract_original_coords_with_h(ligand_file: Path):
    """
    从原始配体文件提取包含氢的坐标

    Returns:
        coords_original_with_h: numpy array [N_original, 3] 或 None
    """
    if not ligand_file.exists():
        return None

    try:
        # 加载分子
        mol = None
        if ligand_file.suffix == '.sdf':
            mol = ppmol.load_sdf(str(ligand_file))
            if mol is None:
                suppl = Chem.SDMolSupplier(str(ligand_file), sanitize=False, removeHs=False)
                mol = next(suppl, None)
        elif ligand_file.suffix == '.mol2':
            mol = ppmol.load_mol2(str(ligand_file))

        if mol is None:
            return None

        # 检查是否有3D坐标
        if mol.GetNumConformers() == 0:
            return None

        # 获取原始坐标（包含原始文件中的所有原子）
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()

        return coords

    except Exception as e:
        print(f"  ⚠️  提取坐标失败 {ligand_file.name}: {e}")
        return None


def process_single_file(args_tuple):
    """处理单个 pt 文件"""
    pt_file, ligand_root, overwrite = args_tuple

    try:
        # 加载现有的 pt 文件
        data = torch.load(pt_file, weights_only=False)

        # 检查是否已有 coords_original_with_h
        if 'coords_original_with_h' in data and not overwrite:
            return (True, f"SKIP {pt_file.stem} (已有氢坐标)")

        # 找到对应的原始配体文件
        # pt文件路径: ligand_features_v3/YEAR/PDB_ID/PDB_ID_ligand.pt
        # 原始文件路径: P-L/YEAR/PDB_ID/PDB_ID_ligand.sdf
        pdb_id = pt_file.stem.replace('_ligand', '')
        year_range = pt_file.parent.parent.name

        # 尝试不同的扩展名
        ligand_file = None
        for ext in ['.sdf', '.mol2', '.pdb', '.mol']:
            test_path = ligand_root / year_range / pdb_id / f'{pdb_id}_ligand{ext}'
            if test_path.exists():
                ligand_file = test_path
                break

        if ligand_file is None:
            return (False, f"{pdb_id}: 找不到原始配体文件")

        # 提取包含氢的原始坐标
        coords_with_h = extract_original_coords_with_h(ligand_file)

        if coords_with_h is None:
            return (False, f"{pdb_id}: 无法提取氢坐标")

        # 更新 pt 文件
        data['coords_original_with_h'] = torch.from_numpy(coords_with_h).float()
        torch.save(data, pt_file)

        return (True, f"{pdb_id}: 添加氢坐标 {coords_with_h.shape[0]} 原子")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{pt_file.stem}: {str(e)[:100]}\n{error_msg[:200]}")


def batch_add_h_coords(
    ligand_features_dir: Path,
    ligand_root: Path,
    overwrite: bool = False,
    n_workers: int = None
):
    """批量添加氢坐标"""

    # 收集所有 pt 文件
    pt_files = list(ligand_features_dir.rglob('*_ligand.pt'))

    if not pt_files:
        print(f"未找到 pt 文件: {ligand_features_dir}")
        return

    print(f"找到 {len(pt_files)} 个 pt 文件")

    # 确定进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(pt_files))

    print(f"使用 {n_workers} 个工作进程")

    # 准备任务
    tasks = [(f, ligand_root, overwrite) for f in pt_files]

    success_count = 0
    fail_count = 0
    skip_count = 0

    if n_workers == 1:
        # 单进程模式
        for task in tqdm(tasks, desc="处理中"):
            success, message = process_single_file(task)
            if "SKIP" in message:
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1
                print(f"  ✗ {message}")
    else:
        # 多进程模式
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, tasks),
                total=len(tasks),
                desc="处理中"
            ))

            for success, message in results:
                if "SKIP" in message:
                    skip_count += 1
                elif success:
                    success_count += 1
                else:
                    fail_count += 1
                    if fail_count <= 10:  # 只打印前10个错误
                        print(f"  ✗ {message}")

    print(f"\n=== 总结 ===")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skip_count}")
    print(f"总计: {len(pt_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="给现有的 ligand_features_v3 文件添加氢原子坐标"
    )
    parser.add_argument(
        "--ligand-features",
        type=str,
        required=True,
        help="ligand_features_v3 目录路径"
    )
    parser.add_argument(
        "--ligand-root",
        type=str,
        required=True,
        help="原始配体文件根目录 (P-L)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的氢坐标"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认=CPU核心数）"
    )

    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("❌ RDKit和pypropel是必需的")
        sys.exit(1)

    ligand_features_dir = Path(args.ligand_features)
    ligand_root = Path(args.ligand_root)

    if not ligand_features_dir.exists():
        raise FileNotFoundError(f"目录不存在: {ligand_features_dir}")

    if not ligand_root.exists():
        raise FileNotFoundError(f"目录不存在: {ligand_root}")

    print(f"Ligand features: {ligand_features_dir}")
    print(f"Ligand root: {ligand_root}")

    batch_add_h_coords(
        ligand_features_dir=ligand_features_dir,
        ligand_root=ligand_root,
        overwrite=args.overwrite,
        n_workers=args.workers
    )


if __name__ == "__main__":
    main()
