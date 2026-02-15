#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取配体特征 V3 (PoC风格 - 64维特征)

从P-L目录读取配体文件，提取64维per-atom特征（包含45维全局特征）
✅ 使用pypropel的get_atom_features_extended
✅ 保留原始实验坐标用于距离计算
✅ 添加多种坐标版本和分子描述符

特征维度：
- V2: 35维 per-atom特征
- V3: 80维 per-atom特征 (35维V2特征 + 45维全局特征)

全局特征 (45维):
- [0-42] Morgan指纹 (43维)
- [43] LogP
- [44] TPSA (归一化)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence
import multiprocessing as mp
import torch
import numpy as np

# 添加pypropel路径
pypropel_path = Path(__file__).parent / 'pypropel_unibio'
if pypropel_path.exists():
    sys.path.insert(0, str(pypropel_path))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import pypropel.mol as ppmol
    from rdkit import Chem
    PYPROPEL_AVAILABLE = True
except ImportError as e:
    print(f"❌ pypropel或RDKit不可用: {e}")
    import traceback
    traceback.print_exc()
    PYPROPEL_AVAILABLE = False

# 导入V2特征提取模块
try:
    from ligand_features_v2 import extract_ligand_features_from_file as extract_v2_features
    V2_AVAILABLE = True
except ImportError:
    print("❌ ligand_features_v2不可用")
    V2_AVAILABLE = False


def _try_load_rdkit_sdf(ligand_file: str, *, sanitize: bool) -> Optional["Chem.Mol"]:
    try:
        suppl = Chem.SDMolSupplier(ligand_file, sanitize=sanitize, removeHs=False)
        return next(suppl, None)
    except Exception:
        return None


def _try_load_rdkit_mol2(ligand_file: str) -> Optional["Chem.Mol"]:
    try:
        return Chem.MolFromMol2File(ligand_file, removeHs=False)
    except Exception:
        return None


def _load_mol_for_global_features(ligand_file: str) -> "Chem.Mol":
    """
    Load a molecule robustly for global-feature extraction.

    Important: V3 needs a valid `mol` *in addition* to V2 extraction, so we
    implement the same fallback strategy here (sanitize=False and mol2 fallback).
    """
    if not PYPROPEL_AVAILABLE:
        raise ImportError("pypropel和RDKit是必需的")

    ligand_path = Path(ligand_file)
    suffix = ligand_path.suffix.lower()

    mol = None

    if suffix == ".sdf":
        # Primary: pypropel loader (fast path)
        try:
            mol = ppmol.load_sdf(str(ligand_path))
        except Exception:
            mol = None

        # Fallback 1: RDKit with sanitize=False (handles special atoms / valence)
        if mol is None:
            mol = _try_load_rdkit_sdf(str(ligand_path), sanitize=False)

        # Fallback 2: try sibling mol2 if SDF is corrupted / inconsistent
        if mol is None:
            mol2_path = ligand_path.with_suffix(".mol2")
            if mol2_path.exists():
                try:
                    mol = ppmol.load_mol2(str(mol2_path))
                except Exception:
                    mol = None
                if mol is None:
                    mol = _try_load_rdkit_mol2(str(mol2_path))

    elif suffix == ".mol2":
        try:
            mol = ppmol.load_mol2(str(ligand_path))
        except Exception:
            mol = None
        if mol is None:
            mol = _try_load_rdkit_mol2(str(ligand_path))

    elif suffix == ".mol":
        try:
            mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
        except Exception:
            mol = None
        if mol is None:
            try:
                mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False, removeHs=False)
            except Exception:
                mol = None

    elif suffix == ".pdb":
        try:
            mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=False)
        except Exception:
            mol = None

    else:
        raise ValueError(f"Unsupported file format: {ligand_file}")

    if mol is None:
        raise ValueError(f"无法加载配体: {ligand_file}")

    return mol


def extract_ligand_features_v3(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50
) -> dict:
    """
    提取配体特征 V3 (80维 = 35维V2 + 45维全局特征)

    Parameters
    ----------
    ligand_file : str
        配体文件路径 (.sdf, .mol2, etc.)
    include_conformers : bool
        是否生成构象用于柔性分析
    num_conformers : int
        生成的构象数量

    Returns
    -------
    dict
        包含以下字段:
        - atom_features: (N, 80) 原子特征（35维V2 + 45维全局特征）
        - coords: (N, 3) 当前坐标（包含氢原子）
        - coords_original: (N_heavy, 3) 原始实验坐标（仅重原子）
        - coords_heavy_current: (N_heavy, 3) 当前重原子坐标
        - gasteiger_charges: (N,) Gasteiger电荷
        - logp_contribs: (N,) LogP贡献
        - tpsa_contribs: (N,) TPSA贡献
        - rmsf: (N,) 原子柔性
        - flexibility_flag: (N,) 柔性标记
        - num_atoms: int 总原子数
        - num_heavy_atoms: int 重原子数
        - has_original_coords: bool 是否有原始坐标
        - molecular_descriptors: dict 分子描述符
        - global_features: (45,) 全局特征
        - ligand_path: str 文件路径
    """
    if not PYPROPEL_AVAILABLE or not V2_AVAILABLE:
        raise ImportError("pypropel、RDKit和ligand_features_v2是必需的")

    # 1. 使用V2方法提取35维特征和其他信息
    v2_features = extract_v2_features(
        ligand_file,
        include_conformers=include_conformers,
        num_conformers=num_conformers
    )

    # 2. 加载分子以提取全局特征（需要单独的robust加载）
    mol = _load_mol_for_global_features(ligand_file)

    # 3. 提取45维全局特征
    global_features = ppmol.get_global_tag(mol, output_dim=45)  # (45,)

    # 4. 合并特征：35维V2特征 + 45维全局特征 = 80维
    v2_atom_features = v2_features['atom_features']  # (N, 35)
    n_atoms = v2_atom_features.shape[0]

    # 将全局特征广播到每个原子
    global_features_broadcast = np.tile(global_features, (n_atoms, 1))  # (N, 45)

    # 拼接特征
    atom_features_v3 = np.concatenate([v2_atom_features, global_features_broadcast], axis=1)  # (N, 80)

    # 5. 返回结果（保留V2的所有其他字段，只替换atom_features）
    result = v2_features.copy()
    result['atom_features'] = atom_features_v3.astype(np.float32)  # (N, 80)
    result['global_features'] = global_features.astype(np.float32)  # (45,)

    return result


def extract_ligand_features_single(
    ligand_files: Sequence[Path] | Path,
    ligand_root: Path,
    output_dir: Path,
    num_conformers: int,
    overwrite: bool,
    no_conformers: bool = False
) -> tuple:
    """
    处理单个配体文件
    """
    try:
        if isinstance(ligand_files, Path):
            candidates: Sequence[Path] = [ligand_files]
        else:
            candidates = list(ligand_files)
        if not candidates:
            return (False, "empty input candidates")

        primary_file = candidates[0]

        # 构建输出路径
        rel_path = primary_file.relative_to(ligand_root)
        out_path = (output_dir / rel_path).with_suffix('.pt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否已存在
        if out_path.exists() and not overwrite:
            return (True, f"SKIP {rel_path}")

        # 提取特征（如果SDF失败，尝试mol2等备选）
        last_error: Optional[Exception] = None
        features = None
        used_file: Optional[Path] = None
        for candidate in candidates:
            try:
                features = extract_ligand_features_v3(
                    str(candidate),
                    include_conformers=(not no_conformers and num_conformers > 0),
                    num_conformers=num_conformers
                )
                used_file = candidate
                break
            except Exception as e:
                last_error = e

        if features is None:
            raise last_error if last_error is not None else RuntimeError("feature extraction failed")

        # 转换为torch张量
        data = {
            # 原子级别特征 (64维)
            'atom_features': torch.from_numpy(features['atom_features']),

            # 坐标信息（多个版本）
            'coords': torch.from_numpy(features['coords']),
            'coords_heavy_current': torch.from_numpy(features['coords_heavy_current']),

            # 电子效应特征
            'gasteiger_charges': torch.from_numpy(features['gasteiger_charges']),
            'logp_contribs': torch.from_numpy(features['logp_contribs']),
            'tpsa_contribs': torch.from_numpy(features['tpsa_contribs']),

            # 柔性特征
            'rmsf': torch.from_numpy(features['rmsf']),
            'flexibility_flag': torch.from_numpy(features['flexibility_flag']),

            # 元数据
            'num_atoms': features['num_atoms'],
            'num_heavy_atoms': features['num_heavy_atoms'],
            'has_original_coords': features['has_original_coords'],

            # 分子级别描述符
            'molecular_descriptors': features['molecular_descriptors'],

            # 全局特征 (45维)
            'global_features': torch.from_numpy(features['global_features']),

            # 路径
            'ligand_path': str(used_file if used_file is not None else primary_file)
        }

        # ✅ 保存原始实验坐标（关键！）
        if features['coords_original'] is not None:
            data['coords_original'] = torch.from_numpy(features['coords_original'])

        # ✅ 保存包含氢的原始坐标（用于全局特征计算）
        if features.get('coords_original_with_h') is not None:
            data['coords_original_with_h'] = torch.from_numpy(features['coords_original_with_h'])

        # 保存
        torch.save(data, out_path)

        ligand_name = primary_file.stem.replace('_ligand', '')
        if used_file is not None and used_file != primary_file:
            return (True, f"{ligand_name} ({features['num_atoms']} atoms, 80-dim) [fallback:{used_file.suffix}]")
        return (True, f"{ligand_name} ({features['num_atoms']} atoms, 80-dim)")

    except Exception as e:
        if isinstance(ligand_files, Path):
            ligand_name = ligand_files.stem.replace('_ligand', '')
        else:
            ligand_name = (ligand_files[0].stem.replace('_ligand', '') if ligand_files else "unknown")
        import traceback
        error_msg = traceback.format_exc()
        return (False, f"{ligand_name}: {str(e)[:100]}\n{error_msg[:200]}")


def batch_extract_ligand_features(
    ligand_root: Path,
    output_dir: Path,
    pattern: str = "*_ligand.*",
    num_conformers: int = 50,
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None,
    no_conformers: bool = False,
    only_ids: Optional[Iterable[str]] = None,
) -> None:
    """
    批量提取配体特征
    """
    # 查找所有配体文件，优先选择.sdf格式
    only_ids_set = {s.lower() for s in only_ids} if only_ids else None
    ligand_files_dict: dict[str, list[Path]] = {}  # key=pdb_id -> candidates in priority order

    # 优先级：.sdf > .mol2 > .pdb > .mol
    for ext in ['.sdf', '.mol2', '.pdb', '.mol']:
        for file_path in ligand_root.rglob(f"*_ligand{ext}"):
            # 提取蛋白ID（去掉_ligand和扩展名）
            protein_id = file_path.stem.replace('_ligand', '').lower()
            if only_ids_set is not None and protein_id not in only_ids_set:
                continue
            ligand_files_dict.setdefault(protein_id, [])
            if file_path not in ligand_files_dict[protein_id]:
                ligand_files_dict[protein_id].append(file_path)

    ligand_items = sorted(ligand_files_dict.items(), key=lambda kv: str(kv[1][0]))
    ligand_files: list[Sequence[Path]] = [candidates for _, candidates in ligand_items]

    if max_files:
        ligand_files = ligand_files[:max_files]

    if not ligand_files:
        print(f"No ligand files found in {ligand_root}")
        return

    print(f"Found {len(ligand_files)} ligand files")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(ligand_files))

    print(f"Using {n_workers} worker processes")
    print(f"Extracting 80D per-atom features (35D V2 + 45D global tag) + {num_conformers} conformers")

    if n_workers == 1:
        # 单进程模式
        success_count = 0
        fail_count = 0
        skip_count = 0

        iterator = tqdm(ligand_files, desc="Processing", unit="file") if TQDM_AVAILABLE else ligand_files

        for idx, ligand_file in enumerate(iterator, 1):
            success, message = extract_ligand_features_single(
                ligand_file, ligand_root, output_dir, num_conformers, overwrite, no_conformers
            )

            if TQDM_AVAILABLE:
                if "SKIP" in message:
                    skip_count += 1
                elif success:
                    success_count += 1
                else:
                    fail_count += 1
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"[{idx}/{len(ligand_files)}] {status} {message}")
                if "SKIP" in message:
                    skip_count += 1
                elif success:
                    success_count += 1
                else:
                    fail_count += 1

    else:
        # 多进程模式
        success_count = 0
        fail_count = 0
        skip_count = 0

        tasks = [(f, ligand_root, output_dir, num_conformers, overwrite, no_conformers) for f in ligand_files]

        with mp.Pool(processes=n_workers) as pool:
            if TQDM_AVAILABLE:
                results = pool.starmap(extract_ligand_features_single, tasks)
                iterator = tqdm(results, total=len(tasks), desc="Processing", unit="file")

                for idx, (success, message) in enumerate(iterator, 1):
                    if "SKIP" in message:
                        skip_count += 1
                    elif success:
                        success_count += 1
                    else:
                        fail_count += 1
                    iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                results = pool.starmap(extract_ligand_features_single, tasks)

                for idx, (success, message) in enumerate(results, 1):
                    status = "✓" if success else "✗"
                    print(f"[{idx}/{len(ligand_files)}] {status} {message}")

                    if success:
                        if "SKIP" in message:
                            skip_count += 1
                        else:
                            success_count += 1
                    else:
                        fail_count += 1

    print(f"\n=== Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total:   {len(ligand_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="批量提取配体特征 V3 (80D = 35D V2 + 45D global)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="配体文件根目录 (P-L)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录 (ligand_features_v3)"
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=50,
        help="生成的构象数量 (默认50，设为0则只使用实验结构)"
    )
    parser.add_argument(
        "--no-conformers",
        action="store_true",
        help="不生成构象，只使用实验结构（更快，但无柔性分析）"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="最大处理文件数"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的文件"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认=CPU核心数）"
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="只处理指定PDB ID（例如: --ids 1lvk 2pll）"
    )

    args = parser.parse_args()

    if not PYPROPEL_AVAILABLE:
        print("❌ pypropel和RDKit是必需的")
        print("请确保pypropel_unibio在当前目录下")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Processing ligands from: {input_path}")
    print(f"Feature dimension: 80 (35 V2 + 45 global)")

    batch_extract_ligand_features(
        ligand_root=input_path,
        output_dir=output_path,
        num_conformers=args.conformers,
        max_files=args.max,
        overwrite=args.overwrite,
        n_workers=args.workers,
        no_conformers=args.no_conformers,
        only_ids=args.ids,
    )


if __name__ == "__main__":
    main()
