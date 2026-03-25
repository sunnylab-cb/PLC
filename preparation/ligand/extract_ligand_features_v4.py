#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取配体特征 V4

从P-L目录读取配体文件，提取96维per-atom特征。

特征维度：
- V2: 35维 per-atom
- V3: 41维 per-atom (35 + 6个高级原子特征)
- V4: 96维 per-atom (41 + 45维global tag + 10维shape)

其中新增高级特征包括：
- 原子SASA与暴露疏水面积
- H-bond Donor/Acceptor
- 原子手性标志
- PMI/NPR/3D跨度等几何形状特征
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence
import multiprocessing as mp

import torch
import numpy as np

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
    pypropel_path = Path(__file__).parent / 'pypropel_unibio'
    if pypropel_path.exists():
        sys.path.insert(0, str(pypropel_path))
        try:
            import pypropel.mol as ppmol
            from rdkit import Chem
            PYPROPEL_AVAILABLE = True
        except ImportError as fallback_error:
            print(f"❌ pypropel或RDKit不可用: {fallback_error}")
            import traceback
            traceback.print_exc()
            PYPROPEL_AVAILABLE = False
    else:
        print(f"❌ pypropel或RDKit不可用: {e}")
        import traceback
        traceback.print_exc()
        PYPROPEL_AVAILABLE = False

try:
    from ligand_features_v3 import extract_ligand_features_from_file as extract_v3_features
    V3_AVAILABLE = True
except ImportError:
    print("❌ ligand_features_v3不可用")
    V3_AVAILABLE = False


def _load_mol_for_global_features(ligand_file: str) -> "Chem.Mol":
    if not PYPROPEL_AVAILABLE:
        raise ImportError("pypropel和RDKit是必需的")

    ligand_path = Path(ligand_file)
    suffix = ligand_path.suffix.lower()
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=False)
        mol = next(suppl, None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_path), sanitize=False, removeHs=False)
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand_path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {ligand_file}")

    if mol is None:
        raise ValueError(f"无法加载配体: {ligand_file}")

    return mol


def _shape_feature_vector(features: dict) -> np.ndarray:
    """
    Extract and normalize shape features.

    Shape features (10-dim):
    - 0: PMI1 (主惯性矩1)
    - 1: PMI2 (主惯性矩2)
    - 2: PMI3 (主惯性矩3)
    - 3: Spherocity (球形度) - already normalized
    - 4: Radius of Gyration (回转半径)
    - 5: NPR1 (归一化主惯性比1) - already normalized
    - 6: NPR2 (归一化主惯性比2) - already normalized
    - 7: Span L (主轴跨度-长)
    - 8: Span W (主轴跨度-宽)
    - 9: Span H (主轴跨度-高)

    Normalization parameters computed from 19,037 samples:
    """
    # Normalization parameters (mean, std) for features that need normalization
    NORM_PARAMS = {
        0: (3879.901123, 6650.009766),    # PMI1
        1: (13954.526367, 28178.578125),  # PMI2
        2: (15758.538086, 30137.193359),  # PMI3
        4: (4.633801, 1.638986),          # Radius of Gyration
        7: (14.585329, 5.682965),         # Span L
        8: (8.701250, 3.143703),          # Span W
        9: (5.480688, 2.575974),          # Span H
    }

    shape_features = np.asarray(features.get('shape_features', np.zeros(10, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if shape_features.size != 10:
        output = np.zeros(10, dtype=np.float32)
        copy_size = min(10, shape_features.size)
        output[:copy_size] = shape_features[:copy_size]
        shape_features = output

    # Apply z-score normalization to specified dimensions
    for dim, (mean, std) in NORM_PARAMS.items():
        if std > 0:
            shape_features[dim] = (shape_features[dim] - mean) / std

    return shape_features


def extract_ligand_features_v4(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> dict:
    if not PYPROPEL_AVAILABLE or not V3_AVAILABLE:
        raise ImportError("pypropel、RDKit和ligand_features_v3是必需的")

    v3_features = extract_v3_features(
        ligand_file,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )

    mol = _load_mol_for_global_features(ligand_file)
    global_features = np.asarray(ppmol.get_global_tag(mol, output_dim=45), dtype=np.float32)
    shape_features = _shape_feature_vector(v3_features)

    v3_atom_features = np.asarray(v3_features['atom_features'], dtype=np.float32)
    n_atoms = v3_atom_features.shape[0]

    global_features_broadcast = np.tile(global_features, (n_atoms, 1))
    shape_features_broadcast = np.tile(shape_features, (n_atoms, 1))

    atom_features_v4 = np.concatenate(
        [v3_atom_features, global_features_broadcast, shape_features_broadcast],
        axis=1
    ).astype(np.float32)

    result = v3_features.copy()
    result['atom_features'] = atom_features_v4
    result['global_features'] = global_features
    result['shape_features'] = shape_features
    return result


def extract_ligand_features_single(
    ligand_file: Path,
    ligand_root: Path,
    output_dir: Path,
    num_conformers: int,
    overwrite: bool,
    no_conformers: bool = False,
    strict: bool = True,
) -> tuple:
    try:
        primary_file = ligand_file

        rel_path = primary_file.relative_to(ligand_root)
        out_path = (output_dir / rel_path).with_suffix('.pt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            return (True, f"SKIP {rel_path}")

        features = extract_ligand_features_v4(
            str(primary_file),
            include_conformers=(not no_conformers and num_conformers > 0),
            num_conformers=num_conformers,
            strict=strict,
            verbose=False,
            num_threads=0,
        )

        data = {
            'atom_features': torch.from_numpy(features['atom_features']),
            'edge_index': torch.from_numpy(features['edge_index']).long(),
            'edge_attr': torch.from_numpy(features['edge_attr']).float(),
            'coords': torch.from_numpy(features['coords']),
            'coords_heavy_current': torch.from_numpy(features['coords_heavy_current']),
            'gasteiger_charges': torch.from_numpy(features['gasteiger_charges']),
            'logp_contribs': torch.from_numpy(features['logp_contribs']),
            'tpsa_contribs': torch.from_numpy(features['tpsa_contribs']),
            'rmsf': torch.from_numpy(features['rmsf']),
            'flexibility_flag': torch.from_numpy(features['flexibility_flag']),
            'atom_sasa': torch.from_numpy(features['atom_sasa']),
            'lipophilic_sasa': torch.from_numpy(features['lipophilic_sasa']),
            'hbond_donor': torch.from_numpy(features['hbond_donor']),
            'hbond_acceptor': torch.from_numpy(features['hbond_acceptor']),
            'chirality_cw': torch.from_numpy(features['chirality_cw']),
            'chirality_ccw': torch.from_numpy(features['chirality_ccw']),
            'shape_features': torch.from_numpy(features['shape_features']),
            'global_features': torch.from_numpy(features['global_features']),
            'num_atoms': features['num_atoms'],
            'num_heavy_atoms': features['num_heavy_atoms'],
            'has_original_coords': features['has_original_coords'],
            'molecular_descriptors': features['molecular_descriptors'],
            'ligand_path': str(primary_file)
        }

        if features['coords_original'] is not None:
            data['coords_original'] = torch.from_numpy(features['coords_original'])

        if features.get('coords_original_with_h') is not None:
            data['coords_original_with_h'] = torch.from_numpy(features['coords_original_with_h'])

        torch.save(data, out_path)

        ligand_name = primary_file.stem.replace('_ligand', '')
        return (True, f"{ligand_name} ({features['num_atoms']} atoms, 96-dim)")

    except Exception as e:
        ligand_name = ligand_file.stem.replace('_ligand', '')
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
    strict: bool = True,
) -> None:
    only_ids_set = {s.lower() for s in only_ids} if only_ids else None
    ligand_files_dict: dict[str, Path] = {}

    for ext in ['.sdf', '.mol2', '.pdb', '.mol']:
        for file_path in ligand_root.rglob(f"*_ligand{ext}"):
            protein_id = file_path.stem.replace('_ligand', '').lower()
            if only_ids_set is not None and protein_id not in only_ids_set:
                continue
            if protein_id not in ligand_files_dict:
                ligand_files_dict[protein_id] = file_path

    ligand_files = sorted(ligand_files_dict.values())

    if max_files:
        ligand_files = ligand_files[:max_files]

    if not ligand_files:
        print(f"No ligand files found in {ligand_root}")
        return

    print(f"Found {len(ligand_files)} ligand files")
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(ligand_files))

    print(f"Using {n_workers} worker processes")
    print(f"Extracting 96D per-atom features (41D V3 + 45D global tag + 10D shape) + {num_conformers} conformers")

    if n_workers == 1:
        success_count = 0
        fail_count = 0
        skip_count = 0

        iterator = tqdm(ligand_files, desc="Processing", unit="file") if TQDM_AVAILABLE else ligand_files

        for idx, ligand_file in enumerate(iterator, 1):
            success, message = extract_ligand_features_single(
                ligand_file, ligand_root, output_dir, num_conformers, overwrite, no_conformers, strict
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
        success_count = 0
        fail_count = 0
        skip_count = 0

        tasks = [(f, ligand_root, output_dir, num_conformers, overwrite, no_conformers, strict) for f in ligand_files]

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
        description="批量提取配体特征 V4 (96D = 41D V3 + 45D global + 10D shape)"
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
        default=None,
        help="输出目录 (默认: 输入目录同级 ligand_features_v4)"
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
    parser.add_argument(
        "--loose",
        action="store_true",
        help="宽松模式：允许构象/MMFF降级，不删除氢原子"
    )

    args = parser.parse_args()

    if not PYPROPEL_AVAILABLE:
        print("❌ pypropel和RDKit是必需的")
        print("请确保pypropel_unibio在当前目录下")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else (input_path.resolve().parent / "ligand_features_v4")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Processing ligands from: {input_path}")
    print(f"Output directory: {output_path}")
    print("Feature dimension: 96 (41 V3 + 45 global + 10 shape)")

    batch_extract_ligand_features(
        ligand_root=input_path,
        output_dir=output_path,
        num_conformers=args.conformers,
        max_files=args.max,
        overwrite=args.overwrite,
        n_workers=args.workers,
        no_conformers=args.no_conformers,
        only_ids=args.ids,
        strict=not args.loose,
    )


if __name__ == "__main__":
    main()
