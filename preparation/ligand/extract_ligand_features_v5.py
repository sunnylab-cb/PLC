#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extract ligand features V5.

Independent pipeline:
- local atom features from ligand_features_v4.py
- global molecular tag (45D)
- normalized shape features (10D)

Final per-atom feature dimension:
- 54D local atom features
- 45D global tag
- 10D shape
= 109D
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

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
    pypropel_path = Path(__file__).parent / "pypropel_unibio"
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
    from ligand_features_v4 import extract_ligand_features_from_file as extract_v4_features
    V4_AVAILABLE = True
except ImportError:
    print("❌ ligand_features_v4不可用")
    V4_AVAILABLE = False


def _load_mol_for_global_features(ligand_file: str) -> "Chem.Mol":
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


def _shape_feature_vector(shape_features: np.ndarray) -> np.ndarray:
    norm_params = {
        0: (3879.901123, 6650.009766),
        1: (13954.526367, 28178.578125),
        2: (15758.538086, 30137.193359),
        4: (4.633801, 1.638986),
        7: (14.585329, 5.682965),
        8: (8.701250, 3.143703),
        9: (5.480688, 2.575974),
    }

    feats = np.asarray(shape_features, dtype=np.float32).reshape(-1)
    if feats.size != 10:
        output = np.zeros(10, dtype=np.float32)
        output[:min(10, feats.size)] = feats[:min(10, feats.size)]
        feats = output

    for dim, (mean, std) in norm_params.items():
        if std > 0:
            feats[dim] = (feats[dim] - mean) / std
    return feats


def extract_ligand_features_v5(
    ligand_file: str,
    include_conformers: bool = True,
    num_conformers: int = 50,
    strict: bool = True,
    verbose: bool = False,
    num_threads: int = 0,
) -> dict:
    if not PYPROPEL_AVAILABLE or not V4_AVAILABLE:
        raise ImportError("pypropel、RDKit和ligand_features_v4是必需的")

    v4_features = extract_v4_features(
        ligand_file,
        include_conformers=include_conformers,
        num_conformers=num_conformers,
        use_existing_coords=True,
        strict=strict,
        verbose=verbose,
        num_threads=num_threads,
    )

    mol = _load_mol_for_global_features(ligand_file)
    global_features = np.asarray(ppmol.get_global_tag(mol, output_dim=45), dtype=np.float32)
    shape_features = _shape_feature_vector(v4_features["shape_features"])

    local_atom_features = np.asarray(v4_features["atom_features"], dtype=np.float32)
    n_atoms = local_atom_features.shape[0]

    global_broadcast = np.tile(global_features, (n_atoms, 1))
    shape_broadcast = np.tile(shape_features, (n_atoms, 1))

    atom_features_v5 = np.concatenate(
        [local_atom_features, global_broadcast, shape_broadcast],
        axis=1,
    ).astype(np.float32)

    result = v4_features.copy()
    result["atom_features"] = atom_features_v5
    result["global_features"] = global_features
    result["shape_features"] = shape_features
    result["feature_dim_total"] = atom_features_v5.shape[1]
    return result


def extract_ligand_features_single(
    ligand_file: Path,
    ligand_root: Path,
    output_dir: Path,
    num_conformers: int,
    overwrite: bool,
    no_conformers: bool = False,
    strict: bool = True,
    num_threads: int = 0,
) -> tuple:
    try:
        rel_path = ligand_file.relative_to(ligand_root)
        out_path = (output_dir / rel_path).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            return True, f"SKIP {rel_path}"

        features = extract_ligand_features_v5(
            str(ligand_file),
            include_conformers=(not no_conformers and num_conformers > 0),
            num_conformers=num_conformers,
            strict=strict,
            verbose=False,
            num_threads=num_threads,
        )

        data = {
            "atom_features": torch.from_numpy(features["atom_features"]),
            "edge_index": torch.from_numpy(features["edge_index"]).long(),
            "edge_attr": torch.from_numpy(features["edge_attr"]).float(),
            "coords": torch.from_numpy(features["coords"]),
            "coords_heavy_current": torch.from_numpy(features["coords_heavy_current"]),
            "gasteiger_charges": torch.from_numpy(features["gasteiger_charges"]),
            "logp_contribs": torch.from_numpy(features["logp_contribs"]),
            "tpsa_contribs": torch.from_numpy(features["tpsa_contribs"]),
            "rmsf": torch.from_numpy(features["rmsf"]),
            "flexibility_flag": torch.from_numpy(features["flexibility_flag"]),
            "atom_sasa": torch.from_numpy(features["atom_sasa"]),
            "lipophilic_sasa": torch.from_numpy(features["lipophilic_sasa"]),
            "hbond_donor": torch.from_numpy(features["hbond_donor"]),
            "hbond_acceptor": torch.from_numpy(features["hbond_acceptor"]),
            "chirality_cw": torch.from_numpy(features["chirality_cw"]),
            "chirality_ccw": torch.from_numpy(features["chirality_ccw"]),
            "rotatable_bond_context": torch.from_numpy(features["rotatable_bond_context"]),
            "formal_charge_class": torch.from_numpy(features["formal_charge_class"]),
            "metal_binding_flag": torch.from_numpy(features["metal_binding_flag"]),
            "stereochemistry_flag": torch.from_numpy(features["stereochemistry_flag"]),
            "shape_features": torch.from_numpy(features["shape_features"]),
            "global_features": torch.from_numpy(features["global_features"]),
            "num_atoms": features["num_atoms"],
            "num_heavy_atoms": features["num_heavy_atoms"],
            "has_original_coords": features["has_original_coords"],
            "molecular_descriptors": features["molecular_descriptors"],
            "ligand_path": str(ligand_file),
            "feature_dim_local": features["feature_dim_local"],
            "feature_dim_total": features["feature_dim_total"],
        }

        if features["coords_original"] is not None:
            data["coords_original"] = torch.from_numpy(features["coords_original"])
        if features.get("coords_original_with_h") is not None:
            data["coords_original_with_h"] = torch.from_numpy(features["coords_original_with_h"])

        torch.save(data, out_path)

        ligand_name = ligand_file.stem.replace("_ligand", "")
        return True, f"{ligand_name} ({features['num_atoms']} atoms, {features['feature_dim_total']}-dim)"
    except Exception as e:
        ligand_name = ligand_file.stem.replace("_ligand", "")
        import traceback
        error_msg = traceback.format_exc()
        return False, f"{ligand_name}: {str(e)[:100]}\n{error_msg[:200]}"


def batch_extract_ligand_features(
    ligand_root: Path,
    output_dir: Path,
    num_conformers: int = 50,
    max_files: int = None,
    overwrite: bool = False,
    n_workers: int = None,
    no_conformers: bool = False,
    only_ids: Optional[Iterable[str]] = None,
    strict: bool = True,
    num_threads: int = 0,
    flat: bool = False,
) -> None:
    only_ids_set = {s.lower() for s in only_ids} if only_ids else None
    ligand_files_dict: dict[str, Path] = {}

    for ext in [".sdf", ".mol2", ".pdb", ".mol"]:
        pattern = f"*{ext}" if flat else f"*_ligand{ext}"
        for file_path in ligand_root.rglob(pattern):
            if flat:
                protein_id = file_path.stem.split("_", 1)[0].lower()
                dedupe_key = file_path.stem.lower()
            else:
                protein_id = file_path.stem.replace("_ligand", "").lower()
                dedupe_key = protein_id
            if only_ids_set is not None and protein_id not in only_ids_set:
                continue
            if dedupe_key not in ligand_files_dict:
                ligand_files_dict[dedupe_key] = file_path

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
    print(
        "Extracting 109D per-atom features "
        "(54D local + 45D global tag + 10D shape)"
        f" + {num_conformers} conformers"
    )
    print(f"Flat scan mode: {flat}")
    print(f"Strict mode: {strict}")
    print(f"RDKit threads per task: {num_threads}")

    success_count = 0
    fail_count = 0
    skip_count = 0

    if n_workers == 1:
        iterator = tqdm(ligand_files, desc="Processing", unit="file") if TQDM_AVAILABLE else ligand_files
        for ligand_file in iterator:
            success, message = extract_ligand_features_single(
                ligand_file, ligand_root, output_dir, num_conformers, overwrite,
                no_conformers, strict, num_threads
            )
            if "SKIP" in message:
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1
            if TQDM_AVAILABLE:
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"{status} {message}")
    else:
        tasks = [
            (f, ligand_root, output_dir, num_conformers, overwrite, no_conformers, strict, num_threads)
            for f in ligand_files
        ]
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(extract_ligand_features_single, tasks)
        iterator = tqdm(results, total=len(tasks), desc="Processing", unit="file") if TQDM_AVAILABLE else results
        for result in iterator:
            success, message = result
            if "SKIP" in message:
                skip_count += 1
            elif success:
                success_count += 1
            else:
                fail_count += 1
            if TQDM_AVAILABLE:
                iterator.set_postfix({"success": success_count, "fail": fail_count, "skip": skip_count})
            else:
                status = "✓" if success else "✗"
                print(f"{status} {message}")

    print("\n=== Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total:   {len(ligand_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="批量提取配体特征 V5 (109D = 54D local + 45D global + 10D shape)"
    )
    parser.add_argument("--input", type=str, required=True, help="配体文件根目录 (P-L)")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (默认: 输入目录同级 ligand_features_v5)")
    parser.add_argument("--conformers", type=int, default=50, help="生成的构象数量 (默认50)")
    parser.add_argument("--no-conformers", action="store_true", help="不生成构象，只使用实验结构")
    parser.add_argument("--max", type=int, default=None, help="最大处理文件数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数")
    parser.add_argument("--rdkit-threads", type=int, default=0, help="每个任务内部 RDKit 线程数，0=全部线程")
    parser.add_argument("--ids", nargs="*", default=None, help="只处理指定PDB ID")
    parser.add_argument("--loose", action="store_true", help="宽松模式：允许构象/MMFF降级")
    parser.add_argument("--flat", action="store_true", help="平铺扫描模式：直接处理输入目录下的 *.sdf/*.mol2/*.pdb/*.mol")

    args = parser.parse_args()

    if not PYPROPEL_AVAILABLE:
        print("❌ pypropel和RDKit是必需的")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else (input_path.resolve().parent / "ligand_features_v5")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Processing ligands from: {input_path}")
    print(f"Output directory: {output_path}")
    print("Feature dimension: 109 (54 local + 45 global + 10 shape)")

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
        num_threads=args.rdkit_threads,
        flat=args.flat,
    )


if __name__ == "__main__":
    main()
