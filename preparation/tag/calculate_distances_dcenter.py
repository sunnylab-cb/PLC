"""
计算残基到配体的最小重原子中心距离 (d_center)

方法：蛋白质每个残基的所有重原子 vs 配体所有重原子，取最小欧几里得距离
（与 calculate_distances_v2.py 相同算法，但路径参数化，覆盖 New/ 目录全量数据）

输出格式（每个 complex_id）：
    {
        'distances':     FloatTensor[N_res],  # 最小中心距 (Å)
        'n_residues':    int,
        'min_distance':  float,
        'max_distance':  float,
        'mean_distance': float,
    }

用法：
    python preparation/tag/calculate_distances_dcenter.py \
        --pl-dir /home/husrcf/Code/ProtBind/New \
        --ligand-feat-dir /home/husrcf/Code/ProtBind/PLC/data/ligand_features_v5 \
        --output /home/husrcf/Code/ProtBind/PLC/data/residue_ligand_distances_dcenter.pt \
        --workers 64
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from preparation.protein.extract_physchem_from_pdb import parse_pdb_protein

warnings.filterwarnings('ignore')

YEAR_DIRS = ['1981-2000', '2001-2010', '2011-2019']


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def _extract_protein_heavy_coords(protein_pdb: str) -> list:
    """
    返回每个残基的重原子坐标列表 List[np.ndarray[N_atoms, 3]]。

    这里显式复用 protein feature 流的 residue enumeration 逻辑，
    以保证 distance label 与 protein feature `.pt` 的残基顺序完全一致。
    """
    with contextlib.redirect_stdout(io.StringIO()):
        pdb_info = parse_pdb_protein(Path(protein_pdb), chain_id=None, mode='complex')

    residue_coords = pdb_info.get('all_atom_coords', [])
    if not residue_coords:
        return []

    return [np.asarray(coords, dtype=np.float32) for coords in residue_coords if len(coords) > 0]


def _extract_ligand_coords_from_pdb(ligand_pdb: str) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', ligand_pdb)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue.get_atoms():
                    if atom.element != 'H':
                        coords.append(atom.get_coord())
    return np.array(coords)


def _extract_ligand_coords_from_pt(pt_path: Path) -> np.ndarray:
    data = torch.load(pt_path, weights_only=False)
    if 'coords_original' in data and data['coords_original'] is not None:
        return data['coords_original'].numpy()

    if 'coords_heavy_current' in data and data['coords_heavy_current'] is not None:
        print(f"[fallback] {pt_path}: using coords_heavy_current")
        return data['coords_heavy_current'].numpy()

    coords = data['coords'].numpy()

    if 'atom_symbols' in data:
        atom_symbols = data['atom_symbols']
        heavy_mask = np.array([str(symbol) != 'H' for symbol in atom_symbols], dtype=bool)
        if len(heavy_mask) == len(coords):
            print(f"[fallback] {pt_path}: filtering coords by atom_symbols to heavy-only")
            return coords[heavy_mask]

    atom_features = data.get('atom_features')
    if atom_features is not None:
        if isinstance(atom_features, torch.Tensor):
            atom_features = atom_features.numpy()
        if atom_features.ndim == 2 and atom_features.shape[0] == coords.shape[0] and atom_features.shape[1] >= 11:
            atom_type_idx = np.argmax(atom_features[:, :11], axis=1)
            heavy_mask = atom_type_idx != 9  # index 9 = H in ligand feature extraction
            print(f"[fallback] {pt_path}: filtering coords by atom_features to heavy-only")
            return coords[heavy_mask]

    print(f"[fallback] {pt_path}: using raw coords (unable to derive heavy-only mask)")
    return coords


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

def _calc_min_distances(residue_coords_list: list, ligand_coords: np.ndarray) -> np.ndarray:
    """每个残基到配体的最小重原子中心距。"""
    n_res = len(residue_coords_list)
    min_dists = np.full(n_res, np.inf, dtype=np.float32)
    for i, res_coords in enumerate(residue_coords_list):
        # [N_res_atoms, N_lig_atoms]
        diff = res_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists[i] = dists.min()
    return min_dists


# ---------------------------------------------------------------------------
# Per-complex worker
# ---------------------------------------------------------------------------

def _process_complex(args: tuple) -> Tuple[Optional[str], Optional[Dict]]:
    complex_dir, year_dir, ligand_feat_root = args
    complex_id = complex_dir.name

    try:
        protein_pdb = complex_dir / f'{complex_id}_protein.pdb'
        if not protein_pdb.exists():
            return None, None

        # --- 找配体坐标 ---
        ligand_pdb = complex_dir / f'{complex_id}_ligand.pdb'
        for suffix in ['.pdb', '.sdf', '.mol2']:
            candidate = complex_dir / f'{complex_id}_ligand{suffix}'
            if candidate.exists():
                ligand_pdb = candidate
                break
        else:
            ligand_pdb = None

        if ligand_pdb is not None and ligand_pdb.suffix == '.pdb':
            ligand_coords = _extract_ligand_coords_from_pdb(str(ligand_pdb))
        else:
            # 从 .pt 特征文件读取
            pt_path = ligand_feat_root / year_dir / complex_id / f'{complex_id}_ligand.pt'
            if not pt_path.exists():
                return None, None
            ligand_coords = _extract_ligand_coords_from_pt(pt_path)

        if len(ligand_coords) == 0:
            return None, None

        residue_coords_list = _extract_protein_heavy_coords(str(protein_pdb))
        if len(residue_coords_list) == 0:
            return None, None

        distances = _calc_min_distances(residue_coords_list, ligand_coords)

        result = {
            'distances':    torch.from_numpy(distances).float(),
            'n_residues':   len(distances),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'mean_distance': float(distances.mean()),
        }
        return complex_id, result

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_calculate(pl_dir: str, ligand_feat_dir: str, output_file: str, num_workers: int):
    pl_root = Path(pl_dir)
    ligand_feat_root = Path(ligand_feat_dir)

    tasks = []
    print("收集任务...")
    for year_dir in YEAR_DIRS:
        year_path = pl_root / year_dir
        if not year_path.exists():
            continue
        for complex_dir in year_path.iterdir():
            if complex_dir.is_dir():
                tasks.append((complex_dir, year_dir, ligand_feat_root))

    print(f"共 {len(tasks)} 个复合物，{num_workers} 进程")

    distances_dict = {}
    success = 0

    if num_workers == 1:
        for task in tqdm(tasks, desc="计算 d_center"):
            cid, data = _process_complex(task)
            if cid is not None:
                distances_dict[cid] = data
                success += 1
    else:
        with mp.Pool(processes=num_workers) as pool:
            for cid, data in tqdm(
                pool.imap_unordered(_process_complex, tasks),
                total=len(tasks),
                desc="计算 d_center"
            ):
                if cid is not None:
                    distances_dict[cid] = data
                    success += 1

    torch.save(distances_dict, output_file)
    print(f"\n成功: {success}/{len(tasks)}")
    print(f"输出: {output_file}")

    if distances_dict:
        min_dists = [v['min_distance'] for v in distances_dict.values()]
        print(f"最小距离统计: mean={np.mean(min_dists):.2f} Å, "
              f"min={np.min(min_dists):.2f} Å, max={np.max(min_dists):.2f} Å")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="计算蛋白质-配体残基级最小重原子中心距 (d_center)"
    )
    parser.add_argument(
        '--pl-dir',
        default='/home/husrcf/Code/ProtBind/New',
        help='P-L 复合物根目录（含 1981-2000 等年份子目录）'
    )
    parser.add_argument(
        '--ligand-feat-dir',
        default='/home/husrcf/Code/ProtBind/PLC/data/ligand_features_v5',
        help='配体 .pt 特征文件根目录（含年份子目录）'
    )
    parser.add_argument(
        '--output',
        default='/home/husrcf/Code/ProtBind/PLC/data/residue_ligand_distances_dcenter.pt',
        help='输出 .pt 文件路径'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=64,
        help='并行进程数'
    )
    args = parser.parse_args()

    batch_calculate(
        pl_dir=args.pl_dir,
        ligand_feat_dir=args.ligand_feat_dir,
        output_file=args.output,
        num_workers=args.workers,
    )
