"""
计算残基到配体的距离

从PDB文件中提取坐标并计算每个残基到配体的最小距离

V2 改进:
1. 使用全重原子计算距离（更准确）
2. 按 (chain, res_id) 排序，确保确定性
3. ✅ 自动使用 coords_original 字段（原始实验坐标）
4. ⚠️  如果特征文件缺少 coords_original 会回退到 coords（旧版本）

注意：
- 需要配体特征文件包含 coords_original 字段（V2格式）
- 如果使用旧版本特征文件，坐标可能不准确
"""
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from typing import Dict, Tuple, Optional
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')


def calculate_residue_ligand_distances(
    protein_pdb: str,
    ligand_coords: np.ndarray,
    ca_only: bool = False  # 改为默认全重原子，更准确
) -> np.ndarray:
    """
    计算每个残基到配体的最小距离
    
    改进:
    1. 默认使用所有重原子 (ca_only=False)
    2. 只处理第一个模型 (model=0)
    3. 返回按 (chain, res_id) 排序的距离数组，确保确定性
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)
    
    if len(structure) == 0:
        return np.array([])
        
    model = structure[0] # 只处理第一个模型

    residue_data = []

    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':  # 跳过HETATM (水、离子等)
                continue

            # 获取残基的原子坐标
            if ca_only:
                if 'CA' in residue:
                    residue_coords = residue['CA'].get_coord().reshape(1, 3)
                else:
                    continue
            else:
                # 使用所有重原子
                residue_coords = np.array([
                    atom.get_coord() for atom in residue.get_atoms()
                    if atom.element != 'H'
                ])

            if len(residue_coords) == 0:
                continue

            # 计算到配体所有原子的距离
            diff = residue_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2) # [N_res_atoms, N_lig_atoms]

            # 取最小距离
            min_distance = dists.min()
            
            # 记录残基标识
            residue_data.append({
                'chain': chain.id,
                'res_id': residue.id[1],
                'distance': min_distance
            })

    # 排序：先按 Chain ID，再按 Residue ID (PDB编号)
    residue_data.sort(key=lambda x: (x['chain'], x['res_id']))
    
    # 提取距离
    distances = np.array([item['distance'] for item in residue_data])

    return distances


def extract_ligand_coords_from_pdb(ligand_pdb: str) -> np.ndarray:
    """从配体PDB文件提取坐标"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', ligand_pdb)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue.get_atoms():
                    if atom.element != 'H':  # 跳过氢原子
                        coords.append(atom.get_coord())

    return np.array(coords)


def extract_ligand_coords_from_pt(ligand_data: Dict, source: Optional[str] = None) -> np.ndarray:
    """优先返回 heavy-only 坐标，避免将含氢的 coords 直接用于距离计算。"""
    source_label = source or "<in-memory ligand_data>"

    if 'coords_original' in ligand_data and ligand_data['coords_original'] is not None:
        return ligand_data['coords_original'].numpy()

    if 'coords_heavy_current' in ligand_data and ligand_data['coords_heavy_current'] is not None:
        print(f"[fallback] {source_label}: using coords_heavy_current")
        return ligand_data['coords_heavy_current'].numpy()

    coords = ligand_data['coords'].numpy()

    if 'atom_symbols' in ligand_data:
        atom_symbols = ligand_data['atom_symbols']
        heavy_mask = np.array([str(symbol) != 'H' for symbol in atom_symbols], dtype=bool)
        if len(heavy_mask) == len(coords):
            print(f"[fallback] {source_label}: filtering coords by atom_symbols to heavy-only")
            return coords[heavy_mask]

    atom_features = ligand_data.get('atom_features')
    if atom_features is not None:
        if isinstance(atom_features, torch.Tensor):
            atom_features = atom_features.numpy()
        if atom_features.ndim == 2 and atom_features.shape[0] == coords.shape[0] and atom_features.shape[1] >= 11:
            atom_type_idx = np.argmax(atom_features[:, :11], axis=1)
            heavy_mask = atom_type_idx != 9  # index 9 = H in ligand feature extraction
            print(f"[fallback] {source_label}: filtering coords by atom_features to heavy-only")
            return coords[heavy_mask]

    print(f"[fallback] {source_label}: using raw coords (unable to derive heavy-only mask)")
    return coords


def process_single_complex(args) -> Tuple[Optional[str], Optional[Dict]]:
    """处理单个复合物的任务函数"""
    complex_dir, year_dir, data_root_path = args
    complex_id = complex_dir.name
    
    try:
        # 查找蛋白和配体PDB文件
        protein_pdb = complex_dir / f'{complex_id}_protein.pdb'
        ligand_pdb = complex_dir / f'{complex_id}_ligand.pdb'

        # 也尝试其他格式
        if not ligand_pdb.exists():
            ligand_pdb = complex_dir / f'{complex_id}_ligand.sdf'
        if not ligand_pdb.exists():
            ligand_pdb = complex_dir / f'{complex_id}_ligand.mol2'

        if not protein_pdb.exists() or not ligand_pdb.exists():
            return None, None

        # 提取配体坐标
        if ligand_pdb.suffix == '.pdb':
            ligand_coords = extract_ligand_coords_from_pdb(str(ligand_pdb))
        else:
            # 对于SDF/MOL2，从已提取的特征中读取
            ligand_feat_file = data_root_path / 'PLK' / 'ligand_features' / year_dir / complex_id / f'{complex_id}_ligand.pt'

            if ligand_feat_file.exists():
                ligand_data = torch.load(ligand_feat_file, weights_only=False)
                ligand_coords = extract_ligand_coords_from_pt(ligand_data, source=str(ligand_feat_file))
            else:
                return None, None

        # 计算距离
        distances = calculate_residue_ligand_distances(
            str(protein_pdb),
            ligand_coords,
            ca_only=False  # 使用全重原子
        )

        if len(distances) == 0:
            return None, None

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
    批量计算所有复合物的残基-配体距离 (多进程版)
    """
    from tqdm import tqdm

    data_root_path = Path(data_root)
    pl_dir = data_root_path / 'P-L'

    distances_dict = {}
    tasks = []

    # 1. 收集所有任务
    print("收集任务...")
    for year_dir in ['1981-2000', '2001-2010', '2011-2019']:
        year_path = pl_dir / year_dir
        if not year_path.exists():
            continue
            
        for complex_dir in year_path.iterdir():
            if not complex_dir.is_dir():
                continue
            tasks.append((complex_dir, year_dir, data_root_path))
            
    print(f"总共有 {len(tasks)} 个复合物待处理")
    print(f"使用 {num_workers} 个进程进行计算...")

    # 2. 并行处理
    with mp.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 实时获取结果
        results = list(tqdm(
            pool.imap_unordered(process_single_complex, tasks),
            total=len(tasks),
            desc="计算距离"
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


if __name__ == '__main__':
    # 测试单个复合物
    print("测试距离计算 (v2 - 全原子 & 排序 & 原始坐标)...")

    # 示例：使用已有的坐标
    protein_pdb = '/home/husrcf/Code/ProtBind/P-L/1981-2000/10gs/10gs_protein.pdb'
    ligand_feat = '/home/husrcf/Code/ProtBind/PLK/ligand_features/1981-2000/10gs/10gs_ligand.pt'

    if Path(protein_pdb).exists() and Path(ligand_feat).exists():
        # 加载配体特征
        ligand_data = torch.load(ligand_feat, weights_only=False)
        ligand_coords = extract_ligand_coords_from_pt(ligand_data, source=ligand_feat)
        print(f"✅ 使用 ligand heavy-only 坐标: {ligand_coords.shape}")

        # 计算距离 (使用全重原子)
        distances = calculate_residue_ligand_distances(
            protein_pdb,
            ligand_coords,
            ca_only=False
        )

        print(f"\n复合物: 10gs")
        print(f"残基数: {len(distances)}")
        print(f"距离范围: {distances.min():.2f} - {distances.max():.2f} Å")
        print(f"平均距离: {distances.mean():.2f} Å")
        print(f"结合位点残基 (<5Å): {(distances < 5.0).sum()}")

    # 批量处理
    print("\n开始批量处理 (v2 - 使用原始坐标)...")
    batch_calculate_distances(
        data_root='/home/husrcf/Code/ProtBind',
        output_file='/home/husrcf/Code/ProtBind/PLK/residue_ligand_distances_v2.pt'
    )
