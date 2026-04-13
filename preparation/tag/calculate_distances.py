"""
计算残基到配体的距离

从PDB文件中提取坐标并计算每个残基到配体的最小距离
"""
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_residue_ligand_distances(
    protein_pdb: str,
    ligand_coords: np.ndarray,
    ca_only: bool = True
) -> np.ndarray:
    """
    计算每个残基到配体的最小距离

    Args:
        protein_pdb: 蛋白PDB文件路径
        ligand_coords: 配体原子坐标 [N_atoms, 3]
        ca_only: 是否只使用Cα原子 (True) 或所有原子 (False)

    Returns:
        distances: [N_residues] 每个残基到配体的最小距离 (Å)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)

    residue_distances = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # 跳过HETATM
                    continue

                # 获取残基的原子坐标
                if ca_only:
                    # 只使用Cα原子
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
                # [N_residue_atoms, N_ligand_atoms]
                distances = np.linalg.norm(
                    residue_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
                    axis=2
                )

                # 取最小距离
                min_distance = distances.min()
                residue_distances.append(min_distance)

    return np.array(residue_distances)


def extract_ligand_coords_from_pdb(ligand_pdb: str) -> np.ndarray:
    """
    从配体PDB文件提取坐标

    Args:
        ligand_pdb: 配体PDB文件路径

    Returns:
        coords: [N_atoms, 3]
    """
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


def batch_calculate_distances(
    data_root: str,
    output_file: str
):
    """
    批量计算所有复合物的残基-配体距离

    Args:
        data_root: 数据根目录 (包含P-L/)
        output_file: 输出文件路径 (.pt)
    """
    from tqdm import tqdm

    data_root = Path(data_root)
    pl_dir = data_root / 'P-L'

    distances_dict = {}

    for year_dir in ['1981-2000', '2001-2010', '2011-2019']:
        year_path = pl_dir / year_dir

        if not year_path.exists():
            continue

        for complex_dir in tqdm(list(year_path.iterdir()), desc=f"处理 {year_dir}"):
            if not complex_dir.is_dir():
                continue

            complex_id = complex_dir.name

            # 查找蛋白和配体PDB文件
            protein_pdb = complex_dir / f'{complex_id}_protein.pdb'
            ligand_pdb = complex_dir / f'{complex_id}_ligand.pdb'

            # 也尝试其他格式
            if not ligand_pdb.exists():
                ligand_pdb = complex_dir / f'{complex_id}_ligand.sdf'
            if not ligand_pdb.exists():
                ligand_pdb = complex_dir / f'{complex_id}_ligand.mol2'

            if not protein_pdb.exists() or not ligand_pdb.exists():
                continue

            try:
                # 提取配体坐标
                if ligand_pdb.suffix == '.pdb':
                    ligand_coords = extract_ligand_coords_from_pdb(str(ligand_pdb))
                else:
                    # 对于SDF/MOL2，从已提取的特征中读取
                    ligand_feat_file = data_root / 'PLK' / 'ligand_features' / year_dir / complex_id / f'{complex_id}_ligand.pt'
                    if ligand_feat_file.exists():
                        ligand_data = torch.load(ligand_feat_file, weights_only=False)
                        ligand_coords = ligand_data['coords'].numpy()
                    else:
                        continue

                # 计算距离
                distances = calculate_residue_ligand_distances(
                    str(protein_pdb),
                    ligand_coords,
                    ca_only=True
                )

                distances_dict[complex_id] = {
                    'distances': torch.from_numpy(distances).float(),
                    'n_residues': len(distances),
                    'min_distance': float(distances.min()),
                    'max_distance': float(distances.max()),
                    'mean_distance': float(distances.mean())
                }

            except Exception as e:
                print(f"错误: {complex_id} - {e}")
                continue

    # 保存
    torch.save(distances_dict, output_file)
    print(f"\n保存到: {output_file}")
    print(f"总共处理: {len(distances_dict)} 个复合物")


if __name__ == '__main__':
    # 测试单个复合物
    print("测试距离计算...")

    # 示例：使用已有的坐标
    protein_pdb = '/home/husrcf/Code/ProtBind/P-L/1981-2000/10gs/10gs_protein.pdb'
    ligand_feat = '/home/husrcf/Code/ProtBind/PLK/ligand_features/1981-2000/10gs/10gs_ligand.pt'

    if Path(protein_pdb).exists() and Path(ligand_feat).exists():
        # 加载配体坐标
        ligand_data = torch.load(ligand_feat, weights_only=False)
        ligand_coords = ligand_data['coords'].numpy()

        # 计算距离
        distances = calculate_residue_ligand_distances(
            protein_pdb,
            ligand_coords,
            ca_only=True
        )

        print(f"\n复合物: 10gs")
        print(f"残基数: {len(distances)}")
        print(f"距离范围: {distances.min():.2f} - {distances.max():.2f} Å")
        print(f"平均距离: {distances.mean():.2f} Å")
        print(f"结合位点残基 (<5Å): {(distances < 5.0).sum()}")
        print(f"近邻残基 (5-10Å): {((distances >= 5.0) & (distances < 10.0)).sum()}")
        print(f"远距离残基 (>10Å): {(distances >= 10.0).sum()}")

    # 批量处理
    print("\n开始批量处理...")
    batch_calculate_distances(
        data_root='/home/husrcf/Code/ProtBind',
        output_file='/home/husrcf/Code/ProtBind/PLK/residue_ligand_distances.pt'
    )
