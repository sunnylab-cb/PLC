#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1：从PDB文件提取蛋白质序列并生成FASTA文件

用于MMseqs2聚类的序列提取
"""

from pathlib import Path
from typing import Dict, List
import sys

try:
    from biotite.structure.io.pdb import PDBFile
    import biotite.structure as bs
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    print("❌ Biotite未安装，请运行: conda install -c conda-forge biotite")
    sys.exit(1)

# 氨基酸三字母到单字母映射
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def extract_sequence_from_pdb(pdb_path: Path) -> str:
    """
    从PDB文件提取蛋白质序列

    Args:
        pdb_path: PDB文件路径

    Returns:
        序列字符串（单字母代码）
    """
    try:
        # 读取PDB文件
        pdb_file = PDBFile.read(str(pdb_path))
        structure = pdb_file.get_structure(model=1)

        # 只保留氨基酸，去除HETATM
        atom_array = structure[bs.filter_amino_acids(structure) & ~structure.hetero]

        # 提取C-alpha原子
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]

        # 提取序列
        sequence = []
        for residue in bs.residue_iter(ca_atoms):
            res_name = residue.res_name[0]  # 三字母代码
            aa_1 = AA_3TO1.get(res_name, 'X')  # 转换为单字母，未知残基用X
            sequence.append(aa_1)

        return ''.join(sequence)

    except Exception as e:
        print(f"  ⚠️  提取序列失败 {pdb_path.name}: {e}")
        return None


def extract_all_sequences(data_dir: Path, output_fasta: Path):
    """
    提取所有蛋白质序列并保存为FASTA文件

    Args:
        data_dir: 数据目录（包含所有PDB文件）
        output_fasta: 输出FASTA文件路径
    """
    print("="*70)
    print("步骤1：提取蛋白质序列")
    print("="*70)

    # 查找所有蛋白质PDB文件
    pdb_files = sorted(data_dir.rglob("*_protein.pdb"))
    print(f"\n找到 {len(pdb_files)} 个蛋白质PDB文件")

    # 提取序列（去重）
    sequences = {}  # {pdb_id: sequence}
    duplicates = {}  # {sequence: [pdb_ids]} 用于检测重复序列

    print("\n提取序列中...")
    success_count = 0
    fail_count = 0

    for pdb_file in pdb_files:
        pdb_id = pdb_file.stem.replace('_protein', '')

        sequence = extract_sequence_from_pdb(pdb_file)

        if sequence:
            sequences[pdb_id] = sequence
            success_count += 1

            # 记录重复序列
            if sequence in duplicates:
                duplicates[sequence].append(pdb_id)
            else:
                duplicates[sequence] = [pdb_id]

            if success_count % 1000 == 0:
                print(f"  已处理: {success_count}/{len(pdb_files)}")
        else:
            fail_count += 1

    print(f"\n提取完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  唯一序列数: {len(duplicates)}")

    # 统计完全相同的序列
    exact_duplicates = {seq: ids for seq, ids in duplicates.items() if len(ids) > 1}
    if exact_duplicates:
        print(f"  完全相同的序列: {len(exact_duplicates)} 组")
        print(f"    示例（前3组）:")
        for i, (seq, ids) in enumerate(list(exact_duplicates.items())[:3]):
            print(f"      {ids[:5]}... ({len(ids)}个)")

    # 写入FASTA文件
    print(f"\n写入FASTA文件: {output_fasta}")
    with open(output_fasta, 'w') as f:
        for pdb_id, sequence in sequences.items():
            f.write(f">{pdb_id}\n")
            # 每行60个字符（FASTA标准格式）
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')

    print(f"✅ FASTA文件已保存: {len(sequences)} 条序列")

    # 保存PDB ID到序列的映射（用于后续步骤）
    mapping_file = output_fasta.with_suffix('.mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("pdb_id\tsequence_length\tsequence\n")
        for pdb_id, sequence in sequences.items():
            f.write(f"{pdb_id}\t{len(sequence)}\t{sequence}\n")

    print(f"✅ 映射文件已保存: {mapping_file}")

    return sequences


if __name__ == "__main__":
    # 配置路径
    data_dir = Path("/home/husrcf/Code/ProtBind/New")
    output_dir = Path("/home/husrcf/Code/ProtBind/dataset_split")
    output_dir.mkdir(exist_ok=True)

    output_fasta = output_dir / "dataset_proteins.fasta"

    # 提取序列
    sequences = extract_all_sequences(data_dir, output_fasta)

    print("\n" + "="*70)
    print("✅ 步骤1完成！")
    print("="*70)
    print(f"\n下一步：运行MMseqs2聚类")
    print(f"  cd {output_dir}")
    print(f"  mmseqs easy-cluster dataset_proteins.fasta cluster_res tmp_dir \\")
    print(f"      --min-seq-id 0.3 -c 0.8 --cov-mode 1 --threads 16")
