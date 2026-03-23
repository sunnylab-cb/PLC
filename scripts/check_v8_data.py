#!/usr/bin/env python3
"""
检查 V8 训练所需的数据文件是否就绪

检查项目:
1. V3 蛋白质特征文件 (complete_features_v3/)
2. V4 配体特征文件 (ligand_features_v4/)
3. VDW 距离标签文件
4. 数据划分文件 (MMseqs2 splits)
"""

from pathlib import Path
import sys


def check_directory(path: Path, name: str, required: bool = True) -> bool:
    """检查目录是否存在且包含文件"""
    if not path.exists():
        status = "❌" if required else "⚠️"
        print(f"{status} {name}: 不存在")
        print(f"   路径: {path}")
        return False

    # 统计文件数量
    file_count = 0
    for year_range in ['1981-2000', '2001-2010', '2011-2019']:
        year_dir = path / year_range
        if year_dir.exists():
            file_count += sum(1 for _ in year_dir.rglob('*.pt'))

    if file_count == 0:
        status = "❌" if required else "⚠️"
        print(f"{status} {name}: 目录存在但为空")
        print(f"   路径: {path}")
        return False

    print(f"✅ {name}: {file_count} 个文件")
    print(f"   路径: {path}")
    return True


def check_file(path: Path, name: str, required: bool = True) -> bool:
    """检查文件是否存在"""
    if not path.exists():
        status = "❌" if required else "⚠️"
        print(f"{status} {name}: 不存在")
        print(f"   路径: {path}")
        return False

    print(f"✅ {name}: 存在")
    print(f"   路径: {path}")
    return True


def main():
    print("=" * 70)
    print("V8 训练数据检查")
    print("=" * 70)

    data_root = Path("data")
    all_ok = True

    # 1. 检查 V3 蛋白质特征
    print("\n1. V3 蛋白质特征 (complete_features_v3/)")
    print("-" * 70)
    v3_ok = check_directory(data_root / "complete_features_v3", "V3 蛋白质特征", required=True)
    all_ok = all_ok and v3_ok

    # 2. 检查 V4 配体特征
    print("\n2. V4 配体特征 (ligand_features_v4/)")
    print("-" * 70)
    v4_ok = check_directory(data_root / "ligand_features_v4", "V4 配体特征", required=True)
    all_ok = all_ok and v4_ok

    # 3. 检查 ESM2 latents
    print("\n3. ESM2 Latents (esm2_latents/)")
    print("-" * 70)
    esm_ok = check_directory(data_root / "esm2_latents", "ESM2 Latents", required=True)
    all_ok = all_ok and esm_ok

    # 4. 检查距离标签文件
    print("\n4. 距离标签文件")
    print("-" * 70)
    vdw_ok = check_file(data_root / "residue_ligand_distances_vdw.pt", "VDW 距离标签", required=True)
    all_ok = all_ok and vdw_ok

    # 5. 检查数据划分文件
    print("\n5. 数据划分文件")
    print("-" * 70)
    splits_ok = check_file(Path("configs/splits/mmseqs_splits_strict.csv"), "MMseqs2 Splits", required=True)
    all_ok = all_ok and splits_ok

    # 总结
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ 所有必需文件就绪，可以开始训练！")
        print("=" * 70)
        print("\n训练命令:")
        print("  # 单 GPU")
        print("  python scripts/train/train_v8.py --config configs/pocket_gvp_v8.yaml")
        print("\n  # 多 GPU")
        print("  torchrun --nproc_per_node=4 scripts/train/train_v8.py --config configs/pocket_gvp_v8.yaml")
        return 0
    else:
        print("❌ 部分必需文件缺失，请按以下步骤准备数据")
        print("=" * 70)

        if not v3_ok:
            print("\n📋 步骤 1: 提取 V3 蛋白质特征")
            print("-" * 70)
            print("python preparation/protein/extract_complete_features_v3.py \\")
            print("  --input data/processed/ \\")
            print("  --output data/complete_features_v3/ \\")
            print("  --pattern '*_protein.pdb' \\")
            print("  --workers 32")
            print("\n验证:")
            print("python preparation/protein/test_v3_features.py \\")
            print("  --input data/complete_features_v3/1981-2000/1a0q/1a0q_protein.pt")

        if not v4_ok:
            print("\n📋 步骤 2: 提取 V4 配体特征")
            print("-" * 70)
            print("python preparation/ligand/extract_ligand_features_v4.py \\")
            print("  --input data/processed/ \\")
            print("  --output data/ligand_features_v4/ \\")
            print("  --workers 32")

        if not esm_ok:
            print("\n📋 步骤 3: 提取 ESM2 Latents")
            print("-" * 70)
            print("python preparation/protein/extract_esm2_latents.py \\")
            print("  --input data/processed/ \\")
            print("  --output data/esm2_latents/ \\")
            print("  --workers 4")

        if not vdw_ok:
            print("\n📋 步骤 4: 生成 VDW 距离标签")
            print("-" * 70)
            print("python preparation/compute_residue_ligand_distances_vdw.py \\")
            print("  --input data/processed/ \\")
            print("  --output data/residue_ligand_distances_vdw.pt")

        if not splits_ok:
            print("\n📋 步骤 5: 生成数据划分")
            print("-" * 70)
            print("python preparation/create_mmseqs_splits.py \\")
            print("  --input data/processed/ \\")
            print("  --output configs/splits/mmseqs_splits_strict.csv")

        return 1


if __name__ == "__main__":
    sys.exit(main())
