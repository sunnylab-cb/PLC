#!/usr/bin/env python3
"""
重新组织 V3 特征文件结构

将扁平结构:
  data/complete_features_v3/1a0q_protein_complete_v3.pt

转换为嵌套结构:
  data/complete_features_v3/YEAR_RANGE/PDB_ID/PDB_ID_protein.pt

年份范围根据 V2 特征目录结构确定
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def get_year_range_from_v2(pdb_id: str, v2_dir: Path) -> str:
    """从 V2 目录结构中查找 PDB ID 对应的年份范围"""
    year_ranges = ['1981-2000', '2001-2010', '2011-2019']

    for year_range in year_ranges:
        year_dir = v2_dir / year_range / pdb_id
        if year_dir.exists():
            return year_range

    return None


def main():
    v3_dir = Path("data/complete_features_v3")
    v2_dir = Path("data/complete_features_v2")

    if not v3_dir.exists():
        print(f"错误: V3 目录不存在: {v3_dir}")
        return

    if not v2_dir.exists():
        print(f"错误: V2 目录不存在 (用于确定年份范围): {v2_dir}")
        return

    # 查找所有扁平的 V3 特征文件
    v3_files = list(v3_dir.glob("*_protein_complete_v3.pt"))

    if len(v3_files) == 0:
        print("没有找到需要重新组织的文件")
        return

    print(f"找到 {len(v3_files)} 个 V3 特征文件")
    print("开始重新组织...")

    success_count = 0
    failed_count = 0
    failed_files = []

    for v3_file in tqdm(v3_files, desc="重新组织"):
        # 提取 PDB ID (去掉 _protein_complete_v3.pt 后缀)
        pdb_id = v3_file.stem.replace("_protein_complete_v3", "")

        # 从 V2 目录查找年份范围
        year_range = get_year_range_from_v2(pdb_id, v2_dir)

        if year_range is None:
            failed_count += 1
            failed_files.append((pdb_id, "无法确定年份范围"))
            continue

        # 创建新的目录结构
        new_dir = v3_dir / year_range / pdb_id
        new_dir.mkdir(parents=True, exist_ok=True)

        # 新的文件名 (与 V2 保持一致)
        new_file = new_dir / f"{pdb_id}_protein.pt"

        # 移动文件
        try:
            shutil.move(str(v3_file), str(new_file))
            success_count += 1
        except Exception as e:
            failed_count += 1
            failed_files.append((pdb_id, str(e)))

    print(f"\n完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")

    if failed_files:
        print(f"\n失败的文件:")
        for pdb_id, error in failed_files[:10]:  # 只显示前 10 个
            print(f"  {pdb_id}: {error}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败")


if __name__ == "__main__":
    main()
