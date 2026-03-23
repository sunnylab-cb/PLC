# V3 特征提取脚本改进总结

## 改进内容

### 1. 输出目录结构（与 V2 保持一致）

**修改前（扁平结构）**:
```
data/complete_features_v3/
  ├── 1a0q_protein_complete_v3.pt
  ├── 1a1b_protein_complete_v3.pt
  └── ...
```

**修改后（嵌套结构）**:
```
data/complete_features_v3/
  ├── 1981-2000/
  │   ├── 1a0q/
  │   │   └── 1a0q_protein.pt
  │   ├── 1a1b/
  │   │   └── 1a1b_protein.pt
  │   └── ...
  ├── 2001-2010/
  │   └── ...
  └── 2011-2019/
      └── ...
```

### 2. 文件命名（与 V2 保持一致）

- **修改前**: `{pdb_id}_protein_complete_v3.pt`
- **修改后**: `{pdb_id}_protein.pt`

### 3. 特征字段完备性

**V3 特征文件包含的字段**:
```python
{
    # 核心特征（V3 新增）
    'physchem_features': torch.Tensor,          # [N, 62] - 物理化学特征
    'spatial_scalar_features': torch.Tensor,    # [N, 72] - 空间标量特征
    'spatial_vector_features': torch.Tensor,    # [N, 6, 3] - 空间矢量特征

    # 坐标信息（必需，用于 GVP）
    'ca_coords': torch.Tensor,                  # [N, 3] - Cα 坐标
    'all_atom_coords': List[torch.Tensor],      # List of [n_atoms, 3] - 全原子坐标

    # 元数据
    'residue_names': List[str],                 # 残基名称
    'residue_indices': List[int],               # 残基索引
    'chain_ids': List[str],                     # 链 ID
    'n_residues': int                           # 残基数量
}
```

### 4. 关键代码修改

#### 4.1 `process_single_pdb` 函数

```python
def process_single_pdb(args_tuple):
    pdb_path, output_dir, chain_id, mode = args_tuple
    pdb_id = pdb_path.stem

    # 提取特征
    features = extract_complete_features_v3(pdb_path, chain_id, mode)

    # 从输入路径中提取年份范围信息
    year_range = None
    for part in pdb_path.parts:
        if '-' in part and any(c.isdigit() for c in part):
            parts_split = part.split('-')
            if len(parts_split) == 2 and parts_split[0].isdigit() and parts_split[1].isdigit():
                year_range = part
                break

    # 创建嵌套的输出目录结构: output_dir/YEAR_RANGE/PDB_ID/
    if year_range:
        pdb_output_dir = output_dir / year_range / pdb_id
    else:
        pdb_output_dir = output_dir / pdb_id

    pdb_output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为 .pt 文件，使用与 V2 相同的命名格式
    output_path = pdb_output_dir / f"{pdb_id}_protein.pt"
    torch.save(features, output_path)

    return (pdb_id, True, None)
```

#### 4.2 特征字典返回

```python
return {
    'physchem_features': torch.from_numpy(physchem_features).float(),
    'spatial_scalar_features': torch.from_numpy(spatial_scalar_features).float(),
    'spatial_vector_features': torch.from_numpy(spatial_vector_features).float(),
    'ca_coords': torch.from_numpy(pdb_info['ca_coords']).float(),  # 必需
    'all_atom_coords': pdb_info['all_atom_coords'],  # 必需（用于计算 GVP vectors）
    'residue_names': pdb_info['residue_names'],
    'residue_indices': pdb_info['residue_indices'],
    'chain_ids': pdb_info['chain_ids'],
    'n_residues': n_residues
}
```

## 使用方法

### 提取新的 V3 特征

```bash
# 批量提取（自动识别年份范围并创建嵌套结构）
python preparation/protein/extract_complete_features_v3.py \
  --input data/processed/ \
  --output data/complete_features_v3/ \
  --pattern '*_protein.pdb' \
  --workers 32

# 单个文件提取
python preparation/protein/extract_complete_features_v3.py \
  --input data/processed/1981-2000/1a0q/1a0q_protein.pdb \
  --output data/complete_features_v3/ \
  --workers 1
```

### 重新组织现有的扁平 V3 特征

如果已经有扁平结构的 V3 特征，可以使用重组脚本：

```bash
# 将扁平结构转换为嵌套结构
python scripts/reorganize_v3_features.py
```

### 更新现有 V3 特征（添加缺失字段）

如果 V3 特征缺少 `ca_coords` 或 `all_atom_coords` 字段：

```bash
# 从 V2 特征复制缺失字段
python scripts/update_v3_features.py
```

## 验证

### 检查目录结构

```bash
# 检查年份范围目录
ls data/complete_features_v3/

# 检查 PDB ID 目录
ls data/complete_features_v3/1981-2000/ | head -5

# 检查文件命名
ls data/complete_features_v3/1981-2000/1a0q/
```

### 检查特征文件内容

```python
import torch

# 加载特征文件
data = torch.load('data/complete_features_v3/1981-2000/1a0q/1a0q_protein.pt',
                  weights_only=False)

# 检查字段
print('Keys:', sorted(data.keys()))

# 检查维度
print('physchem_features:', data['physchem_features'].shape)  # [N, 62]
print('spatial_scalar_features:', data['spatial_scalar_features'].shape)  # [N, 72]
print('spatial_vector_features:', data['spatial_vector_features'].shape)  # [N, 6, 3]
print('ca_coords:', data['ca_coords'].shape)  # [N, 3]
print('all_atom_coords:', len(data['all_atom_coords']))  # N
```

### 使用数据检查脚本

```bash
# 检查所有数据是否就绪
python scripts/check_v8_data.py
```

## 与 V2 的对比

| 特性 | V2 | V3 |
|------|----|----|
| 目录结构 | `YEAR_RANGE/PDB_ID/` | `YEAR_RANGE/PDB_ID/` ✅ |
| 文件命名 | `{pdb_id}_protein.pt` | `{pdb_id}_protein.pt` ✅ |
| 特征维度 | 94D (单一特征) | 62D + 72D + 6×3D (分类特征) |
| ca_coords | ✅ | ✅ |
| all_atom_coords | ✅ | ✅ |
| 元数据 | ✅ | ✅ |

## 常见问题

### Q1: 如何确定年份范围？

A: 脚本会自动从输入路径中提取年份范围。输入路径应包含类似 `1981-2000`, `2001-2010`, `2011-2019` 的目录。

### Q2: 如果无法确定年份范围怎么办？

A: 如果输入路径中没有年份范围信息，脚本会使用扁平结构（直接在 PDB_ID 目录下保存）。

### Q3: 如何处理已存在的文件？

A: 使用 `--overwrite` 参数强制覆盖，否则会跳过已存在的文件。

### Q4: 特征提取失败怎么办？

A: 检查错误信息，常见原因：
- PDB 文件格式问题
- 缺少必需的原子（如 Cα）
- 内存不足（减少 workers 数量）

## 相关脚本

- **提取脚本**: `preparation/protein/extract_complete_features_v3.py`
- **重组脚本**: `scripts/reorganize_v3_features.py`
- **更新脚本**: `scripts/update_v3_features.py`
- **验证脚本**: `preparation/protein/test_v3_features.py`
- **数据检查**: `scripts/check_v8_data.py`
