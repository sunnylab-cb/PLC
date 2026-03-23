# 推理输出格式说明

## 输出目录结构

```
outputs/infer_v7_20260220_143052/
├── 1a2b_predictions.npz          # NPZ 格式（原始格式）
├── 1a2b_predictions.tsv          # TSV 格式（详细预测）
├── 1c3d_predictions.npz
├── 1c3d_predictions.tsv
├── ...
└── pdb/                          # PDB 格式预测口袋
    ├── 1981-2000/
    │   ├── 1a2b/
    │   │   ├── 1a2b_protein.pdb
    │   │   ├── 1a2b_pocket.pdb
    │   │   ├── 1a2b_ligand.sdf
    │   │   └── 1a2b_pocket_predicted.pdb  # 新生成
    │   └── 1c3d/
    │       ├── 1c3d_protein.pdb
    │       ├── 1c3d_pocket.pdb
    │       ├── 1c3d_ligand.sdf
    │       └── 1c3d_pocket_predicted.pdb
    └── 2001-2010/
        └── ...
```

## 1. NPZ 格式 (`.npz`)

**用途**: Python 程序读取，保留完整的 numpy 数组

**读取方式**:
```python
import numpy as np

data = np.load('1a2b_predictions.npz')
pdb_id = str(data['pdb_id'])
predictions = data['predictions']  # [L] 预测类别
probabilities = data['probabilities']  # [L, 3] 类别概率
```

## 2. TSV 格式 (`.tsv`)

**用途**: 表格软件（Excel/LibreOffice）或数据分析

**格式示例**:
```tsv
residue_id	predicted_class	predicted_label	prob_contact	prob_interface	prob_non_binding	confidence
0	0	Contact	0.8523	0.1234	0.0243	0.8523
1	2	Non-binding	0.0456	0.0789	0.8755	0.8755
2	0	Contact	0.9123	0.0567	0.0310	0.9123
3	1	Interface	0.2345	0.6789	0.0866	0.6789
4	0	Contact	0.7856	0.1567	0.0577	0.7856
```

**读取方式**:
```python
import pandas as pd

df = pd.read_csv('1a2b_predictions.tsv', sep='\t')

# 筛选高置信度的 Contact 残基
high_conf_contacts = df[(df['predicted_class'] == 0) & (df['confidence'] > 0.8)]

# 筛选低置信度预测（可能需要人工检查）
low_conf = df[df['confidence'] < 0.5]
```

**在 Excel 中打开**:
1. 打开 Excel
2. 文件 -> 打开 -> 选择 `.tsv` 文件
3. 选择"分隔符" -> "制表符"

## 3. PDB 格式预测口袋 (`.pdb`)

**用途**: 结构可视化和分析

### 文件说明

| 文件名 | 说明 | 来源 |
|--------|------|------|
| `{pdb_id}_protein.pdb` | 完整蛋白质结构 | 从数据集复制 |
| `{pdb_id}_pocket.pdb` | 真实口袋残基 | 从数据集复制 |
| `{pdb_id}_ligand.sdf` | 配体结构 | 从数据集复制 |
| `{pdb_id}_pocket_predicted.pdb` | **预测的口袋残基** | **新生成** |

### 在 PyMOL 中可视化

```python
# 加载结构
load outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/1a2b_protein.pdb
load outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/1a2b_pocket.pdb
load outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/1a2b_pocket_predicted.pdb
load outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/1a2b_ligand.sdf

# 设置显示样式
hide everything
show cartoon, 1a2b_protein
color gray80, 1a2b_protein

# 真实口袋（绿色）
show sticks, 1a2b_pocket
color green, 1a2b_pocket

# 预测口袋（蓝色）
show sticks, 1a2b_pocket_predicted
color blue, 1a2b_pocket_predicted

# 配体（红色）
show sticks, 1a2b_ligand
color red, 1a2b_ligand

# 对比真实和预测
# 重叠部分会显示为青色（绿+蓝）
```

### 在 Chimera 中可视化

```
File -> Open -> 选择 pdb 文件
Actions -> Atoms/Bonds -> show
Select -> Residue -> by attribute value -> 选择特定残基
Actions -> Color -> 选择颜色
```

## 4. 使用场景

### 场景 1: 快速查看预测结果
```bash
# 查看 TSV 文件
column -t -s $'\t' outputs/infer_v7_20260220_143052/1a2b_predictions.tsv | less -S

# 统计各类别数量
awk -F'\t' 'NR>1 {count[$3]++} END {for (c in count) print c, count[c]}' \
    outputs/infer_v7_20260220_143052/1a2b_predictions.tsv
```

### 场景 2: 分析低置信度预测
```python
import pandas as pd

df = pd.read_csv('outputs/infer_v7_20260220_143052/1a2b_predictions.tsv', sep='\t')

# 找出置信度 < 0.5 的残基
low_conf = df[df['confidence'] < 0.5]
print(f"低置信度残基数: {len(low_conf)}")
print(low_conf[['residue_id', 'predicted_label', 'confidence']])
```

### 场景 3: 对比真实和预测口袋
```bash
# 在 PyMOL 中加载并对比
pymol outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/*.pdb \
      outputs/infer_v7_20260220_143052/pdb/1981-2000/1a2b/*.sdf
```

### 场景 4: 批量分析
```python
import pandas as pd
from pathlib import Path

output_dir = Path('outputs/infer_v7_20260220_143052')

# 读取所有 TSV 文件
all_results = []
for tsv_file in output_dir.glob('*_predictions.tsv'):
    df = pd.read_csv(tsv_file, sep='\t')
    pdb_id = tsv_file.stem.replace('_predictions', '')
    df['pdb_id'] = pdb_id
    all_results.append(df)

# 合并所有结果
combined = pd.concat(all_results, ignore_index=True)

# 分析整体置信度分布
print(combined['confidence'].describe())

# 找出所有低置信度预测
low_conf_all = combined[combined['confidence'] < 0.5]
print(f"总共 {len(low_conf_all)} 个低置信度预测")
```

## 5. 数据完整性

每次推理会生成以下文件：
- ✅ `{pdb_id}_predictions.npz` - 总是生成
- ✅ `{pdb_id}_predictions.tsv` - 总是生成
- ⚠️ `pdb/{year}/{pdb_id}/{pdb_id}_pocket_predicted.pdb` - 仅当能找到原始 PDB 文件时生成

如果某个 PDB ID 的原始文件不存在，会显示警告但不影响其他文件的生成。
