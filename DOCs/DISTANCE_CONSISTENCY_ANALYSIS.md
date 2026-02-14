# 距离文件残基数一致性分析报告

## 执行摘要

检查了 **19,037** 个复合物的VDW距离文件和原始距离文件，发现 **81个不匹配样本 (0.43%)**。

## 数据集概览

| 指标 | 数值 |
|------|------|
| VDW距离文件复合物数 | 19,037 |
| 原始距离文件复合物数 | 19,037 |
| 共同复合物数 | 19,037 |
| 不匹配样本数 | 81 |
| 不匹配比例 | 0.43% |

## 长度差异分布

### 统计摘要

| 差异范围 | 样本数 | 占比 |
|---------|--------|------|
| 原始更长 (-13 to -1) | 53 | 65.4% |
| VDW更长 (+1 to +42) | 28 | 34.6% |

### 详细分布

| 差异 (残基) | 样本数 | 占比 | 说明 |
|------------|--------|------|------|
| -13 | 1 | 1.2% | 原始文件多13个残基 |
| -11 | 1 | 1.2% | |
| -10 | 1 | 1.2% | |
| -9 | 1 | 1.2% | |
| -8 | 1 | 1.2% | |
| -4 | 2 | 2.5% | |
| -3 | 1 | 1.2% | |
| -2 | 3 | 3.7% | |
| **-1** | **43** | **53.1%** | **最常见：原始文件多1个残基** |
| +1 | 1 | 1.2% | VDW文件多1个残基 |
| +5 | 1 | 1.2% | |
| +6 | 3 | 3.7% | |
| +7 | 6 | 7.4% | |
| +8 | 2 | 2.5% | |
| +9 | 4 | 4.9% | |
| +10 | 3 | 3.7% | |
| +17 | 2 | 2.5% | |
| +20 | 1 | 1.2% | |
| +21 | 1 | 1.2% | |
| +30 | 1 | 1.2% | |
| +36 | 1 | 1.2% | |
| +42 | 1 | 1.2% | 最大差异 |

## 字段差异分析

### VDW距离文件特有字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `pdb_id` | str | PDB标识符 |
| `distances_vdw_adjusted` | Tensor | 扣除VDW半径后的距离 |
| `is_binding_vdw` | Tensor[bool] | 是否为结合位点 |
| `n_binding` | int | 结合位点残基数 |
| `n_total` | int | 总残基数 |
| `binding_ratio` | float | 结合位点比例 |

### 原始距离文件特有字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `n_residues` | int | 残基总数 |
| `min_distance` | float | 最小距离 |
| `max_distance` | float | 最大距离 |
| `mean_distance` | float | 平均距离 |

### 共同字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `distances` | Tensor | 残基-配体最小距离 |

## 典型不匹配样例分析

### 样例1: 1i6v (VDW更长 +36残基)

**基本信息**：
- VDW长度: 2857
- 原始长度: 2821
- 差异: +36 (VDW多36个残基)

**VDW数据**：
```
distances: [2857]
distances_vdw_adjusted: [2857], min=-0.76Å, max=129.79Å, mean=35.01Å
n_binding: 15
binding_ratio: 0.53%
```

**原始数据**：
```
distances: [2821]
min_distance: 1.75Å
max_distance: 84.70Å
mean_distance: 37.71Å
```

**分析**：VDW文件包含了更多的残基，可能是因为：
- VDW计算时使用了更完整的蛋白质结构
- 原始文件在预处理时过滤了某些残基

### 样例2: 1dar (原始更长 -2残基)

**基本信息**：
- VDW长度: 618
- 原始长度: 620
- 差异: -2 (原始多2个残基)

**VDW数据**：
```
distances: [618]
distances_vdw_adjusted: [618], min=-1.06Å, max=71.13Å, mean=26.62Å
n_binding: 12
binding_ratio: 1.94%
```

**原始数据**：
```
distances: [620]
min_distance: 1.85Å
max_distance: 74.17Å
mean_distance: 29.34Å
```

**分析**：原始文件包含了更多的残基，可能是因为：
- VDW计算时过滤了某些无法计算VDW半径的残基
- 不同的缺失残基处理策略

### 样例3: 6d6t (VDW更长 +42残基)

**基本信息**：
- VDW长度: 2085
- 原始长度: 2043
- 差异: +42 (VDW多42个残基，最大差异)

**VDW数据**：
```
distances: [2085]
distances_vdw_adjusted: [2085], min=-0.48Å, max=95.79Å, mean=43.08Å
n_binding: 11
binding_ratio: 0.53%
```

**原始数据**：
```
distances: [2043]
min_distance: 2.37Å
max_distance: 98.61Å
mean_distance: 45.80Å
```

**分析**：这是差异最大的样例，VDW文件多了42个残基。

## 不匹配原因分析

### 1. 预处理差异

两个距离文件可能在不同时间点、使用不同的预处理流程生成：

- **VDW距离文件** (`residue_ligand_distances_vdw.pt`):
  - 使用 `calculate_distances_vdw.py` 生成
  - 考虑VDW半径
  - 可能使用了更新的PDB文件或特征文件

- **原始距离文件** (`residue_ligand_distances_poc_style.pt`):
  - 使用固定阈值
  - 可能基于较早的数据版本

### 2. 缺失残基处理

不同的处理策略：
- VDW计算可能跳过无法确定VDW半径的残基
- 原始计算可能跳过坐标缺失的残基
- 不同的链断裂处理方式

### 3. 特征文件对齐

残基数应该与以下文件对齐：
- `complete_features/` (理化特征)
- `esm2_latents/` (ESM嵌入)
- `ligand_features_v3/` (配体特征)

不匹配可能是因为距离文件与这些特征文件的对齐方式不同。

## 影响评估

### 对训练的影响

1. **数据加载**：
   - 修复前：可能导致维度不匹配错误
   - 修复后：使用 `min(lengths)` 确保一致性

2. **标签质量**：
   - 0.43%的样本可能使用截断的标签
   - 对整体训练影响较小（99.57%的样本正常）

3. **评估指标**：
   - 原始标签评估可能在81个样本上略有偏差
   - 但由于比例很小，对整体指标影响可忽略

### 风险等级

| 风险类型 | 等级 | 说明 |
|---------|------|------|
| 训练稳定性 | 🟢 低 | 已通过代码修复 |
| 标签质量 | 🟡 中 | 0.43%样本可能有轻微偏差 |
| 模型性能 | 🟢 低 | 影响可忽略 |

## 建议措施

### 短期措施（已实施）✅

1. **代码层面防御**：
   ```python
   # 数据集层面
   lengths = [len(esm), len(protein_scalar), ..., len(labels)]
   if labels_original is not None:
       lengths.append(len(labels_original))
   n_residues = min(lengths)

   # Collate层面
   n_orig = min(len(labels_orig), n_res)
   labels_original[i, :n_orig] = labels_orig[:n_orig]
   ```

2. **日志记录**：
   - 保存不匹配列表到 `data/distance_mismatches.txt`
   - 便于后续追踪和分析

### 中期措施（建议）

1. **重新生成原始距离文件**：
   ```bash
   # 使用与VDW相同的特征文件重新计算
   python scripts/calculate_distances_original.py \
       --data-root data \
       --output data/residue_ligand_distances_poc_style_v2.pt \
       --align-with data/residue_ligand_distances_vdw.pt
   ```

2. **添加数据验证步骤**：
   ```python
   # 在训练前验证
   python scripts/validate_data_consistency.py \
       --check-distances \
       --check-features \
       --check-embeddings
   ```

### 长期措施（可选）

1. **统一数据生成流程**：
   - 创建单一的距离计算脚本
   - 同时生成VDW和原始标签
   - 确保完全对齐

2. **版本控制**：
   - 为数据文件添加版本号
   - 记录生成时间和参数
   - 便于追溯和重现

## 不匹配样本列表

完整的81个不匹配样本已保存到：
```
data/distance_mismatches.txt
```

格式：
```
PDB_ID VDW_LENGTH ORIG_LENGTH DIFF
1dar 618 620 -2
1h1d 213 214 -1
...
```

## 结论

1. **问题规模**：0.43%的样本存在残基数不匹配，影响范围很小
2. **主要模式**：53.1%的不匹配是原始文件多1个残基
3. **已修复**：代码层面已实现多层防御，确保训练稳定性
4. **建议**：可选择重新生成原始距离文件以完全消除不匹配

---

**分析日期**: 2026-02-11
**分析工具**: `scripts/check_distance_consistency.py`
**数据版本**: VDW (19,037), Original (19,037)
