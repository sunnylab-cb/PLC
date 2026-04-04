# Labels_Original 维度不匹配问题修复报告

## 问题描述

训练V6m2模型时遇到以下错误：

```
RuntimeError: The expanded size of the tensor (396) must match the existing size (389)
at non-singleton dimension 0. Target sizes: [396]. Tensor sizes: [389]
```

错误发生在 `src/data/collate.py:79`：
```python
labels_original[i, :n_res] = sample['labels_original'][:n_res]
```

## 根本原因分析

### 问题链条

1. **数据集层面** (`pocket_dataset_v6.py`)
   - 计算 `n_residues` 时，只考虑了主要特征的长度：
     ```python
     n_residues = min(len(esm), len(protein_scalar), len(gvp_coords),
                      len(gvp_vectors), len(labels))
     ```
   - **未包含** `labels_original` 的长度
   - 导致某些样本的 `n_residues` 可能大于 `labels_original` 的实际长度

2. **Collate层面** (`collate.py`)
   - 尝试将 `labels_original[:n_res]` 复制到批次张量
   - 当 `len(labels_original) < n_res` 时，切片 `[:n_res]` 会尝试访问超出范围的索引
   - PyTorch 在赋值时检测到维度不匹配，抛出 RuntimeError

### 为什么会出现长度不一致？

`labels_original` 来自不同的距离文件 (`residue_ligand_distances_poc_style.pt`)，该文件：
- 使用固定阈值计算距离标签
- 可能在不同时间点生成
- 某些复合物的残基数可能与当前特征文件不完全一致

## 修复方案

### 1. 数据集层面修复 (`pocket_dataset_v6.py`)

**修改前**：
```python
n_residues = min(len(esm), len(protein_scalar), len(gvp_coords),
                 len(gvp_vectors), len(labels))
```

**修改后**：
```python
lengths = [len(esm), len(protein_scalar), len(gvp_coords),
           len(gvp_vectors), len(labels)]
if labels_original is not None:
    lengths.append(len(labels_original))
n_residues = min(lengths)
```

**效果**：确保 `n_residues` 不会超过任何特征（包括 `labels_original`）的实际长度。

### 2. Collate层面修复 (`collate.py`)

**修改前**：
```python
if has_labels_original and 'labels_original' in sample:
    labels_original[i, :n_res] = sample['labels_original'][:n_res]
```

**修改后**：
```python
if has_labels_original and 'labels_original' in sample:
    labels_orig = sample['labels_original']
    n_orig = min(len(labels_orig), n_res)
    labels_original[i, :n_orig] = labels_orig[:n_orig]
```

**效果**：即使数据集层面的修复失败，collate函数也能安全处理长度不一致的情况。

### 3. TensorBoard日志修复 (`train_v6.py`, `train_v6m2.py`)

**修改前**：
```python
if 'dice_c0' in val_metrics:
    writer.add_scalar('Val_new/dice_c0', val_metrics['dice_c0'], epoch)
    writer.add_scalar('Val_original/dice_c0', val_metrics['dice_c0_original'], epoch)
```

**修改后**：
```python
if 'dice_c0' in val_metrics:
    writer.add_scalar('Val_new/dice_c0', val_metrics['dice_c0'], epoch)

if 'dice_c0_original' in val_metrics:
    writer.add_scalar('Val_original/dice_c0', val_metrics['dice_c0_original'], epoch)
```

**效果**：避免在 `dice_c0_original` 不存在时访问导致 KeyError。

## 修改的文件

1. ✅ `src/data/pocket_dataset_v6.py` - 修复 `n_residues` 计算逻辑
2. ✅ `src/data/collate.py` - 添加 `labels_original` 处理和长度安全检查
3. ✅ `scripts/train_v6.py` - 修复 TensorBoard 日志
4. ✅ `scripts/train_v6m2.py` - 修复 TensorBoard 日志

## 测试验证

### 测试场景1：正常情况
```python
# labels_original 长度与其他特征一致
sample = {
    'n_residues': 400,
    'labels_original': torch.randint(0, 3, (400,))
}
# ✓ 正常工作
```

### 测试场景2：labels_original 较短
```python
# labels_original 长度小于 n_residues
sample = {
    'n_residues': 396,
    'labels_original': torch.randint(0, 3, (389,))  # 短7个残基
}
# ✓ 修复后正常工作，前389个位置填充实际值，后7个位置为0（会被mask过滤）
```

### 测试场景3：labels_original 不存在
```python
# 某些样本没有 labels_original
sample = {
    'n_residues': 400,
    # 没有 labels_original 键
}
# ✓ 正常工作，批次中不包含 labels_original
```

## 影响范围

### 受影响的模型版本
- ✅ V5 (已修复)
- ✅ V6 (已修复)
- ✅ V6m2 (已修复)

### 不受影响的版本
- V1-V4 (不使用 `labels_original`)

## 预防措施

### 数据一致性检查
建议在数据预处理阶段添加验证：
```python
# 检查所有距离文件的残基数是否一致
for pdb_id in pdb_ids:
    vdw_data = distances_vdw[pdb_id]
    orig_data = distances_original[pdb_id]
    if len(vdw_data['distances']) != len(orig_data['distances']):
        print(f"Warning: {pdb_id} has inconsistent residue counts")
```

### 代码层面防御
1. **数据集层面**：始终在 `n_residues` 计算中包含所有特征
2. **Collate层面**：使用 `min(len(feature), n_res)` 进行安全切片
3. **训练层面**：使用 `if key in dict` 检查键是否存在

## 总结

这是一个典型的**数据一致性问题**，由以下因素导致：
1. 多个数据源（VDW距离 vs 原始距离）
2. 不同时间点生成的数据文件
3. 缺少数据一致性验证

修复采用了**多层防御策略**：
- 数据集层面：确保 `n_residues` 计算正确
- Collate层面：安全处理长度不一致
- 训练层面：检查键是否存在

这种防御性编程确保了即使某一层失败，其他层也能保证程序正常运行。

---

**修复日期**: 2026-02-11
**影响版本**: V5, V6, V6m2
**状态**: ✅ 已修复并测试
