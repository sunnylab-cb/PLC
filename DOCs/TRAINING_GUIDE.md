# 训练指南 - Pocket-GVP-Fusion v2

## 概述

本指南说明如何使用ESM3特征训练Pocket-GVP-Fusion v2模型。

## 当前状态

✅ **已完成的工作**：
1. 创建了支持嵌套目录结构的数据集类 (`src/data/pocket_dataset_v3.py`)
2. 更新了collate函数以自动检测ESM维度
3. 创建了新的训练脚本 (`scripts/train_v3.py`)
4. 创建了配置文件 (`configs/pocket_gvp_v2_nested.yaml`)

⚠️ **当前问题**：
- ESM嵌入长度与蛋白质特征长度不匹配
- 需要修复数据集中的长度一致性检查

## 数据结构

您的数据按年份分组：

```
data/
├── complete_features/
│   ├── 1981-2000/
│   │   └── {pdb_id}/
│   │       └── {pdb_id}_protein.pt
│   ├── 2001-2010/
│   └── 2011-2019/
├── esm3_latents/
│   ├── 1981-2000/
│   │   └── {pdb_id}/
│   │       └── {pdb_id}_protein.pt
│   └── ...
├── ligand_features/
│   └── ...
└── residue_ligand_distances_v2.pt
```

**数据统计**：
- 总复合物数：19,024
- 训练集：15,219
- 验证集：1,902
- 测试集：1,903

## 快速修复方案

需要修复 `src/data/pocket_dataset_v3.py` 中的长度一致性问题：

```python
# 在 __getitem__ 方法中，第199行附近
# 当前代码：
n_residues = int(data.get('n_residues', len(protein_scalar)))

# 应该改为：
n_residues = min(len(esm), len(protein_scalar), len(gvp_coords))
```

## 训练命令

修复后，运行：

```bash
cd /home/husrcf/Code/ProtBind/PLC

# 使用新的v3训练脚本
python scripts/train_v3.py --config configs/pocket_gvp_v2_nested.yaml
```

## 配置说明

**关键配置** (`configs/pocket_gvp_v2_nested.yaml`):

```yaml
data:
  data_root: "data"                 # 数据根目录
  distances_file: "data/residue_ligand_distances_v2.pt"

model:
  d_esm: 1536                       # ESM3维度
  d_ligand_atom: 35                 # 配体特征维度

training:
  batch_size: 8                     # 批大小
  num_workers: 0                    # 先用0调试
```

## 监控训练

```bash
# TensorBoard
tensorboard --logdir outputs/pocket_gvp_fusion_v2_*/runs

# 查看输出
ls outputs/pocket_gvp_fusion_v2_*/checkpoints/
```

## 下一步

1. **修复长度不匹配问题**（见上面的快速修复）
2. **测试单个batch**：设置 `num_workers: 0` 调试
3. **增加workers**：调试成功后改为 `num_workers: 4`
4. **调整批大小**：根据GPU内存调整 `batch_size`

## 模型架构

- **参数量**：31.16M
- **ESM维度**：1536 (ESM3)
- **配体维度**：35
- **输出类别**：3 (Contact/Near/Far)

## 常见问题

### Q: 训练时出现维度不匹配？
A: 检查配置文件中的 `d_esm` 和 `d_ligand_atom` 是否与实际数据匹配。

### Q: DataLoader worker错误？
A: 先设置 `num_workers: 0` 调试，成功后再增加。

### Q: 内存不足？
A: 减小 `batch_size` 或 `max_seq_len`。

## 联系

如有问题，请检查：
1. 数据文件是否完整
2. ESM3特征是否正确生成
3. 配置文件参数是否匹配数据

---

**最后更新**：2026-02-05
