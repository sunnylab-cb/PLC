# 数据加载器对齐测试报告

**测试日期**: 2026-02-06
**测试脚本**: `scripts/test_dataloader.py`
**数据集大小**: 19,037 个蛋白质-配体复合物
**测试结果**: ✅ 所有测试通过

---

## 测试概览

本测试验证了数据加载器的完整流程，确保所有特征正确对齐，维度匹配，并且可以正确输入到模型中。

---

## 测试 1: 单个样本加载 ✅

### 测试内容
- 验证数据集可以正确加载单个样本
- 检查所有特征的维度是否正确
- 验证特征对齐

### 测试结果

**样本信息**:
- PDB ID: 第一个样本
- 序列长度: 416 个残基
- 配体原子数: 59 个原子

**蛋白质特征**:
| 特征 | 形状 | 期望维度 | 状态 |
|------|------|----------|------|
| ESM embeddings | [416, 1280] | 1280 | ✅ |
| Protein scalar | [416, 82] | 82 | ✅ |
| GVP vectors | [416, 4, 3] | [4, 3] | ✅ |
| GVP coords | [416, 3] | 3 | ✅ |

**配体特征**:
| 特征 | 形状 | 期望维度 | 状态 |
|------|------|----------|------|
| Ligand features | [59, 35] | 35 | ✅ |
| Ligand coords | [59, 3] | 3 | ✅ |

**标签分布**:
- Class 0 (Contact, <3.5Å): 2 个残基 (0.48%)
- Class 1 (Near, 3.5-6.0Å): 7 个残基 (1.68%)
- Class 2 (Far, >6.0Å): 407 个残基 (97.84%)

**对齐验证**:
- ✅ ESM embeddings 长度 = 416
- ✅ Protein scalar 长度 = 416
- ✅ GVP vectors 长度 = 416
- ✅ Labels 长度 = 416
- ✅ 所有特征长度一致

---

## 测试 2: Batch Collation ✅

### 测试内容
- 验证多个样本可以正确组成batch
- 检查padding和mask是否正确
- 验证配体图构建

### 测试结果

**Batch配置**:
- Batch size: 4
- Max sequence length: 418
- Max ligand atoms: 66

**每个样本的有效长度**:
| 样本 | 蛋白质长度 | 配体原子数 |
|------|-----------|-----------|
| 0 | 416 | 59 |
| 1 | 416 | 66 |
| 2 | 418 | 41 |
| 3 | 415 | 57 |

**Batch形状验证**:
| 特征 | 形状 | 状态 |
|------|------|------|
| ESM embeddings | [4, 418, 1280] | ✅ |
| Protein scalar | [4, 418, 82] | ✅ |
| Protein vectors | [4, 418, 4, 3] | ✅ |
| Protein coords | [4, 418, 3] | ✅ |
| Protein mask | [4, 418] | ✅ |
| Labels | [4, 418] | ✅ |
| Ligand features | [4, 66, 35] | ✅ |
| Ligand coords | [4, 66, 3] | ✅ |
| Ligand mask | [4, 66] | ✅ |

**Batch标签分布**:
- Class 0 (Contact): 8 (0.48%)
- Class 1 (Near): 19 (1.14%)
- Class 2 (Far): 1638 (98.38%)

**配体图构建**:
- ✅ Ligand x: [223, 35] - 总共223个原子节点
- ✅ Ligand edge_index: [2, 524] - 524条边
- ✅ Ligand edge_attr: [524, 6] - 每条边6维特征
- ✅ Ligand batch: [223] - batch索引

---

## 测试 3: 模型输入准备 ✅

### 测试内容
- 验证batch可以正确转换为模型输入
- 检查所有张量在正确的设备上
- 验证数据类型正确

### 测试结果

**设备检查**:
- ✅ 所有张量都在 CUDA 设备上

**模型输入形状**:
| 输入 | 形状 | Dtype | 设备 |
|------|------|-------|------|
| esm_embeddings | [2, 416, 1280] | float32 | cuda:0 |
| protein_scalar | [2, 416, 82] | float32 | cuda:0 |
| protein_vectors | [2, 416, 4, 3] | float32 | cuda:0 |
| protein_coords | [2, 416, 3] | float32 | cuda:0 |
| protein_mask | [2, 416] | bool | cuda:0 |
| ligand_x | [125, 35] | float32 | cuda:0 |
| ligand_edge_index | [2, 298] | int64 | cuda:0 |
| ligand_edge_attr | [298, 6] | float32 | cuda:0 |
| ligand_batch | [125] | int64 | cuda:0 |
| ligand_to_protein | [2, 66, 35] | float32 | cuda:0 |
| ligand_mask | [2, 66] | bool | cuda:0 |

**数据类型验证**:
- ✅ 浮点特征: float32
- ✅ 整数索引: int64
- ✅ Mask: bool

---

## 测试 4: 对齐一致性检查 ✅

### 测试内容
- 测试10个样本的特征对齐
- 验证所有蛋白质特征长度一致
- 验证所有配体特征长度一致
- 检查标签值范围

### 测试结果

**样本 0 详细信息**:
- 序列长度: 416
- 配体原子数: 59

**一致性验证**:
- ✅ 所有10个样本的ESM、scalar、vectors、coords、labels长度一致
- ✅ 所有10个样本的ligand features和coords长度一致
- ✅ 所有标签值在 [0, 2] 范围内

---

## 测试 5: 边界情况 ✅

### 测试内容
- 扫描数据集查找最短和最长序列
- 测试截断逻辑是否正确工作

### 测试结果

**序列长度统计** (前100个样本):
- 蛋白质: min=74, max=1000
- 配体: min=10, max=100

**截断测试** (max_seq_len=100, max_atoms=20):
- ✅ 截断后蛋白质长度: 100 (符合限制)
- ✅ 截断后配体长度: 20 (符合限制)

---

## 关键发现

### 1. 特征维度正确 ✅
- ESM-2 embeddings: 1280维 (正确)
- Protein scalar: 82维 (42D physchem + 14D surface + 26D geometry)
- GVP vectors: [4, 3] (4个方向向量，每个3维)
- Ligand features: 35维 (30D atom features + 5D physicochemical)

### 2. 数据对齐完美 ✅
- 所有蛋白质特征在序列维度上完全对齐
- 所有配体特征在原子维度上完全对齐
- Padding和mask正确应用

### 3. 标签分布符合预期 ✅
- 极度不平衡: ~98% Far, ~1% Near, ~0.5% Contact
- 这与蛋白质-配体结合的实际情况一致
- 需要使用Focal Loss和类别权重来处理

### 4. 图构建正确 ✅
- 配体分子图正确构建
- 边基于距离阈值(<2Å)连接
- Batch索引正确分配

### 5. 设备和数据类型正确 ✅
- 所有张量正确移动到CUDA
- 浮点特征使用float32
- 索引使用int64
- Mask使用bool

---

## 数据流程总结

```
1. PocketDatasetV3.__getitem__()
   ↓
   返回单个样本字典:
   - 'esm': [N, 1280]
   - 'protein_scalar': [N, 82]
   - 'gvp_vectors': [N, 4, 3]
   - 'gvp_coords': [N, 3]
   - 'ligand_features': [M, 35]
   - 'ligand_coords': [M, 3]
   - 'labels': [N]

2. pocket_collate_fn()
   ↓
   组合成batch并padding:
   - 'esm_embeddings': [B, L, 1280]
   - 'protein_scalar': [B, L, 82]
   - 'protein_vectors': [B, L, 4, 3]
   - 'protein_coords': [B, L, 3]
   - 'protein_mask': [B, L]
   - 'ligand_features': [B, M, 35]
   - 'ligand_coords': [B, M, 3]
   - 'ligand_mask': [B, M]
   - 'labels': [B, L]

3. build_ligand_graphs()
   ↓
   构建配体分子图:
   - 'ligand_x': [N_total, 35]
   - 'ligand_edge_index': [2, E_total]
   - 'ligand_edge_attr': [E_total, 6]
   - 'ligand_batch': [N_total]

4. prepare_model_inputs()
   ↓
   移动到设备并准备模型输入:
   - 所有张量 → device
   - 添加 'ligand_to_protein' (padding版本用于交叉注意力)
```

---

## 结论

✅ **所有测试通过！数据加载器工作正常。**

数据加载器正确实现了：
1. ✅ 特征加载和对齐
2. ✅ Batch collation和padding
3. ✅ 配体分子图构建
4. ✅ 设备转移和类型转换
5. ✅ 边界情况处理

数据现在可以安全地用于模型训练。

---

## 下一步

1. ✅ 数据加载器已验证
2. ✅ 黑名单已清空（所有19,037个样本可用）
3. ✅ TensorBoard日志已恢复
4. 🚀 **准备开始训练！**

推荐训练命令：
```bash
# 单GPU + AMP
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp

# 双GPU + AMP + torch.compile
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp --compile
```
