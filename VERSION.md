# Pocket-GVP-Fusion 版本说明

本文档记录了Pocket-GVP-Fusion模型的各个版本及其主要变化。

---

## V5 (当前版本) - 2026-02-08

### 主要特性
- ✅ **VDW半径距离阈值** (BioLiP标准)
- ✅ **物理准确的结合位点定义**
- ✅ 80维配体特征 (35D V2 + 45D全局特征)
- ✅ 使用所有原子（包括侧链）计算距离
- ✅ 自适应阈值: d < r_vdw(P) + r_vdw(L) + 0.5 Å

### 距离计算改进
**V4问题**: 固定3.5 Å阈值忽略原子大小差异
- 对大原子（O、N）太严格
- 对小原子（H）太宽松
- 忽略非水接触

**V5改进**: BioLiP标准VDW半径
```python
# VDW半径 (Bondi半径, Å)
C: 1.70, N: 1.55, O: 1.52, S: 1.80
P: 1.80, F: 1.47, Cl: 1.75, Br: 1.85, I: 1.98

# 距离阈值
threshold = r_vdw(protein_atom) + r_vdw(ligand_atom) + 0.5 Å

# 示例
C-C: 1.70 + 1.70 + 0.5 = 3.9 Å
O-O: 1.52 + 1.52 + 0.5 = 2.54 Å
N-C: 1.55 + 1.70 + 0.5 = 3.75 Å
```

### 数据统计
**完整数据集 (19,032个复合物)**:
- 平均结合位点比例: 4.7%
- 中位数: 3.8%
- 范围: 0.1% - 70.8%
- ✅ 所有蛋白质都有结合位点

**结合位点分布**:
- 0-1%: 1,327 (7.0%)
- 1-5%: 11,084 (58.2%)
- 5-10%: 5,053 (26.6%)
- 10-20%: 1,428 (7.5%)
- >20%: 140 (0.7%)

### 文件结构
```
configs/
  └── pocket_gvp_v5.yaml          # V5配置文件

src/data/
  └── pocket_dataset_v5.py        # V5数据加载器

scripts/
  ├── train_v5.py                 # V5训练脚本
  └── calculate_distances_vdw.py  # VDW距离计算

data/
  ├── residue_ligand_distances_vdw.pt  # VDW距离标签
  └── ligand_features_v3/              # 80维配体特征
```

### 使用方法
```bash
# 训练
python scripts/train_v5.py --config configs/pocket_gvp_v5.yaml --amp

# 分布式训练
torchrun --nproc_per_node=2 scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml --amp
```

### 关键改进
1. **物理准确性**: 考虑原子范德华半径，更符合物理实际
2. **使用所有原子**: 包括蛋白质侧链原子，不只是Cα
3. **自适应阈值**: 不同原子对有不同的距离阈值
4. **更合理的标签**: 避免固定阈值的偏差

### 配置变化
```yaml
data:
  distances_file: "data/residue_ligand_distances_vdw.pt"  # V5: VDW距离
```

---

## V4 - 2026-02-08

### 主要特性
- ✅ **80维配体特征** (35D V2 + 45D全局特征)
- ✅ 保留原始实验坐标用于距离计算
- ✅ 构象柔性分析 (RMSF)
- ✅ 电子效应特征 (Gasteiger电荷、LogP、TPSA)
- ✅ 分子级别描述符 (10维)

### 配体特征详情
**80维特征组成**:
- **35维V2特征**:
  - 原子符号 (11维)
  - 杂化类型 (6维)
  - 芳香性 (1维)
  - 环特征 (7维)
  - 度特征 (5维)
  - Gasteiger电荷 (1维)
  - LogP贡献 (1维)
  - TPSA贡献 (1维)
  - RMSF (1维)
  - 柔性标志 (1维)

- **45维全局特征** (pypropel):
  - Morgan指纹 (43维)
  - LogP (1维)
  - TPSA归一化 (1维)

### 文件结构
```
configs/
  └── pocket_gvp_v4.yaml          # V4配置文件

src/data/
  └── pocket_dataset_v4.py        # V4数据加载器

scripts/
  └── train_v4.py                 # V4训练脚本

data/
  └── ligand_features_v3/         # 80维配体特征
```

### 使用方法
```bash
# 训练
python scripts/train_v4.py --config configs/pocket_gvp_v4.yaml --amp

# 分布式训练
torchrun --nproc_per_node=2 scripts/train_v4.py \
    --config configs/pocket_gvp_v4.yaml --amp
```

### 关键改进
1. **更丰富的化学信息**: 45维全局特征提供分子指纹和全局描述符
2. **保留实验坐标**: 使用`coords_original`确保距离计算准确
3. **柔性分析**: RMSF和柔性标志提供动态信息
4. **电子效应**: Gasteiger电荷、LogP、TPSA提供更准确的相互作用建模

### 配置变化
```yaml
model:
  d_ligand_atom: 80  # V4: 80维 (V3: 35维)

data:
  ligand_features_dir: "ligand_features_v3"  # V4: 新目录
```

---

## V3 - 2026-02-07

### 主要特性
- ✅ **35维配体特征** (30D原子 + 3D电子效应 + 2D柔性)
- ✅ ESM-2 embeddings (1280维)
- ✅ 完整蛋白质特征 (82维)
- ✅ GVP几何特征
- ✅ 嵌套目录结构

### 配体特征详情
**35维特征组成**:
- 原子符号 (11维)
- 杂化类型 (6维)
- 芳香性 (1维)
- 环特征 (7维)
- 度特征 (5维)
- Gasteiger电荷 (1维)
- LogP贡献 (1维)
- TPSA贡献 (1维)
- RMSF (1维)
- 柔性标志 (1维)

### 文件结构
```
configs/
  └── pocket_gvp_v2_nested.yaml   # V3配置文件

src/data/
  └── pocket_dataset_v3.py        # V3数据加载器

scripts/
  └── train_v3_optimized.py       # V3优化训练脚本

data/
  ├── complete_features/          # 82维蛋白质特征
  ├── esm2_latents/               # ESM-2 embeddings
  └── ligand_features/            # 35维配体特征
```

### 关键改进
1. **ESM-2迁移**: 从ESM-3 (1536D) 迁移到ESM-2 (1280D)
2. **训练优化**: torch.compile、DDP、AMP支持
3. **完整日志**: TensorBoard健康诊断和分类指标
4. **数据清理**: 修复134个损坏的ESM2文件

---

## V2 - 2026-02-05

### 主要特性
- ✅ 7模块混合架构
- ✅ ESM-3 embeddings (1536维)
- ✅ GVP结构编码器
- ✅ 双路径投影 (GVP + ESM)
- ✅ 多层交叉注意力

### 架构组件
1. **GVP编码器**: 3D几何特征提取
2. **GVP投影**: 256D → 256D
3. **ESM投影**: 1536D → 256D
4. **蛋白质上下文编码器**: 4层Transformer
5. **配体编码器**: 3层GINEConv
6. **交叉注意力**: 4层蛋白质-配体交互
7. **分类器**: 3类预测 (Contact/Near/Far)

### 文件结构
```
configs/
  └── pocket_gvp_v2.yaml          # V2配置文件

src/models/
  └── pocket_gvp_fusion.py        # V2模型架构

scripts/
  └── train_v3.py                 # V2训练脚本
```

---

## V1 - 2026-01-XX

### 主要特性
- ✅ 基础GVP-Fusion架构
- ✅ 单路径特征融合
- ✅ 简单交叉注意力

---

## 版本对比

| 特性 | V1 | V2 | V3 | V4 | V5 |
|------|----|----|----|----|-----|
| **配体特征维度** | 64 | 64 | 35 | 80 | 80 |
| **ESM模型** | ESM-3 | ESM-3 | ESM-2 | ESM-2 | ESM-2 |
| **ESM维度** | 1536 | 1536 | 1280 | 1280 | 1280 |
| **蛋白质特征** | 基础 | 完整82D | 完整82D | 完整82D | 完整82D |
| **距离阈值** | 固定 | 固定 | 固定 | 固定 | VDW自适应 |
| **使用原子** | Cα | Cα | Cα | Cα | 所有原子 |
| **GVP编码器** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **双路径投影** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **多层交叉注意力** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **torch.compile** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **分布式训练** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **AMP支持** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **构象柔性** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **全局特征** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **原始坐标** | ✗ | ✗ | ✓ | ✓ | ✓ |

---

## 配体特征演进

### V1/V2: 64维 (PoC风格)
```
[19D 基础特征] + [45D 全局标签]
```
**问题**: 45维全局标签对每个原子重复，信息冗余

### V3: 35维 (优化版)
```
[30D 原子特征] + [3D 电子效应] + [2D 柔性]
```
**改进**: 每个原子独立特征，无冗余

### V4: 80维 (混合版)
```
[35D V2特征] + [45D 全局特征]
```
**优势**: 结合V3的详细原子特征和全局分子信息
**距离**: 固定3.5 Å阈值

### V5: 80维 + VDW距离 (物理准确版)
```
[35D V2特征] + [45D 全局特征] + VDW自适应阈值
```
**优势**:
- 保留V4的丰富特征
- 使用物理准确的VDW半径距离阈值
- 考虑所有原子（包括侧链）
- 更合理的结合位点定义

---

## 数据目录结构

### 嵌套结构 (V3/V4)
```
data/
├── complete_features/
│   ├── 1981-2000/
│   │   └── {pdb_id}/
│   │       └── {pdb_id}_protein.pt
│   ├── 2001-2010/
│   └── 2011-2019/
├── esm2_latents/
│   ├── 1981-2000/
│   ├── 2001-2010/
│   └── 2011-2019/
├── ligand_features/          # V3: 35维
│   ├── 1981-2000/
│   ├── 2001-2010/
│   └── 2011-2019/
└── ligand_features_v3/       # V4: 80维
    ├── 1981-2000/
    ├── 2001-2010/
    └── 2011-2019/
```

---

## 训练命令对比

### V2
```bash
python scripts/train_v3.py --config configs/pocket_gvp_v2.yaml
```

### V3
```bash
# 单GPU + AMP
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml --amp

# 双GPU + torch.compile
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml --compile --amp
```

### V4
```bash
# 单GPU + AMP
python scripts/train_v4.py \
    --config configs/pocket_gvp_v4.yaml --amp

# 双GPU + torch.compile
torchrun --nproc_per_node=2 scripts/train_v4.py \
    --config configs/pocket_gvp_v4.yaml --compile --amp
```

### V5
```bash
# 单GPU + AMP
python scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml --amp

# 双GPU + torch.compile
torchrun --nproc_per_node=2 scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml --compile --amp
```

---

## 性能预期

| 版本 | 配体特征 | 距离标签 | 训练速度 | 内存占用 | 预期性能 |
|------|---------|---------|---------|---------|---------|
| V2 | 64维 | 固定阈值 | 基准 | 基准 | 基准 |
| V3 | 35维 | 固定阈值 | +10% | -15% | +2-3% |
| V4 | 80维 | 固定阈值 | -5% | +10% | +3-5% |
| V5 | 80维 | VDW自适应 | -5% | +10% | +5-8% |

**V5性能提升来源**:
- ✅ 更物理准确的结合位点定义
- ✅ 考虑原子大小差异（VDW半径）
- ✅ 使用所有原子（包括侧链）
- ✅ 更合理的正负样本分布
- ✅ 保留V4的丰富化学信息

---

## 迁移指南

### 从V4迁移到V5

1. **重新计算VDW距离标签** (已完成):
```bash
python scripts/calculate_distances_vdw.py \
    --data-root data \
    --output data/residue_ligand_distances_vdw.pt \
    --workers 32
```

2. **使用V5配置**:
```bash
python scripts/train_v5.py --config configs/pocket_gvp_v5.yaml --amp
```

3. **模型无需修改**: 只是距离标签变化，模型架构保持不变

### 从V3迁移到V4

1. **提取新配体特征**:
```bash
python /home/husrcf/Code/ProtBind/extract_ligand_features_v3.py \
    --input data/raw \
    --output data/ligand_features_v3 \
    --conformers 50 \
    --workers 32
```

2. **使用V4配置**:
```bash
python scripts/train_v4.py --config configs/pocket_gvp_v4.yaml --amp
```

3. **模型无需修改**: 只是输入维度变化，模型架构保持不变

---

## 文档索引

- **架构文档**: `MODEL_ARCHITECTURE.md`
- **特征对比**: `FEATURE_EXTRACTION_COMPARISON.md`
- **配体特征升级**: `LIGAND_FEATURES_V2_UPGRADE.md`
- **数据加载器测试**: `DATALOADER_TEST_REPORT.md`
- **训练指南**: `TRAINING_GUIDE.md`

---

## 更新日志

### 2026-02-08 (V5)
- ✅ 创建V5版本
- ✅ 实现BioLiP标准VDW半径距离计算
- ✅ 使用所有原子（包括侧链）计算距离
- ✅ 生成19,032个复合物的VDW距离标签
- ✅ 创建pocket_dataset_v5.py
- ✅ 创建train_v5.py
- ✅ 创建pocket_gvp_v5.yaml
- ✅ 验证所有蛋白质都有合理的结合位点比例

### 2026-02-08 (V4)
- ✅ 创建V4版本
- ✅ 添加80维配体特征支持
- ✅ 创建pocket_dataset_v4.py
- ✅ 创建train_v4.py
- ✅ 创建pocket_gvp_v4.yaml
- ✅ 测试数据加载器对齐

### 2026-02-07
- ✅ 恢复TensorBoard完整日志
- ✅ 清空文件黑名单
- ✅ 创建数据加载器测试报告

### 2026-02-06
- ✅ ESM-2迁移完成
- ✅ 修复134个损坏文件
- ✅ 添加torch.compile支持
- ✅ 修复AMP dtype问题

---

**当前推荐版本**: V5 (VDW自适应距离阈值)
**稳定版本**: V4 (80维配体特征)
