# ProtBind 处理流程对比：PoC vs PLC vs PLK

## 📁 目录结构对比

| 项目 | 路径 | 用途 | 状态 |
|------|------|------|------|
| **PoC** | `protein-ligand-PoC/` | 原型验证 | ✅ 完成 |
| **PLC** | `PLC/` | 生产版本（当前） | 🔄 开发中 |
| **PLK** | `PLK/` | 实验版本 | 🔬 研究中 |

---

## 🔄 主要差异

### **1. 数据结构**

#### **PoC (原型)**
```
database/features/
├── {pdb_id}_features.npz  # 扁平结构
└── ...
```
- ✅ 简单直接
- ❌ 不支持大规模数据
- 使用ESM-2 (1280维)

#### **PLC (当前生产)**
```
data/
├── complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt
├── esm3_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
├── ligand_features/YEAR/PDB_ID/{pdb_id}_ligand.pt
└── residue_ligand_distances_v2.pt
```
- ✅ 按年份分组，易管理
- ✅ 使用ESM-3 (1536维)
- ✅ 完整82维特征
- ✅ 独立距离标签文件

#### **PLK (实验)**
```
data/
├── complete_features/  # 同PLC
├── esm3_latents/       # 同PLC
└── ligand_features/    # 同PLC
```
- 与PLC数据结构相同
- 实验不同的模型架构

---

### **2. 特征维度**

| 特征类型 | PoC | PLC | PLK |
|---------|-----|-----|-----|
| **ESM** | 1280 (ESM-2) | 1536 (ESM-3) | 1536 (ESM-3) |
| **蛋白质标量** | 35 | 82 | 82 |
| **配体** | 64 | 35 | 35 |
| **GVP向量** | 4×3 | 4×3 | 4×3 |

**PLC特征组成** (82维):
```
42维 物理化学特征
+ 14维 表面特征
+ 26维 几何特征
= 82维 完整特征
```

---

### **3. 模型架构**

#### **核心架构（三者相同）**

```
GVP编码器 → 桥接层 → Transformer → 交叉注意力 → 分类器
```

#### **关键参数对比**

| 参数 | PoC | PLC | 说明 |
|------|-----|-----|------|
| `d_esm` | 1280 | 1536 | ESM维度 |
| `d_phys` | 35 | 82 | 蛋白质特征维度 |
| `d_model` | 512 | 512 | Transformer维度 |
| `n_cross_attention_layers` | 1 | 4 | 交叉注意力层数 |

**GVP路径输入维度**:
- PoC: `256 + 136 + 35 = 427`
- PLC: `256 + 136 + 82 = 474`

**分类器输入维度**:
- PoC: `512 + 35 = 547`
- PLC: `512 + 82 = 594`

---

### **4. 数据加载**

#### **PoC**
```python
# src/data/pocket_dataset.py
class PocketDataset:
    def __init__(self, feature_dir, esm_dir, ...):
        # 扁平结构，直接加载.npz和.pt
        self.feature_dir = Path(feature_dir)
        self.esm_dir = Path(esm_dir)
```

#### **PLC**
```python
# src/data/pocket_dataset_v3.py
class PocketDatasetV3:
    def __init__(self, data_root, ...):
        # 嵌套结构，按年份分组
        self.physchem_dir = data_root / 'complete_features'
        self.esm3_dir = data_root / 'esm3_latents'
        self.ligand_dir = data_root / 'ligand_features'

    def _find_file(self, pdb_id, subdir, filename):
        # 在多个年份目录中搜索
        for year_range in ['1981-2000', '2001-2010', '2011-2019']:
            ...
```

---

### **5. 标签定义**

#### **共同点**
```python
# 3类距离标签
Contact (<3.5Å):  0
Near (3.5-6.0Å):  1
Far (>6.0Å):      2
```

#### **差异**

| 项目 | PoC | PLC |
|------|-----|-----|
| **距离计算** | 内嵌在特征提取 | 独立文件 `residue_ligand_distances_v2.pt` |
| **距离方法** | Cα only | 全重原子 |
| **标签存储** | 在`.npz`中 | 独立字典文件 |

---

### **6. 训练配置**

#### **PoC** (`configs/pocket_gvp_v2.yaml`)
```yaml
data:
  feature_dir: "database/features"
  esm_dir: "data/esm3_latents"

model:
  d_esm: 1280
  d_phys: 35
  d_ligand_atom: 64

training:
  batch_size: 112
  num_workers: 4
```

#### **PLC** (`configs/pocket_gvp_v2_nested.yaml`)
```yaml
data:
  data_root: "data"
  distances_file: "data/residue_ligand_distances_v2.pt"

model:
  d_esm: 1536
  d_phys: 82
  d_ligand_atom: 35

training:
  batch_size: 32
  num_workers: 16
```

---

### **7. 训练脚本**

#### **主要差异**

| 功能 | PoC | PLC |
|------|-----|-----|
| **脚本** | `train.py` | `train_v3.py` |
| **数据集类** | `PocketDataset` | `PocketDatasetV3` |
| **ESM路径** | 参数传递 | 配置文件指定 |
| **距离标签** | 从`.npz`加载 | 从独立文件加载 |

**代码差异**:
```python
# PoC
train_loader = create_dataloader(
    feature_dir=feature_dir,
    esm_dir=esm_dir,  # 显式传递
    ...
)

# PLC
train_loader = create_dataloader_v3(
    data_root=data_root,  # 单一根目录
    distances_file=distances_file,
    ...
)
```

---

## 🔬 PLK 特殊功能

PLK目录包含额外的实验性功能：

### **距离相关**
```
PLK/
├── calculate_distances_v2.py      # 改进的距离计算
├── analyze_distance_distribution.py  # 距离分布分析
├── test_distance_binning.py       # 距离分箱测试
└── models/
    ├── distance_aware_moe.py      # 距离感知MoE
    ├── distance_model.py          # 距离预测模型
    └── distance_binning_head.py   # 距离分箱头
```

### **训练改进**
```
PLK/training/
├── distance_loss.py               # 距离损失函数
└── distance_loss_improved.py      # 改进的距离损失
```

---

## 📊 性能对比

| 指标 | PoC | PLC | 说明 |
|------|-----|-----|------|
| **数据集大小** | ~1000 | 19,024 | PLC支持更大规模 |
| **特征维度** | 35 | 82 | PLC信息更丰富 |
| **ESM质量** | ESM-2 | ESM-3 | PLC使用更新模型 |
| **批大小** | 112 | 32 | PLC更保守（调试） |
| **参数量** | ~31M | ~31.2M | 略有增加 |

---

## 🔄 迁移路径

### **从PoC迁移到PLC**

1. **数据准备**:
   ```bash
   # 1. 提取ESM3特征
   python esm3_extract_latents.py

   # 2. 提取完整特征（82维）
   python PLK/preparation/extract_complete_features.py

   # 3. 提取配体特征
   python PLK/preparation/extract_ligand_features.py

   # 4. 计算距离标签
   python PLK/calculate_distances_v2.py
   ```

2. **配置更新**:
   ```yaml
   # 从 pocket_gvp_v2.yaml
   # 改为 pocket_gvp_v2_nested.yaml

   model:
     d_esm: 1536  # 1280 → 1536
     d_phys: 82   # 35 → 82
     d_ligand_atom: 35  # 64 → 35
   ```

3. **代码更新**:
   ```python
   # 从
   from src.data.pocket_dataset import PocketDataset

   # 改为
   from src.data.pocket_dataset_v3 import PocketDatasetV3
   ```

4. **训练**:
   ```bash
   # 从
   python scripts/train.py --config configs/pocket_gvp_v2.yaml

   # 改为
   python scripts/train_v3.py --config configs/pocket_gvp_v2_nested.yaml
   ```

---

## 💡 推荐使用

### **使用PLC如果**:
- ✅ 需要处理大规模数据（>10K复合物）
- ✅ 想使用最新的ESM-3特征
- ✅ 需要完整的82维蛋白质特征
- ✅ 数据按年份组织

### **使用PoC如果**:
- ✅ 快速原型验证
- ✅ 小规模数据集（<1K）
- ✅ 简单的扁平文件结构

### **使用PLK如果**:
- ✅ 实验新的模型架构
- ✅ 研究距离预测方法
- ✅ 测试不同的损失函数

---

## 📝 关键改进（PoC → PLC）

1. **数据组织**: 扁平 → 嵌套年份结构
2. **ESM升级**: ESM-2 (1280) → ESM-3 (1536)
3. **特征丰富**: 35维 → 82维完整特征
4. **距离计算**: Cα → 全重原子
5. **标签管理**: 嵌入式 → 独立文件
6. **可扩展性**: 1K → 19K+ 复合物

---

**生成时间**: 2026-02-05
**对比版本**: PoC (原型) vs PLC (生产) vs PLK (实验)
