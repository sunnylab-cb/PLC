# 蛋白质特征提取 V3 - 完整文档

## 概述

V3 特征提取是对 V2 的重大升级，引入了**空间矢量特征**，并重新组织了特征分类。

### 核心改进

1. **特征分类重组**：
   - 物理化学特征 (62维)：纯化学属性 → PhysChem MLP Encoder
   - 空间标量特征 (72维)：几何标量 → GVP node_s
   - 空间矢量特征 (6×3维)：方向矢量 → GVP node_v

2. **新增 6 个空间矢量**：
   - 侧链基础伸展矢量
   - 侧链质心矢量（药效团）
   - 芳香环π体系法向量
   - 氢键供体矢量
   - 氢键受体矢量
   - 空腔/溶剂逃逸矢量

3. **安全性增强**：
   - 安全归一化函数（避免 NaN）
   - 多链复合物质心计算
   - 零向量处理

---

## 特征详细说明

### 1. 物理化学特征（62 维）

**组成**：
- V2 Basic (0-19): 20 维 - 疏水性、分子量、极性、芳香性、电荷、pKa
- V2 Sequence (26-33): 8 维 - 氨基酸类型、局部二级结构倾向、相对位置
- V2 H-bond Network (34-41): 8 维 - 氢键供受体数量（标量）
- **新增**：残基类型 One-hot: 20 维
- **新增**：二级结构 One-hot: 3 维 (Helix/Sheet/Coil)
- **新增**：极性分类 One-hot: 3 维 (Nonpolar/Polar/Charged)

**用途**：进入 PhysChem MLP Encoder，与 ESM embeddings 融合

---

### 2. 空间标量特征（72 维）

**组成**：
- V2 SASA (20-25): 6 维 - 溶剂可及表面积
- V2 Surface (42-55): 14 维 - 曲率、凸度、粗糙度、密度、深度、静电势
- V2 Structural Geometry (56-81): 26 维 - DSSP 二级结构、曲率/挠率 RBF
- V2 Enhanced V2 (82-93): 12 维 - Pocketness、Flexibility、Environmental
- **新增**：局部密度统计: 4 维 (5Å/10Å/15Å 半径 + 密度梯度)
- **新增**：到质心距离: 1 维
- **新增**：表面法向量模长: 1 维
- **新增**：局部电荷密度: 4 维 (5Å/10Å 正/负电荷)
- **新增**：局部疏水性: 4 维 (5Å/10Å 疏水/亲水比例)

**用途**：进入 GVP 标量通道 (node_s)

---

### 3. 空间矢量特征（6 个矢量 × 3 维）

#### 矢量 0: Sidechain Base Vector
- **定义**：`normalize(Cβ - Cα)`
- **物理意义**：侧链从主链伸出的基本方向
- **特殊处理**：GLY 使用虚拟 Cβ（基于 N-Cα-C 角平分线）

#### 矢量 1: Sidechain CoM Vector
- **定义**：`normalize(CoM_sidechain - Cα)`
- **物理意义**：指向侧链"药效团"的真实中心
- **优势**：对长侧链（Arg, Lys, Trp）更准确

#### 矢量 2: Aromatic Normal Vector
- **定义**：芳香环平面的法向量（PCA 计算）
- **适用残基**：PHE, TYR, TRP, HIS
- **物理意义**：π-π 堆叠和阳离子-π 相互作用的关键几何信息

#### 矢量 3: H-bond Donor Vector
- **定义**：氢键供体方向（简化版：使用侧链质心方向）
- **适用残基**：SER, THR, TYR, CYS, ASN, GLN, ARG, LYS, HIS, TRP
- **物理意义**：氢键供体的空间取向

#### 矢量 4: H-bond Acceptor Vector
- **定义**：氢键受体方向（简化版：侧链质心反方向 + 主链 C=O）
- **适用残基**：ASP, GLU, SER, THR, TYR, ASN, GLN, HIS, 主链 C=O
- **物理意义**：氢键受体的空间取向

#### 矢量 5: Cavity Pointing Vector
- **定义**：`normalize(Cα - CoM_neighbors)`
- **物理意义**：指向蛋白质表面或口袋的方向
- **用途**：判断残基是否暴露在结合位点

**用途**：进入 GVP 矢量通道 (node_v)

---

## 使用方法

### 1. 单个 PDB 文件提取

```bash
python preparation/protein/extract_complete_features_v3.py \
    --pdb_dir /path/to/pdb/files \
    --output_dir data/complete_features_v3 \
    --mode complex \
    --num_workers 1
```

### 2. 批量提取（多进程）

```bash
python preparation/protein/extract_complete_features_v3.py \
    --pdb_dir /home/husrcf/Code/ProtBind/New \
    --output_dir data/complete_features_v3 \
    --mode complex \
    --num_workers 16 \
    --pdb_list configs/splits/mmseqs_splits_strict.csv
```

### 3. 在 Python 中使用

```python
import torch
from pathlib import Path

# 加载特征
features = torch.load('data/complete_features_v3/1a0q_complete_v3.pt')

# 访问特征
physchem = features['physchem_features']  # [N, 62]
spatial_scalar = features['spatial_scalar_features']  # [N, 72]
spatial_vector = features['spatial_vector_features']  # [N, 6, 3]

# 元数据
residue_names = features['residue_names']
residue_numbers = features['residue_numbers']
chain_ids = features['chain_ids']
n_residues = features['n_residues']
```

---

## 与模型的集成

### V8 模型输入

```python
# PhysChem MLP Encoder
physchem_encoded = physchem_mlp(physchem_features)  # [N, 62] -> [N, 128]

# GVP Encoder
gvp_output = gvp_encoder(
    node_s=spatial_scalar_features,   # [N, 72]
    node_v=spatial_vector_features,   # [N, 6, 3]
    edge_index=edge_index,
    edge_s=edge_s,
    edge_v=edge_v  # 注意：edge_v 需要单独计算
)

# Fusion
fused_features = fusion_layer(physchem_encoded, gvp_output)
```

### 边矢量计算（避坑指南 #5）

```python
# 构建 KNN 图
edge_index = build_knn_graph(ca_coords, k=20)

# 计算边矢量
src, dst = edge_index
edge_v = safe_normalize(ca_coords[dst] - ca_coords[src])  # [E, 3]
```

---

## 避坑指南

### 1. NaN 爆炸（矢量归一化问题）
**问题**：零向量归一化导致 NaN
**解决**：使用 `safe_normalize()` 函数，对零向量返回零向量

```python
def safe_normalize(v, eps=1e-8):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    mask = (norm > eps).astype(np.float32)
    return np.where(mask, v / (norm + eps), 0.0)
```

### 2. 多链复合物的质心计算
**问题**：只使用单链计算质心，导致全局位置特征错误
**解决**：确保使用所有链的 Cα 原子计算全局质心

```python
# 正确：使用所有链
protein_center = all_ca_coords.mean(axis=0)

# 错误：只使用当前链
protein_center = current_chain_ca_coords.mean(axis=0)
```

### 3. 特征冗余微调
**问题**：二级结构 One-hot 与 DSSP 特征冗余
**解决**：可接受的冗余，两者分发到不同网络（PhysChem vs GVP）

### 4. 氢键矢量的近似代价
**问题**：使用侧链质心方向近似氢键方向，对长侧链不够精确
**解决**：Phase 1 使用简化版本，Phase 2 可以提取具体极性原子坐标

### 5. GVP 的边矢量
**问题**：忘记计算 edge_v
**解决**：在构图时计算边矢量 `edge_v = normalize(coord_j - coord_i)`

---

## 测试

运行测试脚本验证特征提取：

```bash
python preparation/protein/test_v3_features.py
```

测试内容：
- ✅ 安全归一化函数
- ✅ 特征维度正确
- ✅ 无 NaN 值
- ✅ 矢量归一化正确
- ✅ 特殊残基（GLY）处理正确

---

## 性能

- **单个蛋白质**：~2-5 秒（取决于大小）
- **批量处理**：~1000 个蛋白质/小时（16 进程）
- **存储大小**：~50-200 KB/蛋白质

---

## 下一步

1. **Phase 1**：测试 V3 特征提取，验证无 NaN
2. **Phase 2**：修改 Dataset 类加载 V3 特征
3. **Phase 3**：创建 V8 模型，集成 V3 特征
4. **Phase 4**：训练 V8 模型，评估性能提升

---

## 参考

- V2 特征文档：`DOCs/V7M3_FEATURE_USAGE_CN.md`
- GVP 论文：Jing et al., "Learning from Protein Structure with Geometric Vector Perceptrons"
- 避坑指南：用户提供的 5 个核心避坑指南
