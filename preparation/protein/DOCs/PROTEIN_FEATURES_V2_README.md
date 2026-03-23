# 蛋白质特征提取 V2 - 完整文档

## 📊 特征概览

### 总维度：94维
- **物理化学特征**: 42维
- **表面特征**: 14维
- **结构几何特征**: 26维
- **增强特征V2**: 12维（新增）

---

## 🆕 新增特征 (12维)

### 1. 显式几何"口袋度"特征 (3维)

弥补GVP对"宏观凹陷"感知不够直接的缺点，通过统计几何学告诉模型"这里是一个坑"。

| 维度 | 特征名称 | 物理含义 | 计算方法 |
|------|---------|---------|---------|
| Dim 1 | CX (Convexity Index) | 凸度指数 | 10Å半径内邻居数 / 预期体积，口袋内残基CX接近1，凸起表面接近0 |
| Dim 2 | Local Packing Density | 局部堆积密度 | 6Å半径内的残基数量，归一化到[0,1] |
| Dim 3 | Shell Density Ratio | 壳层密度比 | (6Å内密度) / (12Å内密度)，深口袋高比率，浅表凹陷低比率 |

**实现位置**: `enhanced_features_v2.py::compute_pocketness_features()`

**关键代码**:
```python
tree = cKDTree(ca_coords)
neighbors_10 = tree.query_ball_point(ca_coords[i], R_10)
CX = min(N_10 / V_expected_10, 1.0)
```

---

### 2. 柔性与置信度特征 (3维)

结合位点通常位于"刚性支架"与"柔性Loop"的交界处（诱导契合）。

| 维度 | 特征名称 | 物理含义 | 计算方法 |
|------|---------|---------|---------|
| Dim 1 | Normalized B-factor | 归一化柔性 | Min-Max归一化B-factor到[0,1]，AlphaFold结构使用(100-pLDDT) |
| Dim 2 | Local Flexibility Variance | 局部柔性方差 | 当前残基与8个最近邻B-factor的标准差，高方差=软硬交界处 |
| Dim 3 | Relative Surface Exposure (RSE) | 相对表面暴露度 | (SASA/MaxSASA) * B-factor，过滤内部高B-factor和表面低B-factor |

**实现位置**: `enhanced_features_v2.py::compute_flexibility_features()`

**关键代码**:
```python
# 读取B-factor（支持AlphaFold的pLDDT）
structure = pdb_file.get_structure(model=1, extra_fields=["b_factor", "occupancy"])
if np.all((b_factors >= 0) & (b_factors <= 100)):
    b_factors = 100.0 - b_factors  # 反转pLDDT
```

---

### 3. 平滑化环境物理化学特征 (6维)

原有的42维PhysChem是"点特征"，结合位点的性质是"场特征"（例如：这一片区域都是疏水的）。

| 维度 | 特征名称 | 物理含义 | 计算方法 |
|------|---------|---------|---------|
| Dim 1 | Env Hydrophobicity | 环境疏水性 | 8Å范围内所有残基疏水性的高斯加权平均 |
| Dim 2 | Hydrophobic Contrast | 疏水对比度 | 环境疏水性 - 自身疏水性，极性残基在疏水环境中=口袋底部锚点 |
| Dim 3 | Env Positive Charge | 环境正电荷密度 | 8Å内正电荷残基(Arg, Lys, His)的加权计数 |
| Dim 4 | Env Negative Charge | 环境负电荷密度 | 8Å内负电荷残基(Asp, Glu)的加权计数 |
| Dim 5 | Env H-bond Donors | 环境氢键供体密度 | 8Å内可提供质子的原子总数 |
| Dim 6 | Env H-bond Acceptors | 环境氢键受体密度 | 8Å内可接受质子的原子总数 |

**实现位置**: `enhanced_features_v2.py::compute_environmental_features()`

**关键代码**:
```python
# 高斯核加权
sigma = 6.0  # Å
w_ij = np.exp(-distances**2 / (2 * sigma**2))
env_feature = np.sum(w_ij * neighbor_features) / np.sum(w_ij)
```

---

## 🔗 Per-Amino Acid 对齐机制

### 对齐信息字段

每个V2特征文件包含以下对齐信息：

```python
{
    'residue_index': [1, 2, 3, ...],      # 残基索引（PDB编号）
    'chain_id': ['A', 'A', 'A', ...],     # 链ID
    'residue_name_3': ['MET', 'LEU', ...], # 三字母残基名
    'residue_name_1': ['M', 'L', ...],     # 单字母残基名
    'sequence': 'MLEQPY...',               # 完整序列
}
```

### 与ESM3对齐

如果提供ESM3参考特征，脚本会自动进行对齐检查：

```python
python3 extract_complete_features_v2.py \
    --input protein.pdb \
    --output output_dir \
    --esm3_reference esm3_features.pt  # 可选
```

对齐逻辑：
1. 检查残基数量是否匹配
2. 检查序列是否一致
3. 按(chain_id, residue_index)对进行匹配
4. 如果不匹配，创建映射并重新排列

---

## 📁 输出格式

### 文件结构

```
protein_features_v2/
├── 1981-2000/
│   ├── 1jmi/
│   │   └── 1jmi_protein_v2.pt
│   └── ...
├── 2001-2010/
└── 2011-2019/
```

### 字段说明

```python
{
    # 主要特征
    'complete_features_v2': torch.Tensor,  # [N, 94] 完整特征

    # 分组特征（用于分析）
    'physicochemical_features': torch.Tensor,  # [N, 42]
    'surface_features': torch.Tensor,          # [N, 14]
    'structural_geometry_features': torch.Tensor,  # [N, 26]
    'pocketness_features': torch.Tensor,       # [N, 3] 新增
    'flexibility_features': torch.Tensor,      # [N, 3] 新增
    'environmental_features': torch.Tensor,    # [N, 6] 新增

    # 对齐信息
    'residue_index': List[int],
    'chain_id': List[str],
    'residue_name_3': List[str],
    'residue_name_1': List[str],
    'sequence': str,

    # 坐标信息
    'ca_coords': torch.Tensor,  # [N, 3]
    'all_atom_coords': List[np.ndarray],  # 每个残基的所有原子坐标

    # 元数据
    'n_residues': int,
    'mode': str,  # 'chain' or 'complex'
    'pdb_path': str,
    'feature_version': 'v2',
    'feature_dim': 94
}
```

---

## 🚀 使用方法

### 单个文件提取

```bash
python3 PLC/preparation/protein/extract_complete_features_v2.py \
    --input /path/to/protein.pdb \
    --output /path/to/output \
    --mode complex
```

### 批量提取

```bash
python3 PLC/preparation/protein/extract_complete_features_v2.py \
    --input /home/husrcf/Code/ProtBind/New \
    --output /home/husrcf/Code/ProtBind/PLC/data/protein_features_v2 \
    --mode complex \
    --workers 8 \
    --pattern "*_protein.pdb"
```

### 参数说明

- `--input`: 输入PDB文件或目录
- `--output`: 输出目录
- `--mode`: 'chain'（单链）或 'complex'（多链，默认）
- `--workers`: 并行进程数（默认4）
- `--pattern`: 文件匹配模式（默认"*_protein.pdb"）
- `--max`: 最大处理文件数（用于测试）
- `--overwrite`: 覆盖已存在的文件
- `--esm3_reference`: ESM3特征文件路径（可选，用于对齐）

---

## ✅ 验证结果

### 测试数据

- **测试文件**: 1jmi_protein.pdb
- **残基数**: 632
- **链**: A, B

### 特征维度验证

| 特征组 | 期望维度 | 实际维度 | 状态 |
|--------|---------|---------|------|
| complete_features_v2 | 94 | 94 | ✅ |
| physicochemical_features | 42 | 42 | ✅ |
| surface_features | 14 | 14 | ✅ |
| structural_geometry_features | 26 | 26 | ✅ |
| pocketness_features | 3 | 3 | ✅ |
| flexibility_features | 3 | 3 | ✅ |
| environmental_features | 6 | 6 | ✅ |

### 批量提取测试

- **测试数量**: 10个蛋白质
- **成功率**: 100% (10/10)
- **平均残基数**: 300-600
- **处理速度**: ~5-10秒/蛋白质

---

## 📝 注意事项

### 1. B-factor处理

- 标准PDB文件：直接使用B-factor列
- AlphaFold结构：自动检测pLDDT并反转为柔性指标
- 缺失B-factor：使用默认值50.0

### 2. 依赖库

必需：
- `biotite`: PDB解析
- `scipy`: KD-Tree空间索引
- `numpy`, `torch`: 数值计算

可选：
- `freesasa`: SASA计算（表面特征）
- `tqdm`: 进度条显示

### 3. 性能优化

- 使用KD-Tree加速邻域查询
- 高斯核加权避免硬切断
- 多进程并行处理

---

## 🔄 与原版本对比

| 项目 | V1 (原版) | V2 (增强版) |
|------|----------|------------|
| 总维度 | 82维 | 94维 (+12维) |
| 口袋识别 | 间接（通过GVP） | 显式3维特征 |
| 柔性信息 | 无 | 3维B-factor特征 |
| 环境特征 | 点特征 | 场特征（高斯平滑） |
| 对齐机制 | 基础 | 完整per-residue对齐 |
| ESM3兼容 | 无 | 支持对齐验证 |

---

## 📚 相关文件

- `enhanced_features_v2.py`: 新增12维特征实现
- `extract_complete_features_v2.py`: 主提取脚本
- `extract_physchem_from_pdb.py`: PDB解析
- `surface_features.py`: 表面特征
- `structural_geometry_features.py`: 结构几何特征
- `physicochemical_features.py`: 物理化学特征

---

## 🎯 下一步

1. **全量提取**: 对所有19,037个蛋白质进行V2特征提取
2. **质量检查**: 验证对齐准确性和特征分布
3. **模型训练**: 使用94维特征训练GVP模型
4. **消融实验**: 评估新增12维特征的贡献

---

生成时间: 2026-02-15
版本: V2.0
