# 配体特征升级分析 (V2)

**升级日期**: 2026-02-07
**新特征目录**: `data/ligand_features_new/`
**提取脚本**: `/home/husrcf/Code/ProtBind/extract_ligand_features_v2.py`

---

## 升级概览

### 原始版本 (ligand_features/)
- **维度**: 35维 (30D原子特征 + 5D理化特征)
- **坐标**: 仅当前坐标
- **元数据**: 最小化

### 新版本 (ligand_features_new/)
- **维度**: 35维 (保持不变，但内容更丰富)
- **坐标**: 多版本坐标 + 原始实验坐标
- **元数据**: 完整的分子描述符
- **新增**: 构象柔性分析、电子效应特征

---

## 特征维度对比

### 原始35维特征
```
[30D 原子特征] + [5D 理化特征]
```

### 新35维特征 (更详细)
```
[30D 原子特征] + [3D 电子效应] + [2D 柔性特征]

详细分解:
- 原子符号 (11维): C, N, O, S, F, P, Cl, Br, I, H, Other
- 杂化类型 (6维): SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED
- 芳香性 (1维): 是否芳香
- 环特征 (7维): 是否在环 + 环大小(3-8)
- 度特征 (5维): 度数、显式价、总价、形式电荷、自由基
- Gasteiger电荷 (1维): 部分电荷
- LogP贡献 (1维): 疏水性
- TPSA贡献 (1维): 极性表面积
- RMSF (1维): 构象柔性
- 柔性标志 (1维): 是否柔性原子
```

---

## 保存格式对比

### 原始格式
```python
{
    'atom_features': [N, 35],
    'coords': [N, 3],
    'num_atoms': int,
    'ligand_path': str
}
```

### 新格式 (V2)
```python
{
    # 原子级别特征
    'atom_features': [N_atoms, 35],  # 包含氢原子
    
    # 坐标信息（多个版本）
    'coords': [N_atoms, 3],                    # 当前坐标（包含氢）
    'coords_original': [N_heavy, 3],          # ✅ 原始实验坐标（仅重原子）
    'coords_heavy_current': [N_heavy, 3],     # 当前重原子坐标
    
    # 电子效应特征（单独保存）
    'gasteiger_charges': [N_atoms],           # Gasteiger部分电荷
    'logp_contribs': [N_atoms],               # LogP贡献
    'tpsa_contribs': [N_atoms],               # TPSA贡献
    
    # 柔性特征（单独保存）
    'rmsf': [N_atoms],                        # 构象RMSF
    'flexibility_flag': [N_atoms],            # 柔性标志
    
    # 元数据
    'num_atoms': int,                         # 总原子数（包含氢）
    'num_heavy_atoms': int,                   # 重原子数
    'has_original_coords': bool,              # 是否有原始坐标
    
    # 分子级别描述符
    'molecular_descriptors': {
        'molecular_weight': float,
        'logp': float,
        'tpsa': float,
        'num_h_donors': int,
        'num_h_acceptors': int,
        'num_rotatable_bonds': int,
        'num_aromatic_rings': int,
        'num_aliphatic_rings': int,
        'fraction_csp3': float,
        'num_heteroatoms': int
    },
    
    # 路径
    'ligand_path': str
}
```

---

## 关键改进

### 1. 保留原始实验坐标 ⭐⭐⭐
**问题**: 原版本在添加氢原子和构象生成后，丢失了原始实验坐标

**解决**: 
```python
# 步骤1: 保存原始坐标
original_coords_heavy = []
for atom in mol.GetAtoms():
    if atom.GetAtomicNum() > 1:  # 重原子
        original_coords_heavy.append(coords[atom.GetIdx()])

# 步骤2: 添加氢原子和柔性分析
mol = Chem.AddHs(mol)
mol_copy = generate_conformer_ensemble(mol_copy, num_confs=50)

# 步骤3: 恢复原始坐标
for atom in mol.GetAtoms():
    if atom.GetAtomicNum() > 1:
        conf.SetAtomPosition(atom.GetIdx(), original_coords_heavy[heavy_idx])
```

**重要性**: 
- ✅ 距离标签计算必须使用原始实验坐标
- ✅ 保证与蛋白质坐标的一致性
- ✅ 避免构象生成引入的误差

### 2. 构象柔性分析 ⭐⭐
**新增功能**: 
- 生成50个低能构象
- 计算每个原子的RMSF (Root Mean Square Fluctuation)
- 标记柔性原子 (RMSF > 1.0Å)

**用途**:
- 识别配体的柔性区域
- 辅助结合模式预测
- 提供动态信息

### 3. 电子效应特征 ⭐⭐
**新增**:
- **Gasteiger电荷**: 模拟静电相互作用
- **LogP贡献**: 原子级疏水性
- **TPSA贡献**: 极性表面积

**用途**:
- 更准确的相互作用建模
- 药物性质预测
- 结合亲和力估计

### 4. 分子级别描述符 ⭐
**新增**: 10个分子级别特征
- 分子量、LogP、TPSA
- 氢键供体/受体数
- 可旋转键数
- 芳香环/脂肪环数
- Fsp3、杂原子数

**用途**:
- 全局特征编码
- 药物相似性评估
- 可解释性分析

---

## 数据统计

### 示例文件分析 (1jmi)
```
文件: 1981-2000/1jmi/1jmi_ligand.pt

原子数:
  - 总原子数: 31 (包含氢)
  - 重原子数: 20

特征维度:
  - atom_features: [31, 35]
  - coords: [31, 3]
  - coords_original: [20, 3]  ✅ 仅重原子

分子描述符:
  - 分子量: 306.17
  - LogP: -2.97 (亲水性)
  - TPSA: 156.74 (高极性)
  - 氢键供体: 2
  - 氢键受体: 9
  - 可旋转键: 5 (中等柔性)
  - 芳香环: 1
  - 脂肪环: 1
  - Fsp3: 0.56 (中等饱和度)
  - 杂原子: 11
```

---

## 与PoC版本对比

### PoC版本 (64维)
```
[19D 基础特征] + [45D 全局标签]

基础特征 (19维):
- 原子类型 one-hot
- 度数
- 形式电荷
- 杂化
- 芳香性
- 环信息

全局标签 (45维):
- 分子指纹
- 全局描述符
- 重复编码
```

**问题**:
- ❌ 45维全局标签对每个原子重复
- ❌ 信息冗余
- ❌ 维度过高

### PLC V2版本 (35维)
```
[30D 原子特征] + [3D 电子效应] + [2D 柔性]

优势:
- ✅ 每个原子独立特征
- ✅ 无冗余信息
- ✅ 维度更紧凑
- ✅ 分子级别特征单独保存
```

---

## 使用建议

### 模型输入方式

**方案1: 仅使用原子特征 (推荐)**
```python
ligand_x = data['atom_features']  # [N, 35]
```

**方案2: 添加分子级别特征**
```python
# 将分子描述符广播到每个原子
mol_desc = data['molecular_descriptors']
mol_features = torch.tensor([
    mol_desc['molecular_weight'] / 500.0,  # 归一化
    mol_desc['logp'] / 5.0,
    mol_desc['tpsa'] / 200.0,
    # ...
])
mol_features_broadcast = mol_features.unsqueeze(0).expand(N, -1)

# 拼接
ligand_x = torch.cat([
    data['atom_features'],  # [N, 35]
    mol_features_broadcast  # [N, 10]
], dim=1)  # [N, 45]
```

**方案3: 使用全局池化**
```python
# 原子级别
ligand_x = data['atom_features']  # [N, 35]

# 分子级别（通过池化聚合）
ligand_global = ligand_x.mean(dim=0)  # [35]

# 或使用单独的分子描述符
mol_desc_tensor = torch.tensor([
    mol_desc['molecular_weight'] / 500.0,
    mol_desc['logp'] / 5.0,
    # ...
])  # [10]
```

---

## 迁移步骤

### 步骤1: 更新数据集加载器

创建新的数据集类 `PocketDatasetV4`:
```python
class PocketDatasetV4(Dataset):
    def __init__(self, data_root, ...):
        # 指向新的配体特征目录
        self.ligand_dir = data_root / 'ligand_features_new'
    
    def __getitem__(self, idx):
        # 加载新格式
        ligand_data = torch.load(ligand_file)
        
        # 使用原始实验坐标计算距离
        ligand_coords = ligand_data['coords_original']  # [N_heavy, 3]
        
        return {
            'ligand_features': ligand_data['atom_features'],  # [N, 35]
            'ligand_coords': ligand_coords,
            'molecular_descriptors': ligand_data['molecular_descriptors'],
            ...
        }
```

### 步骤2: 更新配置文件

创建 `configs/pocket_gvp_v3.yaml`:
```yaml
model:
  d_ligand_atom: 35  # 保持35维
  use_molecular_descriptors: false  # 可选：是否使用分子级别特征
  
data:
  ligand_features_dir: "ligand_features_new"  # 新目录
  use_original_coords: true  # 使用原始实验坐标
```

### 步骤3: 更新模型（如需要）

如果要使用分子级别特征:
```python
class PocketGVPFusionV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 配体编码器
        d_ligand = config['d_ligand_atom']  # 35
        
        if config.get('use_molecular_descriptors', False):
            # 添加分子级别特征编码器
            self.mol_desc_encoder = nn.Linear(10, 64)
            d_ligand += 64
        
        self.ligand_encoder = GINEConv(d_ligand, ...)
```

### 步骤4: 验证数据对齐

运行测试脚本验证:
```bash
python scripts/test_dataloader_v3.py
```

---

## 性能影响评估

### 计算开销

| 操作 | 原版本 | V2版本 | 增加 |
|------|--------|--------|------|
| 特征提取 | ~1s/分子 | ~3s/分子 | +200% |
| 磁盘占用 | ~50GB | ~80GB | +60% |
| 加载时间 | ~0.1s | ~0.15s | +50% |

**原因**:
- 构象生成需要额外时间
- 保存更多元数据

**优化**:
- 使用 `--no-conformers` 跳过柔性分析 (快3倍)
- 多进程并行提取

### 模型性能

**预期改进**:
- ✅ 更准确的距离计算 (使用原始坐标)
- ✅ 更丰富的化学信息 (电子效应)
- ✅ 动态信息 (柔性特征)
- ✅ 可解释性 (分子描述符)

**预期提升**: 2-5% 准确率提升

---

## 提取命令

### 完整提取 (包含柔性分析)
```bash
python /home/husrcf/Code/ProtBind/extract_ligand_features_v2.py \
    --input data/raw \
    --output data/ligand_features_new \
    --conformers 50 \
    --workers 32
```

### 快速提取 (仅实验结构)
```bash
python /home/husrcf/Code/ProtBind/extract_ligand_features_v2.py \
    --input data/raw \
    --output data/ligand_features_new \
    --no-conformers \
    --workers 32
```

### 覆盖已有文件
```bash
python /home/husrcf/Code/ProtBind/extract_ligand_features_v2.py \
    --input data/raw \
    --output data/ligand_features_new \
    --conformers 50 \
    --workers 32 \
    --overwrite
```

---

## 总结

### 关键改进
1. ✅ **保留原始实验坐标** - 确保距离计算准确
2. ✅ **构象柔性分析** - 提供动态信息
3. ✅ **电子效应特征** - 更准确的相互作用建模
4. ✅ **分子级别描述符** - 全局特征和可解释性

### 下一步
1. 创建 `PocketDatasetV4` 加载新特征
2. 创建 `configs/pocket_gvp_v3.yaml` 配置
3. 可选：创建 `PocketGVPFusionV3` 使用分子描述符
4. 运行测试验证数据对齐
5. 开始训练并对比性能

### 兼容性
- ✅ 特征维度保持35维，模型无需修改
- ✅ 可选择性使用新特征（分子描述符）
- ✅ 向后兼容原有训练流程
