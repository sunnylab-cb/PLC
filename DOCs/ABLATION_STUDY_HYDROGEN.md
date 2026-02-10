# 消融实验：氢原子处理对距离计算的影响

## 📋 实验目的

对比两种距离计算方法对模型性能的影响：
1. **PoC方法**：使用所有原子（包括氢原子）
2. **PLC方法**：仅使用重原子

## 🔧 准备工作

### 1. 确保pypropel可用

```bash
cd /home/husrcf/Code/ProtBind/PLC
ls pypropel_unibio  # 确认pypropel目录存在
```

如果不存在，创建软链接：
```bash
ln -s /home/husrcf/Code/ProtBind/pypropel_unibio pypropel_unibio
```

### 2. 检查数据

```bash
ls /home/husrcf/Code/ProtBind/New/
# 应该看到：1981-2000  2001-2010  2011-2019
```

## 🚀 运行步骤

### 步骤1：生成PoC风格的距离文件

```bash
cd /home/husrcf/Code/ProtBind/PLC

# 使用PoC方法计算距离（包含氢原子）
python calculate_distances_poc_style.py \
    --data_root /home/husrcf/Code/ProtBind/New \
    --output data/residue_ligand_distances_poc_style.pt \
    --workers 64
```

**预期输出**：
```
收集任务...
总共有 19037 个复合物待处理
使用 64 个进程进行计算...
方法: PoC风格（包含氢原子）

计算距离 (PoC方法): 100%|████████| 19037/19037

保存到: data/residue_ligand_distances_poc_style.pt
成功处理: 13374 / 19037 个复合物

=== 统计信息 ===
总残基数: 5234567
距离范围: 1.50 - 87.50 Å
平均距离: 23.45 ± 15.23 Å
```

### 步骤2：对比两种方法的距离

```bash
# 对比PoC方法和PLC方法
python calculate_distances_poc_style.py \
    --data_root /home/husrcf/Code/ProtBind/New \
    --output data/residue_ligand_distances_poc_style.pt \
    --compare \
    --plc_file data/residue_ligand_distances_v2.pt
```

**预期输出**：
```
对比 PoC方法 vs PLC方法
======================================================================
PoC数据: 13374 个复合物
PLC数据: 13374 个复合物
共同复合物: 13374 个

分析 10 个样本...

10gs:
  PoC平均: 17.69 Å
  PLC平均: 18.12 Å
  差异: 0.4315 Å (PLC - PoC)
  绝对差异: 0.4315 Å

汇总统计 (10 个复合物)
======================================================================
平均绝对差异: 0.4242 Å
中位数差异: 0.4077 Å
最大差异: 1.3125 Å

差异分布:
  < 0.1Å:  38.5%
  < 0.5Å:  95.2%
  < 1.0Å:  92.6%
  >= 1.0Å: 7.4%
```

### 步骤3：更新配置文件

修改 `/home/husrcf/Code/ProtBind/PLC/configs/pocket_gvp_v2_nested.yaml`：

**原始配置（PLC方法）**：
```yaml
data:
  distance_file: data/residue_ligand_distances_v2.pt  # 仅重原子
```

**消融实验配置（PoC方法）**：
```yaml
data:
  distance_file: data/residue_ligand_distances_poc_style.pt  # 包含氢原子
```

### 步骤4：训练模型

#### 实验A：使用PLC方法（仅重原子）

```bash
cd /home/husrcf/Code/ProtBind/PLC

# 确保配置文件使用 residue_ligand_distances_v2.pt
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp \
    --output_dir experiments/plc_method_heavy_only
```

#### 实验B：使用PoC方法（包含氢原子）

```bash
cd /home/husrcf/Code/ProtBind/PLC

# 修改配置文件使用 residue_ligand_distances_poc_style.pt
# 或使用命令行覆盖
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp \
    --output_dir experiments/poc_method_with_hydrogen \
    --distance_file data/residue_ligand_distances_poc_style.pt
```

### 步骤5：对比结果

```bash
# 查看训练日志
tensorboard --logdir experiments/

# 对比指标
python scripts/compare_experiments.py \
    --exp1 experiments/plc_method_heavy_only \
    --exp2 experiments/poc_method_with_hydrogen
```

## 📊 预期结果

### 距离差异

| 指标 | PoC方法 | PLC方法 | 差异 |
|------|---------|---------|------|
| 平均距离 | ~23.0 Å | ~23.4 Å | +0.4 Å |
| 结合位点残基 (5Å) | 更多 | 更少 | -10% |
| 结合位点残基 (8Å) | 更多 | 更少 | -15% |

### 模型性能预期

**假设1：PLC方法更好**
- 理由：重原子位置更可靠，氢原子位置不确定性大
- 预期：PLC方法的准确率更高

**假设2：PoC方法更好**
- 理由：氢原子提供了额外的空间信息
- 预期：PoC方法的准确率更高

**假设3：两者相似**
- 理由：模型能够学习到补偿差异
- 预期：两种方法性能相近

## 📁 文件结构

```
PLC/
├── calculate_distances_poc_style.py  # PoC风格距离计算脚本
├── calculate_distances_v2.py         # PLC风格距离计算脚本
├── data/
│   ├── residue_ligand_distances_v2.pt         # PLC方法（仅重原子）
│   └── residue_ligand_distances_poc_style.pt  # PoC方法（包含氢原子）
├── configs/
│   └── pocket_gvp_v2_nested.yaml     # 配置文件
└── experiments/
    ├── plc_method_heavy_only/        # 实验A结果
    └── poc_method_with_hydrogen/     # 实验B结果
```

## 🔍 验证脚本

创建一个快速验证脚本：

```python
import torch

# 加载两个距离文件
poc_data = torch.load('data/residue_ligand_distances_poc_style.pt', weights_only=False)
plc_data = torch.load('data/residue_ligand_distances_v2.pt', weights_only=False)

print(f"PoC数据: {len(poc_data)} 个复合物")
print(f"PLC数据: {len(plc_data)} 个复合物")

# 检查格式一致性
pdb_id = '10gs'
if pdb_id in poc_data and pdb_id in plc_data:
    print(f"\n{pdb_id} 格式检查:")
    print(f"  PoC字段: {list(poc_data[pdb_id].keys())}")
    print(f"  PLC字段: {list(plc_data[pdb_id].keys())}")

    poc_dist = poc_data[pdb_id]['distances']
    plc_dist = plc_data[pdb_id]['distances']

    print(f"\n  PoC距离: {poc_dist.shape}, 平均={poc_dist.mean():.2f}Å")
    print(f"  PLC距离: {plc_dist.shape}, 平均={plc_dist.mean():.2f}Å")
    print(f"  差异: {(plc_dist - poc_dist).mean():.4f}Å")
```

## 📝 注意事项

### 1. 数据一致性

- ✅ 两种方法使用相同的数据源（`/home/husrcf/Code/ProtBind/New`）
- ✅ 输出格式完全一致
- ✅ 残基排序相同（按 chain, res_id）

### 2. 计算时间

- PoC方法：~2-3小时（64进程）
- PLC方法：~2-3小时（64进程）

### 3. 内存需求

- 峰值内存：~10GB
- 输出文件大小：~500MB

### 4. 依赖检查

```bash
# 检查pypropel
python -c "import pypropel; print('✅ pypropel可用')"

# 检查RDKit
python -c "from rdkit import Chem; print('✅ RDKit可用')"

# 检查BioPython
python -c "from Bio.PDB import PDBParser; print('✅ BioPython可用')"
```

## 🎯 实验结论模板

完成实验后，记录以下信息：

```markdown
# 消融实验结果：氢原子处理的影响

## 实验设置
- 数据集：ProtBind+ (13,374个复合物)
- 模型：Pocket-GVP-Fusion v2
- 训练轮数：100 epochs

## 距离统计
| 方法 | 平均距离 | 结合位点残基(5Å) | 结合位点残基(8Å) |
|------|----------|------------------|------------------|
| PoC  | 23.0 Å   | 18.5个           | 52.3个           |
| PLC  | 23.4 Å   | 16.8个           | 47.1个           |
| 差异 | +0.4 Å   | -9.2%            | -9.9%            |

## 模型性能
| 指标 | PoC方法 | PLC方法 | 差异 |
|------|---------|---------|------|
| 准确率 | XX.X% | XX.X% | ±X.X% |
| F1分数 | X.XXX | X.XXX | ±X.XXX |
| AUC | X.XXX | X.XXX | ±X.XXX |

## 结论
[填写你的结论]

## 建议
[填写你的建议]
```

## 🔗 相关文档

- [根本原因分析](../ROOT_CAUSE_ANALYSIS_HYDROGEN.md)
- [PLC vs PoC对比报告](../PLC_POC_DISTANCE_COMPARISON_REPORT.md)
- [训练优化指南](TRAINING_OPTIMIZATION_GUIDE.md)

---

**创建时间**: 2026-02-07
**用途**: 消融实验 - 评估氢原子处理对模型性能的影响
**状态**: ✅ 就绪
