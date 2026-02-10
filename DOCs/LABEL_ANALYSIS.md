# 标签计算与合理性分析

## 📊 当前标签定义

### **3类距离标签**

```python
# src/data/pocket_dataset_v3.py:278-283
labels[distances < 3.5] = 0      # Contact (接触)
labels[(distances >= 3.5) & (distances < 6.0)] = 1  # Near (邻近)
labels[distances >= 6.0] = 2     # Far (远离)
```

| 类别 | 阈值 | 生物学意义 |
|------|------|-----------|
| **Contact (0)** | < 3.5Å | 直接结合位点，范德华接触 |
| **Near (1)** | 3.5-6.0Å | 第一壳层，可能的相互作用区域 |
| **Far (2)** | > 6.0Å | 非结合区域 |

---

## 📈 实际数据分布

### **统计数据** (10,451,713 个残基)

```
最小距离: 0.04 Å
最大距离: 1185.61 Å
平均距离: 63.29 Å
中位数:   47.92 Å
```

### **标签分布**

| 类别 | 数量 | 百分比 | 类别权重 |
|------|------|--------|---------|
| **Contact** | 80,738 | 0.77% | 5.0 |
| **Near** | 90,573 | 0.87% | 3.0 |
| **Far** | 10,280,402 | 98.36% | 2.0 |

**类别不平衡比例**: 1 : 1.1 : 127

---

## ⚠️ 问题分析

### **1. 极度不平衡**

**问题**：
- Far类占98.36%，Contact和Near合计仅1.64%
- 不平衡比例达到 **1:127**
- 模型容易退化为"全预测Far"

**影响**：
- 即使模型全预测Far，准确率也有98.36%
- Contact类（最重要）样本极少，难以学习
- 梯度主要来自Far类，Contact类信号被淹没

---

### **2. 阈值合理性评估**

#### **3.5Å 阈值（Contact边界）**

✅ **合理**：
- 范德华半径和：~3.0-4.0Å
- 氢键距离：2.5-3.5Å
- 符合结构生物学定义

❌ **问题**：
- 样本量太少（0.77%）
- 可能遗漏一些弱相互作用

#### **6.0Å 阈值（Near边界）**

⚠️ **需要讨论**：
- 第一壳层通常到5-6Å
- 但Near类样本也很少（0.87%）
- 6.0-10.0Å区域可能也有生物学意义

---

## 💡 改进建议

### **方案1：调整阈值（推荐）**

```python
# 更平衡的阈值
labels[distances < 4.0] = 0      # Contact: 扩大到4.0Å
labels[(distances >= 4.0) & (distances < 8.0)] = 1  # Near: 扩大到8.0Å
labels[distances >= 8.0] = 2     # Far: 从8.0Å开始
```

**优势**：
- 增加Contact和Near样本量
- 8.0Å内通常是功能相关区域
- 更符合药物设计中的"结合口袋"定义

**预期分布**（需验证）：
- Contact: ~2-3%
- Near: ~3-5%
- Far: ~92-95%

---

### **方案2：使用百分位数阈值**

```python
# 基于数据分布的阈值
# 10%分位: 17.28Å
# 5%分位: ~12Å (估计)
# 2%分位: ~8Å (估计)

labels[distances < 4.0] = 0      # Contact: 直接接触
labels[(distances >= 4.0) & (distances < 12.0)] = 1  # Near: 前5%
labels[distances >= 12.0] = 2    # Far: 其余95%
```

**优势**：
- 保证一定的样本量
- 适应数据分布

**劣势**：
- 失去生物学解释性
- 12Å可能太远，无生物学意义

---

### **方案3：增加类别权重（当前方案）**

```yaml
# configs/pocket_gvp_v2_nested.yaml
class_weights: [5.0, 3.0, 2.0]  # Contact, Near, Far
```

**当前权重分析**：
- Contact权重5.0，但样本量0.77% → 有效权重 = 5.0 × 0.0077 = 0.0385
- Far权重2.0，样本量98.36% → 有效权重 = 2.0 × 0.9836 = 1.967

**问题**：权重不够大！

**建议调整**：
```yaml
class_weights: [100.0, 50.0, 1.0]  # 更激进的权重
# 或使用逆频率权重
class_weights: [127, 114, 1]  # 基于样本比例
```

---

### **方案4：Focal Loss + 高gamma**

```yaml
focal_gamma: 5.0  # 当前2.0，增加到5.0
```

**Focal Loss公式**：
```
FL = -(1 - p_t)^gamma * log(p_t)
```

**效果**：
- gamma越大，对难分类样本的关注越高
- 自动降低简单样本（Far类）的权重
- 增强对Contact类的学习

---

## 🔬 距离计算方法

### **当前实现** (`calculate_distances_v2.py`)

```python
# 1. 使用所有重原子（不仅Cα）
residue_coords = [atom.coord for atom in residue if atom.element != 'H']

# 2. 计算到配体所有原子的距离矩阵
diff = residue_coords[:, None, :] - ligand_coords[None, :, :]
dists = norm(diff, axis=2)  # [N_res_atoms, N_lig_atoms]

# 3. 取最小距离
min_distance = dists.min()
```

**合理性**：✅
- 使用全重原子更准确（vs 仅Cα）
- 最小距离代表最近接触点
- 符合结构生物学标准

---

## 📋 推荐方案

### **短期方案（立即可用）**

1. **增加类别权重**：
   ```yaml
   class_weights: [100.0, 50.0, 1.0]
   ```

2. **增加Focal Loss gamma**：
   ```yaml
   focal_gamma: 5.0
   ```

3. **监控指标**：
   - 不仅看总体准确率
   - 重点看Contact类的Recall和F1
   - 使用DICE/IoU评估Contact区域

---

### **中期方案（需要重新生成标签）**

**调整阈值为**：
```python
labels[distances < 4.0] = 0      # Contact
labels[(distances >= 4.0) & (distances < 8.0)] = 1  # Near
labels[distances >= 8.0] = 2     # Far
```

**步骤**：
1. 修改 `src/data/pocket_dataset_v3.py:281-283`
2. 重新运行训练（无需重新计算距离）
3. 对比两种阈值的性能

---

### **长期方案（研究方向）**

1. **多任务学习**：
   - 主任务：3类分类
   - 辅助任务：距离回归
   - 共享表示，互相增强

2. **层次化标签**：
   ```
   Level 1: Binding (< 8Å) vs Non-binding (≥ 8Å)
   Level 2: Contact (< 4Å) vs Near (4-8Å)
   ```

3. **软标签**：
   - 使用距离的连续值
   - 高斯核平滑
   - 避免硬边界问题

---

## 🎯 评估指标建议

**不要只看准确率！**

### **推荐指标**

1. **Contact类专项指标**：
   - Precision@Contact
   - Recall@Contact
   - F1@Contact
   - DICE@Contact
   - IoU@Contact

2. **平衡指标**：
   - Macro-averaged F1（三类平均）
   - Matthews Correlation Coefficient (MCC)

3. **实用指标**：
   - Top-K准确率（预测概率最高的K个残基中有多少是真Contact）
   - AUPRC（Precision-Recall曲线下面积）

---

## 📝 总结

### **当前标签定义**
- ✅ 生物学上合理（3.5Å, 6.0Å）
- ❌ 数据分布极度不平衡（98.36% Far）
- ⚠️ 需要特殊处理

### **立即行动**
1. 增加类别权重到 `[100.0, 50.0, 1.0]`
2. 增加focal_gamma到 `5.0`
3. 监控Contact类的Recall/F1，不只看准确率

### **可选改进**
- 考虑调整阈值到 `[4.0Å, 8.0Å]`
- 添加距离回归辅助任务
- 使用更多评估指标

---

**生成时间**: 2026-02-05
**数据集**: 19,024 complexes, 10,451,713 residues
