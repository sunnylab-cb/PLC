# Pocket-GVP-Fusion v6m5 模型与训练摘要（报告速览版）

本文档面向“快速审阅 + 报告附录”，基于当前代码实现总结 v6m5 的：
- 完整模型架构
- 训练脚本与损失函数设计
- 标签体系与评估口径

---

## 1. 代码范围（Source of Truth）

- 模型定义：`src/models/pocket_gvp_fusion_v6m5.py`
- 训练脚本：`scripts/train_v6m5.py`
- 数据与标签：`src/data/pocket_dataset_v6.py`
- 注意力/Transformer组件：`src/models/transformer_layers.py`
- 当前配置：`configs/pocket_gvp_v6.yaml`

---

## 2. 一句话结论

**v6m5 = v6 双向交叉注意力主干 + 物化特征 MLP 编码（LayerNorm + GELU）+ Dice/Focal 组合损失**，核心目标是提升类别不平衡下的分割表现与训练稳定性。

---

## 3. 输入与标签定义（训练实际口径）

### 3.1 输入特征（按当前配置）

- 蛋白 ESM：`[B, L, 1280]`
- 蛋白标量特征（complete features）：`[B, L, 82]`
- 蛋白向量特征：`[B, L, 4, 3]`
- 蛋白坐标：`[B, L, 3]`
- 配体原子特征：`[N_atoms_total, 80]`
- 配体边特征：`edge_dim=6`

> 说明：模型默认注释里常见 “35D” 是旧描述；实际运行维度由配置 `d_phys=82` 覆盖。

### 3.2 标签体系

- **训练主标签（VDW）**：来自 `distances_vdw_adjusted`，划分为
  - `c0`: `< 0.5`
  - `c1`: `0.5 ~ 4.5`
  - `c2`: `>= 4.5`
- **对比标签（Original）**：固定阈值
  - `c0`: `< 3.5Å`
  - `c1`: `3.5Å ~ 6.0Å`
  - `c2`: `>= 6.0Å`

训练与 best model 选择都以 **VDW 主标签**为准；Original 仅用于验证对比。

---

## 4. 完整模型架构（v6m5）

## 4.1 总体流水线

1. **PhysChem Encoder（新增强化）**
   - `protein_scalar: [B,L,82]`
   - MLP：`82 -> 64 -> 64`
   - 每层后：`LayerNorm + GELU`
   - 输出：`protein_scalar_encoded: [B,L,64]`

2. **Protein Geometric Encoder（GVP）**
   - 输入：标量 `82` + 向量 `4x3` + kNN 图（k=20, radius=10Å）
   - 输出：
     - `h_s_gvp: [B,L,256]`
     - `h_v_gvp: [B,L,16,3]`

3. **Pairwise Bridge + Early Fusion**
   - PairwiseBridge：`[16,3] -> 136`（上三角内积展开）
   - 拼接：`h_geo = [h_s_gvp(256), v_inv(136), protein_scalar_encoded(64)]`
   - 得：`h_geo: [B,L,456]`

4. **双路投影与蛋白上下文建模**
   - `gvp_proj: 456 -> 256`
   - `esm_proj: 1280 -> 256`
   - 拼接：`h_seq: [B,L,512]`
   - `ProteinContextEncoder`（4层 Gated Transformer + PE）
   - 输出：`h_prot_global: [B,L,512]`

5. **Ligand Encoder（GINEConv）**
   - 节点 `80` 维输入，3层 GINEConv，输出 `512` 维原子嵌入
   - 再按 batch scatter 到 padded 张量：`h_ligand: [B,M,512]`

6. **Bidirectional Cross-Attention（关键）**
   - 堆叠 `n_cross_attention_layers=4`
   - `Prot -> Lig` 更新蛋白特征，`Lig -> Prot` 更新配体特征
   - 输出：
     - `h_prot_fused: [B,L,512]`
     - `h_lig_fused: [B,M,512]`

7. **Late Injection + 分类头**
   - 拼接：`x_final = [h_prot_fused(512), protein_scalar_encoded(64)] = [B,L,576]`
   - 分类器：`576 -> 256 -> 3`
   - 输出 logits：`[B,L,3]`

## 4.2 参数规模

- 基于 `configs/pocket_gvp_v6.yaml` 实例化：
  - 总参数量：`45,049,075`
  - 可训练参数：`45,049,075`

---

## 5. 损失函数设计（train_v6m5）

v6m5 使用 **Dice + Focal 组合损失**：

\[
\mathcal{L}_{total} = w_{dice}\cdot \mathcal{L}_{dice} + w_{focal}\cdot \mathcal{L}_{focal}
\]

默认：`w_dice = 0.5`, `w_focal = 0.5`（可在 config 覆盖）。

### 5.1 Dice Loss

- 在 `softmax(logits)` 上按类别计算 Dice
- 使用 `smooth=1.0` 防止除零
- 对所有类别取 mean（macro）

### 5.2 Focal Loss

- 先计算逐 token CE
- 调制项：`(1 - p_t)^gamma`
- 乘类别权重（默认 `[5.0, 3.0, 1.0]`）
- 使用 `mask` 去掉 padding 后取均值

---

## 6. 训练脚本关键机制

1. **优化器/调度器**
   - `AdamW(lr=1e-4, weight_decay=0.01)`
   - `Linear warmup (5 epochs) + CosineAnnealingLR`

2. **稳定性策略**
   - 梯度裁剪：`clip_grad_norm_=1.0`
   - 可选 AMP、可选 DDP、可选 `torch.compile`

3. **验证与模型选择**
   - 每 epoch 同时评估 VDW 与 Original（若提供 original labels）
   - **best model 判据：`val_metrics['loss']`（VDW 口径）**
   - 早停依据同上（patience）

4. **TensorBoard 命名（v6m5）**
   - `Train/*`：训练指标（VDW）
   - `Val_new/*`：验证 VDW 指标
   - `Val/*`：验证 Original 指标（对比口径）
   - `Val_original/*`：Original 分割指标补充
   - `Health/*`：特征范数/门控/注意力熵等健康指标

---

## 7. 与前版差异（用于报告）

- 相比 v6：
  - 引入 PhysChem 编码器（前版多为直接拼接）
  - 引入组合损失（Dice + Focal）
  - LayerNorm + GELU 改善特征尺度与激活平滑性

- 相比 v6m2/v6m4：
  - v6m2 的 ReLU 在负值区间有截断；v6m5 改为 GELU，理论上保留更多细粒度信息。

---

## 8. 报告可直接引用的简短总结

**Pocket-GVP-Fusion v6m5** 采用“蛋白几何（GVP）+ 序列语义（ESM）+ 配体图（GINE）”三模态融合框架，通过双向交叉注意力实现蛋白-配体互相条件化建模；并在蛋白物化特征支路引入 LayerNorm+GELU 的 MLP 编码器，结合 Dice/Focal 组合损失，以提升类别不平衡场景下的分割鲁棒性与收敛稳定性。模型在训练中以 VDW 标签为主目标，Original 标签仅用于横向对比评估。

