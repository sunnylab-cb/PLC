# ProtBind PLC TODO

本文档以当前主训练链路为准，按执行优先级整理特征工程、训练链路和模型架构改造事项。

## Source of Truth

当前主链路以以下文件为准：

- `scripts/train/train_v9m2.py`
- `configs/pocket_gvp_v9.yaml`
- `src/models/pocket_gvp_fusion_v9m2.py`
- `src/models/transformer_layers.py`
- `src/data/pocket_dataset_v8.py`
- `src/data/collate.py`
- `preparation/protein/extract_complete_features_v3.py`
- `preparation/protein/extract_complete_features_v3_nma.py`
- `preparation/ligand/extract_ligand_features_v4.py`

## 当前状态说明

- `split_file` 问题已确认是配置里把 `txt` 写成了 `csv`，不是代码逻辑错误。
- `Flash Attention` 目前不能视为真正完成：
  - 配置默认关闭。
  - 当前实现条件判断导致 flash 分支实际不可达。
- `Boundary Loss` 已有实现，但在 NMA 标签噪声未修正前不建议作为主要卖点。
- 当前 TODO 以“先修真实主链问题，再做下一代架构升级”为原则。

---

## P0 立即处理

### P0-1. 修复 Flash Attention 死路径或移除卖点

**优先级**: P0  
**影响**: 高  
**难度**: 中

**问题**
- `GatedMultiHeadAttention.forward()` 中，`kv is None` 会先被改写为 `x`，后续 `if mask is None and kv is None` 永远不成立。
- 主模型蛋白自注意力始终传入 `protein_mask`，即使修正前一个判断，也依然进不了当前 flash 分支。
- 配置默认 `use_flash_attn: false`，文档与实现状态不一致。

**相关文件**
- `src/models/transformer_layers.py`
- `src/models/pocket_gvp_fusion_v9m2.py`
- `configs/pocket_gvp_v9.yaml`
- `scripts/test_v9m2_setup.py`

**执行方案**
1. 明确策略二选一：
   - 方案 A：真正接成 varlen/unpadding 版本，支持带 padding mask 的 self-attention。
   - 方案 B：短期先移除 V9m2 中关于 Flash Attention 已完成的表述，只保留普通 attention。
2. 如果走方案 A：
   - 在 attention 前基于 `protein_mask` 做 unpad。
   - 仅对 self-attention 开启 flash。
   - cross-attention 继续走标准路径。
3. 补测试：
   - 增加一个无 mask 和一个带 mask 的单元测试。
   - 验证 flash 分支是否真的被命中，而不是只验证能实例化。

**验收标准**
- 明确打印或日志中可确认 flash 分支是否命中。
- `scripts/test_v9m2_setup.py` 不再读取不存在的配置文件。
- 文档、配置、实现三者表述一致。

---

### P0-2. 修复 DDP 下 cluster-weighted loss 仅 rank0 生效的问题

**优先级**: P0  
**影响**: 高  
**难度**: 简单

**问题**
- `cluster_sizes` 只在 rank 0 读取。
- 广播时只发了 `train_ids/val_ids/test_ids`，没有发 `cluster_sizes`。
- 结果是 rank 0 使用 cluster-weighted loss，其他 rank 回退到标准 loss，DDP 梯度语义不一致。

**相关文件**
- `scripts/train/train_v9m2.py`

**执行方案**
1. 在 DDP 广播对象中加入 `cluster_sizes`。
2. 所有 rank 统一构造 `ClusterWeightedLoss`。
3. 如果 split 文件不带 cluster 信息，所有 rank 一致回退到标准 loss。
4. 在日志中打印：
   - `cluster weighting enabled/disabled`
   - `cluster_sizes loaded entries`

**验收标准**
- 多卡训练时所有 rank 的 loss 分支一致。
- rank 0 与其他 rank 不再出现条件分叉。

---

### P0-3. 让 NMA 增强变成标签感知，而不是复用原始构象标签

**优先级**: P0  
**影响**: 高  
**难度**: 高

**问题**
- 当前 `PocketDatasetV8` 对所有 `conf_idx` 都按 `pdb_id` 读取同一份距离标签。
- 对边界残基和阈值附近样本，这会引入系统性标签噪声。
- 该问题会直接削弱 `Boundary Loss` 和所有 distance-aware 类设计。

**相关文件**
- `src/data/pocket_dataset_v8.py`
- `scripts/train/train_v9m2.py`
- `preparation/tag/calculate_distances_vdw.py`
- `preparation/protein/extract_complete_features_v3_nma.py`

**执行方案**
1. 最优方案：
   - 为每个 NMA 构象重新计算 residue-ligand distance。
   - 标签从 `pdb_id + conf_idx` 读取。
2. 过渡方案：
   - 保留硬标签主任务。
   - 为阈值附近残基增加 soft label 或连续距离回归辅助头。
3. 如果短期无法重算全部标签：
   - 先统计 NMA 引起的边界翻转比例。
   - 针对高不确定样本降低 loss 权重。

**验收标准**
- 训练集中的 `conf_idx != 0` 样本不再直接复用 `conf0` 标签。
- 能报告 NMA 前后标签一致率和边界翻转率。

---

### P0-4. 重建 ligand branch，停止使用全连接假边

**优先级**: P0  
**影响**: 高  
**难度**: 中

**问题**
- 配体预处理阶段提取了丰富原子特征，但没有把真实化学拓扑保存下来。
- `collate.py` 在训练时为每个配体构建 `O(M^2)` 全连接图，并用固定伪 edge feature 填充。
- 这既慢，又稀释了真实化学信息，如键阶、芳香性、共轭、环结构、立体信息。

**相关文件**
- `preparation/ligand/extract_ligand_features_v4.py`
- `preparation/ligand/ligand_features_v3.py`
- `src/data/collate.py`
- `src/models/ligand_encoder_3d.py`

**执行方案**
1. 在配体预处理时直接保存：
   - `edge_index`
   - `edge_attr`
   - 建议包含 `bond order / aromatic / conjugated / ring / stereo`
2. 模型侧使用双图：
   - 化学键图：表达真实化学拓扑
   - 空间半径图：表达 3D 邻接关系
3. `collate.py` 不再现场构造全连接图。
4. 如需保留空间边，改为 `radius_graph`，不要用全连接。

**验收标准**
- 配体图构图不再是 `O(M^2)` 全连接。
- 训练输入中保留真实 bond graph。
- 数据加载时间和显存占用有可测量下降。

---

## P1 高优先级

### P1-1. 重做在线 `gvp_vectors`，改成离线、规范化几何基

**优先级**: P1  
**影响**: 中高  
**难度**: 中

**问题**
- 当前在线 `gvp_vectors` 中，3 个向量来自“前 3 个原子相对 CA 的方向”。
- 这对原子顺序敏感，并且容易产生 `CA-CA = 0` 之类的无信息通道。
- 你已经有 6 个更有语义的离线空间向量，当前在线补充更像噪声源。

**相关文件**
- `src/data/pocket_dataset_v8.py`
- `preparation/protein/extract_complete_features_v3.py`
- `preparation/protein/extract_complete_features_v3_nma.py`

**建议目标向量**
- `N - CA`
- `C - CA`
- `CB - CA` 或 `virtual CB - CA`
- `sidechain centroid - CA`

**执行方案**
1. 在蛋白预处理阶段直接保存这 4 个向量。
2. 数据集侧不再在线遍历 `all_atom_coords` 构造 `gvp_vectors`。
3. 对所有向量做稳定归一化，并显式处理 Gly / 缺失原子场景。

**验收标准**
- `PocketDatasetV8.__getitem__()` 不再现场构造 `gvp_vectors`。
- 向量语义与残基几何定义一一对应。

---

### P1-2. 统一蛋白 V3 标量特征的归一化口径

**优先级**: P1  
**影响**: 中高  
**难度**: 中

**问题**
- 目前多处按单蛋白自身最大值归一化：
  - `local density`
  - `distance_to_center`
  - `charge density`
- 这会抹掉跨蛋白的绝对尺度差异。

**相关文件**
- `preparation/protein/extract_complete_features_v3.py`
- `preparation/protein/extract_complete_features_v3_nma.py`

**执行方案**
1. 对稳定连续特征改成训练集级别统计归一化：
   - `mean/std`
   - 或 `median/IQR`
2. 保留少量天然在 `[0, 1]` 的比例型特征。
3. 导出 normalization stats 文件，训练/推理共用。
4. 增加特征分布检查脚本：
   - train/val/test 分布对比
   - NMA 前后分布漂移

**验收标准**
- 不再出现“同一维特征在每个蛋白内部重新缩放到 0~1”的情况，除非该维天然就是比例。

---

### P1-3. 修正 `test_v9m2_setup.py`，覆盖真实主路径

**优先级**: P1  
**影响**: 中  
**难度**: 简单

**问题**
- 当前测试脚本读取不存在的 `configs/pocket_gvp_v9m2.yaml`。
- Flash 测试传了 mask，不会命中当前实现中的 flash 路径。
- 没有覆盖真实主训练链路中的数据集、collate、模型前向组合。

**相关文件**
- `scripts/test_v9m2_setup.py`

**执行方案**
1. 改为读取当前实际使用的 `configs/pocket_gvp_v9.yaml`。
2. 加一个最小前向 smoke test：
   - dataset sample
   - collate
   - model forward
3. 为 flash 分支单独做路径级测试。
4. 补 DDP cluster-weighted loss 的最小一致性测试。

**验收标准**
- 测试脚本与实际训练配置一致。
- 能覆盖主链最容易漂移的逻辑分支。

---

### P1-4. 重新定位 `Boundary Loss`

**优先级**: P1  
**影响**: 中  
**难度**: 简单

**问题**
- 当前 `Boundary Loss` 实现本身可用，但数据标签噪声尚未治理。
- 在 NMA 标签未修复前，过早强调 boundary loss 容易掩盖主问题。

**相关文件**
- `src/losses/boundary_loss.py`
- `scripts/train/train_v9m2.py`

**执行方案**
1. 在 P0-3 完成前，保持默认关闭。
2. 将其重新定义为“可选实验分支”，不是当前主卖点。
3. 增加 boundary residue ratio 与 boundary-specific F1 的评估。

**验收标准**
- 开关状态、文档描述和实验结论一致。

---

## P2 架构升级

### P2-1. 引入显式 residue-atom pair 表示

**优先级**: P2  
**影响**: 很高  
**难度**: 高

**问题**
- 当前主干只有 token-level cross-attention。
- 缺少显式的 residue-atom pair tensor 或 pair bias，无法直接编码“哪个残基与哪个原子如何相互作用”。

**建议方向**
- 构造 `pair_ij` 表示：
  - residue token
  - ligand atom token
  - pair distance / direction / chemistry bias
- 做 pair update，再把 pair summary 回流到 residue head。

**相关文件**
- `src/models/pocket_gvp_fusion_v9m2.py`
- `src/models/transformer_layers.py`
- `src/models/ligand_encoder_3d.py`

**参考方向**
- FABind
- LABind

---

### P2-2. 分类头从“仅 residue token”升级为“residue token + local pair summary”

**优先级**: P2  
**影响**: 高  
**难度**: 中高

**问题**
- 当前最终分类基本只依赖 residue token。
- 对结合位点任务来说，局部 ligand context 往往比更深的全局 transformer 更关键。

**执行方案**
1. 汇聚每个残基邻近 ligand atom 的 pair summary。
2. 分类头输入改为：
   - `h_residue`
   - `pair_summary`
   - 可选 `distance-aware local context`

**验收标准**
- 分类头明确依赖局部蛋白-配体交互摘要，而非仅靠全局 token。

---

### P2-3. 探索 equivariant residue-atom message passing

**优先级**: P2  
**影响**: 高  
**难度**: 很高

**问题**
- 当前 cross-attention 仍是标量主导。
- 蛋白侧和配体侧都已经具备 3D 信息，继续只堆标量 attention 的上限有限。

**建议方向**
- 对 residue 和 atom 做联合的 3D equivariant message passing。
- 将方向、距离、局部框架信息显式注入交互层。
- 让 NMA 从“数据增强”升级为“构象建模”。

**参考方向**
- DynamicBind

---

## 后续研究路线图

以下事项保留为次级路线图，但默认排在上面的主链修复之后。

### 数据与增强
- 自适应 NMA 采样
- Hard example mining / OHEM
- 一致性正则：同一蛋白不同构象预测一致
- NMA 质量控制：RMSD、能量、物理合理性过滤

### 损失函数
- Distance-aware loss
- Contrastive loss
- 动态 focal gamma

### 训练策略
- Layer-wise LR decay
- AMP 默认化
- Gradient checkpointing
- Gradient accumulation

### 架构微调
- Multi-scale fusion
- Structural attention bias
- Residual scaling

### 推理与评估
- Per-residue confidence
- DCC / overlap volume
- 多 checkpoint 集成

---

## 推荐执行顺序

### 第一阶段：修真实主链
1. P0-1 Flash Attention 路径问题
2. P0-2 DDP cluster-weighted loss 一致性
3. P0-3 NMA 标签感知
4. P0-4 ligand 真实 bond graph

### 第二阶段：稳固特征基础
1. P1-1 `gvp_vectors` 离线化
2. P1-2 标量特征归一化重做
3. P1-3 主路径测试补齐
4. P1-4 重新定位 boundary loss

### 第三阶段：做下一代架构
1. P2-1 residue-atom pair 表示
2. P2-2 pair-aware 分类头
3. P2-3 equivariant residue-atom message passing

---

## 参考文献

### 通用方法

1. Flash Attention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. DropBlock: Ghiasi et al., "DropBlock: A regularization method for convolutional networks", NeurIPS 2018
3. Contrastive Learning: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
4. Layer-wise LR Decay: Clark et al., "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", ICLR 2020
5. Stochastic Depth: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016

### 蛋白-配体结合位点/复合体建模

6. LABind: ligand-aware binding site identification  
   https://www.nature.com/articles/s41467-025-62899-0

7. FABind: 联合建模蛋白与配体，并显式处理相互作用  
   https://openreview.net/forum?id=PnWakgg1RL

8. DynamicBind: 强调蛋白-配体共同适配与动态结构更新  
   https://www.nature.com/articles/s41467-024-45461-2

---

最后更新：2026-03-23
