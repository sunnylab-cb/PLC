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

## V11m4/V11m5 后续实验 TODO（2026-05-26）

### 当前结论

- `V11m4` 是当前最强稳定基座：best F1_c0 约 `0.5814 @ e61`。
- `V11m5` 将 local54 pair evidence 前移到 P→L attention bias，100 epoch best F1_c0 约 `0.5780`，确认不优于 V11m4；该路线归档，不再作为主线继续堆复杂度。
- 旧 `V11m5t`（pair-bias + global gate）100 epoch best F1_c0 约 `0.5830`，与 `V11m4t` 持平；收益主要来自 global gate，而不是 pair-bias。
- 当前 `V11m5t` 已改为 `V11m4t + group-wise balanced ligand input adapters`：local54/global55/UniMol 分组 LN/MLP/LN 后再进入 Ligand3DEncoder，用于验证入口 `RMS(W_group x_group)` 是否平衡。
- Ligand dependency eval 显示主要有效信号来自 local54 / ligand graph；global55 更像 precision calibration / delta amplitude control。

### 下一步实验

1. `V11m4t`: 用 `best_f1_c0_model.pt` 重新跑 ligand dependency eval，确认 global gate 是 calibration 还是增强 ligand causality。
2. `V11m4tp`: probe-only V11m4t control，不改变 logits/参数，只记录旧入口 `RMS(W_group x_group)`，作为 V11m5t-balanced 的直接对照。
3. `V11m5t-balanced`: 跑新配置 `configs/pocket_gvp_v11m5t_hpc.yaml`，重点看：
   - `Health/ligand_group_*_proj_rms`
   - `Health/ligand_*_nodeproj_contrib_rms`
   - `Gradients/grad_norm_ligand_*_adapter`
   - full F1 是否至少不低于 V11m4t
4. 用 `scripts/diagnostics/check_v11_input_balance.py` 对 V11m4tp 与 V11m5t-balanced 做 no-train entry-balance probe，比较：
   - `ligand_local_nodeproj_contrib_rms`
   - `ligand_global_nodeproj_contrib_rms`
   - `ligand_unimol_nodeproj_contrib_rms`
   - `protein_gvp_proj_rms` / `protein_esm_proj_rms`
5. Ligand dependency eval 重点比较：
   - full / shuffle / zero
   - zero_local54 / zero_global55 / zero_unimol
6. `V11m6`: 作为 `V11m4t` 的受控后继运行，不继续 pair-bias 前移：
   - local54 + ligand graph hidden states 只产生 late residue-level delta。
   - global55 + UniMol CLS 只做 delta amplitude calibration。
   - 默认 detach protein feature in delta branch，保护 V11 protein-side floor。
   - 训练后必须跑同一套 dependency eval，重点比较 `Δshuffle`、`Δzero_local54`、`Δzero_unimol` 是否优于 V11m4t/V11m5t-balanced。

---

## Ligand Encoding 方案记录与后续 TODO（2026-05-26）

详细设计记录见 `DOCs/Co-attention/ligand_encoding_review.md`。

### 当前判断

- 同行主流仍以 atom-level graph/token 为 ligand 表示底座；one-hot / categorical embedding 对 atom type、hybridization、chirality、ring/aromatic 等非序数特征仍合理。
- `temperature/thermometer` 或 ordinal encoding 只适合 degree、valence、H count、formal charge bin、ring size 等有序离散特征，不适合 atom type。
- `target encoding` 不应作为训练特征：当前 residue-level target 强依赖 protein-ligand pair，target mean encoding 容易引入 ligand identity / protein family shortcut。
- 不应把 ligand 合并成单一全局向量；单一 ligand vector 可做 calibration/global prior，但无法提供 residue-specific evidence。
- 现有 ablation 已证明 local54 / ligand graph 是主要 ligand dependency 来源；global55 更像 precision calibration / delta amplitude control。
- 最有价值的下一步 ligand 表示升级是 functional group / pharmacophore tokens，用中间语义层缩小 ligand atom feature 与 protein ESM/GVP/physchem 表征之间的语义密度差。

### 分步任务

1. **P0: 等 V11m4tp / V11m5t-balanced 完成后做入口平衡判断**
   - 比较旧入口 `RMS(W_local x_local)`、`RMS(W_global x_global)`、`RMS(W_unimol x_unimol)`。
   - 同时比较 protein 端 `RMS(h_gvp_proj)` / `RMS(h_esm_proj)` 和 modality gate 后贡献。
   - 若入口明显不平衡且 balanced adapter 改善或至少不伤 F1，则保留 group-wise ligand adapters 为 V11m5t 主线。
2. **P1: 若 balanced adapter 有收益，固定 group-wise ligand adapter 设计**
   - local54 / global55 / UniMol 分别 `LN/MLP/LN` 后再 concat。
   - 目标是训练起点 projected RMS 同数量级，不是长期强制同权。
   - 后续 dependency eval 必须包含 `zero_local54`、`zero_global55`、`zero_unimol`。
3. **P2: 设计 V11m7 functional group / pharmacophore tokens**
   - 保留 atom-level local54，不替换为全局 ligand embedding。
   - 用 RDKit SMARTS / pharmacophore detection 生成 ligand-only FG tokens。
   - 不引入 protein-ligand cross-distance，不使用 holo leakage，不使用 target encoding。
   - 最小 ablation: `atom-only`、`atom+FG`、`atom+FG+global gate`。
4. **P3: 如果 V11m7 显示 FG token 有效，再考虑更强 ligand prior**
   - UniMol atom-level memory adapter。
   - ChemBERTa / MoLFormer CLS token。
   - KPGT/GROVER-style graph prior token。
   - conformer ensemble / microstate features。

### 暂停方向

- 不继续把旧 V11m5 pair-bias 前移路线作为主线堆复杂度。
- 不做 target encoding 进入训练特征。
- 不用 single ligand embedding 替代 atom-level tokens。
- 不优先增加 co-attention 深度或重复加入 pair-bias，除非 dependency eval 证明其带来稳定 ligand causality 增益。

---

## V12m3 / V12m4 实验 TODO（2026-05-22）

### 当前结论

- `V12m3 pair old128` 仍是当前最强主线：best F1 `0.4781 @ e56`，best Dice `0.4799 @ e56`。
- `V12m3 pair256` 早期更快：best F1 `0.4698 @ e28`，但截至 e31 仍未超过 old128 best；继续观察到 e50-e60。
- `V12m4 pair_aux old128` 暂不作为主线：best F1 `0.4569 @ e26`，低于 `pair256` 与 old128。
- V12m4 的 pair branch 单独输出强于 main logits：例如 e26 `pair_f1_c0=0.4638` vs main `0.4569`，说明 pair evidence 有效，但没有被主 logits 正确吸收。

### V12m4 问题定位

- 结构上 base classifier 已经 concat `pair_context + pair_features + h_int_broadcast`，同时又使用 `logits = base_logits + pair_delta_gate * pair_logits_delta`，存在重复注入 pair evidence 的风险。
- `pair_aux.weight=0.1` 直接监督 `pair_logits_delta`，可能让 auxiliary branch 单独变强，但与 main logits 校准产生 negative transfer。
- `pair_delta_gate` 只从约 `0.100` 增到 `0.116`，pair branch 明显更好但主 logits 只吃到很小一部分 delta。
- TB 表现：V12m4 val focal loss 从 e9 后持续上升，train loss 继续下降，说明校准/过拟合问题比 pair256 更明显。

### 下一步实验优先级

1. 继续当前 `V12m3 pair256` 和 `V12m4 pair_aux old128` 到 e50-e60，确认长期趋势。
2. 若新增实验，优先做 `V12m4-lite`：
   - `pair_aux.weight: 0.05`
   - 保留 `use_pair_logits_delta: true`
   - 可考虑 `pair_delta_gate_init: 0.2`
3. 更干净的结构 ablation：
   - `classifier_pair_concat: false`
   - pair evidence 只通过 `pair_logits_delta` 影响 logits
   - 目的：验证 pair delta 是否真正能作为主 logits correction，而不是与 base classifier 重复竞争。
4. 如果 `pair256` 到 e50 仍低于 old128，回到 old128 bottleneck，下一步只调 calibration / checkpoint policy，不继续扩大 pair bottleneck。

---

## V11m2 → V12 候选路线（2026-05-13）

这组事项以当前 V11m2 为基线：ESM2 + PhysChem + protein GVP + RoPE/DistanceBias transformer + ligand GVP/self-attn + bidirectional cross-attn + ordinal smoothing/CNC。目标不是继续堆模块，而是优先选择能提高 ligand-aware residue scoring、标签鲁棒性和几何先验的改动。

### V12-A. 先做的低风险 ablation

**A1. 配体全局物化条件，但必须复用 V6 provenance/mask**

- 价值：模型现在有 ligand global token，但全局物化性质没有作为干净的 graph-level conditioning 明确进入模型。
- 注意：`LogP/TPSA/Morgan` 已经部分存在于 V5/V6 ligand features，不能直接再用一套裸 RDKit 描述符无脑拼接，否则会重复且污染 metal/inorganic/multifragment ligand。
- 建议：从 V6 的 `global_features`、`x_group_mask/source`、`ligand_type`、`operation_status` 中构建 masked graph-level ligand summary，再通过小 MLP 加到 ligand global token 或 gate。
- 验收：普通 organic、metal、inorganic、multifragment 四类样本都不能出现 silent-zero global props。

**A2. 表面曲率 / 局部形状特征**

- 价值：binding pocket 是表面凹陷，局部 PCA 曲率是低成本几何先验。
- 建议：先离线预计算 2-4D，例如 `lambda3/sum(lambda)`、`lambda2/lambda1`、邻域半径内 density-corrected curvature。
- 风险：CA-only PCA 对多链界面和柔性 loop 会有噪声，需要按 protein length / chain 分开归一化。
- 验收：加入 spatial scalar 后，特征归一化统计固定，不能按单蛋白自身最大值归一化。

**A3. 对称 CNC，但要 warmup + cap**

- 价值：BioLiP 正负标签都可能有噪声，正侧错误 contact 也会污染训练。
- 风险：如果早期模型还没学好，`p_far` 高的 contact 可能是真难例，不是错标；直接下权会牺牲 recall。
- 建议：只在 warmup 后启用，用 EMA confidence 或连续 N 次高置信 far，再 clip；并限制每 batch 正样本 clip 比例。
- 验收：必须单独记录 `negative_clip_frac` 和 `positive_clip_frac`，并监控 Contact recall 是否下降。

### V12-B. 架构核心候选

**B1. Cross-attention 后的 ligand FiLM / gate**

- 价值：V11m1 已经证明“ligand global token 调制 protein representation”是干净方向。V12 可以把 gate 从最终 classifier 前推进到 cross-attn block 间。
- 建议：每层用 ligand global token 生成轻量 gate，优先用 zero-init / near-identity init，避免一开始扰乱已稳定的 protein branch。
- 风险：FiLM 比最终 gate 更强，会改变优化动力学；必须和 V11m1 做 controlled ablation。

**B2. 非对称 cross-attention 深度需要谨慎**

- 价值：可能减少计算量。
- 主要疑点：两个方向的 attention 都是 `L*M` 级别，减少 ligand update 不一定带来 30% 实测加速；而 ligand global token 当前被 classifier/gate 使用，削弱 ligand 更新可能伤害 ligand-aware scoring。
- 建议：不要作为 V12 默认核心。可做 ablation：`4P+4L` vs `4P+2L` vs `4P+1L`，以 wall-clock、F1、Contact recall 一起评估。

**B3. ESM2 多层融合是高价值但不是低风险**

- 价值：多层 ESM 表征通常能补足局部化学语义和远程结构语义。
- 成本：需要重新抽取/存储 ESM2 多层 latent，数据体积约按层数放大；dataset schema 会变。
- 建议：先抽 `last4` 或固定若干层到一个小 pilot split，训练可学习 scalar mix；不要立刻全量重跑。

### V12-C. 暂缓或独立项目

**C1. 多尺度配体图边**

- 暂缓理由：V11m2 已经有 ligand self-attn，天然提供 atom-level long-range mixing；额外多关系 GVP 会增加 encoder 复杂度，收益不确定。
- 更稳替代：先给 ligand self-attn 加/保留 3D distance bias，并把 V6 ligand type/status/global summary 喂给 global token。

**C2. Uni-Mol / ChemBERTa ligand embedding**

- 价值可能高，尤其对稀有配体。
- 暂缓理由：BioLiP 含 metal、inorganic、peptide、nucleotide、multifragment；通用 small-molecule LM 可能无法覆盖这些类别，缺失本身会成为新 bias。
- 建议：作为独立 V13 数据工程项目，先定义 coverage、failure mask 和 fallback，再谈接入模型。

**C3. 虚拟口袋节点**

- 暂缓理由：需要改 GVP graph building 和训练时动态几何，调试成本高；当前更大的收益来自数据契约、loss 噪声和 ligand conditioning。

### 推荐执行顺序

1. V11m3: `A1 + A2 + A3`，只改 feature/loss/conditioning 输入，保留主架构。
2. V12: `B1`，把 V11m1 的 ligand gate 扩展到 cross-attn block 间。
3. V12 ablation: `B2` 非对称 cross-attn，只作为速度/精度权衡实验。
4. V12 data pilot: `B3` ESM 多层 scalar mix，先小规模验证收益再全量重抽。
5. V13+: `C2/C3`，作为独立数据和架构项目。

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

---

## V12 開發狀態（2026-05-15）

### 已完成

- Interleaved Co-Attention 架構（N=6 CoAttentionBlock，Evoformer 風格）
- FiLM conditioning（Uni-Mol → protein，每個 block，zero-init identity）
- Parallel bidirectional cross-attention（P→L + L→P 同時讀 pre-update h）
- Ligand SA RoPE neutralization（all-zero position_ids，避免假序列偏差）
- CrossModalDistanceBias 移除（protein/ligand 在各自坐標系，跨模態距離是偽特徵）
- SASA residual gate（tanh-bounded，zero-init，classifier 前，rsa_mask 防缺失偏差）
- protein_rsa_mask（1=有 RSA，0=舊 .pt 缺失，防 RSA=0 被誤判為 buried）
- RSA pre-injection（Innovation #3：co-attention 前注入，zero-init，6 blocks 都能看到 surface prior）
- Uni-Mol → global token augmentation（Innovation #4：mean-pool token + zero-init proj，讓 cross-attn 可主動查詢 Uni-Mol）

### 待驗證

- 環境問題修復後跑完整 forward smoke test（torch_scatter 與 ROCm 7.2 相容性）

### 待實作

- **dynamic FiLM（#3 ablation）**：blocks 2-5 改用 `h_lig[:, 0, :]`（protein-context-aware global token）做 FiLM conditioning，不是靜態 unimol_embed。方向合理但會改變優化動力學，需 controlled ablation，不和 #1/#2 同時上。
- **V12 訓練配置**：`configs/pocket_gvp_v12.yaml`
- **V12 訓練腳本**：`scripts/train/train_v12.py`（基於 train_v11.py，加 unimol_embed 載入 + prepare_model_inputs_v12）
- **Uni-Mol 嵌入接入 dataset**：PocketDatasetV8 加載 `data/unimol_embeddings/{yr}/{pdb}/{pdb}_unimol.pt`

最后更新：2026-05-15
