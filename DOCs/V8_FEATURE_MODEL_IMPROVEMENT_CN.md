# V8 特征输入 × 模型架构 联动改进建议（中文）

## 1) 当前实现的关键事实（基于代码）

### 1.1 三路蛋白特征 + 一路配体特征

- 蛋白理化特征：`62D`，进 `physchem_encoder`。
- 蛋白空间标量：`72D`，作为 GVP 的 `node_s`。
- 蛋白空间矢量：`6×3D`，与在线计算的 `4×3D gvp_vectors` 拼接后作为 GVP 的 `node_v`（共 `10` 个向量通道）。
- 配体特征：`96D atom features + 3D coords`，经 `Ligand3DEncoder` 后与蛋白做双向 cross-attention。

对应配置见：`configs/pocket_gvp_v8.yaml:27`、`configs/pocket_gvp_v8.yaml:28`、`configs/pocket_gvp_v8.yaml:29`、`configs/pocket_gvp_v8.yaml:34`。

### 1.2 V3 特征文件已包含 GVP 在线计算所需字段

- `extract_complete_features_v3.py` 输出包含：
  - `physchem_features`
  - `spatial_scalar_features`
  - `spatial_vector_features`
  - `ca_coords`
  - `all_atom_coords`

见：`preparation/protein/extract_complete_features_v3.py:950`。

### 1.3 `gvp_vectors` 当前是在线算，不是 npz 离线读

- 数据集加载 `all_atom_coords` 后调用 `_compute_gvp_vectors()` 在线构造 `4×3D` 向量。
- 当前构造方式为：`CA->质心` + `CA->前3个原子`（依赖原子顺序）。

见：`src/data/pocket_dataset_v7.py:244`、`src/data/pocket_dataset_v7.py:352`。

### 1.4 模型融合路径（现在是有效的）

- `physchem_encoder` 编码后既参与早期几何融合，也在分类头前做 late injection。
- GVP 路径：`node_s=72D spatial scalar`，`node_v=(4+6)×3D`。
- ESM 与几何路径经 channel-wise gate 融合。
- 再通过多层双向 cross-attention 融合配体上下文。

见：
- `src/models/pocket_gvp_fusion_v8.py:134`
- `src/models/pocket_gvp_fusion_v8.py:287`
- `src/models/pocket_gvp_fusion_v8.py:343`
- `src/models/pocket_gvp_fusion_v8.py:370`
- `src/models/pocket_gvp_fusion_v8.py:435`
- `src/models/pocket_gvp_fusion_v8.py:482`

---

## 2) 主要改进机会（按优先级）

## P0（低风险，高收益，优先做）

### P0-1. 修复 V3 理化特征中的“二级结构恒定”问题

问题：当前 `physchem_features[:,56:59]` 直接写死为 `[0,0,1]`（默认 Coil），对模型几乎无信息增益。

见：`preparation/protein/extract_complete_features_v3.py:188`。

建议：
- 从 DSSP/已有结构几何模块读取真实二级结构并写入三维 one-hot（Helix/Sheet/Coil）。
- 若缺失再 fallback 到 Coil，不要全量默认。

预期收益：
- 提升局部结构敏感性，尤其对 β-sheet 富集口袋与 loop 暴露位点判别更稳定。

### P0-2. 提升在线 `gvp_vectors` 的几何语义稳定性

问题：`CA->前3个原子` 强依赖原子存储顺序，跨 PDB/解析器一致性不足。

见：`src/data/pocket_dataset_v7.py:352`。

建议（替代方案）：
- 向量1：`CA->残基重原子质心`
- 向量2：`N->CA`
- 向量3：`CA->C`
- 向量4：`C->O`（缺失时置零并附带有效位 mask）

预期收益：
- 减少噪声与样本间漂移，GVP 的 vector branch 更“结构可比”。

### P0-3. 配体图构建从“全连接伪边”改为“化学键+局部空间边”

问题：当前 collate 用全连接 `O(M^2)`，且 `bond_type` 基本是固定伪值，既慢又弱化化学先验。

见：`src/data/collate.py:416`、`src/data/collate.py:427`。

建议：
- 主边：真实化学键边（从配体特征文件读取）。
- 补边：半径图/knn 图（如 `r=4.5Å`），用于非键空间相互作用。
- `edge_attr` 区分“键边/空间边”并保留真实键类型。

预期收益：
- 训练吞吐提升；配体编码器学到更真实的局部化学拓扑。

### P0-4. 序列截断从“头部截断”升级为“口袋感知截断”

问题：当前超长蛋白直接 `[:max_seq_len]`，可能把真正口袋位点截掉。

见：`src/data/pocket_dataset_v7.py:314`。

建议：
- 基于 `distances_vdw_adjusted` 或最近配体距离，选取“最相关 K 个残基”或“以候选口袋为中心的窗口”。
- 同时保留少量全局上下文残基（例如 top-K + stride 采样）。

预期收益：
- 提升长链样本表现，减少无关残基噪声。

---

## P1（中风险，中高收益）

### P1-1. 三模态显式门控（ESM / 几何 / 理化）

问题：当前门控是 `ESM` vs `GVP+physchem(已拼接)` 二路；理化分支既早融又 late injection，存在“重复注入但不可控”的风险。

见：`src/models/pocket_gvp_fusion_v8.py:343`、`src/models/pocket_gvp_fusion_v8.py:370`、`src/models/pocket_gvp_fusion_v8.py:482`。

建议：
- 把 `physchem` 从 `h_geo` 中拆成独立支路，做三路 softmax gate：
  - `w_esm * h_esm + w_geo * h_geo + w_phys * h_phys`
- late injection 改为残差式轻注入（如 FiLM 或 gated residual），避免重复放大。

预期收益：
- 融合可解释性更好，避免个别模态“压制”其他模态。

### P1-2. 对空间矢量引入有效性 mask 与置信度门控

问题：部分残基矢量可能为零向量（缺失原子或计算失败），当前直接与有效矢量同权输入。

建议：
- 增加 `vector_valid_mask`（每个向量通道一个 bit）。
- 在 GVP 前增加 lightweight gate：`v = gate(validity, norm, residue_type) * v`。

预期收益：
- 降低缺失/低质量矢量对训练稳定性的影响。

### P1-3. 在 protein↔ligand cross-attn 加入几何偏置

问题：当前 cross-attn 主要基于特征相似度，未显式注入残基-原子几何距离偏置。

见：`src/models/pocket_gvp_fusion_v8.py:435`。

建议：
- 构造 residue-atom 距离 `d_ij`，加入 attention bias（如 RBF(d_ij) 线性投影后加到 attention logits）。
- 对过远 pair 施加 soft mask。

预期收益：
- 强化“空间邻近优先”归纳偏置，尤其有利于 Contact 类。

---

## P2（实验型，收益不确定但值得探索）

### P2-1. 多任务学习：分类 + 连续距离回归

建议：
- 在现有 3-class logits 外，新增一个距离回归头（预测 `min distance` 或 `vdw-adjusted distance`）。
- 训练损失：`L = L_cls + λ * L_reg`。

价值：
- 提升边界样本（Near/Far）排序一致性，通常会改善 PR-AUC。

### P2-2. 模态 dropout 课程学习（Curriculum）

问题：当前 `esm_dropout_prob`、`physchem_dropout_prob` 为常量。

见：`configs/pocket_gvp_v8.yaml:50`、`configs/pocket_gvp_v8.yaml:51`。

建议：
- 前期高 dropout 促融合鲁棒性，后期逐步降低以追求上限性能。
- 或按 batch 难度自适应调整。

---

## 3) 建议的最小实验路线（4 组就能看方向）

1. **Baseline**：当前 V8。
2. **+ DSSP真实二级结构**（只改特征提取）。
3. **+ 稳定 gvp_vectors**（只改 `_compute_gvp_vectors`）。
4. **+ 配体稀疏图（键边+半径边）**（只改 collate/配体图构建）。

评估指标建议：
- 主指标：Contact 类 `F1 / IoU / PR-AUC`。
- 辅助：Near/Far macro-F1、按蛋白长度分桶表现、推理耗时。

如果第 2~4 组都带来稳定增益，再进入 P1（门控与几何偏置）会更稳。

---

## 4) 与你当前仓库状态相关的补充

- `train_v8` 的输出目录命名重复（`..._v8_v7m2_...`）问题已修复为仅保留一个 `v8`。
- 修复位置：`scripts/train/train_v8.py:168`。

