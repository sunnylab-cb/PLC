# Pocket-GVP-Fusion V8 模型架构与输入特征详解（中文）

本文基于仓库当前实现代码整理，覆盖：

- V8 端到端数据流
- 7个核心模块的结构与张量维度
- 蛋白/配体输入特征的组成与语义
- 训练标签与实现注意点

---

## 1. V8 总体任务与输出

- 任务：蛋白残基级三分类（每个残基预测一个类别）
- 输出：`logits [B, L, 3]`
  - `B`：batch size
  - `L`：该 batch 内最大残基长度（padding 后）
  - `3`：三分类（Contact / Near / Far）

核心返回结构：

- `logits`
- `h_prot_global`（蛋白上下文编码后特征）
- `h_fused`（蛋白与配体交互后的融合特征）
- `h_ligand_fused`（配体融合特征，含 global token）

---

## 2. 端到端流程（数据到模型）

### 2.1 数据集阶段（`PocketDatasetV7`）

每个样本读取：

- 蛋白 V3 特征文件：`data/complete_features_v3/.../{pdb}_protein.pt`
- ESM2 特征文件：`data/esm2_latents/.../{pdb}_protein.pt`
- 配体 V4 特征文件：`data/ligand_features_v4/.../{pdb}_ligand.pt`

生成样本字段：

- 蛋白：`esm`、`protein_physchem`、`protein_spatial_scalar`、`protein_spatial_vector`
- 几何：`gvp_coords`、`gvp_vectors`
- 配体：`ligand_features`、`ligand_coords`
- 标签：`labels`

> 注：`gvp_vectors [N,4,3]` 在数据集中在线计算，不是直接从 V3 文件读取。

### 2.2 Collate 阶段（`pocket_collate_fn_v8`）

对可变长蛋白与配体做 padding，得到 batch 级张量：

- `esm_embeddings [B, L, 1280]`
- `protein_physchem [B, L, 62]`
- `protein_spatial_scalar [B, L, 72]`
- `protein_spatial_vector [B, L, 6, 3]`
- `protein_vectors [B, L, 4, 3]`（即 GVP vectors）
- `protein_mask [B, L]`
- `ligand_features [B, M, 96]`
- `ligand_coords [B, M, 3]`
- `ligand_mask [B, M]`

### 2.3 模型输入准备（`prepare_model_inputs_v8`）

- 将配体构造成 batched graph：`ligand_x, ligand_edge_index, ligand_edge_attr, ligand_batch`
- 将批次字段映射到模型 forward 参数

---

## 3. V8 架构分解（7个模块）

## 3.1 模块1：PhysChem 编码器（62D -> 128D）

结构：

1. `Linear(62,128)`
2. `LayerNorm(128)`
3. `GELU`
4. `Linear(128,128)`
5. `LayerNorm(128)`
6. `GELU`

输出：`protein_physchem_encoded [B, L, 128]`

训练时可做模态 dropout（`physchem_dropout_prob`）。

## 3.2 模块2：蛋白几何编码（GVP）

输入到 GVP：

- 标量节点 `node_s = protein_spatial_scalar [B,L,72]`
- 向量节点 `node_v = cat(gvp_vectors[4], spatial_vector[6]) = [B,L,10,3]`

处理流程：

- 按样本有效残基（`protein_mask`）构图
- 使用 `build_knn_graph`（`k=20`, `radius=10Å`）构边
- `GVPEncoder` 堆叠 3 层 GVPConv

输出：

- `h_s_gvp [B, L, 256]`
- `h_v_gvp [B, L, 20, 3]`

## 3.3 模块3：Bridge + ESM/GVP 门控融合

### (1) PairwiseBridge（向量不变特征）

- 对 `h_v_gvp [B,L,20,3]` 做 Gram 矩阵
- 取上三角展开：`20 * (20+1) / 2 = 210`
- 得到 `v_inv [B,L,210]`

### (2) 几何主分支拼接

- `h_geo = concat(h_s_gvp[256], v_inv[210], physchem_encoded[128])`
- `h_geo [B, L, 594]`

### (3) 与 ESM 融合

- GVP分支投影：`594 -> 256`
- ESM分支投影：`1280 -> 256`
- 门控网络输出 `gate [B,L,256]`
- 融合：
  - `h_gvp_gated = gate * h_gvp_proj`
  - `h_esm_gated = (1-gate) * h_esm_proj`
  - `h_seq = concat(h_gvp_gated, h_esm_gated) = [B,L,512]`

训练时 ESM 也可做模态 dropout（`esm_dropout_prob`）。

## 3.4 模块4：蛋白上下文编码器（Transformer）

`ProteinContextEncoder`：

- 位置编码（默认 sinusoidal）
- 4 层 `GatedTransformerBlock`
  - `LayerNorm -> Gated MHA -> Residual`
  - `LayerNorm -> SwiGLU FFN -> Residual`
- 最终 `LayerNorm`

输出：`h_prot_global [B,L,512]`

## 3.5 模块5：配体 3D 编码器 + 全局 token

`Ligand3DEncoder` 输入：

- `ligand_x [N_total_atoms,96]`
- `ligand_pos [N_total_atoms,3]`
- `ligand_edge_index` / `ligand_edge_attr`
- `ligand_batch`

处理后输出：

- `h_ligand_nodes [N_total_atoms,512]`

然后 scatter 到 batch：

- `h_ligand [B,Max_M,512]`

再拼接可学习全局 token：

- `h_ligand_with_global [B, 1+Max_M, 512]`
- `ligand_mask_with_global [B, 1+Max_M]`

## 3.6 模块6：顺序双向 Cross-Attention

每层执行两步（可堆叠多层，配置中 V8 为 4 层）：

1. `prot_to_lig`：蛋白 query 配体，更新蛋白
2. `lig_to_prot`：配体 query 更新后的蛋白，更新配体

并在 CrossAttentionBlock 中两次应用 `q_mask`，降低 padding 噪声传播。

输出：

- `h_prot_fused [B,L,512]`
- `h_lig_fused [B,1+Max_M,512]`

## 3.7 模块7：Late Injection + 分类头

- 将融合蛋白特征与 physchem 编码再次拼接：
  - `x_final = concat(h_prot_fused[512], physchem_encoded[128]) = [B,L,640]`
- 分类头：`640 -> 256 -> 3`

最终输出 `logits [B,L,3]`。

---

## 4. 输入特征详细拆解

## 4.1 蛋白理化特征 `62D`（`protein_physchem`）

组成：

- `0-19`：V2 basic 理化属性（20维）
- `20-27`：V2 sequence 局部序列特征（8维）
- `28-35`：V2 hbond network 特征（8维）
- `36-55`：残基类型 one-hot（20维）
- `56-58`：二级结构 3态 one-hot（Helix/Sheet/Coil）
- `59-61`：极性分类 one-hot（Nonpolar/Polar/Charged）

## 4.2 蛋白空间标量 `72D`（`protein_spatial_scalar`）

组成：

- `0-5`：SASA 特征（6维）
- `6-19`：Surface 特征（14维）
- `20-45`：Structural Geometry（26维）
  - 含 DSSP 8类 one-hot + Frenet 微分几何
- `46-57`：Enhanced V2（12维）
  - 口袋度3 + 柔性3 + 环境理化6
- `58-61`：局部密度统计（4维）
- `62`：到质心距离（1维）
- `63`：表面法向量幅值（1维）
- `64-67`：局部电荷密度（4维）
- `68-71`：局部疏水性（4维）

## 4.3 蛋白空间向量 `6×3`（`protein_spatial_vector`）

每个残基 6 个方向向量：

1. Sidechain Base Vector
2. Sidechain CoM Vector
3. Aromatic Normal Vector
4. H-bond Donor Vector
5. H-bond Acceptor Vector
6. Cavity Pointing Vector

## 4.4 运行时 GVP 向量 `4×3`（`protein_gvp_vectors`）

由数据集在线计算：

- 向量1：`CA -> 该残基原子质心`
- 向量2-4：`CA -> 前3个原子`

与 `6×3` 空间向量拼接后送入 GVP，共 `10×3`。

## 4.5 配体原子特征 `96D`（`ligand_x`）

组成：

- `41D`（V3 原子级）
  - 其中 `41 = 35(V2) + 6(高级原子特征)`
- `45D` 全局 tag（对每个原子广播）
- `10D` 形状特征（对每个原子广播）

并配合 `ligand_pos [N,3]` 与图边特征进入 `Ligand3DEncoder`。

---

## 5. 关键超参数（当前配置）

来自 `configs/pocket_gvp_v8.yaml`：

- `d_esm=1280`
- `d_phys=62`
- `d_spatial_scalar=72`
- `d_ligand_atom=96`
- `n_gvp_vectors=10`
- `n_gvp_vector_hidden=20`
- `d_model=512`
- `n_gvp_layers=3`
- `n_transformer_layers=4`
- `n_cross_attention_layers=4`

---

## 6. 标签与训练目标

数据集优先使用 VDW 调整距离映射标签：

- 类0：`distance_vdw_adjusted < 0.5Å`
- 类1：`0.5Å <= distance_vdw_adjusted < 4.5Å`
- 类2：`>= 4.5Å`

如无该字段，回退到旧阈值规则。

---

## 7. 实现注意点（阅读/调参必看）

1. `prepare_model_inputs_v8` 里配体图目前是全连接构边，`edge_attr` 近似模板特征，不是严格化学键图。  
2. `ligand_to_protein` 目前主要作为 shape 占位（用于读取 `B, Max_M`），内容本身未直接参与核心融合计算。  
3. V8 同时有“早期注入 physchem（进 h_geo）”和“晚期注入 physchem（分类前 concat）”，属于双注入策略。

---

## 8. 一句话总结

V8 的本质是：**ESM 语义 + 蛋白几何（72D标量 + 10向量GVP）+ 配体3D图表示** 的多模态分层融合网络，其中蛋白侧通过 GVP 与门控Transformer建模，蛋白-配体侧通过顺序双向Cross-Attention交互，最终做残基级三分类。
