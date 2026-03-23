# V7m3 模型特征使用说明（Complete / ESM2 / Ligand）

## 1. 先说结论

- `v7m3` **不是新模型结构**，而是在 `v7/v7m2` 的同一模型结构上增加了 **Cluster-Weighted Loss**（按簇大小加权损失）。
- 因此，`v7m3` 的特征输入方式与 `PocketGVPFusionV7` 一致：
  - 蛋白 `Complete feature`（`protein_scalar`）
  - 蛋白 `ESM2 feature`（`esm_embeddings`）
  - 配体 `Ligand feature`（`atom_features + coords`）
- 在当前实现中，这三路输入是联合使用的，数据集会同时检查三类文件是否存在。

---

## 2. V7m3 代码入口与真实模型

- 训练脚本：`scripts/train/train_v7m3.py`
- 模型类：`src/models/pocket_gvp_fusion_v7.py` 中的 `PocketGVPFusionV7`
- 数据集：`src/data/pocket_dataset_v6.py` 中的 `PocketDatasetV6`
- 批处理与模型输入拼装：`src/data/collate.py`

这意味着：你训练 `v7m3` 时，真正前向传播的仍是 `PocketGVPFusionV7`，`v7m3` 主要改的是 loss 计算策略，不是输入模态定义。

---

## 3. 三类特征分别怎么被读取与使用

### 3.1 Complete feature（蛋白结构/理化特征）

#### 数据来源

- 目录来自配置：`data.protein_features_dir`
- 推荐在 `v7m3` 使用：`complete_features_v2`
- 每个样本文件形如：
  - `data/<protein_features_dir>/<YEAR>/<PDB_ID>/<pdb_id>_protein.pt`

#### 读取逻辑

`PocketDatasetV6` 会优先读取：

1. `complete_features_v2`（通常是 94 维）
2. 若不存在，再退回 `complete_features`（通常是 82 维）

#### 进入模型后的作用

- 一路进入 `physchem_encoder`（MLP 编码）
- 一路作为 GVP 的 `node_s` 输入（几何编码）
- 最后在分类头前做 late injection（与融合后的蛋白表示拼接）

> 简单说：`complete feature` 在 v7 里不是“辅助输入”，而是主干输入之一。

---

### 3.2 ESM2 feature（蛋白语言模型特征）

#### 数据来源

- 固定目录：`data/esm2_latents`
- 文件形如：
  - `data/esm2_latents/<YEAR>/<PDB_ID>/<pdb_id>_protein.pt`

#### 读取逻辑

`PocketDatasetV6` 支持两种键：

- `embeddings`
- `latents`

常见维度是 `1280`（ESM2）。

#### 进入模型后的作用

- 输入 `esm_proj` 投影到融合维度
- 训练时会进行 ESM 模态 dropout（`esm_dropout_prob`）
- 与 GVP/physchem 分支通过 channel-wise gate 融合（`modality_gate`）

> 简单说：ESM2 是与几何分支并列的核心模态，通过门控自适应融合。

---

### 3.3 Ligand feature（配体特征）

#### 数据来源

- 目录来自配置：`data.ligand_features_dir`
- 推荐在 `v7m3` 使用：`ligand_features_v4`
- 文件形如：
  - `data/<ligand_features_dir>/<YEAR>/<PDB_ID>/<pdb_id>_ligand.pt`

#### 读取逻辑

`PocketDatasetV6` 主要读取：

- `atom_features`（例如 v4 常见 96 维）
- `coords`（3D 坐标）

`collate.py` 会把配体原子拼成 batched graph（节点、边、坐标、batch 索引）。

#### 进入模型后的作用

- `Ligand3DEncoder` 对配体图进行 3D 编码
- 加一个 learnable global ligand token
- 与蛋白 token 做双向 cross-attention（protein ↔ ligand）

> 简单说：ligand 分支决定了“蛋白-配体交互上下文”，是 v7 架构性能关键点之一。

---

## 4. 你最关心的“怎么用”：配置联动规则

在这个项目里，三类特征能否正常训练，核心在于**目录 + 维度配置一致**。

### 推荐组合（V7/V7m3）

```yaml
data:
  protein_features_dir: "complete_features_v2"
  ligand_features_dir: "ligand_features_v4"

model:
  d_phys: 94
  d_esm: 1280
  d_ligand_atom: 96
```

### 可兼容旧组合（V6 风格）

```yaml
data:
  protein_features_dir: "complete_features"
  ligand_features_dir: "ligand_features_v3"

model:
  d_phys: 82
  d_esm: 1280
  d_ligand_atom: 80
```

### 关键一致性检查

1. `protein_features_dir` 选了哪套特征，`d_phys` 必须匹配其真实维度。
2. `ligand_features_dir` 选了哪套特征，`d_ligand_atom` 必须匹配 `atom_features` 维度。
3. ESM2 通常为 `1280`，`d_esm` 一般保持 `1280`。

---

## 5. 训练与推理如何落地

### 训练（v7m3）

```bash
python scripts/train/train_v7m3.py --config configs/pocket_gvp_v7.yaml
```

多卡示例：

```bash
torchrun --nproc_per_node=4 scripts/train/train_v7m3.py \
  --config configs/pocket_gvp_v7.yaml --compile
```

### 推理（v7 推理脚本，读取 checkpoint 内 config）

```bash
python scripts/infer/infer_v7.py \
  --checkpoint <your_checkpoint.pt> \
  --pdb_id 1jmi \
  --data_root data
```

注意：`infer_v7.py` 会优先读取 checkpoint 里的 `config`，其中 `data.protein_features_dir` / `data.ligand_features_dir` 会直接影响加载路径。

---

## 6. 常见报错与排查（实用）

1. **`Missing files for <pdb_id>`**
   - 三路文件至少缺一路（complete / esm2 / ligand）。
   - 检查目录层级是否是 `<YEAR>/<PDB_ID>/xxx.pt`。

2. **线性层维度不匹配（matmul shape error）**
   - 大概率是 `d_phys` 或 `d_ligand_atom` 和真实特征维度不一致。

3. **切换到旧特征后训练崩溃**
   - 记得同时改 `data.*_dir` 与 `model.d_*`，不能只改目录不改维度。

---

## 7. 一句话总览

`v7m3 = v7 模型输入（Complete + ESM2 + Ligand 三模态融合） + Cluster-Weighted Loss`。你真正需要保证的是：**特征目录和模型维度严格对齐**。

---

## 8. `complete_features_v2` 的 94 维逐维索引（0-based）

下面是按**实际拼接顺序**列出的 94 维明细（索引从 0 开始）：

- 拼接定义来源：`preparation/protein/extract_complete_features_v2.py`
- 拼接公式：`42 + 14 + 26 + 3 + 3 + 6 = 94`

### 8.1 Dim 0-41：Physicochemical（42维）

#### A. Basic（Dim 0-19）

- `0`: `hydrophobicity_norm`（Kyte-Doolittle / 4.5）
- `1`: `molecular_weight_norm`（分子量 / 204.2）
- `2`: `side_chain_volume_norm`（侧链体积 / 227.8）
- `3`: `polarity_norm`（极性类别 / 3）
- `4`: `hbond_donors_norm`（供体数 / 4）
- `5`: `hbond_acceptors_norm`（受体数 / 2）
- `6`: `aromaticity_flag`（是否芳香）
- `7`: `charge_at_ph7`（pH=7 净电荷）
- `8`: `polarity_onehot_nonpolar`
- `9`: `polarity_onehot_polar_uncharged`
- `10`: `polarity_onehot_positive`
- `11`: `polarity_onehot_negative`
- `12`: `has_pka`（是否可电离）
- `13`: `pka_norm`（pKa / 14）
- `14`: `acidic_flag`（pKa < 7）
- `15`: `basic_flag`（pKa > 7）
- `16`: `hydro_class_strong_hydrophobic`
- `17`: `hydro_class_weak_hydrophobic`
- `18`: `hydro_class_weak_hydrophilic`
- `19`: `hydro_class_strong_hydrophilic`

#### B. SASA（Dim 20-25）

- `20`: `total_sasa_norm`（total_sasa / 300）
- `21`: `relative_sasa_clipped`（`min(total_sasa/max_sasa, 1)`）
- `22`: `polar_sasa_norm`（polar_sasa / 200）
- `23`: `apolar_sasa_norm`（apolar_sasa / 200）
- `24`: `is_exposed`（relative_sasa > 0.5）
- `25`: `is_buried`（relative_sasa < 0.2）

#### C. Sequence（Dim 26-33）

- `26`: `is_aromatic_residue`
- `27`: `is_charged_residue`
- `28`: `is_polar_residue`
- `29`: `local_aromaticity`（窗口统计）
- `30`: `local_helix_fraction`（窗口统计）
- `31`: `local_turn_fraction`（窗口统计）
- `32`: `local_sheet_fraction`（窗口统计）
- `33`: `relative_position`（残基序号 / 序列长度）

> 说明：代码里计算了 `is_terminal`，但**未写入最终8维**，所以不在 26-33 中。

#### D. H-bond Network（Dim 34-41）

- `34`: `hbond_feature_0`
- `35`: `hbond_feature_1`
- `36`: `hbond_feature_2`
- `37`: `hbond_feature_3`
- `38`: `hbond_feature_4`
- `39`: `hbond_feature_5`
- `40`: `hbond_feature_6`
- `41`: `hbond_feature_7`

说明（很重要）：

- 这 8 维来自 `compute_hbond_features(...)`（可选外部模块）。
- 当前仓库内未包含该模块实现文件；若模块不可用，代码会保留这 8 维为默认值（通常为 0）。

### 8.2 Dim 42-55：Surface（14维）

- `42`: `local_curvature`
- `43`: `convexity`
- `44`: `surface_roughness`
- `45`: `local_density`
- `46`: `surface_depth`
- `47`: `electrostatic_potential`
- `48`: `charge_density`
- `49`: `polarity_ratio`
- `50`: `hydrophobicity_local`
- `51`: `hydrophobic_ratio`
- `52`: `surface_hydrophobicity`
- `53`: `mapping_nearest_charge`
- `54`: `mapping_weighted_charge`
- `55`: `mapping_distance_weighted_charge`

### 8.3 Dim 56-81：Structural Geometry（26维）

#### A. DSSP one-hot（Dim 56-63）

- `56`: `dssp_H`（α-helix）
- `57`: `dssp_B`（β-bridge）
- `58`: `dssp_E`（β-strand）
- `59`: `dssp_G`（3-10 helix）
- `60`: `dssp_I`（π-helix）
- `61`: `dssp_T`（Turn）
- `62`: `dssp_S`（Bend）
- `63`: `dssp_coil`（`-` / loop）

#### B. Differential geometry（Dim 64-81）

- `64`: `curvature_raw`
- `65`: `torsion_raw`
- `66`: `kappa_rbf_center_0.000`
- `67`: `kappa_rbf_center_0.286`
- `68`: `kappa_rbf_center_0.571`
- `69`: `kappa_rbf_center_0.857`
- `70`: `kappa_rbf_center_1.143`
- `71`: `kappa_rbf_center_1.429`
- `72`: `kappa_rbf_center_1.714`
- `73`: `kappa_rbf_center_2.000`
- `74`: `tau_rbf_center_-1.000`
- `75`: `tau_rbf_center_-0.714`
- `76`: `tau_rbf_center_-0.429`
- `77`: `tau_rbf_center_-0.143`
- `78`: `tau_rbf_center_0.143`
- `79`: `tau_rbf_center_0.429`
- `80`: `tau_rbf_center_0.714`
- `81`: `tau_rbf_center_1.000`

### 8.4 Dim 82-93：Enhanced V2（12维）

#### A. Pocketness（Dim 82-84）

- `82`: `pocket_cx`（Convexity Index）
- `83`: `local_packing_density`
- `84`: `shell_density_ratio`

#### B. Flexibility（Dim 85-87）

- `85`: `b_factor_norm`
- `86`: `local_flexibility_std`
- `87`: `rse`（Relative Surface Exposure）

#### C. Environmental（Dim 88-93）

- `88`: `env_hydrophobicity`
- `89`: `hydrophobic_contrast`
- `90`: `env_positive_charge`
- `91`: `env_negative_charge`
- `92`: `env_hbond_donors`
- `93`: `env_hbond_acceptors`

---

## 9. 这 94 维如何进入模型（补充）

- 94维 `complete_features_v2` 在数据集中读取为 `protein_scalar`。
- 这 94 维全部作为 GVP 的 `node_s` 标量通道输入。
- 同一份 `protein_scalar` 也会进入 `physchem_encoder` 做 MLP 编码（得到 `protein_scalar_encoded`）。
- 最终在 v7 中会做两次融合：
  1. **中间融合**：`concat(h_s_gvp, v_inv, protein_scalar_encoded)`
  2. **分类前融合**：`concat(h_prot_fused, protein_scalar_encoded)`

这也是为什么同一个 `complete_features_v2` 会同时影响几何分支与语义分支。
