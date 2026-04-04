# Codex Modification Summary

最后更新：2026-03-30

本文档汇总本轮对 `/home/husrcf/Code/ProtBind/PLC` 做过的所有主要代码修改，按模块划分，并明确说明新增文件、修改文件、设计目的、当前状态与已知注意事项。

---

## 1. 训练链路与配置

### 1.1 新增 V10 训练配置与模型

新增文件：

- [configs/pocket_gvp_v10.yaml](/home/husrcf/Code/ProtBind/PLC/configs/pocket_gvp_v10.yaml)
- [src/models/pocket_gvp_fusion_v10.py](/home/husrcf/Code/ProtBind/PLC/src/models/pocket_gvp_fusion_v10.py)
- [scripts/train/train_v10.py](/home/husrcf/Code/ProtBind/PLC/scripts/train/train_v10.py)

主要改动：

- 新增 `v10` 配置，显式切换到：
  - 蛋白特征目录：`complete_features_v4`
  - 配体特征目录：`ligand_features_v5`
- 新增 `PocketGVPFusionV10`，继承 `PocketGVPFusionV9m2`，只更新输入维度：
  - `d_spatial_scalar = 89`
  - `d_ligand_atom = 109`
  - `n_gvp_vectors = 12`
- 新增 `train_v10.py`：
  - 使用 `PocketGVPFusionV10`
  - 使用 `pocket_collate_fn_v10`
  - 使用 `prepare_model_inputs_v10`
  - 默认 split 文件改为 `configs/splits/splits.txt`

### 1.2 V10 AMP 与 resume 逻辑补齐

修改文件：

- [scripts/train/train_v10.py](/home/husrcf/Code/ProtBind/PLC/scripts/train/train_v10.py)

主要改动：

- 新增 `--amp` CLI 参数
- AMP 开关逻辑修改为：
  - CLI `--amp` 优先
  - 否则回退到 `config['training']['use_amp']`
- resume 时恢复：
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict`
  - `scaler_state_dict`（仅在 `use_amp=True` 时）
  - `epoch`
  - `best_val_loss`
  - `patience_counter`

### 1.3 V10 验证逻辑修复

修改文件：

- [scripts/train/train_v10.py](/home/husrcf/Code/ProtBind/PLC/scripts/train/train_v10.py)

问题：

- `validate()` 里错误地向 `compute_segmentation_metrics()` 传递了 `logits=...`
- 当前 `src/metrics/metrics.py` 的 `compute_segmentation_metrics()` 不接受这个参数

修复：

- 改为：
  - `compute_segmentation_metrics(preds, labels, mask)`
  - `compute_enhanced_metrics(logits, labels, mask, ...)`
- 再将两者结果合并

### 1.4 V10 当前与 V9m2 的差异

现状：

- `v10` 已可训练、可 resume、可使用 AMP
- 但日志系统仍比 `v9m2` 简化，尚未完全补齐：
  - `checkpoint_latest.pt`
  - `Health/*`
  - `Gradients/*`
  - `Val_new / Val_original` 命名空间

说明：

- 当前 `v10` 目标是先打通新特征链路
- 日志体系仍然偏“最小可用”

---

## 2. 数据批处理与 collate

### 2.1 兼容多代特征的 collate 层

修改文件：

- [src/data/collate.py](/home/husrcf/Code/ProtBind/PLC/src/data/collate.py)

主要改动：

- 保留原有：
  - `pocket_collate_fn`
  - `prepare_model_inputs`
  - `pocket_collate_fn_v8`
  - `prepare_model_inputs_v8`
- 新增：
  - `pocket_collate_fn_v10`
  - `prepare_model_inputs_v10`
- 新增兼容入口：
  - `pocket_collate_fn_auto`
  - `prepare_model_inputs_auto`

其中：

- `v10` 支持：
  - 动态蛋白 `spatial_scalar` 维度（当前 89D）
  - 动态蛋白 `spatial_vector` 通道数（当前 8x3）
  - 动态配体特征维度（当前 109D）
- `auto` 入口只作为备用，主要用于未来消融/兼容实验

### 2.2 修复旧 collate 中重复追加边列表的问题

修改文件：

- [src/data/collate.py](/home/husrcf/Code/ProtBind/PLC/src/data/collate.py)

问题：

- 旧版 `pocket_collate_fn()` 中曾出现重复初始化与重复 append：
  - `ligand_edge_index_list`
  - `ligand_edge_attr_list`

修复：

- 删除重复代码，只保留一组边列表的初始化和 append

---

## 3. 配体特征链路

### 3.1 在旧链路中下沉真实 bond graph 到 V2

修改文件：

- [preparation/ligand/ligand_features_v2.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v2.py)

主要改动：

- 新增 `extract_bond_graph(mol)`
- 新增 `get_bond_type_features(bond)`
- 在 `extract_ligand_atom_features()` 中直接基于最终工作分子 `mol` 提取：
  - `edge_index`
  - `edge_attr`

边特征定义（6D）：

- single
- double
- triple
- aromatic
- conjugated
- in_ring

目的：

- 确保 `edge_index/edge_attr` 与 `atom_features/coords` 原子顺序完全一致

### 3.2 修复 Gasteiger/LogP 口径

修改文件：

- [preparation/ligand/ligand_features_v2.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v2.py)

问题：

- 原先直接在未 `sanitize` 的 RDKit 分子上计算：
  - `Gasteiger charge`
  - `Crippen LogP contribution`
- 导致某些样本出现：
  - `NaN`
  - 被置零
  - 与历史数据不一致

修复：

- 新增 `make_sanitized_descriptor_mol(mol)`
- 电子描述符改为在 sanitize 副本上计算：
  - `gasteiger_charges`
  - `logp_contribs`
  - `tpsa_contribs`

结果：

- 对 `1a5g` 的验证中：
  - `gasteiger_charges` 已与历史数据完全一致
  - `logp_contribs` 已与历史数据完全一致

### 3.3 构象柔性分析默认改为宽松可降级

修改文件：

- [preparation/ligand/ligand_features_v2.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v2.py)
- [preparation/ligand/ligand_features_v3.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v3.py)
- [preparation/ligand/extract_ligand_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/extract_ligand_features_v4.py)

主要改动：

- 增加 `verbose` / `num_threads` 参数透传
- 默认 suppress 逐分子打印，避免多进程刷屏
- 支持宽松模式：
  - `MMFF` 全失败时保留 embedded conformers
  - 不再整条样本直接报废（仅在 `strict=False` 下）

验证结果：

- `1a5g` 在 `strict=False` 下可以得到非零：
  - `rmsf`
  - `flexibility_flag`

### 3.4 调整 pypropel 导入优先级

修改文件：

- [preparation/ligand/extract_ligand_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/extract_ligand_features_v4.py)

主要改动：

- 现在优先尝试导入环境里的 `pypropel`
- 只有环境里不存在时，才 fallback 到本地 `pypropel_unibio`

备注：

- 后续排查表明：
  - 当前环境版与本地 `pypropel_unibio` 在 `get_global_tag()` 上行为一致
  - `global_features` 与历史数据的偏差不是由“环境版 vs 本地版”造成的

### 3.5 新建独立 ligand 特征 V4 模块

新增文件：

- [preparation/ligand/ligand_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v4.py)

设计目标：

- 完全独立，不依赖旧的 `ligand_features_v2.py / ligand_features_v3.py`
- 仅依赖 RDKit 和必要基础库

局部原子特征设计：

- 30D 基础 atom features
- 3D 电子特征
- 2D 柔性特征
- 6D 高级 atom features
- 3D formal charge class
- 3D rotatable-bond context
- 4D metal-binding flags
- 3D stereochemistry flags

合计：

- `54D local atom features`

新增实现内容：

- `formal charge class`
- `rotatable-bond context`
- `metal-binding flag`
- `stereochemistry atom-level flag`

### 3.6 新建独立 ligand 提取 V5 脚本

新增文件：

- [preparation/ligand/extract_ligand_features_v5.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/extract_ligand_features_v5.py)

设计：

- 基于独立 `ligand_features_v4.py` 的 `54D local`
- 再追加：
  - `45D global`
  - `10D shape`

最终：

- `109D per-atom features`

默认行为：

- 默认宽松模式（`strict=False`）
- 显式传 `--strict` 才严格失败

验证结果：

- `1a5g` smoke test 成功
- 输出：
  - `atom_features (87, 109)`
  - `edge_index (2, 180)`
  - `edge_attr (180, 6)`
  - 新增特征字段均存在且非零

### 3.7 清理与扫描 ligand v5 坏文件

执行结果：

- 扫描 [data/ligand_features_v5](/home/husrcf/Code/ProtBind/PLC/data/ligand_features_v5) 后发现仅 1 个坏文件
- 空文件：
  - [4jwr_ligand.pt](/home/husrcf/Code/ProtBind/PLC/data/ligand_features_v5/2011-2019/4jwr/4jwr_ligand.pt)
- 已删除该空文件

目的：

- 修复训练时 `DataLoader` 中的 `EOFError: Ran out of input`

---

## 4. 蛋白特征链路

### 4.1 新建独立 protein 特征 V4 模块

新增文件：

- [preparation/protein/protein_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py)

设计目标：

- 仅使用蛋白自身信息，不引入任何 ligand 侧信息
- 保持旧训练链路所需关键字段一致
- 保留坐标与原子信息：
  - `ca_coords`
  - `all_atom_coords`
  - `all_atom_names`
- 新增更强的 backbone / packing / environment 表征

保留字段：

- `physchem_features`
- `spatial_scalar_features`
- `spatial_vector_features`
- `ca_coords`
- `all_atom_coords`
- `all_atom_names`
- `residue_names`
- `residue_indices`
- `insertion_codes`
- `chain_ids`
- `n_residues`

新增：

- `spatial_scalar_raw_features`
- `spatial_scalar_feature_names`

### 4.2 新增的蛋白特征

新增特征类别：

1. 局部 backbone frame 标量
- `phi_sin`
- `phi_cos`
- `psi_sin`
- `psi_cos`
- `omega_sin`
- `omega_cos`

2. side-chain packing / clash-aware 标量
- 最近侧链质心距离
- packing score
- 局部空隙率 proxy

3. 残基级环境组成直方图
- 邻域芳香残基比例
- 邻域 H-bond donor 残基比例
- 邻域 H-bond acceptor 残基比例
- 邻域 Gly/Pro 比例

4. 残基级金属结合潜力
- self metal binder
- neighbor metal binder density (6A / 10A)

5. 局部各向异性
- 基于邻域 PCA 的 anisotropy 标量

6. 多尺度 cavity direction
- 原先 1 个 cavity direction 升级成 3 个尺度
- `spatial_vector_features` 从 `6x3` 升级为 `8x3`

### 4.3 归一化口径改成训练集级 stats

新增机制：

- `fit_normalization_stats(...)`
- `save_normalization_stats(...)`
- `load_normalization_stats(...)`
- `normalize_spatial_scalar_features(...)`

设计：

- 先提取 `spatial_scalar_raw_features`
- 再通过 JSON 中固定的 `mean/std` 做标准化
- 推理时复用同一份 stats 文件

### 4.4 新建独立 protein 提取 V4 脚本

新增文件：

- [preparation/protein/extract_complete_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_complete_features_v4.py)

提供两个模式：

1. `fit-stats`
- 收集 `spatial_scalar_raw_features`
- 生成 stats JSON

2. `extract`
- 读取 stats JSON
- 正式提取归一化后的 V4 特征

### 4.5 `extract_complete_features_v4.py` 覆盖/跳过机制

修改文件：

- [preparation/protein/extract_complete_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_complete_features_v4.py)

行为：

- 默认：
  - 输出文件已存在时，**直接跳过**
  - 不会先做昂贵提取再判断
- 显式传 `--overwrite`：
  - 才重写

这个逻辑已经前移到真正提取之前：

- 先构造目标 `out_file`
- 再判断 `out_file.exists() and not overwrite`
- 满足就立刻 `SKIP`

### 4.6 蛋白 stats 与小规模提取验证

已生成：

- [protein_v4_stats_new_test.json](/home/husrcf/Code/ProtBind/PLC/configs/protein_v4_stats_new_test.json)

检查结果：

- `feature_names = 89`
- `scalar_means = 89`
- `scalar_stds = 89`
- 无 `NaN/Inf`
- 无全零标准差

并已对 [complete_features_v4_test](/home/husrcf/Code/ProtBind/PLC/data/complete_features_v4_test) 做小规模数值检查：

- 无 `NaN/Inf`
- 无 shape mismatch
- `physchem_features (N, 62)`
- `spatial_scalar_features (N, 89)`
- `spatial_scalar_raw_features (N, 89)`
- `spatial_vector_features (N, 8, 3)`

---

## 5. TODO 文档重写

修改文件：

- [TODO.md](/home/husrcf/Code/ProtBind/PLC/TODO.md)

主要改动：

- 重写为按优先级组织的执行清单
- 引入：
  - `P0 / P1 / P2`
- 包含：
  - 训练链路问题
  - 特征工程改造
  - 架构升级建议
  - 参考文献

后续还补回了原本需要保留的参考文献区段，便于写作引用。

---

## 6. 当前仍保留但未完全收尾的事项

### 6.1 `train_v10.py` 仍未完全追平 `train_v9m2.py`

虽然已具备：

- AMP
- resume
- DDP
- checkpoint 保存
- best model 保存

但仍未完全追平 `v9m2` 的完整实验诊断体系，例如：

- `checkpoint_latest.pt`
- 更细粒度的 `Train/Val/Val_new/Val_original`
- `Health/*`
- `Gradients/*`

### 6.2 `global_features` / `shape_features` 的历史口径仍未完全考古完成

已知：

- `gasteiger/logp` 已修复并对齐历史数据
- `global_features` 仍与老数据不完全一致
- `shape_features` 的原始口径也与老数据存在偏差

但用户已要求当前阶段不再聚焦这部分问题，因此本轮未继续追到完全复刻。

---

## 7. 涉及的主要文件清单

### 新增文件

- [DOCs/CODEX_MODIFICATION_SUMMARY_20260330.md](/home/husrcf/Code/ProtBind/PLC/DOCs/CODEX_MODIFICATION_SUMMARY_20260330.md)
- [configs/pocket_gvp_v10.yaml](/home/husrcf/Code/ProtBind/PLC/configs/pocket_gvp_v10.yaml)
- [preparation/ligand/ligand_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v4.py)
- [preparation/ligand/extract_ligand_features_v5.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/extract_ligand_features_v5.py)
- [preparation/protein/protein_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py)
- [preparation/protein/extract_complete_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_complete_features_v4.py)
- [scripts/train/train_v10.py](/home/husrcf/Code/ProtBind/PLC/scripts/train/train_v10.py)
- [src/models/pocket_gvp_fusion_v10.py](/home/husrcf/Code/ProtBind/PLC/src/models/pocket_gvp_fusion_v10.py)

### 修改文件

- [TODO.md](/home/husrcf/Code/ProtBind/PLC/TODO.md)
- [preparation/ligand/ligand_features_v2.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v2.py)
- [preparation/ligand/ligand_features_v3.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/ligand_features_v3.py)
- [preparation/ligand/extract_ligand_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/ligand/extract_ligand_features_v4.py)
- [src/data/collate.py](/home/husrcf/Code/ProtBind/PLC/src/data/collate.py)

---

## 8. 推荐后续动作

如果继续推进，建议按下面顺序：

1. 用 `ligand_features_v5` 重提取缺失/损坏样本
2. 用 `complete_features_v4` 完成全量蛋白提取
3. 做一次 `v10` 单 batch smoke test
4. 再补齐 `train_v10.py` 的日志/checkpoint 体系，使其接近 `v9m2`

