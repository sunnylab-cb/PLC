# PLC vs PoC 失效分析报告

## 1. 报告目的

这份文档回答一个具体问题：

`PLC` 当前主线为什么在实际表现上不如更早、更简单的 `PoC`？

这里的结论不是基于主观印象，而是基于当前仓库里能验证到的四类证据：

1. 当前有效训练线
2. 当前有效模型定义
3. 当前有效数据与标签路径
4. 当前已有 benchmark 结果

同时，这份报告也包含一个可执行的最小回退实验方案，目标不是“讨论可能原因”，而是把问题尽快定位到具体改动。

---

## 2. 当前有效主线

## 2.1 PLC 的 Source of Truth

当前 `PLC` 真正应当被视为主线的，不是历史文档里的 v2/v6/v8 摘要，而是 v10m4 训练线：

- 训练入口：`scripts/train/train_v10m4.py`
- 模型定义：`src/models/pocket_gvp_fusion_v10m4.py`
- 数据集：`src/data/pocket_dataset_v8.py`
- batch 组装：`src/data/collate.py` 里的 `pocket_collate_fn_v10` 和 `prepare_model_inputs_v10`
- 配置：`configs/pocket_gvp_v10m4.yaml`
- 实际实跑配置：`outputs/pocket_gvp_fusion_v10m4_20260405_205804/config.yaml`

从当前代码可还原出一句话版 pipeline：

```text
complete_features_v4
+ esm2_latents
+ ligand_features_v5
+ PocketDatasetV8
+ pocket_collate_fn_v10
+ PocketGVPFusionV10m4
+ Focal + Contact-only Tversky
```

## 2.2 PoC 的 Source of Truth

当前 `PoC` 主线则更集中：

- 特征提取：`scripts/extract_features.py`
- 训练：`scripts/train.py`
- 数据集：`src/data/pocket_dataset.py`
- batch：`src/data/collate.py`
- 模型：`src/models/pocket_gvp_fusion.py`
- 配置：`configs/pocket_gvp_v2.yaml`

它的 pipeline 远短于 `PLC`：

```text
protein/ligand structure
-> extract_features.py
-> *_features.npz
-> PocketDataset
-> pocket_collate_fn
-> PocketGVPFusion
-> focal loss
```

---

## 3. 结论先行

如果只保留一句结论：

`PLC` 不如 `PoC`，最主要不是因为“模型还不够复杂”，而是因为它在同一条主线上同时改变了监督定义、训练目标、模型结构和正则化强度，导致模型学到的不是更精确的 pocket，而是更宽、更不稳定、优化更困难的 decision boundary。

我认为影响大小从高到低大致是：

1. 标签与训练目标漂移
2. 过强正则化与更保守优化
3. 丢失 PoC 中很重要的 multi-chain inductive bias
4. 架构膨胀过度，缺少受控逐项消融
5. pipeline 版本漂移，训练事实和文档/命名长期不一致

---

## 4. Part I: 差异矩阵

## 4.1 数据表示与标签定义

| 维度 | PoC | PLC v10m4 | 对性能的潜在影响 |
|------|-----|-----------|------------------|
| 样本格式 | 单个 `*_features.npz` 同时含 protein / ligand / labels | protein / esm / ligand / distance labels 分离存放 | PLC 更容易出现静默不一致 |
| 标签生成位置 | 特征提取阶段直接生成 | 训练阶段从独立距离字典生成 | PLC 的监督更容易与特征版本漂移 |
| Contact 阈值 | `<4.0Å` | `<5.0Å` | PLC 会学到更大的正样本区域 |
| Near 阈值 | `4.0-6.0Å` | `5.0-8.0Å` | PLC 的正负边界更模糊 |
| 标签实现 | `extract_features.py` 里直接 `thresholds=[4.0, 6.0]` | `PocketDatasetV8._distances_to_labels()` 里按 `5.0/8.0` 切 | 两边 supervision 不是一个任务 |
| 距离语义 | 训练特征与标签在同一条抽取链中生成 | 文件名写 `dcenter`，但当前主线监督实际是 `d_min` 语义 | 命名与真实监督不一致，增加误判成本 |

### 关键判断

`PoC` 训练的是更“紧”的 contact residue 检测问题；`PLC` 训练的是更“宽”的 residue proximity 问题。  
这会直接改变模型的 pocket 几何偏好。

---

## 4.2 蛋白与配体输入复杂度

| 维度 | PoC | PLC v10m4 | 备注 |
|------|-----|-----------|------|
| ESM | 1280 | 1280 | 当前 v10m4 实跑仍是 ESM2，不是更强的外部语言模型优势 |
| 蛋白标量 | 35D | 62D physchem + 89D spatial scalar | PLC 输入更宽 |
| 蛋白向量 | 4x3 | 12x3 | PLC 向量通道大幅增加 |
| 配体特征 | 64D | 109D | PLC 配体表示显著更复杂 |
| 配体边特征 | 6D | 6D chemical + 8D RBF = 14D | PLC 在线又加一层几何 |

### 关键判断

`PLC` 的问题不是“信息不够”，而是“信息过多且同时变化”。  
如果这些附加特征没有被更强监督和更清晰 ablation 约束，模型更容易学到 dataset-specific shortcut，而不是更好的泛化规律。

---

## 4.3 架构差异

| 维度 | PoC | PLC v10m4 | 影响 |
|------|-----|-----------|------|
| GVP 输入 | 35D scalar + 4x3 vector | 89D scalar + 12x3 vector | PLC 的 GVP 学习问题更难 |
| 蛋白上下文 | 4 层 Transformer + chain-aware RoPE | 4 层 RoPE Transformer + protein distance bias | PLC 更复杂，但未保留链信息 |
| 多链建模 | 有 `chain_ids`、`is_same_chain`、chain-gap RoPE | 无 chain-aware 机制 | 这是明确回退 |
| 配体编码 | `LigandEncoder` | `Ligand3DEncoder` + global token | PLC 更强但更难训稳 |
| 跨模态融合 | 单向 protein query ligand | 双向 cross-attn + AttnRes | PLC 参数和耦合度更高 |
| 分类头输入 | `512 + 35` | `512 + 128` | PLC 晚期再次强化 physchem 路径 |

### 关键判断

`PoC` 的结构是“足够强但仍然短路径”的设计。  
`PLC` 则进入了“每一层都想再补一点信息”的阶段。  
在没有严密对照实验时，这种架构膨胀非常容易变成训练不稳定和真实增益不清。

---

## 4.4 训练与优化差异

| 维度 | PoC | PLC v10m4 | 影响 |
|------|-----|-----------|------|
| batch size | 32 | 18 | PLC 梯度统计更噪 |
| learning rate | `1e-4` | `5e-5` | PLC 更保守 |
| weight decay | `0.01` | `0.05` | PLC 正则更强 |
| 基础 dropout | `0.1` | `0.15` | PLC 更强 |
| ESM dropout | `0.0` | `0.15` | PLC 更强 |
| physchem dropout | 无 | `0.2` | PLC 更强 |
| GVP output dropout | 无 | `0.15` | PLC 更强 |
| coordinate noise | 无 | `0.5Å` | PLC 更强 |
| stochastic depth | 无 | `0.15` | PLC 更强 |
| loss | focal | focal + contact-only Tversky | PLC 目标更偏 recall |

### 关键判断

`PLC` 不是“正则稍强”，而是几乎把所有常见 regularization 都叠上去了。  
如果模型已经因为标签更宽而倾向 over-predict，再配 recall-oriented Tversky，它就很容易出现：

- train 指标看起来不错
- val / external benchmark precision 下降
- DCC / IoU 改善不明显

---

## 4.5 实际 benchmark 表现的信号

当前 `PLC` 已有的 benchmark 结果：

### COACH420

- `DCC_Success = 0.3002`
- `Mean_Prec = 0.3573`
- `Mean_Rec = 0.6209`
- `Mean_F1 = 0.4323`

### HOLO4K

- `DCC_Success = 0.1478`
- `Mean_Prec = 0.2274`
- `Mean_Rec = 0.3537`
- `Mean_F1 = 0.2656`

### 这些数字说明什么

COACH420 上最明显的信号是：

- recall 明显高于 precision

这与当前主线的监督和 loss 是高度一致的：

- Contact 定义更宽
- Tversky 设定 `beta > alpha`
- class weight 继续拉高 Contact

也就是说，当前模型行为并不是“完全没学到”，而是“学到了偏大的 pocket”。

这是一种典型的 supervision / objective mismatch，不是单纯网络容量不足。

---

## 5. Part II: 我认为最可能拉低性能的改动，按优先级排序

## 5.1 第一名：标签漂移 + 目标漂移

这是我认为最关键的原因。

### 证据

- `PoC`：`Contact < 4Å`, `Near < 6Å`
- `PLC`：`Contact < 5Å`, `Near < 8Å`
- `PLC` loss 还进一步对 Contact recall 施加偏置

### 影响机制

更宽的 Contact / Near 定义会带来三个直接后果：

1. 正样本区域扩大
2. 正负边界更模糊
3. 残基级三分类与 pocket-level localization 的耦合变差

最终模型会更倾向于：

- 覆盖更多 pocket 周边残基
- 召回上升
- 精确定位下降

在 external benchmark 上，这通常表现为：

- Precision 降
- DCC 不一定升
- IoU 不稳定

### 我对这项的判断

如果只能先改一件事，我会先改这里，而不是先改模型。

---

## 5.2 第二名：过强正则叠加导致欠收敛

当前 `PLC` 同时打开了：

- dropout
- attn dropout
- classifier dropout
- stochastic depth
- esm dropout
- physchem dropout
- gvp output dropout
- coordinate noise
- 较高 weight decay
- 更小 learning rate

### 影响机制

这些手段单独看都合理，但同时开启时会带来两个问题：

1. 有效监督信号被削弱
2. 不同模态在训练期被随机破坏，融合层更难学到稳定分工

特别是 `PLC` 的输入本来已经更宽、更杂，再加大量 sample-level dropout，本质上是在让一个更复杂的系统用更弱的梯度学习。

### 为什么 PoC 更稳

`PoC` 的多模态设计更保守：

- 没有这么多同时启用的 dropout / augmentation
- 没有 recall-biased auxiliary loss
- 优化更激进、路径更短

所以它更容易收敛到一个“朴素但稳”的解。

---

## 5.3 第三名：丢掉了 PoC 的 chain-aware inductive bias

这是一个非常重要但容易被忽略的回退。

`PoC` 显式引入了：

- `chain_ids`
- GVP edge 的 `is_same_chain`
- chain-gap RoPE position ids

`PLC v10m4` 当前主线没有这套东西。

### 为什么这很重要

对于多链复合物：

- 顺序邻近不一定等于同链邻近
- 空间接近也不一定表示同一条多肽链上的局部关系

没有 chain-aware 建模时，模型更容易：

- 在 protein self-attention 里跨链混淆
- 在图边上把“跨链近距离”当成与“同链局部结构”同等重要

这会削弱 protein context encoder 的结构归纳偏置。

### 我的判断

这项不是第一原因，但它是一个非常具体、非常像“PLC 为了升级主线而意外丢掉的有效偏置”。

---

## 5.4 第四名：架构同时改太多，缺乏干净控制变量

`PLC v10m4` 当前主线不是只加了一个模块，而是同时引入了：

- 更宽的蛋白特征
- 更宽的配体特征
- 更强的配体图
- protein distance bias
- bidirectional cross-attention
- protein-side AttnRes
- ligand-side AttnRes
- 多层正则
- 新 loss

### 影响机制

当主线一次改太多时，即使最终效果下降，也很难知道到底是哪一项造成的：

- supervision 变差？
- 正则过重？
- 配体图更复杂但没带来增益？
- bidirectional fusion 更难训稳？
- distance bias 与 GVP 表示冗余？

这会造成一种假象：

- 每项改动单独看都“合理”
- 组合后却没有形成更好的整体

### 我的判断

`PLC` 当前最缺的不是新模块，而是“删回去，看哪一项真的有用”的纪律。

---

## 5.5 第五名：pipeline 版本漂移造成高风险的静默不一致

从工程角度看，`PLC` 还有一个问题：  
它不是单一主线稳步演进，而是历史版本长期共存：

- v2/v3/v5/v6/v7/v8/v9/v10/v10m1/v10m2/v10m3/v10m4

这带来三个风险：

1. 名字和事实不一致
2. 文档和代码不一致
3. 数据目录与训练脚本不一致

最典型的例子就是：

- 文件名还叫 `dcenter`
- 但当前真实监督语义已经是 `d_min`

这种漂移即使不是直接 bug，也会让：

- 实验设计更难对齐
- 错误更难排查
- “为什么变差”更难归因

---

## 5.6 第六名：一些复杂机制当前并没有真正提供收益

这里指的是两类东西：

- NMA augmentation 框架存在，但当前实际运行 `nma_n_confs=1`
- cluster-weighted loss 框架存在，但当前 split 没有 cluster size 第 4 列，实际上不生效

它们本身不是当前性能下降的主因，因为基本没真正起作用。  
但它们会增加维护复杂度，使人误以为系统已经做了更充分的 regularization / reweighting。

---

## 6. 为什么我认为问题首先在 supervision，而不是首先在模型

如果一个模型只是“容量不够”，你通常会看到：

- recall 和 precision 都不高
- 各种复杂改造有机会带来明显提升

但 `PLC` 的现有行为更像：

- recall 被推高
- precision 跟不上
- pocket 更容易做大而不是做准

这更符合“任务定义和优化方向发生偏移”的画像，而不符合“模型本身还不够强”的画像。

换句话说：

- `PoC` 更像在解一个清楚的 contact residue detection 问题
- `PLC` 更像在解一个带 recall 偏置的 broader residue proximity 问题

当 benchmark 最终按 pocket-level 准确定位打分时，后者不一定占优。

---

## 7. Part III: 最小回退实验方案

目标不是“大改”，而是用最少实验次数确定到底哪一项让 `PLC` 掉点。

原则：

1. 一次只改一类因素
2. 先改 supervision / objective，再改 architecture
3. 所有实验固定同一个 split、同一个 benchmark 脚本、同一个 random seed
4. 每次只关心 external benchmark，不只看 val loss

---

## 7.1 实验总路线

建议顺序：

### Stage A: 先验证是不是 supervision / objective 问题

1. `A0`: 当前 v10m4 复现实验
2. `A1`: 只把标签阈值从 `5/8` 改回 `4/6`
3. `A2`: 在 `A1` 基础上去掉 Tversky，只保留 focal
4. `A3`: 在 `A2` 基础上把 class weight 改得接近 `PoC`

### Stage B: 再验证是不是 regularization 过重

5. `B1`: 去掉 `coord_noise`
6. `B2`: 去掉 `gvp_dropout`
7. `B3`: 去掉 `physchem_dropout`
8. `B4`: 把全局 dropout / attn_dropout / stochastic_depth 降回 `PoC` 水平

### Stage C: 再看结构复杂度

9. `C1`: 去掉 distance-biased attention
10. `C2`: 把 bidirectional cross-attn 改回 protein<-ligand 单向
11. `C3`: 关掉 AttnRes
12. `C4`: 在 PLC 上恢复 chain-aware 机制

---

## 7.2 具体实验设计

## 实验 A0：当前 v10m4 基线

目的：

- 固定一条你以后都拿来对比的 reference

固定条件：

- 当前实跑 config：`outputs/pocket_gvp_fusion_v10m4_20260405_205804/config.yaml`
- split：`configs/splits/splits.txt`
- benchmark：同一套 COACH420 / HOLO4K 脚本

记录：

- Val loss
- Val dice / iou / f1_c0
- COACH420: DCC_Success / Mean_Prec / Mean_Rec / Mean_F1 / Mean_AP
- HOLO4K: 同上

---

## 实验 A1：只回退标签阈值

改动：

- `label_contact_thresh: 5.0 -> 4.0`
- `label_near_thresh: 8.0 -> 6.0`

其余保持完全不变。

目的：

- 单独测试 supervision 是否是主因

预期：

- 如果 external precision / DCC 明显提升，说明问题首先在标签定义
- 如果只改阈值就能明显追近 `PoC`，后面很多结构性改动都不需要优先处理

---

## 实验 A2：只回退 loss 目标

基于 `A1`，再做：

- `tversky_weight: 0.3 -> 0.0`
- `focal_weight: 0.7 -> 1.0`

也就是回到纯 focal。

目的：

- 检查 recall-biased objective 是否在推大口袋

预期：

- 如果 precision 上升而 recall 小幅下降、但 F1 / DCC 变好，说明 Tversky 是当前回退方向之一

---

## 实验 A3：回退 class weight

基于 `A2`，再改：

- `class_weights: [8,3,1] -> [8,0,2]` 或一个更接近 `PoC` 的保守版本

目的：

- 检查对 `Near` 的显式加权是否把模型推向更宽的 positive belt

---

## 实验 B1-B4：剥离正则

建议顺序：

1. `coord_noise_std = 0`
2. `gvp_dropout_prob = 0`
3. `physchem_dropout_prob = 0`
4. `dropout/attn_dropout/classifier_dropout/stochastic_depth` 全部降到接近 `PoC`

为什么这样排：

- `coord_noise` 和 `gvp_dropout` 直接作用在几何主干，最可能改变 pocket 轮廓
- `physchem_dropout` 会破坏晚期注入路径
- 全局 dropout 最后再统一收缩

预期：

- 如果剥掉这些正则后 val 和 external 同时改善，说明当前主线存在明显欠收敛

---

## 实验 C1-C3：回退高耦合结构

### C1: 关掉 protein distance bias

目的：

- 验证这个 bias 是否与 GVP 已编码的局部几何冗余，反而让 self-attn 学坏

### C2: 单向 cross-attn 替代双向 cross-attn

目的：

- 测试当前 bidirectional fusion 是否“太重但增益不稳定”

### C3: 去掉 AttnRes

目的：

- 测试多 block residual attention mixing 是否增加了训练难度，但没有带来足够收益

---

## 实验 C4：在 PLC 上恢复 chain-aware 建模

这是我建议在 supervision / regularization 之后优先做的结构实验。

最小实现：

1. 数据集增加 `chain_ids`
2. GVP 图边增加 `is_same_chain`
3. protein self-attn 用 chain-gap position ids

目的：

- 测试 `PoC` 中这个明确有效的 inductive bias 是否在 `PLC` 被丢失

如果这一项带来稳定提升，那说明 `PLC` 不是“架构不够高级”，而是“丢掉了真正有用的结构偏置”。

---

## 7.3 推荐的最小实验子集

如果你现在只愿意做最少的实验，我建议先跑这 5 个：

1. `A0` 当前 v10m4
2. `A1` 改回 `4/6` 标签
3. `A2` 纯 focal
4. `B1` 去掉 coord noise
5. `C4` 恢复 chain-aware

这 5 个实验足以回答 80% 的问题。

---

## 7.4 我预期最可能出现的结果

我对结果的先验排序是：

1. `A1` 会显著改善 precision / DCC
2. `A2` 会进一步改善 precision，代价是 recall 略降
3. `B1` 和 `B2` 会带来次一级改善
4. `C4` 会在多链复合物或复杂 external benchmark 上带来稳定增益
5. `C1-C3` 的收益不一定大，但至少能帮助你证明当前复杂结构是否值得保留

---

## 8. 最终判断

我不认为 `PLC` 当前失败的根因是“想法太差”，恰恰相反，很多局部改动都很合理。

问题在于：

- 你把更宽的标签
- 更偏 recall 的 loss
- 更重的正则
- 更复杂的融合结构

放进了同一条主线上。

这样一来，即使某些模块本身有价值，也很容易被更大的 supervision / optimization 偏差掩盖。

如果要把 `PLC` 拉回到真正优于 `PoC` 的状态，我的建议非常明确：

先把任务定义和训练目标拉回清楚，再决定保留哪些“高级模块”。

不是先继续加模块。

---

## 9. 附录：一句话式对照

### PoC

一个 supervision 更清楚、路径更短、结构偏置更干净的模型。

### PLC

一个把很多合理升级叠在一起、但当前主线更像“目标漂移 + 过度工程化”的系统。

### 为什么 PLC 现在输给 PoC

因为它当前更像在优化“更宽的 recall-friendly residue coverage”，而不是“更准的 pocket localization”。

