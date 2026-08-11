# PLC 架构演化、瓶颈与下一轮路线（2026-08-11）

## 1. 结论摘要

当前实验已经把主线从“大规模 fusion 搜索”收敛为：

> 保留稳定的 protein decision floor，用 late ligand-local delta 注入配体证据；下一阶段的主要缺口不是更深的 cross/co-attention，而是显式区分“候选口袋检测”和“给定 ligand 的候选口袋选择”。

PDBbind residue F1 与 HOLO4K ligand assignment 是两个相关但不同的目标：

- PDBbind 主线已经证明模型使用 ligand，且有效来源主要是 ligand graph 和 local54。
- HOLO4K 的主要失败集中在同蛋白多 ligand、多口袋样本；单 ligand 子集并不比 COACH420 差。
- 因此，只增强逐 residue 的局部分类器可能继续提高 PDBbind F1，却未必改善 HOLO4K LAA。

下一轮最高优先级应是 `V11m16_candidate_pocket_rank_gate`：先从 ligand-independent protein hidden 产生多个 pocket candidates，再用 ligand embedding 排序，并通过零初始化、有界 residual gate 回写 residue logits。现有 V11m15 只训练 embedding auxiliary loss、未接入推理 logits，正好是这一方向尚未完成的部分。

## 2. 实验演化与提升趋势

下表只列关键转折点。不同 checkpoint selector 和 gate 口径存在差异，因此数值用于路线判断，不应直接替代统一 protocol 下的最终表格。

| 阶段 | 代表结果 | ligand dependency | 得到的结论 |
|---|---:|---:|---|
| V11m3 | 稳定 protein trunk | ligand 作用有限 | protein-side pocket prior 很强，后续必须保护 |
| V11m6n `grad060_no_gate` | test real F1 0.6082 | real-shuffle 0.0813 | fractional protein gradient 比 detach/full no-detach 更平衡 |
| V11m13 `group_rescompat` | test real F1 0.6158 | real-shuffle 0.0802 | local54 分组 K/V 与 residue-chem compatibility 能在不损失 ligand gap 的情况下涨点 |
| V13 scratch residue-atom/triangle probe | real F1 约 0.6090 | gap 可到约 0.0868 | 更强 ligand sensitivity 不等于更高 real F1；scalar bias/弱 triangle 会扰动稳定 decision geometry |
| ESM3 global prior scratch seed44 | test real F1 0.6286 | real-shuffle 0.0739 | 当前最高单 seed；protein global prior 有迁移收益，但均值和方差仍需按 seed 报告 |
| ESM-C warm full-finetune | val F1 0.6101，val gated F1 0.6146 | test 尚待统一确认 | 更换 PLM 表征时必须 warm-start 后全模型适配；冻结 adapter 明显不足 |

### 2.1 稳定结论

1. **local54/graph 是主要 ligand evidence。** `grad060` source ablation 中，zero graph 和 zero local54 的 drop 约 0.15，而 global55 约 0.03、UniMol 约 0.002。
2. **late residual 是正确接入位置。** 复杂 fusion 接管 trunk 时，ligand gap 可能上升，但 real F1 与 protein floor 下滑。
3. **梯度强度存在 sweet spot。** `0.60` 优于 `0.70`；full no-detach 更脆，detach 又过于保守。
4. **group_rescompat 的收益来自 value/readout 侧的局部化学条件化。** 当前实现先对 local54 group 做 K/V residual，再按 residue hidden 与 ligand context 产生 group gate，见 `src/models/pocket_gvp_fusion_v11m10.py:285-357`。
5. **PLM 表征不是直接替换即可。** ESM-C scratch weighted sum 约 0.600，warm full-finetune 才到 0.610；concat 和 frozen adapter 均失败。

### 2.2 已经接近判负的方向

- 继续增加 self-attention/cross-attention 深度或宽度：收益不稳定，显存显著增加。
- 用 V12-style block 直接替换稳定 co-attn：历史 probe 未超过 co-attn 主线。
- UniMol/global55 作为强 gate：source ablation 不支持。
- raw ESM multi-layer concat：优化不稳定且 val F1 明显下降。
- 只增加 scalar residue-atom score bias 或 heuristic triangle smoothing：能提高 ligand sensitivity，但未提高 real ceiling。
- frozen base + 小 adapter：无法消化 PLM representation shift。

## 3. 当前代码的结构性瓶颈

### 3.1 local delta 是单次 residue-to-atom readout

当前 V11m13 继承的 local delta 路径为：

```text
protein hidden -> Q
ligand hidden + local54 group features -> K,V
softmax over ligand atoms
ligand read + pair summary stats
late delta logits
```

优点是稳定、便宜、可解释；局限是：

- 每个 residue 独立对全部 atom 做 softmax，没有显式的多 pocket competition。
- 没有持续更新的 residue-atom pair state；V13 只在 score 上加 scalar correction，表达力仍有限。
- residue coherence 主要由 protein trunk 提供，ligand-conditioned pocket-level coherence 很弱。

### 3.2 V11m15 assignment head 尚未进入 inference path

`src/models/pocket_gvp_fusion_v11m15.py:1-12` 已明确说明 auxiliary loss 与 residue logits 分离。当前做法还有两个关键 train/inference gap：

1. pocket key 用 GT positive mask pooling，推理时没有 GT mask。
2. pocket hidden 使用 `h_fused`，它可能已经被当前 ligand 条件化；这会使 contrastive head 学到 shortcut，而不是 ligand-independent pocket representation。

因此，即便 V11m15 retrieval accuracy 上升，HOLO4K residue-level LAA 也可能不变。

### 3.3 ESM-C MLC 只有全局 scalar mixing

`src/models/pocket_gvp_fusion_v11.py:979-1009` 对所有样本、所有 residue 共用同一组 layer weights。它能选择总体最有用的 block，却不能让 pocket residue 与非 pocket residue采用不同的层组合。raw concat 已经失败，因此下一步若继续 MLC，应只测试有界的 residue-conditioned deviation，而不是扩大 concat projection。

## 4. 原始论文证据与对 PLC 的含义

### 4.1 候选口袋 + ligand 排序

- [VN-EGNN](https://doi.org/10.1186/s13321-025-01127-9) 用带坐标的 virtual nodes 表示多个 binding sites，并用 segmentation 与 center loss 共同训练。
- [VN-EGNNrank](https://openreview.net/forum?id=86TgAWEkaD) 将候选 pocket 与 ligand 编码到共同空间，专门做 ligand-conditioned pocket ranking。
- [ConGLUDe](https://arxiv.org/abs/2601.09693) 进一步把 ligand 与 whole-protein、多个隐式 binding-site embedding 做三轴对比学习。

这三项工作和 HOLO4K 的失败模式高度吻合：PLC 缺的不是再预测一遍 residue propensity，而是为一个 protein 保留多个 candidate site，并让当前 ligand 决定使用哪一个。

### 4.2 显式 pair state 与局部几何约束

- [FABind](https://proceedings.neurips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html) 显式维护 residue-atom pair embedding，交替进行独立 message passing、cross-attention 与 interface update；其 ligand-informed pocket prediction 还使用 center constraint。
- [TANKBind](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html) 对多个 protein blocks 分别与 ligand 建立 interaction matrix，再用局部区域 negative 做 contrastive ranking。
- [LaMPSite](https://arxiv.org/abs/2312.03016) 在无 bound pose 的设置下，用 protein contact map 与 ligand 内部 distance map 更新 residue-atom interaction embedding。

这些结果支持 pair interaction 的方向，但不支持直接把 bound residue-atom distance 输入 PLC：我们的任务在推理时没有真实 ligand pose，使用晶体复合物距离会造成标签泄漏。可迁移的是 protein contact/structure geometry、ligand graph/distance，以及低秩 pair state，不是 bound cross-distance。

### 4.3 protein pair representation

[AF2BIND](https://www.nature.com/articles/s41592-026-03011-2) 报告 AlphaFold2 pair representation 对 binding-residue prediction 优于 ESM2/ESM1-IF 单流表征，且 pair feature 与 ESM feature 组合略有进一步提升。它支持一条独立的 protein-floor 支线：缓存并压缩 pair representation，而不是继续堆叠 PLM final/MLC feature。

### 4.4 contrastive learning 的边界

- [DrugCLIP](https://proceedings.neurips.cc/paper_files/paper/2023/file/8bd31288ad8e9a31d519fdeede7ee47d-Paper-Conference.pdf) 证明 BioLiP/ChEMBL 的 pocket-ligand pair 可用于 dense-retrieval contrastive training。
- [BindCLIP](https://arxiv.org/abs/2602.15236) 则指出纯 CLIP embedding 可能依赖 shortcut、缺少细粒度 interaction sensitivity，并采用 hard negatives 与额外结构监督缓解。

因此 PLC 不应把 contrastive head 当最终预测器。更合理的作用是提供 candidate ranking score，再通过小幅 gate 影响原有 residue classifier，并以 real/shuffle/zero 与 BP/PDR/LAA 共同验收。

## 5. 下一轮架构实验矩阵

### P0: V11m16 candidate-pocket rank gate

这是当前最高 ROI 的结构改动。

这不是重复 V11m8 的 `pocket_virtual_tokens`。旧模块先用 **ligand-conditioned** tokens 对整条 protein 做一次全局 read，再把 token value 回写到 `lig_read`；它没有独立的空间中心、candidate membership、候选间 diversity，也没有显式的 ligand-to-candidate ranking loss。V11m16 要先由 protein-only 分支形成可审计的多个候选，再让 ligand 选择候选。

```text
pre-cross protein hidden
  -> K=4~8 protein-only site queries / spatial candidates
  -> candidate membership M[k, residue]
  -> candidate pocket embedding P[k]

ligand graph + local54
  -> ligand query Q

cos(Q, P[k])
  -> candidate weights
  -> residue assignment gate = sum_k weight[k] * M[k, residue]
  -> final logits = V11m13 logits + bounded_beta * centered(gate)
```

关键约束：

- candidate proposal 必须 ligand-independent；ligand 只负责 ranking。
- `bounded_beta` 从 0 或 0.02 初始化，上限 0.2~0.3，保证初始行为等于 V11m13。
- pocket key 必须来自 pre-cross/pre-delta protein hidden，避免当前 ligand 泄漏到 candidate representation。
- 第一版使用 V4 的 3 个 protein-only pocketness channels（CX/local packing/shell ratio）加 10 Å spatial NMS 产生 candidates，先验证 ranking 是否真能提高 LAA；验证后再换 learned virtual queries。
- deterministic 第一版的 candidate target 由 GT contact residues 对各 Gaussian candidate membership 的 coverage 决定，训练损失为 residue objective + candidate CE `lambda=0.02`。in-batch/same-protein contrastive 只在 learned-query 版本中再引入，避免首轮混合两个假设。

建议 probe：

| run | 变化 | 主要问题 |
|---|---|---|
| `v11m16_rankgate_detcluster_b002` | deterministic spatial candidates + bounded gate 0.02 | 最低风险验证 inference coupling |
| `v11m16_rankgate_query4_b002` | 4 个 protein-only learned site queries + diversity/center loss | 验证 learned candidates 是否优于 clustering |
| `v11m16_rankgate_query8_sp002` | 8 queries + in-batch contrastive + sparse samevalid bonus | 验证 HOLO4K multi-pocket LAA ceiling |

#### 2026-08-11 执行状态

- 已实现 `src/models/pocket_gvp_fusion_v11m16.py`、训练入口、V11-family inference detection、Star 单任务 launcher 与单元测试。
- 真实 V11m13 checkpoint smoke：84,931,624 总参数，仅 133,889 个 `assignment_candidate_*` 参数可训练；主干、co-attn、local-delta 和 classifier 均冻结。
- Star run：`pocket_gvp_fusion_seed44_candidate_rankgate_k4_aux002_v11m16_20260810_175011`，GPU0，40 epochs；`torch.compile` 明确禁用。
- epoch 1 validation：`F1_c0=0.599056`、`AUPRC=0.609788`、`AUROC=0.950276`、`F1/MCC@0.5=0.603118/0.5927`、val-selected gate `0.42936`。同 seed V11m13 checkpoint 的 best val F1 是 `0.599105`，因此首轮等价于基线，尚无增益也无回归。
- epoch 2 step 662 的 candidate rank accuracy 为 `0.6522`（四候选随机基线 0.25），rank entropy `0.6953`，best soft-Gaussian coverage `0.3117`，valid fraction `0.9583`。这表明 ligand query 已学到候选区分信号，但 soft coverage 不是二值 proposal recall，不能直接解释成“只覆盖 31% 样本”。
- gate alpha 已从 `0.020` 降至 `0.01855`，centered membership abs mean `0.07717`，对应平均直接 logit 扰动约 `0.0014`。因此第一轮主要判据是 candidate 可排序性与 val/test 不回归；若排序成立但 F1/LAA 不动，下一步应增强/归一化 gate coupling，而不是否定 candidate decomposition。
- epoch 2/3 validation 分别为 `F1_c0=0.5990/0.5991`、AUPRC 均约 `0.6098`，仍与同 seed V11m13 的 `0.599105` 持平。当前证据支持“candidate head 可学习且低风险”，尚不支持“已提高 residue F1”。
- epoch 4--6 validation 仍为 `F1_c0=0.5991`、AUPRC `0.6098`、AUROC `0.9503`，没有出现延迟涨点。训练 batch 统计显示 candidate loss 约从 `1.22` 降至 `0.18`、rank accuracy 升至 `0.91--1.00`、entropy 从约 `1.33` 降至 `0.39`，但 alpha 同时收缩到约 `0.0078`；当前瓶颈明确是 ranking 到 residue logits 的 coupling，而不是 candidate representation 没学会。
- `scripts/train/train_v11m3.py` 已加入 validation-only auxiliary diagnostics：不把 auxiliary loss 加入 val loss，仅按 validation samples 汇总并写入 `ValAssignment/*`，同时打印 rank accuracy、valid fraction、coverage、alpha 与 entropy。该改动会作用于后续启动的 V11m16 jobs；已运行的 Star 进程不会被中途修改。
- Star PDBbind-only V11m16 在 epoch 22 early-stop；best checkpoint 来自 epoch 21，val F1 `0.5991707`，仅比同 seed V11m13 的 `0.599105` 高约 `0.000066`。test real global F1 与 V11m13 **完全相同**为 `0.6157781`；V11m16 real/shuffle/zero 为 `0.6157781/0.5355871/0.4473152`，V11m13 为 `0.6157781/0.5355479/0.4473729`。real-shuffle gap 从 `0.0802302` 轻微降至 `0.0801911`，没有可报告增益。
- best checkpoint 的 bounded candidate gate alpha 为 `0.0078204`。结合高 train rank accuracy，可判定 deterministic candidate ranking 表征可学习，但当前 centered scalar gate 被主 residue objective 压到近乎无效；该 PDBbind-only probe 判为 no effect，而不是架构性能提升。

第一验收指标不是 PDBbind F1 单点，而是：

```text
PDBbind test real F1 不下降超过 0.003
real-shuffle gap >= 0.07
HOLO4K protein-balanced LAA 上升
HOLO4K PDR/BP F1 不下降
multi-ligand low-overlap subset 的增益 > single-ligand subset
```

### P1: V11m16 conditional ESM-C layer mixer

在 ESM-C warm full-finetune 基础上，将全局 layer weights 改为有界 residue-conditioned deviation：

```text
alpha_i = softmax(global_logits + eps * tanh(router(structure_hidden_i)))
h_i = sum_l alpha_i,l * LN(h_i,l)
```

设置：`eps=0.1/0.2`，router 最后一层零初始化，禁止 ligand 输入。这样初始严格复现当前 weighted sum，避免 concat 的尺度和优化问题。

建议只跑一个 `warm_condmix_eps01` probe。若 test real/shuffle/zero 不优于 warm weighted sum，MLC 线即可收敛，不再扩大。

### P1: compressed protein pair prior

AF2BIND 提示 pair representation 可能改善 protein floor。PLC 不需要保存完整 `L x L x C`：

1. 从 AF2/ESMFold pair state 缓存每个 residue 的 top-k/attention-pooled 32~64D summary。
2. 通过零初始化 adapter 注入 pre-cross protein hidden。
3. 先冻结 pair encoder，只训练 adapter，再允许全模型小 LR 适配。

该支线优先验证 `protein-only/base F1`、长链 HOLO4K PDR 与低同源 split；它不是 ligand dependency 模块。

### P2: low-rank residue-atom pair value state

若 P0 证明 candidate ranking 有效，再尝试把 V13 scalar score bias 升级为低秩 pair value state：

```text
u_i = Wp(h_residue_i)             # [R, rank, d]
v_a = Wl(h_atom_a, local54_a)     # [A, rank, d]
z_ia = sum_rank u_i * v_a
z_ia <- protein-contact update + ligand-graph update
lig_read_i <- sum_a attention_ia * Value(z_ia)
```

推荐 rank 8/16、只更新 value/readout、alpha 零初始化。不要第一轮就建立 full `R x A x 128` tensor，也不要使用 native complex cross-distance。

## 6. BioLip side path 可行性审计

### 6.1 HPC2 当前实况

远端目录：`/hpc2hdd/home/shuang886/biolip`

| 项目 | 当前观察 |
|---|---:|
| protein-chain bundle 目录 | 345,574 |
| fixed receptor ready list | 137,051 |
| protein V4 feature 文件 | 110,213 |
| ESM2 feature 文件 | 80,525 |
| ligand feature 文件 | 13,348 |
| excluded IDs | 719 |
| protein failed IDs | 707 |

这说明 protein 侧预处理规模很大，但用于 PLC 全量训练的 ligand feature 尚不完整。pair metadata 已包含 `binding_residues`、ligand ID、heavy atom count 和 chain mapping，可以生成 per-ligand residue labels。

抽查 `3p4g_A` 得到 13 个独立 Ca ligand rows，每个 ligand 只有 1 个 heavy atom。这同时说明 BioLip 的价值与风险：它确实含同蛋白多位点监督，但当前总体分布可能被 metal ion/cofactor/peptide/nucleic-acid 样本主导，不能直接与 PDBbind drug-like small molecules 混训。

[BioLiP2](https://academic.oup.com/nar/article/52/D1/D404/7233921) 本身通过几何规则、常见添加剂过滤和文献验证标注 biologically relevant interactions，但数据库覆盖金属、核酸、肽和大型复合物；“生物学相关”不等于“与 PDBbind 小分子分布一致”。

### 6.2 推荐用途

BioLip 最适合先做 **candidate-pocket/assignment pretraining**，而不是直接替换 PDBbind 主训练：

```text
Stage A: BioLip filtered pretrain
  freeze/low-LR protein trunk
  train candidate proposal + ligand-pocket rank head

Stage B: PDBbind finetune
  load V11m13/ESM3 or ESM-C mainline
  unfreeze bounded rank gate
  retain residue CE and ligand-dependency protocol
```

必须先完成以下 QC：

1. 只保留 RDKit-valid、heavy atoms 6~128 的 non-polymer organic ligands；metal/ion、peptide、DNA/RNA、buffer/additive 分桶而非混合。
2. binding residues 必须能无歧义映射到 V4/ESM residue order；positive residues 至少 3 个。
3. protein 结构、V4、ESM、ligand V5 四类 feature 必须全部存在且 finite。
4. 按 protein sequence cluster 和 site cluster 切分，不能按 ligand row 随机切分。
5. 排除 PDBbind val/test、COACH420、HOLO4K 的 PDB ID，并做 sequence/pocket similarity leakage audit。
6. 统计每个 protein 的 distinct low-overlap site 数；只有这部分才是 assignment hard-negative 监督。
7. 使用 PDBbind train stats 对 BioLip feature 做 OOD 检查；必要时使用 source-specific input affine adapter，但共享主干。

### 6.3 最小 pilot

不要先补齐全部 34 万 bundle。当前先用已有全特征数据做 632-pair feasibility pilot；若方向成立，再生成一个 20k~50k drug-like clean subset：

- 先完成 manifest/QC 与 leakage report。
- 补齐该 subset 的 ligand V5 和缺失 ESM feature。
- 跑 `BioLip pretrain -> PDBbind finetune` 与 `PDBbind only` 对照。
- 只有 HOLO4K LAA/multi-pocket subset 明显改善且 PDBbind real F1 不掉，才扩大 BioLip。

### 6.4 BioLip readiness QC 结果

- 工具：`scripts/diagnostics/build_biolip_training_manifest.py`。
- HPC2 CPU job `10175093` 已完成（97.55 s，MaxRSS 约 5.93 GB）；只读取 `pairs_cache_v6.pkl` 和资产文件名，没有修改原始数据。
- 输出分为 all-pair metadata ledger、当前全特征 ready、仅缺 ligand V6、同链 multi-site group 四张表，并标记 PDBbind direct-ID overlap。
- 704,660 个 pair 中，188,124 个满足 regular-organic、heavy atoms 6--128、至少 3 个 binding residues 和 exact complete atom mapping；38,218 个同时具备 fixed receptor、protein V4 与 chain ESM2。
- 当前 ligand V6 仅覆盖 6,674 个 feature keys，导致三路 feature 全齐只有 1,518 对；去掉 direct PDBbind ID 后剩 1,382 对。现阶段首要数据瓶颈是 ligand V6 覆盖，而不是 BioLip 原始 pair 数量。
- 当前 V4 是 full-receptor/PDB 粒度，ESM2 是 `PDB_chain` 粒度。新增 `scripts/diagnostics/inspect_biolip_feature_alignment.py`，按 `(chain_id, residue_index, insertion_code)` 精确切 V4 并核验 ESM2，而不是仅比较长度。
- strict-ready 真实 smoke `11ba_A_BS01_UPA` 已通过：V4 全 receptor 248 residues，目标链/ESM2 均为 124 residues；ligand 为 63 个含氢 atoms、39 个 heavy atoms、134 条边，所有张量 finite 且 residue keys 完全一致。
- 全量 1,382 对 tensor QC（HPC2 job `10175117`）已完成：1,382/1,382 通过，0 issue、0 warning；residue 数 p50/p90/max 为 264/537/1231，ligand atoms p50/p90/max 为 34/71.9/123，均在当前 `max_seq_len=1800`、`max_atoms=128` 内。
- 进一步排除 COACH420/HOLO4K direct PDB ID 后，第一版物化集为 632 pairs、314 PDB、554 PDB-chain；按完整 PDB group 划分为 546 train / 86 val。HPC2 job `10175166` 用时 1:59、MaxRSS 约 541 MB，生产 `PocketDatasetV8` 全量重载 QC 通过。
- 物化数据标签沿用主线定义：protein residue heavy atoms 到 ligand heavy atoms 的最小距离，`contact<5 A`、`near=5--8 A`。标签计数为 contact 9,989、near 14,213、far 191,682；每个 pair 至少 3 个 contact，最大 39 个。
- 每条 chain sidecar 由 full-receptor V4 按 `(chain_id,residue_index,insertion_code)` 精确切片，并从 fixed CIF 补回 `all_atom_coords/all_atom_names`；源 V4、ESM2 和 ligand V6 文件保持只读，pair 目录只使用 symlink。
- 新增 `allow_zero_unimol_fallback`：BioLip 预训练仍保留 V11m13 的全部 UniMol 层和 checkpoint shape，但在没有匹配缓存时显式传入 zero embedding；默认值为 false，因此不改变任何现有训练的严格行为。
- 初始两阶段 jobs `10175174/10175195` 均在零 GPU 消耗时取消：第一项被 alpha-transfer 修正取代，第二项因 MMseqs 发现 PDBbind val/test 同源泄漏而停止。当前干净依赖链为分类审计 `10175198` -> internal-clean 物化/QC `10175200` -> 两阶段 GPU pilot `10175201`。Stage A 从 V11m13 seed44 初始化、冻结主干、只训 133,889 个 `assignment_candidate_*` 参数；Stage B 恢复 PDBbind 与真实 UniMol，并以较低 LR 微调同一 head，最后自动跑 test ligand-dependency。
- PDBbind-only TB 显示 candidate loss 已从约 `1.22` 降到 `0.18`、batch rank accuracy 到 `0.91--1.00`，但 gate alpha 同时从 `0.020` 缩到 `0.0078`。因此 Stage A 结束后只迁移 query/key ranking 表征，并将 alpha 重置为 `0.02` 再进入 Stage B；alpha 被视为 target-domain coupling/calibration 参数，而不是应从 BioLip 迁移的表示。
- direct PDB ID 排除仍不足以支撑 benchmark claim。HPC2 job `10175189` 使用 MMseqs 对 554 条 BioLip chain 检索 PDBbind val/test、COACH420、HOLO4K，门槛为 query coverage `>=0.8` 且 identity `>=0.3`：485/554 chains 命中至少一个评估集；按完整 PDB group 排除后仅剩 99 pairs、46 PDB、69 chains。分项命中为 PDBbind val 204 chains、test 165、COACH420 186、HOLO4K 413，说明 direct-ID filtering 远远不够。
- `10175198` 进一步输出 category-specific exclusion/retained manifests。`internal-clean` 只用于 PDBbind val/test；若未来报告 COACH420/HOLO4K，则必须使用 all-evaluation clean 数据或从 38,218 个待补 ligand feature 的候选中扩展，不能复用 internal-clean pilot。
- PDBbind internal-clean 过滤保留 303 pairs、148 PDB、253 chains；重新按完整 PDB group 划分为 272 train / 31 val。HPC2 job `10175200` 用时 54 s，生产 loader QC 再次全量通过；contact/near/far 标签为 4,916/7,094/94,136。该规模只用于 feasibility，不足以支撑最终 BioLip gain claim。
- 真实 `1coh_D_BS01_COH` V6 smoke 已通过：75 个含氢 atoms、43 heavy atoms、109D features、164 条双向 edges、原始 heavy coordinates `43x3`，feature completeness `0.8571`，所有张量 finite，并正确路由为 metal-complex。HPC2 AF3 环境补齐了缺失的纯 Python 依赖 `click==8.1.7` 和 `packaging==24.2`。
- 扩展数据采用“先过滤、后提 ligand”的顺序。首个 job `10175213` 因一条合法 17-aa 短链被旧 QC 的 `<20 aa` 硬阈值误杀；修复后只拒绝空序列/非法字符，并在 summary 中显式统计短链。重跑 job `10175262` 已完成（4:56，MaxRSS 约 580 MB），1 条 `<20 aa` query 被正常保留，8,519 条 evaluation targets 无短链或缺失。
- 候选 manifest 实际为 36,700 pairs；排除 direct PDBbind 与 COACH420/HOLO4K IDs 后为 23,435 pairs、9,700 PDB、18,986 chains。MMseqs 全评估去同源后保留 **6,999 pairs、2,734 PDB、5,663 chains**；PDBbind-internal-only 可保留 14,289 pairs，external-benchmark-only 可保留 9,721 pairs。最终训练数据使用更严格的 all-evaluation scope。
- 6,999 个 all-evaluation-clean 候选全部为 regular-organic；metadata heavy atoms 的 min/p50/p90/p95/p99/max 为 `6/17/36/41/48/50`，因此这一批不会触发 128-atom 主线的极端大配体问题。
- ligand V6 与完整训练数据依赖链已提交：`10175284` 用单个 `a128m512u` 节点（128 CPU、200 GB、96 workers）只处理 6,999 个 allowlisted pairs，并做 exact-manifest tensor QC；`10175287` 在其结束后重建 ready manifest；`10175288` 再用两轮 sequence exclusion 的 7,227 个 PDB-ID 并集物化 `v11_allclean_20260811` 并全量通过生产 loader QC 后才放行训练。
- `10175284` 已在 53:14 内完成，6,999/6,999 tensors 有效、0 failure；含氢 atom 数 min/p50/p90/p99/max 为 `6/32/64/94/130`。tensor-derived route 实际为 inorganic 811、metal-complex 186、no-conformer biological 208、regular-organic 5,794，说明 pair-cache metadata 会把约 17% 非 regular chemistry 错标成 regular-organic。
- 因此 full-clean 主线在物化前新增真实 tensor 过滤：`ligand_route==5` 且 feature completeness `>=0.8`，loader `max_atoms=256`、兜底 crop 为 `alternating_ends`。首个物化 job `10175288` 仅因 regenerated QC 目录没有复制 direct-ID exclusion 文件而在 6 秒内失败；路径假设已修复，重投 `10179954` 正在物化 5,794 pairs，后续 full-clean GPU job 为 `10179955`。
- 当前 303-pair PDBbind-internal-clean feasibility pilot 为 GPU job `10175201`。完整 all-evaluation-clean 两阶段 run 为依赖 job `10175294`；Stage A/Stage B experiment names 已包含 Slurm job ID，避免并发 run 通过“最新同名目录”串用 checkpoint。

## 7. 推荐执行顺序

考虑 Star 只允许一个 PLC 任务占一张 GPU：

1. CPU 完成 BioLip manifest/QC、配体类型分布、distinct-site coverage 与 leakage audit。
2. 单 GPU 跑 `v11m16_rankgate_detcluster_b002`，这是最快的架构判据。
3. 若 LAA 有信号，跑 `v11m16_rankgate_query4_b002`；否则停止 candidate-query 扩展。
4. 并行准备 BioLip 20k~50k clean subset，随后只跑一个 pretrain/finetune pilot。
5. 再跑一个 ESM-C `warm_condmix_eps01`；不做更多 concat/depth sweep。
6. compressed AF2/ESMFold pair prior 作为独立 P1 支线，先做 cache 成本和线性 probe，再决定是否接主模型。

## 8. 当前最强可辩护的论文叙事

1. PLC 的有效 ligand signal 来自 local atom chemistry 与 ligand graph，而非简单 global/UniMol prior。
2. late local delta 与 fractional protein gradient 在 ligand sensitivity 和 protein floor 之间形成稳定折中。
3. group-wise chemistry compatibility 在保持 real-shuffle gap 的同时提高 residue-level F1。
4. 更强的 pair/scalar bias 可以提高 ligand sensitivity，但若不保护稳定主干，会降低 real ceiling。
5. HOLO4K 暴露的是 candidate-pocket assignment 缺口；下一代模型应显式分解 pocket proposal、ligand ranking 和 boundary refinement。
6. BioLip 可能提供 PDBbind 缺乏的多位点监督，但必须通过 ligand-class、site-overlap 与 leakage QC 后作为 assignment pretraining 数据，而非无过滤混训。
