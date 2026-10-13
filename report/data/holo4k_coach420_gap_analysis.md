# HOLO4K vs COACH420 gap analysis

## 结论

HOLO4K 明显低于 COACH420 不应直接解释为模型整体失效。更合理的解释是：在本仓库当前评估口径下，HOLO4K 是更宽、更难的压力测试，包含更大的多链结构、更多 ligand/pocket 候选、更稀疏的正例 residue，以及更强的多口袋定位/assignment 压力。模型在 HOLO4K 上主要掉在 recall、DCC/PDR 类定位成功率和多 ligand 情形，而 precision 并没有同幅度崩掉。

需要谨慎表述：目前没有可靠来源支持“HOLO4K 是预测 pocket 或弱验证 pocket 数据集”。P2Rank 数据集说明把 HOLO4K 描述为从 PDB 直接下载的大型 protein-ligand complex 数据集，`mlig` 变体的相关 ligand code 来自 MOAD 2013；P2Rank 论文同时提醒，直接由 PDB complex 定义 ground truth 会有未标注真 binding site 和标签噪声问题。

## 外部来源支持的 dataset 差异

1. P2Rank 数据集 README 说明：
   - HOLO4K 是 large protein-ligand complex dataset。
   - HOLO4K 包含从 PDB 直接下载的较大 multi-chain structures。
   - `holo4k.ds` 的正确复现实验规模是 4009 proteins。
   - `*(mlig)*` 变体显式指定 relevant ligands，valid ligand codes 来自 MOAD 2013。
   Source: https://raw.githubusercontent.com/rdk/p2rank-datasets/master/README.md

2. P2Rank 论文指出：
   - COACH420 与 HOLO4K 是其无偏测试集。
   - HOLO4K contains larger proteins with more binding sites than COACH420，因为 HOLO4K mainly multimers，而 COACH420 only single-chain proteins。
   - pocket-centric evaluation 和 residue-centric evaluation 的目标不同；一个较大的正确 pocket 在 pocket-level 可以算成功，但 residue F1 会因为 false positives 被惩罚。
   - 直接从 PDB complex 定义 ground truth 有不完整/噪声风险，很多 negative 区域可能是尚未被该 PDB entry 标出的真实 binding site。
   Source: https://link.springer.com/article/10.1186/s13321-018-0285-8

3. P2Rank 工具本身是 pocket-centric，基于 protein solvent-accessible surface 上的点打分和聚类来预测 pockets。
   Source: https://github.com/rdk/p2rank

因此，paper 中不应写成“HOLO4K 的标签是预测 pocket”或“COACH420 一定是 BioLiP 强验证集”。基于已查来源，更稳妥的说法是：COACH420 是较小的 single-chain benchmark；HOLO4K 是 PDB-derived、larger、multi-chain/multimer-heavy benchmark，且 relevant ligand definition 在 P2Rank `mlig` 口径中与 MOAD ligand codes 有关。

## 本地评估口径

`scripts/infer/infer_v11_benchmark.py` 的当前逻辑显示：

- benchmark 默认是 aggregate-by-PDB；per-ligand mode 主要用于 diagnostics。
- contact label 使用 `contact_thresh=5.0 Å`，near label 使用 `near_thresh=8.0 Å`。
- DCC success 默认阈值是 `4.0 Å`。
- `sample_mean` 是对 CSV rows 的均值。
- `PDR` 是预测 contact center 到同一 protein 任一 true ligand-pocket center 的距离小于阈值。
- `LAA` 在 true pocket 数量不少于 2 时，要求预测中心最近的 true pocket 正好是当前 ligand pocket 且距离小于阈值。

注意：当前可读的 checked-in `benchmark_summary.json` 是较旧格式，没有新版 `sample_mean`/`PDR`/`LAA` 字段。因此这里对 PDR/LAA 的解释来自脚本定义；数值证据主要来自同一 checkpoint 的 F1、recall、precision、DCC、DCC success 与 ligand-count stratification。

## 同一 checkpoint 下的本地数值证据

比较 `checkpoint_epoch_65.pt`：

| Dataset | N | Mean F1 | Precision | Recall | Mean DCC | DCC success@4A | Mean binary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COACH420 | 420 | 0.511 | 0.602 | 0.467 | 6.45 | 0.402 | 0.628 |
| HOLO4K | 4008 | 0.382 | 0.580 | 0.318 | 12.77 | 0.244 | 0.487 |

关键点：HOLO4K precision 只小幅下降，但 recall 从 0.467 降到 0.318，DCC success 从 0.402 降到 0.244，平均 DCC 约翻倍。这更像是“在更大、更复杂结构中找不到/对不准正确 pocket”，而不是模型到处乱报造成 precision 崩溃。

## 为什么 HOLO4K 更低

### 1. HOLO4K 蛋白更大，正例更稀疏

本地 CSV 统计：

| Dataset | mean residues | median residues | mean contact residues | mean contact fraction | mean binding fraction |
|---|---:|---:|---:|---:|---:|
| COACH420 | 284 | 259 | 16.36 | 0.0686 | 0.211 |
| HOLO4K | 510 | 388 | 16.59 | 0.0485 | 0.151 |

两者绝对 contact residue 数差不多，但 HOLO4K 蛋白大很多，所以正例比例更低。对 residue-wise F1 来说，这是更难的 class imbalance/localization task。

长度分层也支持这一点：

| Dataset/bin | N | Mean F1 | Mean DCC | DCC success |
|---|---:|---:|---:|---:|
| COACH 200-399 aa | 237 | 0.550 | 5.74 | 0.468 |
| HOLO 200-399 aa | 1591 | 0.529 | 6.93 | 0.414 |
| COACH 400-799 aa | 68 | 0.380 | 10.02 | 0.147 |
| HOLO 400-799 aa | 1226 | 0.282 | 15.87 | 0.076 |
| HOLO >=800 aa | 702 | 0.131 | 25.64 | 0.010 |

HOLO4K 在 200-399 aa 中与 COACH 接近；主要崩在 400+，尤其 800+ residues。

### 2. HOLO4K ligand/pocket 数更多，多口袋 assignment 更难

按本地 processed ligand `.pt` 文件计数：

| Dataset | unique keys | ligand files | mean ligands/key | median | p90 | p99 | max | multi-ligand % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COACH420 | 420 | 658 | 1.57 | 1 | 3 | 7 | 12 | 29.0% |
| HOLO4K | 4008 | 12968 | 3.24 | 2 | 6 | 24 | 63 | 64.8% |

F1 随 ligand count 下降非常明显：

| Dataset/ligands per key | N | Mean F1 | Recall | Mean DCC |
|---|---:|---:|---:|---:|
| COACH 1 | 298 | 0.548 | 0.521 | 6.15 |
| HOLO 1 | 1411 | 0.561 | 0.527 | 6.18 |
| COACH 2 | 76 | 0.433 | 0.355 | 7.38 |
| HOLO 2 | 1213 | 0.353 | 0.266 | 13.27 |
| COACH 3-4 | 33 | 0.398 | 0.309 | 7.04 |
| HOLO 3-4 | 773 | 0.266 | 0.182 | 17.69 |
| HOLO 5-9 | 402 | 0.193 | 0.123 | 20.90 |
| HOLO >=10 | 209 | 0.136 | 0.086 | 20.60 |

这直接支持“HOLO4K 低分不是单纯 dataset 名称导致，而是多 ligand/multi-pocket case 对模型和评估都更难”。特别是 HOLO4K single-ligand 子集 F1 约 0.56，和 COACH single-ligand 子集几乎一致；差距主要来自 multi-ligand/multi-pocket。

### 3. residue-wise F1 比 pocket-level hit 更严格

P2Rank 原论文专门讨论了 pocket-centric 和 residue-centric evaluation 的差异：pocket-centric 只要预测 pocket center/region 命中已知 binding site 即可，而 residue-centric F1 会惩罚边界过宽、错覆盖和 false positives。本仓库评估同时用 residue labels 和 DCC/PDR/LAA 类 center-based metrics，因此 HOLO4K 这种多 pocket、大 multimer benchmark 上会更容易出现：

- center 接近某个真实 pocket，但 residue 边界不精确，F1 低；
- 预测命中了“某个” pocket，PDR 可能还可以，但 ligand-specific LAA 因 assignment 错误低；
- PDB-derived ground truth 未标注其他真实 pocket，导致模型预测的 plausible region 被算作 false positive。

### 4. 结构/ligand 质量和 ligand 定义可能更宽

本地 processed HOLO4K 的高 ligand-count cases 中可见许多 modified residues/ions/cofactors 类 ligand code，例如 `MSE`、`CO3` 等。这不等同于“标签错误”，但说明当前 processed ligand extraction 可能比 canonical drug-like ligand benchmark 更宽。COACH420 也有多 component ligand，但 HOLO4K 的数量和长尾更明显。

Paper 中应把这称为 “broader ligand annotation/extraction and multi-ligand stress-test”，不要写成“HOLO4K 标签弱/错”，除非后续有 ligand-code QC 统计支持。

## 是否说明模型不好

结论应分两层：

1. 不说明模型在 pocket/contact 任务上整体不好。单 ligand HOLO4K 与 COACH single-ligand 近似，precision 也没有明显崩溃，说明模型仍能识别不少局部 binding residue signal。

2. 说明模型当前在 harder regime 下存在限制：大蛋白、多链、多 ligand、多 pocket assignment 时 recall 和 localization 明显下降。LAA/PDR 若低，应解释为 pocket localization 和 ligand-specific assignment 难度上升，而不是简单归因于“HOLO4K 数据差”。

## Paper 建议表述

可以写：

> The performance gap between COACH420 and HOLO4K is largely attributable to benchmark difficulty rather than a simple failure mode on an alternative dataset. In the P2Rank benchmark suite, HOLO4K consists of larger PDB-derived multi-chain complexes and contains more binding sites than the single-chain COACH420 set. Under our residue-level and ligand-conditioned evaluation, this translates into a substantially harder localization and assignment problem: positive contact residues are sparser, multiple ligand pockets compete within the same structure, and ligand-specific assignment metrics penalize predictions that localize to a nearby but different pocket. Consistent with this interpretation, the largest degradation on HOLO4K appears in recall and center-based success, whereas precision is comparatively stable; single-ligand HOLO4K cases are close to COACH420, while multi-ligand cases account for most of the drop.

需要加限制：

> We therefore treat HOLO4K as a broader stress test for multi-pocket, multi-chain binding-site prediction. The lower HOLO4K scores should not be interpreted as evidence that the model fails to learn binding-site signals, but they do reveal a limitation in ligand-specific localization and assignment under complex multi-ligand settings. We avoid over-interpreting HOLO4K labels: the benchmark is PDB-derived and, as noted in prior work, PDB-complex ground truth can be incomplete or noisy because not all true binding sites are necessarily occupied by annotated ligands in a given structure.

不建议写：

- “HOLO4K 包含 predicted pockets，因此标签弱。”
- “COACH420 是 BioLiP 强验证 gold standard。”
- “HOLO4K 低分完全由 benchmark 噪声导致。”
- “模型在 HOLO4K 差说明模型无效。”

## 建议补充图/表

1. Dataset summary table：N、mean/median residues、contact residues、contact fraction、ligands/protein、multi-ligand rate、F1、precision、recall、DCC/PDR/LAA。

2. Ligand-count stratification：按 `1`、`2`、`3-4`、`5-9`、`>=10` ligands/key 展示 F1、recall、DCC、PDR、LAA。这个是最有力证据，因为 HOLO single-ligand 并不差，多 ligand 长尾才是主因。

3. Protein-length stratification：按 `<200`、`200-399`、`400-799`、`>=800` residues 展示 F1/DCC/PDR。用于证明 HOLO4K 的大蛋白/multimer 难度。

4. Scatter/hexbin：x 轴 ligand count 或 residue count，y 轴 F1/LAA，颜色为 dataset。用来显示 performance gap 随任务复杂度扩大。

5. Ligand-code QC supplemental：列出 HOLO4K top ligand residue codes 和高 ligand-count examples，单独标记 modified residues/ions/cofactors/peptide-like components。目的是支持“broader ligand extraction”而不是主文里直接指责标签质量。

6. 同时报告 `sample_mean` 与 `protein_mean_of_ligand_mean/max`。如果 sample_mean 对多 ligand rows 加权更重，protein-level aggregation 可以说明结论是否被多 ligand proteins 的重复采样放大。

## 检查过的本地路径

- `/home/husrcf/Code/ProtBind/PLC/scripts/infer/infer_v11_benchmark.py`
- `/home/husrcf/Code/ProtBind/PLC/scripts/diagnostics/analyze_benchmark_gap.py`
- `/home/husrcf/Code/ProtBind/PLC/scripts/infer/infer_v10m4_holo4k.py`
- `/home/husrcf/Code/ProtBind/PLC/scripts/infer/infer_v10m4_coach420.py`
- `/home/husrcf/Code/ProtBind/PLC/scripts/infer/infer_v10m6_coach420.py`
- `/home/husrcf/Code/ProtBind/PLC/scripts/infer/infer_v10m3_coach420.py`
- `/home/husrcf/Code/ProtBind/PLC/results/holo4k_v11_20260513_223941/benchmark_summary.json`
- `/home/husrcf/Code/ProtBind/PLC/results/holo4k_v11_20260513_223941/run_meta.json`
- `/home/husrcf/Code/ProtBind/PLC/results/holo4k_v11_20260513_223941/all_results.csv`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_233311/benchmark_summary.json`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_233311/run_meta.json`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_233311/all_results.csv`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_235326/benchmark_summary.json`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_235326/run_meta.json`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420_v11_20260513_235326/all_results.csv`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420/benchmark_summary.json`
- `/home/husrcf/Code/ProtBind/PLC/results/coach420/all_results.csv`
- `/home/husrcf/Code/ProtBind/PLC/results/holo4k/benchmark_summary.json`
- `/home/husrcf/Code/ProtBind/PLC/results/holo4k/all_results.csv`
- `/home/husrcf/Code/ProtBind/PLC/benchmark/p2rank-datasets-fixed_hres/coach420`
- `/home/husrcf/Code/ProtBind/PLC/benchmark/p2rank-datasets-fixed_hres/holo4k`
- `/home/husrcf/Code/ProtBind/coach420`
- `/home/husrcf/Code/ProtBind/holo4k`
