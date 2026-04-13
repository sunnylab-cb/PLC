# 文獻調研：單蛋白多配體的 Train/Test Splitting 最佳實踐

更新日期：2026-04-13

## 背景

PDBbind 中 ~72% 的 entries 與至少一個其他 entry 共享相同蛋白質上下文（L1 exact），
經 MMseqs2 95% 聚類後上升至 ~89%（L2）。本文件整理 SOTA 方法如何處理此問題。

---

## 一、各方法的分割策略

| 方法 | 發表 | 分割策略 | 序列閾值 | 備註 |
|------|------|---------|---------|------|
| ScanNet | Nature Methods, 2022 | 分層測試集：≥70% / CATH superfamily / fold / 全新 | 100/95/90/70% 四輪聚類 | 最嚴格的分層評估設計 |
| EquiPocket | 2023 | MMseqs2 聚類 | test vs train < 90% | |
| GraphBind | PMC, 2021 | CD-HIT 去冗餘 | train 內部 + train-test < 30% | 最嚴格 |
| PocketMiner | Nature Comms, 2023 | USEARCH 聚類，以 cluster 為單元 | 90% | |
| P2Rank | J. Cheminformatics, 2018 | SCOP family 分割 | 結構層級 | |
| UniSite | NeurIPS 2025 Spotlight | UniProt-centric + MMseqs2 三檔 | 50/70/90% | 與 PLC 最相似的設定 |
| LP-PDBBind | arXiv 2023 / J. Phys. Chem. B 2025 | PDB 週序列聚類 30% + 配體 Tanimoto < 0.99 | 30% | |
| PDBbind CleanSplit | Nature Machine Intelligence, 2025 | 結構聚類，同時去除 train 內部冗餘 | 結構+序列 | 證明原有高分很大程度來自 leakage |
| DeepSite | Bioinformatics, 2017 | sc-PDB 非冗餘版 | >25% + 覆蓋度 >90% 移除 | |

### 關鍵發現

1. **CleanSplit (2025)** 移除 train-test 泄漏後，現有 SOTA 模型性能大幅下降，
   證明原有高分很大程度來自 leakage。
2. **ScanNet** 的分層測試集設計（按相似度分組報告）是目前最有資訊量的評估方式。
3. **時間分割不充分**：EquiBind 的 time-split 使 train-test 平均序列相似度降至 0.484，
   但仍有泄漏。時間分割是必要但不充分條件。

---

## 二、多配體條目：保留 vs 去重 vs 聚合

### 三種主流策略

| 策略 | 做法 | 代表 | 適用場景 |
|------|------|------|---------|
| **Aggregation** | 保留所有條目，合併 binding residue 標籤取聯集 | UniSite, LABind | ligand-agnostic binding site prediction |
| **Deduplication** | 每蛋白只留代表性條目 | GraphBind, CleanSplit | 防止過擬合高頻家族 |
| **Cluster-level 保留** | 全部留，但 cluster 不跨 split | LP-PDBBind | 平衡數據量與 leakage |

### 對 PLC 的適用性分析

**Aggregation 不適合 PLC。** PLC 是 ligand-conditioned 模型：

- 輸入：(protein, **specific** ligand)
- 輸出：每個殘基對**這個特定配體**的接觸分類

Aggregation 會把不同配體的接觸殘基合併，破壞 ligand-conditioning 的語義。
例如：殘基 D 只接觸 L2 不接觸 L1，aggregation 會錯誤地把 D 在 P+L1 中標為 contact。

Aggregation 適合 ligand-agnostic 的模型（P2Rank、ScanNet、UniSite），
不適合 ligand-conditioned 的模型（PLC）。

### PLC 的建議策略

**Cluster-level 保留 + Noisy Negative Loss（見下文）**：
- 保留同蛋白多配體的所有條目（每條的標籤對其配體而言是正確的）
- 以 cluster 為單位分割，不跨 split
- 用 loss-level 機制處理 c_2 標籤的不確定性

---

## 三、c_2 標籤不確定性與 Graceful Loss Treatment

### 問題定義

c_2 (far) 的語義是「離**這個配體**遠」，不是「永遠不結合任何東西」。
當同一蛋白有多個配體 entry 時：

```
P + L1 → {A,B,C} = contact,  {D,E,...} = far
P + L2 → {B,D,E} = contact,  {A,C,...} = far
```

殘基 D 在 P+L1 中標記 c_2，但它確實具有結合能力（只是不結合 L1）。
若模型偵測到 D 的 binding-capable 結構特徵並預測 positive，會被錯誤懲罰。

這本質上是 **Positive-Unlabeled (PU) Learning** 問題：
- Positive labels (c_0, c_1) 是可靠的
- Negative labels (c_2) 是不可靠的——可能是真 negative，也可能是 unlabeled positive

### 提議的解決方案：Confident Negative Clipping

當模型對一個 c_2 標籤的殘基高置信預測為 positive（c_0 或 c_1）時，
降低該殘基的 loss 權重，而非完全懲罰。

核心參數：
- `clip_thresh`：置信度閾值（建議 0.85-0.9），僅對高置信預測生效
- `clip_ratio`：降低後的權重（建議 0.05-0.1），保留微弱約束防止 collapse
- `warmup_epochs`：前 N epochs 使用正常 loss

防止 collapse 的機制：
1. 閾值設高，只對非常自信的預測生效
2. 不完全丟棄 loss，保留微弱懲罰
3. Warmup 讓模型先學基本 pattern
4. 可設置每 batch 最多 clip X% 殘基的上限

### 相關文獻框架

- **Confident Learning** (Northcutt et al., JAIR 2021) — 用模型自身置信度識別 noisy labels
- **Bootstrap Loss** (Reed et al., 2015) — 用模型預測修正 target
- **Asymmetric Loss for PU Learning** — 對 negative 樣本使用較低的 loss weight
- **Positive-Unlabeled Learning** — 把 c_2 視為 unlabeled 而非 confirmed negative

---

## 四、分割工具推薦

| 工具 | 功能 | 適用場景 |
|------|------|---------|
| **GraphPart** (NAR Genomics 2023) | 嚴格同源分區，保證 split 間零泄漏 | 推薦首選 |
| **DataSAIL** (Nature Comms 2025) | 蛋白+配體雙維度 ILP 約束 | 蛋白-配體數據集最佳 |
| **Fast-Part** (bioRxiv 2024) | GraphPart 加速版 | 大數據集 |
| **MMseqs2 easy-cluster** | 超快序列聚類 | 已在用（L2） |

### 分割閾值建議

| 閾值 | 用途 | 依據 |
|------|------|------|
| 30% | 最嚴格測試集（homology-free） | ProteinNet, GraphBind |
| 40-50% | 中等嚴格測試集 | LP-PDBBind, UniSite |
| 90-95% | 內部去冗餘 | EquiPocket, PocketMiner |

建議採用 **ScanNet 分層報告模式**：在 <30%, 30-70%, >70% 三個區間分別報告，
比單一混合數字更有科學價值。

---

## 五、建議的 PLC 落地方案

```
L1: contact_unique_chain_signature    → exact protein context grouping ✅ 已完成
L2: MMseqs2 95%                       → 近似序列家族               ✅ 已完成
L3: GraphPart / DataSAIL at 30%       → homology-free split          ⬜ 待實作
Loss: Confident Negative Clipping     → 處理 c_2 標籤不確定性        ⬜ 待實作
Eval: 分層測試集                       → 按相似度分組報告             ⬜ 待實作
```

訓練時 cluster-weighted sampling 防止高頻家族主導梯度。

---

## 參考文獻

- Graber et al. "Resolving data bias improves generalization in binding affinity prediction." Nature Machine Intelligence, 2025.
- Tubiana et al. "ScanNet: an interpretable geometric deep learning model." Nature Methods, 2022.
- Zhang et al. "EquiPocket: an E(3)-Equivariant GNN for Ligand Binding Site Prediction." 2023.
- Xia et al. "GraphBind: protein structural context embedded rules." NAR, 2021.
- Cimermancic et al. "PocketMiner: predicting locations of cryptic pockets." Nature Communications, 2023.
- Gao et al. "UniSite: The First Cross-Structure Dataset." NeurIPS 2025.
- Wang et al. "LP-PDBBind: Leak Proof PDBBind." J. Phys. Chem. B, 2025.
- Northcutt et al. "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR, 2021.
- Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping." ICLR Workshop, 2015.
- Jespersen et al. "GraphPart: homology partitioning for biological sequence analysis." NAR Genomics, 2023.
- Kroll et al. "DataSAIL: Data Splitting Against Information Leakage." Nature Communications, 2025.
- Krivák & Hoksza. "P2Rank: machine learning based tool for prediction of ligand binding sites." J. Cheminformatics, 2018.
