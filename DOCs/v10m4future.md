現在我對整個系統有了完整的理解。讓我做一個系統性的分析。

---

## 全面漲點分析報告

---

### 一、Feature 層

#### 1.1 蛋白質 Feature（當前: 62D physchem + 89D spatial + 12 GVP vectors）

**缺漏項：**

| 特徵 | 當前狀態 | 問題 |
|------|----------|------|
| SASA | ✅ 包含 | 方向不區分，只有 magnitude |
| HSE（Half-Sphere Exposure）| ❌ 缺失 | 比 SASA 更能區分"面向腔體"vs"面向溶劑" |
| 演化保守性（per-position entropy）| ❌ 缺失 | ESM 隱式包含但不顯式 —— pocket 位置往往高度保守 |
| Protrusion Index | ❌ 缺失 | 殘基突出程度，與 binding 有強相關 |
| 二級結構邊界 | 部分 | 只有 SS label，缺 SS transition（loop-helix cap，beta edge，pocket 常在此） |

**HSE 是性價比最高的遺漏特徵**：上半球 / 下半球的鄰居非對稱分佈直接表示殘基是否"朝向空腔"，而這正是 binding site 的幾何本質。計算也不需要額外數據，BioPython 已支持。

**→ 建議調研方向（請你）**：  
- Hamelryck 2005 "An amino acid has two sides: a new 2D measure provides a different view of solvent exposure"（HSE 原論文）  
- 看一下 `preparation/protein/surface_features.py` 裡有沒有 HSE，如果沒有需要添加

#### 1.2 配體 Feature（當前: 109D）

**問題：**

```python
self.ligand_global_token = nn.Parameter(torch.randn(1, 1, c['d_model']))
# ⚠️ 這是一個固定的可學習常數 token，與輸入配體無關！
```

即每個不同配體在 cross-attention 中拿到的 global prefix 是**完全相同的**，沒有配體特異性。這是一個設計缺陷。

**建議**：替換為從 ligand encoder 輸出做 mean pool / attention pool 的動態 global token。

---

### 二、輸入處理

#### 2.1 標籤設計問題（重要）

```yaml
label_contact_thresh: 5.0  # d_center heavy-atom 中心距
label_near_thresh: 8.0
```

**問題1**：d_center（重原子質心距）不等於 d_min（最近重原子距）。對於大側鏈殘基（Trp, Arg），實際接觸原子距可能是 3.5Å 但質心距超過 5Å，被錯誤標成 Near。這個 label noise 系統性地影響訓練。

**問題2**：Near class 邊界模糊。5-8Å 這個範圍的殘基即非真正binding也非完全無關，中間類的學習信號本身就比較弱，加之 class_weights=[8,3,1] 讓 Near 相對被忽視。

**→ 可考慮**：
- 使用 d_min（最近重原子）替代 d_center 重算標籤
- 或收緊 contact thresh（4.5Å）讓 Contact class 更 pure

#### 2.2 超長蛋白截斷問題

```python
max_seq_len: 1500
```

如果 binding site 在蛋白 C 端而截斷從 0 開始保留前 1500 殘基，binding site 會被截掉。需要確認目前的截斷策略。

---

### 三、模型架構

#### 3.1 Ligand Global Token 設計缺陷（高優先）

**當前**：固定可學習常數 `nn.Parameter`  
**問題**：所有配體的 global context 都相同，失去配體差異化能力  
**修復**：

```python
# Forward pass 中加入：
ligand_global = h_ligand_after_encoder[ligand_mask].mean_pool()  # per-sample
h_ligand_with_global = cat([ligand_global.unsqueeze(1), h_ligand], dim=1)
```

這是**不需要論文調研就可以直接改**的結構性問題。

#### 3.2 Multi-Scale GVP（中優先）

當前 GVP: `radius=10Å, k=20`。10Å 只能捕獲局部幾何。口袋的"深度"往往需要 15-20Å 才能感知到。

```
[Protein]
  ├─ GVP-Local  (r=8Å,  k=15) → local geometry
  └─ GVP-Global (r=16Å, k=20) → pocket context
           concat ↓
       [d_gvp_out = 256+256]
```

參考：**GearNet**（Zhang et al. 2022）/ **Protein-GNN** 的多尺度策略。

#### 3.3 Per-Layer Distance Bias（低成本改進）

當前 `DistanceBias` 計算一次後所有 Transformer 層共享。但實際上：
- 淺層 Transformer 更關心局部幾何（短距離 bias 有用）
- 深層 Transformer 更關心語義（bias 影響應該小）

**建議**：為每層 Transformer 使用獨立的 `DistanceBias` 模塊（參數量增加 n_layers 倍，但每個 module 很小）。

#### 3.4 Cross-Attention 缺乏結構指導

**當前**：標準 QK^T 注意力，沒有 protein-ligand 空間關係指導  
**合法改法**（非漏題）：

可以用**配體內部形狀特徵**（如主軸方向、形狀描述符）作為 cross-attention 中 ligand side 的 bias，讓蛋白質"知道"自己在和什麼形狀的分子交互，而無需任何蛋白-配體聯合坐標。

但這個比較複雜，性價比不確定。

---

### 四、輸出頭

**當前問題**：每個殘基獨立分類，沒有利用 3D 空間一致性。

**改進方向**：

1. **Spatial Smoothness Auxiliary Loss**（低成本）：  
   若殘基 i 預測為 Contact 但其 3D 鄰居中無任何 Contact，施加小額懲罰。這抑制孤立的假陽性預測。

2. **Deeper Classifier with Residual**（低成本）：  
   `[d_model + physchem] → 512 → 256 → 3` 加上殘差連接。當前只有 2 層 MLP。

3. **Ligand-Conditioned Classifier**（中成本）：  
   在 Classifier 輸入處額外 concat 一個 per-sample 的 ligand global vector（mean pool of ligand encoder outputs）。讓 classifier 知道"面對這個特定配體時，這個殘基多重要"。

---

### 五、Loss 設計

#### 5.1 SASA 輔助損失（高性價比）

SASA 特徵**已在 89D spatial_scalar 中預計算**，是 free supervision。

```
protein_encoder_output → side_branch_MLP → predicted_SASA
                                           ↓ L2 loss with precomputed SASA
```

訓練編碼器去預測 SASA 強迫它學到"表面暴露"信息，這正是 binding site 最需要的幾何認知。

#### 5.2 距離迴歸輔助頭（需要標籤）

已有 `distances_file: "data/residue_ligand_distances_dcenter.pt"`，可以直接用。

```
protein_encoder_output → regression_head → predicted_distance
                                            ↓ Huber loss with true distance
```

迴歸頭提供比 3-class 更細的梯度信號，特別對 5Å 邊界附近的殘基。

#### 5.3 Near Class Tversky（可選）

目前 Tversky 只針對 Contact class（class=0）。可以加一個 Near class 的 Tversky（較小 weight）：

```python
# Contact: alpha=0.3, beta=0.7 (recall-biased)
# Near:    alpha=0.5, beta=0.5 (balanced Dice)
tversky_near_weight: 0.1
```

這讓 Near class 的邊界識別也有明確梯度方向。

#### 5.4 R-Drop 正則化

訓練時對同一樣本做兩次前向（不同 dropout mask），用 KL 散度懲罰兩次輸出的差異：

```
loss = loss_CE + alpha * KL(p1 || p2)
```

R-Drop 在 NLP 分類任務上普遍有 1-2% 提升，在小數據集上更顯著。  
**→ 建議調研**：Liang et al. NeurIPS 2021 "R-Drop: Regularized Dropout for Neural Networks"

---

### 六、超參數調整

| 問題 | 當前值 | 建議 | 原因 |
|------|--------|------|------|
| LR 分層 | 統一 1e-4 | ESM proj: 5e-5, 其他: 1e-4 | ESM 特徵是預計算的，投影層更新太快容易過擬合 |
| weight_decay 排除項 | 未明確 | 排除 bias + LayerNorm | 這些參數加 WD 有害無益 |
| warmup_epochs | 5 | 建議 8-10 | focal_gamma=2.5 + class_weights=[8,3,1] 組合強，早期不穩定 |
| TTA（測試時增強）| 未啟用 | val/test 用多 NMA conformation 均值 | 現在 val 只用 conf0，NMA 增強的 val 收益未利用 |

---

### 七、調研方向（需要你配合查文獻）

1. **HSE 特徵**：Half-Sphere Exposure — 改善殘基暴露描述
2. **Multi-scale protein graph**：GearNet (Zhang 2022) / HoloProt (Somnath 2021) — 多尺度幾何
3. **R-Drop 正則化**：Liang NeurIPS 2021 — 訓練一致性約束
4. **Pocket prediction 監督信號**：FPocket / P2Rank 的 score 作為輔助標籤預訓練（需公開工具生成）
5. **Label smoothing 替代方案**：用 distance-based soft label 替代 hard threshold（需重算 labels）

---

### 優先級總結

| 優先級 | 改動 | 難度 | 預期收益 |
|--------|------|------|----------|
| 🔴 高 | Fix ligand global token（動態化） | 小 | 配體區分能力顯著提升 |
| 🔴 高 | SASA 輔助迴歸損失 | 小 | 免費 supervision，強化表面感知 |
| 🔴 高 | 距離迴歸輔助頭 | 小 | 邊界殘基更細緻梯度 |
| 🟡 中 | Multi-scale GVP（8Å+16Å） | 中 | 補充 pocket depth 感知 |
| 🟡 中 | R-Drop 正則化 | 小 | 減少泛化 gap |
| 🟡 中 | Ligand-conditioned classifier | 中 | 配體感知分類 |
| 🟢 低 | Per-layer distance bias | 小 | 細節優化 |
| 🟢 低 | Label redesign（d_min vs d_center）| 需重算特徵 | 減少 label noise |

---

最建議先做哪個方向，你有傾向嗎？「Fix ligand global token + SASA 輔助損失」是我認為改動最小但收益最確定的組合，可以作為 v10m5 的主線。