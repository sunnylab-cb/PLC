# Pocket-GVP-Fusion 版本说明

本文档记录了Pocket-GVP-Fusion模型的各个版本及其主要变化。

---

## 总体简介（如何阅读/如何维护）

- **目的**：用"面向维护"的方式记录每次版本迭代的**动机**、**改动点**、**影响范围**与**复现实例**，方便回溯与对齐训练/评测脚本。
- **顺序**：按时间从新到旧（最新版本放最上面）。
- **最小必填**（每个版本建议都包含）：`主要特性`、`关键改进/变化`、`文件结构`、`配置变化`、`使用方法`。
- **以代码为准**：如文档与实现不一致，请在对应版本条目里标注"以哪些文件为准"，并给出路径（例如：`configs/...`、`src/models/...`、`scripts/...`）。
- **关联文档**：模型结构图见 `MODEL_ARCHITECTURE_GRAPH.md`（如需更新，请同步更新对应版本条目中的"架构改进"描述）。

## 总体结论（版本演进主线）

- 迭代主线通常围绕三件事：**标签/物理定义更准确**（例如距离阈值）、**特征更丰富**（例如配体/蛋白输入特征）、**模态融合更强**（例如交叉注意力从单向到双向）。
- 维护建议：每次升级尽量明确回答三问：**为什么改**（问题/瓶颈）、**改了什么**（模块/超参/数据）、**怎么验证**（复现实验命令+预期变化）。

## 维护模板（复制即可）

> 新版本建议直接复制以下模板，填完后放到本文档最上方（在当前最新版本之前）。

````markdown
## V? - YYYY-MM-DD

### 主要特性
- ✅ ...

### 关键改进/变化
1. ...

### 影响范围（改了哪些东西）
- 配置：`configs/...`
- 代码：`src/...`
- 脚本：`scripts/...`
- 数据/标签：`data/...`

### 文件结构
```
configs/
  └── ...
src/
  └── ...
scripts/
  └── ...
```

### 配置变化
```yaml
model:
  # ...
data:
  # ...
```

### 使用方法（最小可复现）
```bash
# 训练
python scripts/train_?.py --config configs/?.yaml --amp

# 评测
python scripts/evaluate.py --config configs/?.yaml --ckpt outputs/.../best.pt
```

### 兼容性/迁移说明（可选）
- 与 V? checkpoint 是否兼容：是/否（如否，给出转换脚本/原因）
- 需要重新生成哪些中间产物：...

### 已知问题（可选）
- ...

---
````

---

## V10m6 - 2026-04-12

### 主要特性
- ✅ **Batched GVP Graph Construction**：per-sample for-loop → 單次 batched GVP forward pass（預估 2-4x 訓練加速）
- ✅ **向量化 Boundary Loss**：雙層 Python 迴圈 → 全 tensor 操作（預估 eval 2-5x 加速）

### 关键改进/变化

**模型層面（model file）：**
1. GVP 處理流程重構：
   - 舊：`for b in range(B)` 逐 sample 建圖 + 逐 sample 跑 GVP encoder
   - 新：所有 sample 的 valid residues flatten 成一個 flat tensor，per-sample 建圖後 offset edge indices 合併，**一次** GVP encoder forward pass，再 scatter 回 `[B, L, d]`
   - `build_protein_graph` 仍 per-sample 調用（KNN + backbone edges 需要 sample 隔離），但 GVP encoder 只跑一次
2. 繼承 V10m5 全部改動（Dynamic Ligand Global Token、gradient checkpointing fix、loss smart gating、cache_intermediates config）

**Loss 層面（boundary_loss.py，所有版本受益）：**
1. `compute_boundary_mask` 向量化重寫：
   - 舊：`for b in range(B): for i in valid_indices:` — O(B × N²) Python 迴圈
   - 新：`torch.cdist` → boolean mask → `.any(dim=2)` — 零 Python 迴圈
   - 輸出完全一致，純加速

### 影響範圍
- 配置：`configs/pocket_gvp_v10m6.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m6.py`（新建）
- 腳本：`scripts/train/train_v10m6.py`（新建）
- Loss：`src/losses/boundary_loss.py`（原地修改，所有版本受益）

### 文件结构
```
configs/
  └── pocket_gvp_v10m6.yaml
src/models/
  └── pocket_gvp_fusion_v10m6.py   # 新建，Batched GVP
src/losses/
  └── boundary_loss.py             # 原地修改，向量化
scripts/train/
  └── train_v10m6.py               # 新建
```

### 配置变化
```yaml
# 無新增配置項，僅 experiment.name 更新
experiment:
  name: "pocket_gvp_fusion_v10m6"
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m6.py --config configs/pocket_gvp_v10m6.yaml

# 分布式訓練
torchrun --nproc_per_node=4 scripts/train/train_v10m6.py --config configs/pocket_gvp_v10m6.yaml
```

### 兼容性/迁移说明
- 與 V10m5 checkpoint **不兼容**（class 名稱改為 `PocketGVPFusionV10m6`）
- 模型權重結構完全相同，僅 forward 計算路徑改變（數值等價）
- 無需重新生成特徵資料或距離標籤

### 已知問題
- `build_protein_graph` 仍 per-sample 調用（含 KNN），完全向量化需改為 batched KNN（PyG `knn_graph` 支援 `batch` 參數），留待後續
- `boundary_loss` 向量化版本在 N 很大時（>2000）會分配 `[B, N, N]` 記憶體，可能 OOM；建議 `max_seq_len ≤ 1800`

---

## V10m5 - 2026-04-12

### 主要特性
- ✅ **Dynamic Ligand Global Token**：固定 `nn.Parameter` → masked mean pool over ligand GNN outputs
- ✅ **Gradient Checkpointing Bug Fix**：啟用 checkpointing 時不再遺失 `coords` 和 `position_ids`
- ✅ **Loss Smart Gating**：weight=0 的 loss 分支完全跳過計算（含 softmax/one_hot）
- ✅ **可配置 cache_intermediates**：TensorBoard Health 診斷的中間 tensor cache 改為 config 控制
- ✅ **Label 閾值對齊 PoC**：Contact < 4.0Å / Near 4.0–6.0Å / Far > 6.0Å

### 关键改进/变化

**模型層面（model file）：**
1. Dynamic Ligand Global Token：
   - 舊：`self.ligand_global_token = nn.Parameter(torch.randn(1, 1, d_model))` — 所有配體共享同一固定向量
   - 新：`global_token = masked_mean_pool(h_ligand, ligand_mask)` — 每個配體有獨立的 global representation
   - 效果：cross-attention 第一層就能區分不同配體，零新增參數
2. Gradient Checkpointing Fix：
   - 舊：`checkpoint(encoder, h_seq, mask)` — 遺失 `coords`（distance bias）和 `position_ids`（chain-gap RoPE）
   - 新：用 closure 包裝 `coords` 和 `position_ids`，完整傳入
3. `_cached_*` 改為 `cache_intermediates` config 控制（預設 `false`，省記憶體）

**Loss 層面（train script）：**
1. `compute_combined_loss_v4` smart gating：
   - `focal_weight=0` → 跳過整個 focal 計算
   - `tversky_weight=0` → 跳過（原有）
   - `weighted_dice_weight=0` → 跳過（原有）
   - 共用 tensor（log_softmax、one_hot）lazy init，多個 loss 啟用時不重複計算

**標籤定義：**
- `label_contact_thresh`: 5.0 → 4.0（對齊 PoC）
- `label_near_thresh`: 8.0 → 6.0（對齊 PoC）

### 影響範圍
- 配置：`configs/pocket_gvp_v10m5.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m5.py`（新建）
- 腳本：`scripts/train/train_v10m5.py`（新建）

### 文件结构
```
configs/
  └── pocket_gvp_v10m5.yaml
src/models/
  └── pocket_gvp_fusion_v10m5.py   # 新建
scripts/train/
  └── train_v10m5.py               # 新建
```

### 配置变化
```yaml
model:
  cache_intermediates: false  # NEW: 控制 _cached_* 中間 tensor（TensorBoard Health 用）

data:
  label_contact_thresh: 4.0   # 5.0 → 4.0（對齊 PoC）
  label_near_thresh: 6.0      # 8.0 → 6.0（對齊 PoC）
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m5.py --config configs/pocket_gvp_v10m5.yaml

# 分布式訓練
torchrun --nproc_per_node=4 scripts/train/train_v10m5.py --config configs/pocket_gvp_v10m5.yaml
```

### 兼容性/迁移说明
- 與 V10m4 checkpoint **不兼容**（class 名稱改為 `PocketGVPFusionV10m5`；`ligand_global_token` 參數被移除）
- 標籤閾值改變（4/6 vs 5/8），**必須重新訓練**，不可直接 finetune V10m4 checkpoint
- 無需重新生成特徵資料；距離標籤檔案不變（閾值在 dataloader 動態計算）

---

## V10m4 - 2026-04-05

### 主要特性
- ✅ 修復 Dice Loss 主導訓練導致泛化災難（global batch Dice → per-sample Tversky）
- ✅ 新增 GVP Output Dropout（sample-level，對稱於 ESM dropout）
- ✅ 新增 CA Coordinate Noise（結構擾動增強，影響 KNN 圖拓撲）
- ✅ 三層 GVP 正則化機制完整化

### 关键改进/变化

**Loss 層面（train script）：**
1. 移除 global batch Dice Loss（根因：全局聚合使少數 Contact class 梯度被 Far 稀釋，訓練不穩定）
2. 新增 `compute_tversky_loss`：per-sample、Contact-only、smooth=1.0、alpha=0.3/beta=0.7
   - alpha < beta：允許適量 FP，強調 recall（不漏 binding site）
   - smooth=1.0：訓練早期梯度穩定，杜絕梯度爆炸
3. 新增 `compute_weighted_dice_loss`：per-sample、三 class、class_weights 正規化加權（可選）
4. 統一 `compute_combined_loss_v4`，支援 sample_weights（兼容 cluster dedup）
5. label_smoothing: 0.15 → 0.05（過高 smoothing 對少數 Contact class 更不利）
6. focal_gamma: 2.0 → 2.5（加強 hard example mining）
7. class_weights: [7,3,2] → [8,3,1]（Contact↑、Far↓）

**模型層面（model file）：**
1. GVP Output Dropout（L2）：訓練時以 `gvp_dropout_prob` 機率將整個樣本的 GVP 輸出 mask 為零
   - 迫使模型不過度依賴幾何特徵，與 ESM dropout 對稱
2. Coordinate Noise（L3）：在 `build_protein_graph` 前對 CA/N/C 座標加相同 Gaussian noise
   - 同時改變 KNN 圖拓撲、邊距離和 local backbone frame，結構正則化更強
   - 僅訓練期間啟用

**蛋白 GVP 邊特徵真實化（`build_protein_graph`）：**
- 舊：`edge_s [E,1]` = 原始距離；`edge_v [E,1,3]` = 全局方向
- 新：`edge_s [E,20]` = RBF(16) + seq offset 4-class；`edge_v [E,2,3]` = global + local backbone frame
- 強制保留 backbone i±1 邊（coord noise 下不丟失）
- Sequential offset 4-class: backbone / short / medium / long range → 模型能區分肽鍵與遠程接觸
- N-CA-C local frame（第 2 個 edge vector）：模型學 frame-relative 方向，有利 H-bond 幾何

**配體側邊特徵增強：**
- 舊：RDKit 化學邊 6D（bond type/ring/aromaticity），缺乏 3D 距離信息
- 新：化學邊 6D + RBF(原子間距, 8 bins, 0-5Å) = 14D
- 在 forward 中在線計算，無需重新生成 ligand feature 資料

**3D Distance-Biased Attention（蛋白自注意力）：**
- 在 protein context encoder（4 層 RoPE Transformer）的 self-attention 加入空間距離偏置
- `DistanceBias` 模塊：CA-CA pairwise distances → RBF(16) → Linear → per-head bias `[B, n_heads, L, L]`
- 零初始化：訓練起點等同標準 attention，偏置逐步學習（訓練穩定）
- 距離矩陣預計算一次，所有 4 層共享
- **無洩題**：僅使用蛋白質自身 CA 座標（protein-internal）；不涉及配體位置

**三層 GVP 正則化完整體系：**
```
L1: internal GVP Dropout  (layer-level scalar+vector channel dropout, drop_rate=dropout=0.1)
L2: gvp_dropout_prob=0.1  (output-level sample mask, NEW in V10m4)
L3: coord_noise_std=0.5Å  (input coord Gaussian noise, NEW in V10m4)
```

### 影響範圍

- 配置：`configs/pocket_gvp_v10m4.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m4.py`（新建，繼承 V10m3 架構）
- 腳本：`scripts/train/train_v10m4.py`（新建）

### 文件结构
```
configs/
  └── pocket_gvp_v10m4.yaml
src/models/
  └── pocket_gvp_fusion_v10m4.py   # 新建，架構同 V10m3 + GVP dropout
scripts/train/
  └── train_v10m4.py               # 新建，新 loss 函數
```

### 配置变化
```yaml
model:
  gvp_dropout_prob: 0.1       # NEW: GVP output sample-level dropout
  coord_noise_std: 0.5        # NEW: CA coord noise (Å)
  gvp_edge_rbf_bins: 16       # NEW: protein edge RBF bins
  gvp_edge_rbf_dmax: 20.0     # NEW: protein edge RBF max dist (Å)
  ligand_edge_rbf_bins: 8     # NEW: ligand edge RBF bins (appended to chemical bond 6D)

training:
  focal_gamma: 2.5          # 2.0 → 2.5
  class_weights: [8.0, 3.0, 1.0]  # [7,3,2] → [8,3,1]
  focal_weight: 0.7
  tversky_weight: 0.3       # NEW: per-sample Contact-only Tversky
  tversky_alpha: 0.3
  tversky_beta: 0.7
  tversky_smooth: 1.0
  weighted_dice_weight: 0.0 # optional, disabled by default
  label_smoothing: 0.05     # 0.15 → 0.05
  dice_weight: 0.0          # deprecated
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m4.py --config configs/pocket_gvp_v10m4.yaml

# 純 Focal 基線對比（在 config 中設置 tversky_weight: 0.0, focal_weight: 1.0）
python scripts/train/train_v10m4.py --config configs/pocket_gvp_v10m4.yaml

# 評測（使用 V10m3 推理腳本，模型架構相同）
python scripts/infer/infer_v10m3.py --checkpoint outputs/.../best_model.pt ...
```

### 兼容性/迁移说明
- 與 V10m3 checkpoint **不兼容**（class 名稱從 `PocketGVPFusionV10m3` 改為 `PocketGVPFusionV10m4`）
- 需重新訓練；無需重新生成特徵資料或距離標籤

### 設計動機（技術細節）
global batch Dice 的失效機制：
1. 全 batch 聚合 → Contact 5% tokens 對 Loss 貢獻遠小於 Far 70% tokens
2. 初期隨機預測 → Dice(Contact)≈0 → 巨大梯度 → 模型瘋狂預測 Contact
3. train Dice↑ 但 precision 崩潰 → test 嚴重 over-prediction → 泛化災難

Tversky 修復：per-sample + Contact-only + smooth=1.0 → 梯度穩定、方向正確

---

## V10m3 - 2026-04-04

### 主要特性
- ✅ **d_center 标签体系回归**: 从 VDW 距离标签切换回经典重原子中心距离（d_center）
- ✅ **主流 benchmark 对齐**: Contact < 5.0Å / Near 5.0–8.0Å / Far > 8.0Å，与 BioLiP / P2Rank / COACH 一致
- ✅ **TensorBoard Val/ 修复**: 移除 `_original` 路由逻辑，d_center 指标直接写入 `Val/`
- ✅ **Health/ 诊断恢复**: 补全 `_cached_*` 属性赋值，contribution_rate 和 gate_openness 正常记录

### 关键改进/变化

#### 1. 标签体系切换
- **动机**: VDW 距离标签（V10m2）与主流 benchmark 不对齐，难以横向比较
- **改动**: `PocketDatasetV8` 新增 `label_contact_thresh` / `label_near_thresh` 参数（默认保持旧值兼容）
- **V10m3 阈值**: Contact < 5.0Å，Near 5.0–8.0Å，Far > 8.0Å
- **距离文件**: `data/residue_ligand_distances_dcenter.pt`（重原子中心最小距离，19029 对）

#### 2. 距离文件重新生成
- **问题**: `residue_ligand_distances_v2.pt` 只有 13374 条（脚本硬编码了错误路径 `P-L/`）
- **修复**: 新建 `preparation/tag/calculate_distances_dcenter.py`，`--pl-dir` 默认指向 `New/`

#### 3. TensorBoard 修复
- **问题**: `Val/` 命名空间被 `if 'loss_original' in val_metrics` 门控，v10m3 无 `distances_file_original` 时永远为空
- **修复**: 移除 `_original` 分支，d_center 指标直接写入 `Val/`

#### 4. Health/ 诊断恢复
- **问题**: v10m2/m3 的 `forward` 未设置 `self._cached_*` 属性，训练脚本 `hasattr` 检查静默跳过
- **修复**: 在 `forward` 中补全 `_cached_gvp_scalar/vector/h_geo/current_esm/h_gvp_proj/h_esm_proj/gate`

### 影响范围
- 配置：`configs/pocket_gvp_v10m3.yaml`（新增）
- 模型：`src/models/pocket_gvp_fusion_v10m3.py`（新增，含 `_cached_*` 修复）
- 脚本：`scripts/train/train_v10m3.py`（新增）、`scripts/infer/infer_v10m3.py`（新增）
- 数据：`preparation/tag/calculate_distances_dcenter.py`（新增）
- 数据集：`src/data/pocket_dataset_v8.py`（新增 `label_contact_thresh/label_near_thresh` 参数）

### 文件结构
```
configs/
  └── pocket_gvp_v10m3.yaml
src/
  ├── models/
  │   └── pocket_gvp_fusion_v10m3.py
  └── data/
      └── pocket_dataset_v8.py          # 新增阈值参数
scripts/
  ├── train/
  │   └── train_v10m3.py
  └── infer/
      └── infer_v10m3.py
preparation/tag/
  └── calculate_distances_dcenter.py
data/
  └── residue_ligand_distances_dcenter.pt  # 需先运行生成脚本
```

### 配置变化
```yaml
data:
  distances_file: "data/residue_ligand_distances_dcenter.pt"
  label_contact_thresh: 5.0   # Contact: d_center < 5.0Å
  label_near_thresh: 8.0      # Near: 5.0–8.0Å; Far: >8.0Å

training:
  class_weights: [4.0, 1.0, 2.0]   # Contact 类更大，权重相应调整
```

### 使用方法（最小可复现）
```bash
# 1. 生成 d_center 距离文件（首次需要）
python preparation/tag/calculate_distances_dcenter.py --workers 64

# 2. 训练
python scripts/train/train_v10m3.py --config configs/pocket_gvp_v10m3.yaml --amp

# 3. 推理（评估模式）
python scripts/infer/infer_v10m3.py \
    --checkpoint outputs/xxx/best_model.pt \
    --split_file configs/splits/splits.txt \
    --evaluate --output results/

# 4. 全局推理（test set 分类）
python scripts/infer/infer_v10m3.py \
    --checkpoint outputs/xxx/best_model.pt \
    --split_file configs/splits/splits.txt \
    --global_infer --output results/
```

### 兼容性/迁移说明
- 与 V10m2 checkpoint **不兼容**（标签体系不同，需重新训练）
- 需要先运行 `calculate_distances_dcenter.py` 生成距离文件
- `PocketDatasetV8` 新参数有默认值，V10m1/m2 训练脚本无需修改

---

## V10m2 - 2026-04-03

### 主要特性
- ✅ **RoPE 维度 Bug 修复**: `cos/sin` 切片到 `head_dim//2` 后再 unsqueeze，消除 [32 vs 64] 广播错误
- ✅ **b_gvp 维度修复**: GVP 内部输出 256D → 投影到 d_model 512D，消除 `stack` size mismatch
- ✅ **Health/ 诊断恢复**: 补全 `_cached_*` 属性赋值

### 关键改进/变化

#### 1. RoPE 维度修复
- **错误**: `apply_rotary_pos_emb` 中 `cos [L, 64]` 直接 unsqueeze 后与 `x1 [B,H,L,32]` 广播失败
- **修复**: `half = x1.shape[-1]; cos = cos[:, :half].unsqueeze(0).unsqueeze(0)`

#### 2. b_gvp 投影修复
- **错误**: `InterBlockAttnRes.forward` 中 `torch.stack([b_gvp, b0, b1])` 要求三者最后维度相同，但 `b_gvp = h_s_gvp` 为 256D，b0/b1 为 512D
- **修复**: `__init__` 中新增 `self.b_gvp_proj = nn.Linear(d_gvp_internal, d_model, bias=False)`

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10m2.py`（修复 + `_cached_*`）
- 脚本：`scripts/train/train_v10m2.py`

### 文件结构
```
src/models/pocket_gvp_fusion_v10m2.py
scripts/train/train_v10m2.py
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v10m2.py --config configs/pocket_gvp_v10.yaml --amp
```

### 兼容性/迁移说明
- 与 V10m1 checkpoint **不兼容**（新增 `b_gvp_proj` 参数）
- 使用与 V10m1 相同的 VDW 距离标签和配置文件

---

## V10m1 - 2026-04-01

### 主要特性
- ✅ **RoPE (Rotary Position Embedding)**: Transformer 中引入相对位置编码
- ✅ **Ligand-side AttnRes**: 配体侧也引入 Inter-Block Attention Residuals
- ✅ **b_gvp 额外 block**: GVP 纯几何输出作为第三个 AttnRes block，几何信号直达后期交叉注意力

### 关键改进/变化

#### 1. RoPE
- 替换 `ProteinContextEncoder` 中的标准 MHA 为 `RoPEGatedMultiHeadAttention`
- 使用 `RotaryEmbedding` 对 Q/K 施加旋转位置编码，支持相对位置感知

#### 2. Ligand-side AttnRes
- V10 只有蛋白侧 AttnRes（b0, b1）；V10m1 新增配体侧 `lig_attn_res`（b_lig0）
- 每层交叉注意力前，配体表征也回溯早期编码器输出

#### 3. b_gvp block
- Block 划分：b_gvp（GVP 输出）、b0（融合后）、b1（Transformer 后）
- 蛋白侧 AttnRes 从 2 个 block 扩展到 3 个

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10m1.py`（新增）
- 脚本：`scripts/train/train_v10m1.py`（新增）

### 文件结构
```
src/models/pocket_gvp_fusion_v10m1.py
scripts/train/train_v10m1.py
```

### 配置变化
```yaml
# 复用 configs/pocket_gvp_v10.yaml，无新增配置项
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v10m1.py --config configs/pocket_gvp_v10.yaml --amp
```

### 兼容性/迁移说明
- 与 V10t checkpoint **不兼容**（新增 `lig_attn_res`、`b_gvp_proj` 参数）

---

## V10t - 2026-03-28

### 主要特性
- ✅ **Inter-Block Attention Residuals (AttnRes)**: 蛋白侧跨 block 注意力残差（参考 Kimi 2025）
- ✅ **BF16 训练支持**: 替代 FP16，在 AMD GPU 上更稳定
- ✅ **梯度累积**: 支持 `accumulation_steps` 配置
- ✅ **异步数据传输**: `non_blocking=True` + `pin_memory`
- ✅ **Tensor 累积器**: 减少 CPU-GPU 同步次数，提升训练吞吐

### 关键改进/变化

#### 1. Inter-Block AttnRes
- 每层交叉注意力前，蛋白表征通过 `InterBlockAttnRes` 融合 b0（融合后）和 b1（Transformer 后）
- 使用可学习的 softmax 权重，初始化为零（保守初始化）
- 允许模型在不同抽象层次的表征间动态选择

#### 2. 训练效率优化
- BF16：`torch.bfloat16` 替代 `torch.float16`，数值范围更大，溢出风险更低
- 梯度累积：等效扩大 batch size，适合显存受限场景
- 异步传输：DataLoader `pin_memory=True` + `.to(device, non_blocking=True)`

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10t.py`（新增）
- 脚本：`scripts/train/train_v10t.py`（新增）
- 配置：`configs/pocket_gvp_v10.yaml`（新增）

### 文件结构
```
configs/
  └── pocket_gvp_v10.yaml
src/models/
  └── pocket_gvp_fusion_v10t.py
scripts/train/
  └── train_v10t.py
```

### 配置变化
```yaml
training:
  use_bf16: false          # V10: BF16 支持（默认关闭，CLI --amp 开启）
  accumulation_steps: 1    # V10: 梯度累积步数
  pin_memory: true         # V10: 异步数据传输
  prefetch_factor: 2
  cudnn_benchmark: false
```

### 使用方法（最小可复现）
```bash
# 单 GPU，BF16
python scripts/train/train_v10t.py --config configs/pocket_gvp_v10.yaml --amp

# 多 GPU
torchrun --nproc_per_node=4 scripts/train/train_v10t.py --config configs/pocket_gvp_v10.yaml
```

### 兼容性/迁移说明
- 与 V9m2 checkpoint **不兼容**（新增 AttnRes 模块）
- 继承 V9m2 的所有特征（complete_features_v4、NMA、VDW 标签）

---

## V9m2 - 2026-03-02

### 主要特性
- ✅ **Flash Attention**: 使用triton版本加速训练（ROCm兼容），预期加速1.5-2x
- ✅ **Enhanced Label Smoothing**: 从0.1提升到0.15，提高泛化能力
- ✅ **Boundary Loss**: 对结合位点边界残基施加额外权重，提升边界预测精度

### 关键改进/变化

#### 1. Flash Attention集成
- **动机**: 加速训练，减少显存占用
- **实现**: 在`GatedMultiHeadAttention`中添加Flash Attention支持
- **特点**:
  - 使用triton版本，兼容ROCm
  - 自动fallback到标准attention（当mask存在或cross-attention时）
  - 仅在训练时启用，推理时使用标准attention
- **性能**: 预期训练速度提升1.5-2x，显存占用减少~20%

#### 2. Enhanced Label Smoothing
- **动机**: 提高模型泛化能力，减少过拟合
- **改进**: V9的0.1 → V9m2的0.15
- **原理**: 软化标签分布 `y_smooth = (1-ε)*y_hard + ε/K`，防止模型对预测过于自信

#### 3. Boundary Loss
- **动机**: 结合位点边界残基最难预测，需要额外关注
- **实现**:
  - 识别边界残基：在8Å范围内存在不同类别邻居的残基
  - 对边界残基施加2x权重
  - 损失组合：`Total = 0.4*Dice + 0.4*Focal + 0.2*Boundary`
- **预期**: 边界预测F1提升2-3%

### 影响范围（改了哪些东西）

#### 新增文件
- `src/models/pocket_gvp_fusion_v9m2.py` - V9m2模型
- `src/losses/boundary_loss.py` - 边界损失实现
- `scripts/train/train_v9m2.py` - V9m2训练脚本（内置默认参数）
- `scripts/test_v9m2_setup.py` - 设置测试脚本
- `TODO.md` - 未来优化方向（10大类，40+具体建议）

#### 修改文件
- `src/models/transformer_layers.py` - 添加Flash Attention支持
  - `GatedMultiHeadAttention`: 新增`use_flash_attn`参数
  - `GatedTransformerBlock`: 传递`use_flash_attn`参数
  - `ProteinContextEncoder`: 传递`use_flash_attn`参数
- `src/losses/__init__.py` - 导出boundary loss函数

#### 配置说明
- V9m2**复用v9配置文件** (`configs/pocket_gvp_v9.yaml`)
- V9m2特性通过训练脚本的默认参数启用
- 要启用完整V9m2功能，需在v9配置中添加相应参数

### 文件结构
```
configs/
  └── pocket_gvp_v9.yaml            # V9m2使用v9配置
src/
  ├── models/
  │   ├── pocket_gvp_fusion_v9m2.py # V9m2模型
  │   └── transformer_layers.py     # 添加Flash Attention支持
  └── losses/
      ├── boundary_loss.py          # 边界损失
      └── __init__.py               # 更新导出
scripts/
  ├── train/
  │   └── train_v9m2.py             # V9m2训练脚本（内置默认参数）
  └── test_v9m2_setup.py            # 设置测试
TODO.md                              # 未来优化方向
```

### 配置变化
```yaml
# V9m2使用v9配置文件: configs/pocket_gvp_v9.yaml
# V9m2的新功能通过训练脚本的默认参数启用

# 在train_v9m2.py中的默认值:
use_boundary_loss: false (默认)      # 设置为true启用boundary loss
boundary_weight: 0.2                 # boundary loss权重
boundary_multiplier: 2.0             # 边界残基权重倍数
boundary_distance_threshold: 8.0     # 边界判定距离阈值(Å)
label_smoothing: 0.15                # 从0.1提升到0.15
use_flash_attn: true                 # 在model配置中设置

# 要启用V9m2的完整功能，在v9配置中添加:
model:
  use_flash_attn: true               # 启用Flash Attention

training:
  label_smoothing: 0.15              # 增强label smoothing
  use_boundary_loss: true            # 启用boundary loss
  boundary_weight: 0.2
  boundary_multiplier: 2.0
  boundary_distance_threshold: 8.0
  dice_weight: 0.4                   # 调整以平衡boundary loss
  focal_weight: 0.4
```

### 使用方法（最小可复现）
```bash
# 测试设置
python scripts/test_v9m2_setup.py

# 单GPU训练（使用v9配置）
python scripts/train/train_v9m2.py --config configs/pocket_gvp_v9.yaml

# 多GPU训练 (DDP)
torchrun --nproc_per_node=4 scripts/train/train_v9m2.py --config configs/pocket_gvp_v9.yaml

# V9m2默认启用以下功能（通过脚本默认参数）:
# - label_smoothing: 0.15
# - 其他V9m2功能需要在配置文件中显式启用

# 要启用完整V9m2功能，修改configs/pocket_gvp_v9.yaml:
# model:
#   use_flash_attn: true
# training:
#   label_smoothing: 0.15
#   use_boundary_loss: true
#   boundary_weight: 0.2
#   dice_weight: 0.4
#   focal_weight: 0.4
```

### 测试结果
```
✓ Flash Attention 已安装 (v2.8.3)
✓ V9m2 模型实例化成功 (23.8M参数)
✓ Boundary Loss 计算成功
✓ 配置文件加载成功
✓ Transformer Layers 测试通过
```

### 预期性能提升
- **训练速度**: +50-100% (Flash Attention)
- **显存占用**: -20% (Flash Attention)
- **边界预测**: +2-3% F1 (Boundary Loss)
- **泛化能力**: +1-2% (Enhanced Label Smoothing)
- **整体性能**: 预期 +3-5% F1

### 兼容性/迁移说明
- 与 V9 checkpoint **不完全兼容**（模型结构相同，但配置不同）
- 可以加载V9权重，但需要调整配置文件
- 不需要重新生成特征（使用相同的complete_features_v3和complete_features_v3_nma）
- 不需要重新生成标签（使用相同的VDW距离标签）

### 已知问题
- Flash Attention在极长序列（>2000残基）时可能不稳定，建议禁用
- Boundary Loss计算对于大batch可能较慢，可考虑降低boundary_weight或禁用

### 下一步优化方向
详见 `TODO.md`，优先级建议：
- **短期** (1-2天): 启用AMP、Layer-wise LR Decay、Per-residue Confidence
- **中期** (1周): Contrastive Loss、NMA质量控制、快照集成
- **长期** (2周+): 半监督学习、Multi-scale Fusion、难样本挖掘

---

## V9 - 2026-02-24

### 主要特性
- ✅ **Stochastic Depth (Drop Path)**: Transformer和Cross-Attention层线性缩放的drop_path
- ✅ **Separate Attention Dropout**: 注意力权重矩阵独立dropout（与FFN dropout分离）
- ✅ **Classifier Dropout**: 分类头dropout提升至0.2
- ✅ **Weight Decay优化**: 从0.01提升到0.05，并通过param groups排除LayerNorm/bias
- ✅ **NMA数据增强**: 每个蛋白质随机采样多个构象（Normal Mode Analysis）

### 关键改进/变化

#### 1. Stochastic Depth (Drop Path)
- **动机**: 提高模型正则化，减少过拟合
- **实现**:
  - Transformer层：线性缩放drop_path_prob从0到0.1（按层递增）
  - Cross-Attention层：同样线性缩放（修复了V9初版的bug）
- **原理**: 训练时随机跳过残差路径，类似Dropout但作用于整个层

#### 2. Separate Attention Dropout
- **动机**: 注意力权重和FFN输出需要不同的dropout策略
- **实现**:
  - `attn_dropout`: 专门用于attention权重矩阵（默认0.1）
  - `dropout`: 用于FFN和其他地方（默认0.1）
- **效果**: 更精细的正则化控制

#### 3. Classifier Dropout增强
- **动机**: 分类头容易过拟合，需要更强的正则化
- **改进**: 从0.1提升到0.2

#### 4. Weight Decay优化
- **动机**: 更强的权重衰减，但排除不应该衰减的参数
- **实现**:
  ```python
  no_decay_keywords = ['bias', 'norm.weight', 'norm.bias', 'LayerNorm']
  param_groups = [
      {'params': decay_params, 'weight_decay': 0.05},
      {'params': no_decay_params, 'weight_decay': 0.0}
  ]
  ```

#### 5. NMA数据增强
- **动机**: 增加蛋白质构象多样性，提高模型对柔性的鲁棒性
- **实现**:
  - 使用ProDy ANM生成10个构象（conf0-conf9）
  - 训练时每个蛋白随机采样3个构象
  - conf0始终包含（原始构象），conf1-9随机采样
  - ESM/Ligand/标签不变，只换蛋白质坐标相关特征
- **数据**: `complete_features_v3_nma/` 目录
- **效果**: 训练集扩大3倍，验证/测试集仍用conf0

### 影响范围（改了哪些东西）

#### 新增文件
- `src/models/pocket_gvp_fusion_v9.py` - V9模型
- `configs/pocket_gvp_v9.yaml` - V9配置
- `scripts/train/train_v9.py` - V9训练脚本
- `preparation/protein/extract_complete_features_v3_nma.py` - NMA特征提取

#### 修改文件
- `src/models/transformer_layers.py` - 添加drop_path和attn_dropout支持
- `src/data/pocket_dataset_v8.py` - NMA数据加载支持

### 文件结构
```
configs/
  └── pocket_gvp_v9.yaml
src/
  ├── models/
  │   ├── pocket_gvp_fusion_v9.py
  │   └── transformer_layers.py      # 添加drop_path, attn_dropout
  └── data/
      └── pocket_dataset_v8.py       # NMA支持
scripts/
  └── train/
      └── train_v9.py
preparation/protein/
  └── extract_complete_features_v3_nma.py
data/
  └── complete_features_v3_nma/      # NMA特征
      └── YEAR_RANGE/
          └── PDB_ID/
              ├── {pdb_id}_protein_conf0.pt  # 原始
              ├── {pdb_id}_protein_conf1.pt
              └── ...
              └── {pdb_id}_protein_conf9.pt
```

### 配置变化
```yaml
# configs/pocket_gvp_v9.yaml

experiment:
  name: "pocket_gvp_fusion_v9"

data:
  nma_features_dir: "complete_features_v3_nma"  # V9: NMA特征目录
  nma_n_confs: 3                                # V9: 每个蛋白采样3个构象
  nma_max_conf: 9                               # V9: 最大构象索引

model:
  # V9: 正则化增强
  attn_dropout: 0.1                             # V9: 独立attention dropout
  classifier_dropout: 0.2                       # V9: 分类头dropout (从0.1提升)
  stochastic_depth_prob: 0.1                    # V9: 最大drop_path概率

training:
  weight_decay: 0.05                            # V9: 从0.01提升 (排除LayerNorm/bias)
  label_smoothing: 0.1                          # 继承V8
```

### 使用方法（最小可复现）
```bash
# 生成NMA特征（如果还没有）
python preparation/protein/extract_complete_features_v3_nma.py \
  --pdb_dir data/pdbs \
  --output_dir data/complete_features_v3_nma \
  --n_modes 10 \
  --n_confs 9 \
  --amplitude 1.0

# 训练
python scripts/train/train_v9.py --config configs/pocket_gvp_v9.yaml

# 多GPU训练
torchrun --nproc_per_node=4 scripts/train/train_v9.py --config configs/pocket_gvp_v9.yaml
```

### NMA特征生成参数
```python
# ProDy ANM参数
n_modes = 10          # 使用前10个低频振动模式
n_confs = 9           # 生成9个扰动构象（加上conf0共10个）
amplitude = 1.0       # 振幅缩放因子
cutoff = 15.0         # ANM cutoff距离(Å)
```

### 兼容性/迁移说明
- 与 V8 checkpoint **不兼容**（模型结构相同，但需要NMA数据）
- 需要生成NMA特征：运行`extract_complete_features_v3_nma.py`
- 不需要重新生成标签（使用相同的VDW距离标签）
- 可以禁用NMA回退到V8模式：设置`nma_n_confs: 1`

### 已知问题
- NMA特征生成较慢（每个蛋白~10-30秒）
- 需要大量存储空间（10x原始特征）
- 某些蛋白质结构可能导致ANM失败（自动跳过）

### 验证结果
- NMA覆盖率: 100% (19,029/19,029 proteins)
- 每个蛋白平均构象数: 10.0
- 数据集扩展: 19,029 → 57,087 samples (3x)
- Drop path线性缩放验证: [0.0, 0.033, 0.067, 0.1] ✓

---

## V8 - 2026-02-24

### 主要特性
- ✅ **Complete Features V3 Integration**: 使用 V3 特征提取（62D PhysChem + 72D Spatial Scalar + 6×3D Spatial Vectors）
- ✅ **Enhanced GVP with 10 Vectors**: 将 6 个 V3 化学矢量与 4 个 GVP 矢量结合（总共 10 个矢量）
- ✅ **Separated Feature Categories**: 物理化学、空间标量、空间矢量三类特征分离处理
- ✅ **Increased GVP Capacity**: GVP 矢量隐藏维度从 16 增至 20

### 关键改进/变化
1. **特征分离与专用处理**:
   - 62D 物理化学特征 → PhysChem MLP Encoder
   - 72D 空间标量特征 → GVP node_s
   - 6×3D 空间矢量特征 + 4×3D GVP 矢量 → GVP node_v (总共 10 个矢量)

2. **GVP 架构增强**:
   - `n_gvp_vectors`: 4 → 10 (增加 6 个 V3 化学矢量)
   - `n_gvp_vector_hidden`: 16 → 20 (增加矢量处理能力)
   - Bridge 输出: 136 → 210 维 (20 个矢量通道)

3. **V3 特征的 6 个空间矢量**:
   - Sidechain Base/CoM Vector: 侧链方向
   - Aromatic Normal Vector: 芳香环法向量
   - H-bond Donor/Acceptor Vector: 氢键方向
   - Cavity Pointing Vector: 空腔指向矢量

### 影响范围（改了哪些东西）
- 配置：`configs/pocket_gvp_v8.yaml`
- 模型：`src/models/pocket_gvp_fusion_v8.py`
- 数据集：`src/data/pocket_dataset_v7.py`
- 特征提取：`preparation/protein/extract_complete_features_v3.py`
- 文档：`DOCs/V3_FEATURE_EXTRACTION.md`

### 文件结构
```
configs/
  └── pocket_gvp_v8.yaml                    # V8 模型配置
src/
  ├── models/
  │   └── pocket_gvp_fusion_v8.py          # V8 模型实现
  └── data/
      └── pocket_dataset_v7.py              # V7 数据集（加载 V3 特征）
preparation/protein/
  ├── extract_complete_features_v3.py      # V3 特征提取脚本
  └── test_v3_features.py                  # V3 特征验证脚本
```

### 配置变化
```yaml
model:
  d_phys: 62                        # V8: 物理化学特征（从 94D 降至 62D）
  d_spatial_scalar: 72              # V8: 空间标量特征（新增）
  n_gvp_vectors: 10                 # V8: 10 个矢量（4 GVP + 6 V3）
  n_gvp_vector_hidden: 20           # V8: 矢量隐藏维度（从 16 增至 20）

data:
  protein_features_dir: "complete_features_v3"  # V8: 使用 V3 特征
  ligand_features_dir: "ligand_features_v4"     # 继续使用 V4 配体特征
```

### 使用方法（最小可复现）

#### 1. 提取 V3 蛋白质特征
```bash
# 批量提取整个目录
python preparation/protein/extract_complete_features_v3.py \
  --input data/processed/ \
  --output data/complete_features_v3/ \
  --pattern "*_protein.pdb" \
  --workers 32

# 验证提取的特征
python preparation/protein/test_v3_features.py \
  --input data/complete_features_v3/1981-2000/1a0q/1a0q_protein.pt
```

#### 2. 训练 V8 模型
```bash
# 基础训练
python scripts/train/train_v8.py \
  --config configs/pocket_gvp_v8.yaml

# 多 GPU 训练
torchrun --nproc_per_node=4 scripts/train/train_v8.py \
  --config configs/pocket_gvp_v8.yaml \
  --distributed
```

#### 3. 评估 V8 模型
```bash
python scripts/evaluate.py \
  --config configs/pocket_gvp_v8.yaml \
  --checkpoint outputs/pocket_gvp_fusion_v8/best.pt \
  --split test
```

### 兼容性/迁移说明
- **与 V7 checkpoint 不兼容**：GVP 输入维度不同（4 → 10 vectors）
- **需要重新提取 V3 蛋白质特征**：使用 `extract_complete_features_v3.py`
- **配体特征可复用**：继续使用 V4 配体特征（无需重新提取）
- **距离标签可复用**：继续使用 VDW-based 距离标签

### 已知问题
- ~~训练脚本待创建~~: ✅ `scripts/train/train_v8.py` 已创建
- **参数量增加**: Bridge 维度增大，需要监控过拟合风险
- **计算复杂度增加**: 10 个矢量 vs 4 个矢量，训练速度可能下降

---

## V7m3 - 2026-02-23

### 主要特性
- ✅ **Cluster-Size Inverse Weighting Loss**: 基于 MMseqs2 聚类大小的动态样本权重
- ✅ **Sample-wise Loss Weighting**: 对小簇（孤儿蛋白）施加更高的 Loss 惩罚
- ✅ **Improved Generalization**: 提高模型对低相似度样本的泛化能力
- ✅ **MMseqs2 30% Sequence Identity Clustering**: 使用更严格的数据划分避免数据泄露

### 关键改进/变化

1. **Cluster-Weighted Loss (V7m3)**
   - 权重公式: `weight = 1 / log(cluster_size + ε)`
   - 小簇（cluster_size 小）获得更高权重
   - 大簇（cluster_size 大）获得较低权重
   - 默认 ε = 2.0 防止权重过大

2. **数据划分更新**
   - 使用 `configs/splits/mmseqs_splits_strict.csv`
   - MMseqs2 参数: `--min-seq-id 0.3 -c 0.8`
   - 新增 cluster_size 列用于权重计算
   - 数据集: Train 15229, Val 1903, Test 1905 (Total 19037)

3. **实现细节**
   - 新增 `ClusterWeightedLoss` 类管理样本权重
   - 新增 `load_splits_with_cluster_size()` 函数读取聚类信息
   - 新增 `compute_combined_loss_with_sample_weights()` 函数
   - 向后兼容：如果没有 cluster_size 信息，自动回退到标准 loss

### 影响范围（改了哪些东西）
- 配置：`configs/pocket_gvp_v7.yaml` (split_file 路径更新)
- 代码：
  - `src/data/pocket_dataset.py` (新增 `load_splits_with_cluster_size`)
  - `scripts/train/train_v7m3.py` (新增 ClusterWeightedLoss 类和相关逻辑)
- 数据：`configs/splits/mmseqs_splits_strict.csv` (4列格式)

### 文件结构
```
configs/
  ├── pocket_gvp_v7.yaml          # 配置文件（split_file 更新）
  └── splits/
      └── mmseqs_splits_strict.csv  # 新格式：split\tpdb_id\trepresentative\tcluster_size
src/
  └── data/
      └── pocket_dataset.py        # 新增 load_splits_with_cluster_size()
scripts/
  └── train/
      └── train_v7m3.py            # V7m3 训练脚本
```

### 配置变化
```yaml
data:
  split_file: "configs/splits/mmseqs_splits_strict.csv"  # 更新为 CSV 格式
```

### 使用方法（最小可复现）
```bash
# 单GPU训练
python scripts/train/train_v7m3.py --config configs/pocket_gvp_v7.yaml

# 单GPU + torch.compile
python scripts/train/train_v7m3.py --config configs/pocket_gvp_v7.yaml --compile

# 多GPU分布式训练
torchrun --nproc_per_node=4 scripts/train/train_v7m3.py \
    --config configs/pocket_gvp_v7.yaml --compile
```

### 权重统计示例
```
Cluster-Weighted Loss Statistics:
  Total samples: 19037
  Weight range: [0.2891, 1.4427]
  Mean weight: 0.4523

  Top 5 highest-weighted samples (smallest clusters):
    1xyz: weight=1.4427, cluster_size=1  (孤儿蛋白)
    2abc: weight=1.2345, cluster_size=2
    ...

  Top 5 lowest-weighted samples (largest clusters):
    3ov1: weight=0.2891, cluster_size=2236  (大簇代表)
    ...
```

### 兼容性/迁移说明
- 与 V7m2 checkpoint 兼容（模型结构未变）
- 与 V7 checkpoint 兼容（模型结构未变）
- 需要使用新的 split 文件：`configs/splits/mmseqs_splits_strict.csv`
- 向后兼容：如果使用旧的 split 文件（无 cluster_size），自动回退到标准 loss

### 已知问题
- 无

---

