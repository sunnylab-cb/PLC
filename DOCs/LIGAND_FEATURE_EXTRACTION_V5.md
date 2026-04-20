# Ligand Feature Extraction V5 — 完整特徵說明

> 對應腳本：`preparation/ligand/extract_ligand_features_v5.py`  
> 依賴模組：`preparation/ligand/ligand_features_v4.py`（本地原子特徵）、`pypropel.mol`（全域標籤）

## 總覽

每個原子的最終特徵維度為 **109D**，由三部分組成：

| 區段 | 維度 | 來源 |
|------|------|------|
| Local atom features | 54D | `ligand_features_v4.py` |
| Global molecular tag | 45D | `pypropel.mol.get_global_tag()` |
| Normalized shape features | 10D | `ligand_features_v4.py` → V5 標準化 |
| **合計** | **109D** | — |

Global tag 和 shape features 以 broadcast 方式複製到每個原子上（同一分子的每個原子共享相同的全域向量）。

---

## 一、Local Atom Features（54D）

### 1.1 Base atom features（33D）

| 子特徵 | 維度 | 編碼方式 | 說明 |
|--------|------|----------|------|
| Atom type | 11 | One-hot | C, N, O, S, F, P, Cl, Br, I, H, Other |
| Hybridization | 6 | One-hot | SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED |
| Aromaticity | 1 | Binary | 是否芳香原子 |
| Ring membership | 7 | Binary | [是否在環上, 3-ring, 4-ring, 5-ring, 6-ring, 7-ring, 8-ring] |
| Degree features | 5 | Normalized float | degree/6, explicit_valence/6, total_valence/6, (formal_charge+2)/4, radical_electrons/2 |
| Formal charge class | 3 | One-hot | [負電荷, 中性, 正電荷] |

### 1.2 Electronic features（3D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Gasteiger charge | 1 | 原子部分電荷（RDKit `ComputeGasteigerCharges`） |
| Crippen LogP contribution | 1 | 原子對分子脂溶性的貢獻（`_CalcCrippenContribs`） |
| TPSA contribution | 1 | 原子對拓撲極性表面積的貢獻（`_CalcTPSAContribs`） |

### 1.3 Flexibility features（2D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| RMSF | 1 | 構象系綜的均方根漲落。生成 50 個構象（ETKDGv3 + MMFF 優化），對齊後計算每個原子的位置波動 |
| Flexibility flag | 1 | Binary，RMSF > 1.0 Å 則為 1（高度柔性原子） |

### 1.4 Advanced atom features（6D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Atom SASA | 1 | 溶劑可及表面積（`rdFreeSASA`），先做力場優化修復重合原子 |
| Lipophilic SASA | 1 | SASA × LogP contribution，衡量疏水暴露面 |
| H-bond donor | 1 | Binary，RDKit `BaseFeatures.fdef` 的 Donor 家族 |
| H-bond acceptor | 1 | Binary，RDKit `BaseFeatures.fdef` 的 Acceptor 家族 |
| Chirality CW | 1 | Binary，順時針手性（`CHI_TETRAHEDRAL_CW`） |
| Chirality CCW | 1 | Binary，逆時針手性（`CHI_TETRAHEDRAL_CCW`） |

### 1.5 Rotatable bond context（3D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Is on rotatable bond | 1 | Binary，原子直接參與可旋轉鍵 |
| Adjacent to rotatable bond | 1 | Binary，鄰接可旋轉鍵的原子 |
| Rotatable bond density | 1 | 參與可旋轉鍵的次數 / 4，clamp [0,1] |

### 1.6 Metal-binding flags（4D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Is metal-binding candidate | 1 | Binary，N(非芳香)/O/S 且非 H |
| Is N-type binder | 1 | Binary，非芳香 N（排除帶正電 N） |
| Is O-type binder | 1 | Binary |
| Is S-type binder | 1 | Binary |

### 1.7 Stereochemistry flags（3D）

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Has CIP code | 1 | Binary，是否有 CIP 立體化學標記 |
| Is R configuration | 1 | Binary |
| Is S configuration | 1 | Binary |

---

## 二、Global Molecular Tag（45D）

來自 `pypropel.mol.get_global_tag(mol, output_dim=45)`。

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Compressed Morgan fingerprint | 43 | Morgan FP（radius=2）的隨機投影壓縮向量，捕捉分子拓撲子結構指紋 |
| MolLogP | 1 | 分子脂水分配係數 |
| TPSA / 100 | 1 | 歸一化後的拓撲極性表面積 |

此向量描述的是**整個分子的全域化學性質**，broadcast 到每一個原子上。

---

## 三、Normalized Shape Features（10D）

由 V4 計算原始 shape 值，V5 對特定維度做 z-score 標準化。

| Index | 特徵 | 說明 | 是否標準化 |
|-------|------|------|-----------|
| 0 | PMI1 | 主慣性矩 1（最小） | 是 (μ=3879.9, σ=6650.0) |
| 1 | PMI2 | 主慣性矩 2 | 是 (μ=13954.5, σ=28178.6) |
| 2 | PMI3 | 主慣性矩 3（最大） | 是 (μ=15758.5, σ=30137.2) |
| 3 | Spherocity Index | 球形度指數 [0,1] | 否 |
| 4 | Radius of Gyration | 迴旋半徑 | 是 (μ=4.63, σ=1.64) |
| 5 | NPR1 | 歸一化主慣性比 1 | 否 |
| 6 | NPR2 | 歸一化主慣性比 2 | 否 |
| 7 | Principal axis span L | 主軸跨度（最長） | 是 (μ=14.59, σ=5.68) |
| 8 | Principal axis span W | 主軸跨度（中等） | 是 (μ=8.70, σ=3.14) |
| 9 | Principal axis span H | 主軸跨度（最短） | 是 (μ=5.48, σ=2.58) |

此向量同樣 broadcast 到每一個原子上。

---

## 四、Edge Features（6D）

鍵特徵用於 GNN 的邊屬性（不計入 109D 原子特徵，但是模型輸入的一部分）：

| 子特徵 | 維度 | 說明 |
|--------|------|------|
| Is single bond | 1 | Binary |
| Is double bond | 1 | Binary |
| Is triple bond | 1 | Binary |
| Is aromatic bond | 1 | Binary |
| Is conjugated bond | 1 | Binary |
| Is in ring | 1 | Binary |

邊為雙向（begin→end 和 end→begin 都加入）。

---

## 五、附帶輸出（非 109D 特徵向量，但存入 `.pt`）

| Key | Shape | 說明 |
|-----|-------|------|
| `coords` | [N, 3] | 所有原子（含 H）的 3D 座標 |
| `coords_heavy_current` | [N_heavy, 3] | 重原子座標 |
| `coords_original` | [N_heavy, 3] or None | 實驗結構原始重原子座標 |
| `coords_original_with_h` | [N_all, 3] or None | 實驗結構原始座標（含 H） |
| `gasteiger_charges` | [N] | 完整 Gasteiger 電荷向量 |
| `logp_contribs` | [N] | 完整 LogP 貢獻向量 |
| `tpsa_contribs` | [N] | 完整 TPSA 貢獻向量 |
| `rmsf` | [N] | 完整 RMSF 向量 |
| `flexibility_flag` | [N] | 完整柔性標記 |
| `atom_sasa` | [N] | 完整 SASA |
| `lipophilic_sasa` | [N] | 完整疏水 SASA |
| `hbond_donor` | [N] | 氫鍵供體標記 |
| `hbond_acceptor` | [N] | 氫鍵受體標記 |
| `chirality_cw` / `chirality_ccw` | [N] | 手性標記 |
| `rotatable_bond_context` | [N, 3] | 可旋轉鍵上下文 |
| `formal_charge_class` | [N, 3] | 形式電荷分類 |
| `metal_binding_flag` | [N, 4] | 金屬結合標記 |
| `stereochemistry_flag` | [N, 3] | 立體化學標記 |
| `shape_features` | [10] | 標準化後 shape |
| `global_features` | [45] | 全域 tag |
| `molecular_descriptors` | dict | 分子層級描述符（MW, LogP, TPSA, H-donors, H-acceptors, rotatable bonds 等 + 10 個 shape 值） |

---

## 六、構象生成流程

1. 載入分子文件（SDF/MOL2/PDB/MOL）
2. 保存原始實驗座標（如有）
3. 添加氫原子（如原始結構未含 H）
4. ETKDGv3 生成 50 個構象（`pruneRmsThresh=0.5`）
5. MMFF 力場優化每個構象（`maxIters=100000`）
6. 對齊所有構象，計算 RMSF
7. 提取所有原子級和分子級特徵
8. 拼接 54D local + 45D global + 10D shape = 109D
