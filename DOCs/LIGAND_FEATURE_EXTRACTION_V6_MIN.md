# Ligand Feature Extraction V6-Min — 完整特徵說明

> 對應腳本：`preparation/ligand/biolip_ligand_extract.py`（V6 pipeline）  
> 資料合約：`preparation/ligand/ligand_biolip_types.py`（enum、分類器、schema）  
> 依賴：`extract_ligand_features_v5.py`（V5 backbone，保持不變）、`ligand_features_v4.py`

---

## 設計原則

1. **V5 x[109] 語義凍結**：109D 向量的維度、位置、意義永不改變。
2. **No silent zero**：V5 的問題在於金屬/無機配體失敗時默默填零，模型無法分辨 "真實零" 與 "提取失敗零"。V6 對高風險 descriptor/conformer/shape 群組用 `x_group_mask` 明確標記，並保留安全的基礎 atom identity/topology。
3. **分類器優先**：`classify_ligand_type()` 在任何 RDKit 化學計算之前執行。
4. **路由與類型分開**：`ligand_route`（int，排他）控制處理路徑；`ligand_type`（14D，多熱）記錄所有屬性。
5. **OperationStatus 取代 bool**：區分 "未嘗試"、"跳過（policy）"、"失敗"、"成功"、"不適用"、"V5 返回但子步驟未驗證"。

---

## 總覽：.pt 檔案輸出欄位

### V5 相容欄位（保持不變）

| Key | Shape | 說明 |
|-----|-------|------|
| `atom_features` | [N, 109] float32 | V5 特徵向量（54D local + 45D global + 10D shape） |
| `edge_index` | [2, E] int64 | 雙向鍵圖 |
| `edge_attr` | [E, 6] float32 | 鍵類型特徵 |
| `coords` | [N, 3] float32 | 最佳構象座標（含 H） |
| `coords_heavy_current` | [N_heavy, 3] float32 | 重原子座標 |
| `coords_original` | [N_heavy, 3] float32 | 實驗結構原始重原子座標 |
| `coords_original_with_h` | [N_all, 3] float32 | 實驗結構原始座標（含 H，如有） |
| `gasteiger_charges` | [N] | Gasteiger 電荷 |
| `logp_contribs` | [N] | Crippen LogP 貢獻 |
| `tpsa_contribs` | [N] | TPSA 貢獻 |
| `rmsf` | [N] | RMSF（構象系綜） |
| `flexibility_flag` | [N] | Binary（RMSF > 1.0 Å） |
| `atom_sasa` | [N] | 溶劑可及表面積 |
| `lipophilic_sasa` | [N] | 疏水 SASA |
| `hbond_donor` / `hbond_acceptor` | [N] | 氫鍵供/受體 |
| `chirality_cw` / `chirality_ccw` | [N] | 手性標記 |
| `rotatable_bond_context` | [N, 3] | 可旋轉鍵上下文 |
| `formal_charge_class` | [N, 3] | 形式電荷分類（負/中/正） |
| `metal_binding_flag` | [N, 4] | 金屬結合候選標記（N/O/S） |
| `stereochemistry_flag` | [N, 3] | CIP 立體化學標記 |
| `shape_features` | [10] | 標準化 3D shape |
| `global_features` | [45] | 全域分子 tag |
| `molecular_descriptors` | dict | 分子層級描述符 |
| `num_atoms` / `num_heavy_atoms` | int | 原子數 |
| `has_original_coords` | bool | 是否有實驗座標 |
| `feature_dim_total` / `feature_dim_local` | int | 109 / 54 |
| `ligand_path` / `pdb_id` / `protein_chain` / `binding_site_id` / `ligand_id` / `ligand_resseq` | str | 身份標識 |

### V6 新增欄位

| Key | Shape / Type | 說明 |
|-----|-------------|------|
| `feature_schema_version` | str | `"biolip_ligand_v6_min_multifragment_route_2026_05"`，偵測 schema 版本漂移 |
| `ligand_type` | [14] float32 | 多熱配體類型（見下方§3） |
| `ligand_route` | int | LigandRoute enum（主要處理路徑，見§2） |
| `ligand_type_from_id_table` | bool | 分類是否來自 CCD code 表（True = 可靠） |
| `ccd_lookup_used` | bool | Phase 2：是否查詢過真實 CCD 字典（目前恆 False） |
| `conformer_policy` | int | ConformerPolicy enum（見§4） |
| `hydrogen_policy` | int | HydrogenPolicy enum（見§4） |
| `operation_status` | [12] int8 | 每個 pipeline 操作的 OperationStatus（見§5） |
| `x_group_mask` | [N, 14] float32 | 每個原子每個特徵群組的有效性（0/1） |
| `x_group_source` | [N, 14] int8 | FeatureSource enum，特徵來自何處（見§6） |
| `x_atom_extra` | [N, 14] float32 | 元素週期表特徵（見§7） |
| `x_atom_extra_mask` | [N, 14] float32 | x_atom_extra 的有效性 |
| `x_atom_extra_source` | [N, 14] int8 | x_atom_extra 的來源（目前有效值為 PERIODIC_TABLE_LOOKUP） |
| `fragment_id` | [N] int64 | 每個當前原子所屬的連通片段編號（不修剪片段，與 `atom_features/coords` 對齊） |
| `fragment_count` | int | 分子圖片段數（未連接成分數） |
| `fragment_pruned` | bool | Phase 2：是否已修剪溶劑小片段 |
| `largest_fragment_atom_fraction` | float | 最大片段佔全部重原子的比例 |
| `feature_completeness_ratio` | float | x_group_mask 的平均值（0~1，越高越完整） |

---

## § 2. 處理路由（LigandRoute，排他）

```
UNKNOWN_OR_FAILED       = 0  載入或分類失敗
PURE_METAL_ION          = 1  單原子金屬（Zn²⁺, Ca²⁺, Mg²⁺ …）→ 完全繞過 V5
INORGANIC_POLYATOMIC    = 2  無機多原子（SO4, PO4, NO3 …）  → 完全繞過 V5
METAL_COMPLEX           = 3  有機 + 金屬（HEM, CIS, CPT …） → V5 strict=False，無構象
NO_CONFORMER_BIOLOGICAL = 4  核苷酸/胺基酸/糖/脂（ATP/NAG…）→ V5 strict=False，無構象
REGULAR_ORGANIC         = 5  一般小分子                      → 完整 V5 pipeline
MULTIFRAGMENT_NO_CONFORMER = 6  多片段斷連圖                  → raw-graph fallback
```

**分類器優先原則**：`classify_ligand_type(pair_meta, mol_summary)` 在任何 RDKit 化學計算之前執行，僅需讀取 pair_meta.json 和一次 `sanitize=False` 的快速讀取。

路由優先順序（依序判斷）：
1. heavy == 1 且是金屬元素 → `PURE_METAL_ION`
2. CCD code 在 `INORGANIC_POLYATOMIC_CODES` 中，或（heavy ≤ 8 且無 C；允許 VO4/WO4 這類 metallo-oxyanion）→ `INORGANIC_POLYATOMIC`
3. CCD code 在 `METAL_COMPLEX_CODES` 中，或（has_metal 且 heavy > 1）→ `METAL_COMPLEX`
4. CCD code 在 `NUCLEOTIDE_COFACTOR_CODES` 中 → `NO_CONFORMER_BIOLOGICAL`
5. CCD code 在 `NUCLEIC_ACID_FRAGMENT_CODES` 中 → `NO_CONFORMER_BIOLOGICAL`
6. CCD code 在 `PEPTIDE_CODES` 中 → `NO_CONFORMER_BIOLOGICAL`
7. CCD code 在 `CARBOHYDRATE_CODES` 中 → `NO_CONFORMER_BIOLOGICAL`
8. CCD code 在 `LIPID_CODES` 中 → `NO_CONFORMER_BIOLOGICAL`
9. 若分子圖不連通（fragment_count > 1）→ `MULTIFRAGMENT_NO_CONFORMER`
10. 其餘（且 load_ok=True）→ `REGULAR_ORGANIC`
11. 無法載入 → `UNKNOWN_OR_FAILED`

若初始路由為 `REGULAR_ORGANIC`，但 heavy > 50 或 rotatable_bonds > 20，路由升級為 `NO_CONFORMER_BIOLOGICAL`。多片段不走這條升級，而是直接走獨立的 `MULTIFRAGMENT_NO_CONFORMER`。

---

## § 3. 配體類型（ligand_type [14] 多熱）

多熱向量，允許多個 bit 同時為 1（例如 HEM 的 `metal_complex=1` 且 `metalloporphyrin=1`）。

| Index | 名稱 | 類別 | 說明 |
|-------|------|------|------|
| 0 | `pure_metal_ion` | primary | 單原子金屬離子 |
| 1 | `inorganic_polyatomic` | primary | SO4/PO4/NO3 等無機多原子 |
| 2 | `metal_complex` | primary | 含金屬的有機錯合物 |
| 3 | `metalloporphyrin` | additive | `metal_complex` 的子類（HEM/CLA/COB 等；FAD 屬於 nucleotide/cofactor） |
| 4 | `nucleotide_cofactor` | primary | ATP/ADP/NAD/FAD 等 |
| 5 | `peptide_ligand` | primary | 胜肽配體 |
| 6 | `nucleic_acid_fragment` | primary | DNA/RNA 片段 |
| 7 | `carbohydrate_glycan` | primary | 糖類（NAG/GLC 等） |
| 8 | `lipid_long_chain` | primary | 長鏈脂肪酸/磷脂 |
| 9 | `unusual_element` | additive | 含 B/Se/As/Si/Te/Ge（RDKit 處理較不可靠） |
| 10 | `multifragment` | additive | 分子圖不連通（多個斷開片段） |
| 11 | `large_flexible` | additive | heavy > 50 或 rotatable_bonds > 20 |
| 12 | `regular_organic` | primary | 標準有機小分子（full V5 pipeline） |
| 13 | `unknown_or_unclassified` | catch-all | 無法載入或不符合任何類型 |

---

## § 4. 政策 Enum

### ConformerPolicy（conformer_policy 欄位）

```
FULL_ETKDG_MMFF    = 0  50 構象，ETKDGv3 + MMFF（V5 預設，REGULAR_ORGANIC）
RAW_COORDS_ONLY    = 1  不生成 conformer，只使用輸入 SDF 座標（BioLiP 中通常是 bound coords；synthetic fixture 不代表實驗構象）
EXPERIMENTAL_ONLY  = 1  backward-compatible alias
FAILED_FALLBACK    = 2  嘗試 full 失敗，降級為無構象
SKIPPED_METAL      = 3  金屬/無機路由，不適用構象生成
```

### HydrogenPolicy（hydrogen_policy 欄位）

```
EXPLICIT_ONLY     = 0  僅保留 SDF/mol2 中的顯式 H（Phase 2 可能使用）
V5_ADD_H          = 1  使用 V5 backbone；V5 會保留顯式 H 或調用 AddHs
ADD_H_FAILED      = 2  嘗試 AddHs 失敗，退回顯式 H
SKIPPED_BY_POLICY = 3  bypass 路由不加 H
UNKNOWN           = 4
```

路由 → 政策對應：

| LigandRoute | ConformerPolicy | HydrogenPolicy |
|-------------|-----------------|----------------|
| PURE_METAL_ION | SKIPPED_METAL | SKIPPED_BY_POLICY |
| INORGANIC_POLYATOMIC | SKIPPED_METAL | SKIPPED_BY_POLICY |
| METAL_COMPLEX | RAW_COORDS_ONLY | V5_ADD_H |
| NO_CONFORMER_BIOLOGICAL | RAW_COORDS_ONLY | V5_ADD_H |
| REGULAR_ORGANIC | FULL_ETKDG_MMFF | V5_ADD_H |
| MULTIFRAGMENT_NO_CONFORMER | RAW_COORDS_ONLY | EXPLICIT_ONLY |

多片段 ligand 不進入 V5 full pipeline；分類器檢出後直接走 raw-graph fallback：保留所有 fragment、填入安全 base atom/topology/x_atom_extra，並將 `hydrogen_policy=EXPLICIT_ONLY`、descriptor/conformer groups 設為無效。

---

## § 5. 操作狀態（operation_status [12] int8）

OperationStatus enum：

```
NOT_ATTEMPTED     = 0  程式碼路徑未到達
SKIPPED_BY_POLICY = 1  策略性跳過（例如金屬 → 無 ETKDG）
FAILED            = 2  嘗試但拋出例外或返回無效結果
SUCCESS           = 3
NOT_APPLICABLE    = 4  此配體類型語義上無意義
UNVERIFIED         = 5  V5 返回了整體結果，但沒有暴露該子步驟的明確成功狀態
```

12 個操作（依序）：

| Index | 操作 | 說明 |
|-------|------|------|
| 0 | `rdkit_parse` | 從 SDF/mol2 載入分子 |
| 1 | `rdkit_full_sanitize` | `Chem.SanitizeMol` 完整模式 |
| 2 | `rdkit_partial_sanitize` | `SANITIZE_ALL ^ SANITIZE_PROPERTIES`（V5 未使用，Phase 2） |
| 3 | `rdkit_add_h` | `Chem.AddHs` |
| 4 | `gasteiger` | `ComputeGasteigerCharges` |
| 5 | `crippen_logp` | `_CalcCrippenContribs` |
| 6 | `tpsa` | `_CalcTPSAContribs` |
| 7 | `morgan` | 全域 tag（Morgan FP + LogP + TPSA） |
| 8 | `etkdg` | ETKDGv3 構象嵌入 |
| 9 | `mmff` | MMFF 力場優化 |
| 10 | `sasa` | `rdFreeSASA` |
| 11 | `shape` | 3D shape 描述符 |

典型 operation_status 值（以 METAL_COMPLEX/HEM 為例）：
```
rdkit_parse=SUCCESS, rdkit_full_sanitize=UNVERIFIED, rdkit_partial_sanitize=SKIPPED,
rdkit_add_h=UNVERIFIED, gasteiger=UNVERIFIED, crippen_logp=UNVERIFIED, tpsa=UNVERIFIED,
morgan=UNVERIFIED, etkdg=SKIPPED_BY_POLICY, mmff=SKIPPED_BY_POLICY,
sasa=UNVERIFIED, shape=SUCCESS  # shape 來自 experimental coords
```

---

## § 6. x_group_mask / x_group_source（特徵群組有效性）

每個原子都有一組 14D mask 和 14D source，指示高風險 V5 派生群組是否可信。穩定的基礎 atom identity/topology 欄位（atom type、hybridization、degree、formal charge class 等）不在這 14 組中逐項 mask；bypass 路徑會盡量從 raw SDF graph 填入這些基礎欄位，而不是全零。

**mask [N, 14] float32**：0 = 無效（勿輸入模型），1 = 有效。  
**source [N, 14] int8**：FeatureSource enum，說明數值來自何處。

FeatureSource enum：
```
UNAVAILABLE            = 0  數值不可用
RDKIT_FULL_SANITIZE    = 1  RDKit 完整 sanitize 後計算
RDKIT_PARTIAL_SANITIZE = 2  RDKit 部分 sanitize（Phase 2）
EXPERIMENTAL_COORDS    = 3  直接來自實驗座標
PERIODIC_TABLE_LOOKUP  = 4  元素週期表查詢
CCD_LOOKUP             = 5  CCD 字典查詢（Phase 2）
CODE_TABLE             = 6  離線 CCD code 列表（Phase 1）
FALLBACK_HEURISTIC     = 7  啟發式估算
NOT_APPLICABLE         = 8  此配體類型語義上無意義
```

14 個群組（對應 X_GROUP_FIELDS）：

| Index | 群組名稱 | 對應 x[109] 維度 | 依賴 conformer |
|-------|---------|-----------------|---------------|
| 0 | `local_gasteiger_charge` | dim 33 | 否 |
| 1 | `local_crippen_logp` | dim 34 | 否 |
| 2 | `local_tpsa_contrib` | dim 35 | 否 |
| 3 | `rmsf_and_flexibility` | dim 36–37 | **是** |
| 4 | `sasa_and_lipophilic_sasa` | dim 38–39 | 否 |
| 5 | `hbond_donor_acceptor` | dim 40–41 | 否 |
| 6 | `chirality` | dim 42–43 | 否 |
| 7 | `stereochemistry` | dim 51–53 | 否 |
| 8 | `rot_bond_context` | dim 44–46 | 否 |
| 9 | `degree_and_formal_charge` | dim 25–32 | 否 |
| 10 | `global_morgan_43` | dim 54–96 | 否 |
| 11 | `global_mollogp` | dim 97 | 否 |
| 12 | `global_tpsa` | dim 98 | 否 |
| 13 | `shape_3d` | dim 99–108 | **是** |

群組 3（rmsf）需要構象系綜才有效。群組 13（shape）不等同於 RMSF：若沒有 ETKDG/MMFF 但有 bound experimental coordinates，shape 仍然有效，mask=1 / source=EXPERIMENTAL_COORDS。

**不同路由的典型 feature_completeness_ratio**：

| 路由 | 有效群組數 | feature_completeness_ratio |
|------|-----------|---------------------------|
| PURE_METAL_ION | 1/14（degree/formal_charge only） | ~0.071 |
| INORGANIC_POLYATOMIC | 1/14（degree/formal_charge only） | ~0.071 |
| METAL_COMPLEX（V5 成功，無構象，但有 experimental shape） | 13/14 | ~0.929 |
| NO_CONFORMER_BIOLOGICAL（同上） | 13/14 | ~0.929 |
| REGULAR_ORGANIC（含構象） | 14/14 | 1.000 |
| REGULAR_ORGANIC（構象失敗但有 experimental shape） | 13/14 | ~0.929 |

---

## § 7. x_atom_extra（元素週期表特徵 [N, 14]）

不依賴 RDKit 化學計算，只用元素符號查週期表。對所有路由（包含純金屬）都有效。

| Index | 特徵 | 正規化方式 | 說明 |
|-------|------|-----------|------|
| 0 | atomic_number | Z / 100 | 原子序 |
| 1 | period | period / 7 | 週期（1–7） |
| 2 | group | group / 18 | 族（1–18，鑭系/錒系 = 0） |
| 3 | block_s | one-hot | s-block（H, He, 鹼金屬, 鹼土金屬） |
| 4 | block_p | one-hot | p-block |
| 5 | block_d | one-hot | d-block（過渡金屬） |
| 6 | block_f | one-hot | f-block（鑭系/錒系） |
| 7 | pauling_en | EN / 4.0 | Pauling 電負性（未知 = 0，mask=0） |
| 8 | covalent_radius | r / 300 pm | Alvarez (2008) 共價半徑 |
| 9 | vdw_radius | r / 400 pm | Bondi/Alvarez vdW 半徑 |
| 10 | is_metal | binary | 是否金屬元素 |
| 11 | is_transition_metal | binary | 是否過渡金屬 |
| 12 | is_halogen | binary | 是否鹵素 |
| 13 | is_metalloid | binary | 是否類金屬（B/Si/Ge/As/Sb/Te） |

mask=1 的條件：
- dims 0–6、10–13：只要元素已知（在週期表中）就有效
- dim 7（pauling_en）：EN ≥ 0（稀有氣體 EN=0，mask 仍為 1）
- dims 8–9（covalent/vdw radius）：半徑 > 0 時有效

---

## § 8. x[109] 維度分布（X109_GROUP_SLICES）

完整的維度索引對應（從 `ligand_biolip_types.X109_GROUP_SLICES` 讀取，由 assert 保證覆蓋全部 109D）：

| 群組名稱 | 維度範圍 | 長度 |
|---------|---------|------|
| `local_atom_type` | 0–10 | 11 |
| `local_hybridization` | 11–16 | 6 |
| `local_aromaticity` | 17 | 1 |
| `local_ring` | 18–24 | 7 |
| `local_degree` | 25–29 | 5 |
| `local_formal_charge_class` | 30–32 | 3 |
| `local_gasteiger_charge` | 33 | 1 |
| `local_crippen_logp` | 34 | 1 |
| `local_tpsa_contrib` | 35 | 1 |
| `local_rmsf` | 36 | 1 |
| `local_flexibility_flag` | 37 | 1 |
| `local_atom_sasa` | 38 | 1 |
| `local_lipophilic_sasa` | 39 | 1 |
| `local_hbond_donor` | 40 | 1 |
| `local_hbond_acceptor` | 41 | 1 |
| `local_chirality` | 42–43 | 2 |
| `local_rot_bond_context` | 44–46 | 3 |
| `local_metal_binding` | 47–50 | 4 |
| `local_stereochemistry` | 51–53 | 3 |
| `global_morgan_43` | 54–96 | 43 |
| `global_mollogp` | 97 | 1 |
| `global_tpsa` | 98 | 1 |
| `shape_3d` | 99–108 | 10 |
| **合計** | **0–108** | **109** |

---

## § 9. 純金屬/無機路徑的特徵行為

對 `PURE_METAL_ION` 和 `INORGANIC_POLYATOMIC` 路由，V5 pipeline 完全繞過：

- `atom_features [N, 109]` 只填入安全的基礎 atom identity/topology 欄位；Gasteiger/LogP/TPSA/RMSF/SASA/global/shape 等派生欄位保持 0
- 除 `degree_and_formal_charge` 外，其餘高風險群組 `mask=0 / source=NOT_APPLICABLE`
- `edge_index`、`edge_attr` = 拓撲鍵（`sanitize=False` 讀取），不含化學序資訊
- `gasteiger_charges`、`logp_contribs` 等全部為零向量
- `x_atom_extra [N, 14]` = 元素週期表特徵（主要可靠化學特徵）
- `feature_completeness_ratio = 1/14 ≈ 0.071`，只代表 `degree_and_formal_charge` 群組可用

模型處理建議：使用 `x_atom_extra` 作為純金屬/無機的主要特徵輸入；x[109] 在 mask=0 時應被遮罩或填充 learnable embedding。

---

## § 10. Schema 版本控制

```python
FEATURE_SCHEMA_VERSION = "biolip_ligand_v6_min_multifragment_route_2026_05"
```

每份 .pt 檔都存有此欄位。dataloader 應在啟動時驗證：
```python
assert data['feature_schema_version'] == FEATURE_SCHEMA_VERSION, \
    f"Schema mismatch: {data['feature_schema_version']}"
```

升版規則：若 x[109] 任何維度語義改變、或 x_group_mask 的群組定義改變，必須 bump 版本號。

---

## § 11. 未來規劃（Phase 2/3）

| 功能 | 說明 |
|------|------|
| CCD 字典查詢 | 真實 CCD dict 補充 metal_complex 的鍵序/電荷 |
| dative bond 重建 | 配位鍵獨立加入 `edge_index_coord`，不改主圖 |
| `rdkit_partial_sanitize` | METAL_COMPLEX 的部分 sanitize 路徑（現為 SKIPPED） |
| protein-metal contact edge | metal 到蛋白 donor atom/residue 的 pair-level coordination 邊 |
| 溶劑片段修剪 | `fragment_pruned=True`，保留最大片段 |
| PropKa 質子化 | 生理 pH 下的質子化狀態 |

---

## § 12. Smoke Test

可用以下命令執行 V6-Min 的 route/schema/mask/source/status 回歸檢查：

```bash
conda run -n AIAA python preparation/ligand/smoke_test_biolip_ligand_v6.py
```

覆蓋案例：

| Case | 覆蓋重點 |
|------|----------|
| BEN | regular organic，V5 109D/edge/global/shape regression |
| ZN | pure metal ion，empty edge，bypass V5 |
| HEM | metal complex，no-conformer，experimental shape |
| ATP / FAD | nucleotide/cofactor，不因 heavy atom 數量誤走 regular organic |
| NAG | glycan no-conformer |
| MSE | unusual element bit + periodic-table features |
| synthetic SO4 / VO4 | inorganic polyatomic / metallo-oxyanion |
| synthetic multifragment | fragment 保留、no prune、V5 failure raw fallback |
| missing SDF | controlled failure，不寫壞 .pt |
