# Latest Complete Features V4

本文件描述目前 PLC 主線使用的 **latest complete protein features**，也就是：

- 生成入口：[extract_complete_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_complete_features_v4.py)
- 核心實作：[protein_features_v4.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py)

這套特徵對應目前主線資料目錄中的 `complete_features_v4` / `complete_features_v4_nma`。

## 總覽

每個 protein `.pt` 目前包含以下主欄位：

- `physchem_features`: `[N_res, 62]`
- `spatial_scalar_features`: `[N_res, 89]`
- `spatial_scalar_raw_features`: `[N_res, 89]`
- `spatial_vector_features`: `[N_res, 8, 3]`
- `ca_coords`: `[N_res, 3]`
- `all_atom_coords`: `List[np.ndarray[n_atoms_i, 3]]`
- `all_atom_names`: `List[List[str]]`
- `residue_names`: `List[str]`
- `residue_indices`: `List[int]`
- `insertion_codes`: `List[str]`
- `chain_ids`: `List[str]`
- `n_residues`: `int`
- `spatial_scalar_feature_names`: `List[str]`

其中模型主用的是：

- `physchem_features`
- `spatial_scalar_features`
- `spatial_vector_features`
- `ca_coords`
- `all_atom_coords`（下游某些流程会用）

## 輸出結構

最新 complete features 不是單一大向量，而是三塊主特徵：

1. `physchem_features`: 62 維
2. `spatial_scalar_features`: 89 維
3. `spatial_vector_features`: 8×3

如果把向量特徵展平成標量維度，則單殘基可視為：

- `62 + 89 + (8×3) = 175` 維

但實際模型中 `spatial_vector_features` 保持向量形式，不會先手動 flatten。

## Residue Enumeration

這套特徵的 residue source of truth 來自：

- [parse_pdb_protein()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_physchem_from_pdb.py#L53)

其規則是：

- 只保留 amino-acid residues
- 去除 `HETATM`
- 以有 `CA` 的 residue list 定義主序
- 保留 `chain_id / res_id / insertion_code`
- 再回填每個 residue 的所有 heavy atoms

這也是目前 `dcenter` / `vdw` 主線標籤腳本已對齊的 residue enumeration。

## 1. `physchem_features` `[N_res, 62]`

定義位置：

- [extract_physchem_features_v4()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L108)

### 維度分解

- `0:20` 基礎物理化學屬性（來自 v2 basic）
- `20:28` 序列局部特徵（8 維）
- `28:36` 氫鍵網路特徵（8 維）
- `36:56` 殘基類型 one-hot（20 種標準胺基酸）
- `56:59` 3-state secondary structure one-hot
- `59:62` residue chemistry class one-hot

### `0:20` 基礎物理化學屬性

來源：

- [get_basic_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/physicochemical_features.py#L146)

依序為：

1. hydrophobicity
2. molecular weight
3. side-chain volume
4. polarity class scalar
5. H-bond donor count
6. H-bond acceptor count
7. aromaticity flag
8. charge at pH 7
9. nonpolar one-hot
10. polar-uncharged one-hot
11. positively charged one-hot
12. negatively charged one-hot
13. ionizable flag
14. pKa normalized
15. acidic flag
16. basic flag
17. strongly hydrophobic flag
18. weakly hydrophobic flag
19. weakly hydrophilic flag
20. strongly hydrophilic flag

### `20:28` 序列局部特徵

來源：

- [compute_sequence_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/physicochemical_features.py#L280)

依序為：

1. current residue is aromatic
2. current residue is charged
3. current residue is polar
4. local-window aromaticity
5. local helix tendency
6. local turn tendency
7. local sheet tendency
8. relative sequence position

註：

- 原始函式內還計算了 `is_terminal`，但沒有放進最後的 8 維向量

### `28:36` 氫鍵網路特徵

來源：

- [compute_hbond_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/hbond_network_features.py#L231)

依序為：

1. donor count
2. acceptor count
3. mean H-bond distance
4. H-bond distance std
5. has strong H-bond (`< 3.0 Å`)
6. has weak H-bond (`3.5–4.0 Å`)
7. number of H-bond partners
8. network connectivity

### `36:56` 殘基類型 one-hot

20 種標準胺基酸 one-hot，順序是：

`ALA, CYS, ASP, GLU, PHE, GLY, HIS, ILE, LYS, LEU, MET, ASN, PRO, GLN, ARG, SER, THR, VAL, TRP, TYR`

### `56:59` Secondary Structure 3-state

從 DSSP 8-state 映射而來：

- `56` helix
- `57` sheet
- `58` coil

映射規則：

- helix = `H/G/I`
- sheet = `B/E`
- coil = `T/S/-`

### `59:62` Residue Chemistry Class

- `59` nonpolar
- `60` polar
- `61` charged

## 2. `spatial_scalar_features` `[N_res, 89]`

定義位置：

- [get_spatial_scalar_feature_names_v4()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L531)
- [extract_spatial_scalar_features_v4()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L550)

注意：

- `spatial_scalar_raw_features` 是未 normalization 的原始值
- `spatial_scalar_features` 是用 train-set stats 正規化後的版本

### 維度分解

- `0:6` v2 SASA features
- `6:20` v2 surface features
- `20:46` v2 structural geometry features
- `46:58` v2 enhanced features
- `58:62` local density raw
- `62` distance to center
- `63` surface normal magnitude
- `64:68` local charge density
- `68:72` local hydrophobicity
- `72:78` backbone torsions
- `78:81` sidechain packing
- `81:85` environment composition histogram
- `85:88` metal-binding potential
- `88` local anisotropy

### `0:6` v2 SASA features

來源：

- [extract_physicochemical_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/physicochemical_features.py#L345)

SASA 6 維為：

1. total SASA
2. relative SASA
3. polar SASA
4. apolar SASA
5. exposed flag
6. buried flag

### `6:20` v2 Surface Features (14 維)

來源：

- [extract_surface_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/surface_features.py#L448)

依序為：

1. local curvature
2. convexity
3. surface roughness
4. local density
5. surface depth
6. electrostatic potential
7. charge density
8. polarity ratio
9. hydrophobicity
10. hydrophobic ratio
11. surface hydrophobicity
12. nearest mapped charge
13. weighted neighbor charge
14. distance-weighted charge

### `20:46` v2 Structural Geometry Features (26 維)

來源：

- [extract_structural_geometry_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/structural_geometry_features.py#L444)

依序為：

- `20:28` DSSP 8-state one-hot
- `28:46` geometry 18 維

其中 geometry 18 維為：

1. curvature
2. torsion
3-10. curvature RBF encoding (8 dims)
11-18. torsion RBF encoding (8 dims)

### `46:58` v2 Enhanced Features (12 維)

來源：

- [enhanced_features_v2.py](/home/husrcf/Code/ProtBind/PLC/preparation/protein/enhanced_features_v2.py)

依序為：

#### `46:49` pocketness (3 維)

1. convexity index (CX)
2. local packing density
3. shell density ratio

#### `49:52` flexibility / confidence (3 維)

1. normalized B-factor
2. local flexibility variance
3. relative surface exposure (RSE)

#### `52:58` environmental physchem (6 維)

1. environment hydrophobicity
2. hydrophobic contrast
3. environment positive charge density
4. environment negative charge density
5. environment H-bond donor density
6. environment H-bond acceptor density

### `58:62` local density raw

來源：

- [compute_local_density_raw()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L155)

依序為：

1. density within 5 Å
2. density within 10 Å
3. density within 15 Å
4. density contrast (`10Å - 5Å`)

### `62` distance to center

來源：

- [compute_distance_to_center_raw()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L169)

定義：

- residue `CA` 到整個 protein `CA` centroid 的距離

### `63` surface normal magnitude

來源：

- [compute_surface_normal_magnitude()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L174)

### `64:68` local charge density

來源：

- [compute_local_charge_density_raw()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L193)

依序為：

1. positive density within 5 Å
2. positive density within 10 Å
3. negative density within 5 Å
4. negative density within 10 Å

### `68:72` local hydrophobicity

來源：

- [compute_local_hydrophobicity()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L211)

依序為：

1. nonpolar ratio within 5 Å
2. nonpolar ratio within 10 Å
3. polar-or-charged ratio within 5 Å
4. polar-or-charged ratio within 10 Å

### `72:78` backbone torsions

來源：

- [compute_backbone_torsion_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L227)

依序為：

1. `phi_sin`
2. `phi_cos`
3. `psi_sin`
4. `psi_cos`
5. `omega_sin`
6. `omega_cos`

### `78:81` sidechain packing

來源：

- [compute_sidechain_packing_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L274)

依序為：

1. nearest sidechain centroid distance
2. packing score (`close6 / 20`)
3. void ratio

### `81:85` environment composition histogram

來源：

- [compute_environment_composition_histograms()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L294)

依序為：

1. aromatic neighbor ratio
2. H-bond donor neighbor ratio
3. H-bond acceptor neighbor ratio
4. gly/pro neighbor ratio

### `85:88` metal-binding potential

來源：

- [compute_metal_binding_potential()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L312)

依序為：

1. intrinsic self metal-binder flag
2. neighboring metal-binder density within 6 Å
3. neighboring metal-binder density within 10 Å

### `88` local anisotropy

來源：

- [compute_local_anisotropy()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py)

## 3. `spatial_vector_features` `[N_res, 8, 3]`

定義位置：

- [extract_spatial_vector_features_v4()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L363)

每個 residue 共有 8 個 3D 向量：

1. sidechain base vector
2. sidechain center-of-mass vector
3. aromatic ring normal vector
4. H-bond donor direction
5. H-bond acceptor direction
6. cavity-pointing vector at 6 Å scale
7. cavity-pointing vector at 10 Å scale
8. cavity-pointing vector at 14 Å scale

### 各向量含義

- `0` sidechain base vector
  - 非 Gly 直接用 `CA -> CB`
  - Gly 用 backbone 幾何近似虛擬 `CB`

- `1` sidechain COM vector
  - `CA -> sidechain heavy atom centroid`

- `2` aromatic normal vector
  - 對 `PHE/TYR/TRP/HIS` 取芳香環法向量

- `3` H-bond donor vector
  - donor residue 時使用 sidechain COM 方向

- `4` H-bond acceptor vector
  - acceptor residue 時優先取 `C -> O` 或 `-v_com`

- `5:8` multi-scale cavity vectors
  - residue 指向局部鄰居重心反方向的 cavity/open-space 指標

## 4. Coordinate / Metadata Fields

### `ca_coords`

- `[N_res, 3]`
- 每個 residue 的 `CA` 座標

### `all_atom_coords`

- `List[np.ndarray[n_atoms_i, 3]]`
- 每個 residue 的 heavy-atom 座標
- 與 feature residue enumeration 完全對齊

### `all_atom_names`

- `List[List[str]]`
- 與 `all_atom_coords` 一一對齊的 atom name

### `residue_names`

- 三字母殘基名稱列表

### `residue_indices`

- PDB residue number

### `insertion_codes`

- PDB insertion code

### `chain_ids`

- 每個 residue 的 chain ID

## Normalization

只有 `spatial_scalar_features` 會做 train-set-level normalization：

- 原始值存在 `spatial_scalar_raw_features`
- 正規化後存在 `spatial_scalar_features`

stats 生成與載入位置：

- [fit_normalization_stats()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L505)
- [load_normalization_stats()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L492)

`physchem_features` 與 `spatial_vector_features` 不經這套 stats normalization。

## Source Map

主流程：

1. [parse_pdb_protein()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/extract_physchem_from_pdb.py#L53)
2. [extract_physicochemical_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/physicochemical_features.py#L345)
3. [extract_surface_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/surface_features.py#L448)
4. [extract_structural_geometry_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/structural_geometry_features.py#L444)
5. [compute_pocketness_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/enhanced_features_v2.py#L48)
6. [compute_flexibility_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/enhanced_features_v2.py#L107)
7. [compute_environmental_features()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/enhanced_features_v2.py#L203)
8. [extract_complete_features_v4()](/home/husrcf/Code/ProtBind/PLC/preparation/protein/protein_features_v4.py#L583)

## Practical Notes

- 如果要看每個 scalar dim 的名稱，直接讀：
  - `spatial_scalar_feature_names`
- 如果要做 label 對齊或 residue-level 距離計算，應與：
  - `residue_names`
  - `residue_indices`
  - `insertion_codes`
  - `chain_ids`
  - `all_atom_coords`
  對齊使用
- 現在主線 `dcenter` / `vdw` 已經改為與這套 residue enumeration 對齊
