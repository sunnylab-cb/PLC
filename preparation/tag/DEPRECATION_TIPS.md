# Tag Scripts Deprecation Tips

以下兩個腳本建議視為**棄用**：

- `preparation/tag/calculate_distances.py`
- `preparation/tag/calculate_distances_v2.py`

## 為什麼棄用

### 1. protein residue enumeration 與 feature 流不一致

這兩支腳本都會自行從 raw `*_protein.pdb` 重新掃 residue。

問題是：

- protein feature 流使用 `parse_pdb_protein()` 作為 residue source of truth
- 其 residue list 是：
  - 只保留 amino-acid residues
  - 以有 `CA` 的 residue list 定義主序
  - 保留 `chain_id / res_id / insertion_code`
- 但這兩支舊腳本沒有直接重用這套邏輯

因此在以下情況可能出現靜默錯位：

- insertion code
- 多鏈文件順序與數字排序不一致
- 缺 `CA` 的 residue
- 非標準但仍在 `ATOM` 區段的 residue

這會導致：

- protein feature `.pt` 的 residue index
- distance label 的 residue index

兩者不一定對應到同一個殘基。

### 2. 舊版 ligand fallback 存在 heavy-only 不夠嚴格的歷史問題

這兩支腳本曾經在沒有 `coords_original` 時直接回退到 ligand `.pt` 的 `coords`。

而 `coords` 可能包含氫原子。

這會讓名義上的 heavy-atom distance 實際上混入氫原子座標。

雖然後續已經對 fallback 做過修補，但這兩支腳本的 protein 端 source-of-truth 問題仍在，因此不建議再作為主線使用。

## 現在應該用什麼

主線建議使用：

- `preparation/tag/calculate_distances_dcenter.py`
- `preparation/tag/calculate_distances_vdw.py`

原因：

- 這兩支現在都已改成重用 protein feature 流的 `parse_pdb_protein()`
- residue enumeration 已與 feature 流對齊
- ligand fallback 也已改成優先使用 heavy-only 座標鏈

## 如果還是要用舊腳本

只有在你明確知道：

- 資料沒有 insertion code
- 沒有多鏈順序問題
- 沒有 missing `CA` / 非標準 residue 進入 `ATOM`
- 且能接受與當前主線不一致

的情況下才建議暫時使用。

否則請不要再用這兩支生成新的 supervision labels。
