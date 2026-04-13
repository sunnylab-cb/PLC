# PDBbind 聚類與篩選現況報告

更新日期：2026-04-13

## 範圍

本報告總結目前在 PLC 內已完成的兩層處理：

- `L1 exact grouping`：以 `contact_unique_chain_signature` 作為同蛋白多配體的 exact 群組鍵。
- `L2 MMseqs2 95%`：以 L1 representative 的 `all_unique_contact_chains` 做近似序列聚類，再回填成 group/entry 級 L2 cluster。

本報告只描述資料現況、群組規模、pair-share 統計，以及篩選鏈條；不涉及 L3 split 實作。

## 結論摘要

- 原始資料共有 `19037` 個 protein-ligand entries。
- 用 `contact_unique_chain_signature` 做 L1 後，共得到 `8291` 個 exact groups。
- 其中 `13660` 個 entries 落在大小大於 1 的 group，佔全資料 `71.76%`。
- 換句話說，單蛋白多配體或同 binding-context 多 entry，不是邊緣現象，而是資料主體。
- 以 L1 group 內部計算，總共有 `103087` 個 entry-pairs 共享同一個 exact protein context。
- 再做 L2 `MMseqs2 95%` 後，`8291` 個 L1 groups 被壓縮成 `4080` 個 L2 cluster signatures，群組數下降 `50.79%`。
- 在 L2 層級，共有 `16985` 個 entries 落在大小大於 1 的近似 cluster，約佔全資料 `89.22%`。
- 以 L2 cluster 內部計算，總共有 `441573` 個 entry-pairs 處於同一個 95% cluster 中。

這些數字支持目前的工程判斷：

1. 同蛋白多配體必須先 group-aware，不能用 random split 直接切。
2. L1 exact grouping 應當保留，因為它定義了最可信的「同一個 binding-context」。
3. L2 95% clustering 也非常必要，因為 exact 不足以覆蓋大量近似序列與突變體。

## 資料來源與輸出位置

本報告使用的中間輸出來自：

- `/tmp/plc_l1_manifest/summary.json`
- `/tmp/plc_l1_manifest/l1_groups.tsv`
- `/tmp/plc_l1_manifest/l1_manifest.tsv`
- `/tmp/plc_l2_mmseqs_input/summary.json`
- `/tmp/plc_l2_mmseqs95/summary.json`
- `/tmp/plc_l2_mmseqs95/l2_manifest_mmseqs95.tsv`
- `/tmp/plc_l2_mmseqs95/l2_groups_mmseqs95.tsv`

相關方法論與引用請見：

- [PDBBIND_DEDUP_SPLIT_REFERENCES_APA.md](/home/husrcf/Code/ProtBind/PLC/DOCs/PDBBIND_DEDUP_SPLIT_REFERENCES_APA.md)

## L1：exact protein-context grouping 現況

### L1 定義

L1 使用 `contact_unique_chain_signature`：

- 先從 `*_protein.pdb` 解析 chain sequence。
- 再用 `*_pocket.pdb` 取得 ligand-contacting chains。
- 對接觸鏈序列做 hash。
- 忽略相同鏈的重複拷貝數，只保留唯一接觸鏈集合。

這個定義能把像 `5qa4`（單體）與 `5qa7`（多聚體）這種接觸鏈序列相同、但 stoichiometry 不同的 entries 放在同一個 exact group。

### L1 核心統計

- 總 entries：`19037`
- L1 groups：`8291`
- duplicate groups：`2914`
- 落在 duplicate groups 的 entries：`13660` (`71.76%`)
- singleton groups：`5377`
- singleton groups 佔全部 groups：`64.85%`
- duplicate groups 佔全部 groups：`35.15%`
- 最大 exact group：`178` entries
- duplicate group 的中位數大小：`3`

### L1 group size 分布

| group 大小 | group 數 |
|---|---:|
| `1` | 5377 |
| `2` | 1276 |
| `3-5` | 1057 |
| `6-10` | 370 |
| `11-20` | 148 |
| `21-50` | 55 |
| `51-100` | 4 |
| `>100` | 4 |

這說明：

- 大部分 exact groups 雖然是 singleton，但真正造成資料偏斜的是尾端的大群組。
- group size 呈現明顯 heavy-tail，少數蛋白擁有非常多 ligand entries。

### L1 pair-share 統計

若把「共享同一個 exact protein context」定義為在同一個 `contact_unique_chain_signature` group 內，則：

- 共享 exact protein context 的 entry-pairs 數：`103087`
- 全資料所有可能 entry-pairs 數：`181194166`
- exact share pair 比例：`0.0569%`

雖然 pair 比例看似小，但這是因為全體 pair 空間極大；從 entry 視角來看，`71.76%` 的資料其實都已經位於非 singleton exact group。

### 最大 L1 groups 範例

| 排名 | group size | representative | 範例成員 |
|---|---:|---|---|
| 1 | 178 | `1aq7` | `1aq7, 1auj, 1bju, 1bjv, 1bty, ...` |
| 2 | 150 | `1avn` | `1avn, 1bn1, 1bn3, 1bn4, 1bnm, ...` |
| 3 | 117 | `1aq1` | `1aq1, 1b38, 1b39, 1di8, 1dm2, ...` |
| 4 | 106 | `3mxf` | `3mxf, 3p5o, 2yel, 3u5j, 3u5l, ...` |
| 5 | 97 | `1bcd` | `1bcd, 1g1d, 1g52, 1g53, 1g54, ...` |

典型例子：

- `5qa4 / 5qa7 / 5qb0` 已被歸入同一個 L1 group。
- 該 group 的 representative 預設選到較簡單的 `5qa4`。

## L2：MMseqs2 95% 近似聚類現況

### L2 定義

L2 不是直接對全部 entries 跑聚類，而是：

1. 先取每個 L1 group 的 representative。
2. 從 representative 擷取 `all_unique_contact_chains`。
3. 把這些鏈序列送進 `MMseqs2 easy-cluster`。
4. 再把鏈級 cluster 反投影回 group 級，形成 `mmseqs95_l2_cluster_signature`。

本次使用參數：

- `--min-seq-id 0.95`
- `-c 0.8`

### L2 輸入規模

- L1 groups：`8291`
- representative 最長鏈模式輸入：`8291` 條序列
- representative 所有唯一接觸鏈模式輸入：`9361` 條序列
- 含多條唯一接觸鏈的 L1 groups：`878`

本次正式執行採用的是 `all_unique_contact_chains` 模式。

### L2 核心統計

- L2 鏈級輸入序列數：`9361`
- MMseqs2 鏈級 cluster 數：`4309`
- 鏈級 cluster 壓縮率：`53.97%`
- 回填後的 L2 cluster signatures：`4080`
- L1 group 數壓縮率：`50.79%`
- 落在 merged L2 clusters 的 L1 groups：`5518` (`66.55%` of L1 groups)
- 落在 merged L2 clusters 的 entries：`16985` (`89.22%` of all entries)
- 最大 L2 cluster signature 含 `66` 個 L1 groups

### L2 pair-share 統計

若把「共享 95% sequence-level context」定義為位於同一個 `mmseqs95_l2_cluster_signature`，則：

- 共享 L2 cluster 的 group-pairs：`24769`
- 共享 L2 cluster 的 entry-pairs：`441573`
- entry-pair 比例（相對所有可能 entry-pairs）：`0.2437%`

這個數字遠高於 L1 的 `103087`，說明 exact grouping 之外仍有大量高同源序列互相連結。

### L2 cluster size 分布

#### 以 group 數計

| L2 cluster 內 group 數 | cluster 數 |
|---|---:|
| `1` | 2773 |
| `2` | 623 |
| `3-5` | 457 |
| `6-10` | 149 |
| `11-20` | 52 |
| `21-50` | 25 |
| `51-100` | 1 |
| `>100` | 0 |

#### 以 entry 數計

| L2 cluster 內 entry 數 | cluster 數 |
|---|---:|
| `1` | 2052 |
| `2` | 683 |
| `3-5` | 670 |
| `6-10` | 331 |
| `11-20` | 192 |
| `21-50` | 113 |
| `51-100` | 28 |
| `>100` | 11 |

### 最大 L2 clusters 觀察

- 最大 L2 cluster signature 含 `66` 個 L1 groups。
- 第二大含 `46` 個 L1 groups。
- 依 entry 數計，最大的 L2 cluster 含 `425` 個 entries。

這說明 L2 之後，很多原本 exact 分開的 groups 其實仍處於同一個高相似序列家族中。對 train/test split 來說，這是強烈訊號：只做 L1 exact 不足以完全避免 leakage。

## 篩選與聚類鏈條

目前實際採用的鏈條是：

1. `raw entries`
2. `parse protein chains from *_protein.pdb`
3. `select ligand-contacting chains from *_pocket.pdb`
4. `contact_unique_chain_signature`
5. `L1 representative selection`
6. `extract all unique contacting chains from representatives`
7. `MMseqs2 easy-cluster 95%`
8. `chain-cluster backfill -> L2 cluster signature`

對應腳本：

- [pdbbind_exact_multiligand_analysis.py](/home/husrcf/Code/ProtBind/PLC/preparation/cluster/pdbbind_exact_multiligand_analysis.py)
- [build_l1_manifest.py](/home/husrcf/Code/ProtBind/PLC/preparation/cluster/build_l1_manifest.py)
- [build_l2_mmseqs_input.py](/home/husrcf/Code/ProtBind/PLC/preparation/cluster/build_l2_mmseqs_input.py)
- [run_l2_mmseqs95.py](/home/husrcf/Code/ProtBind/PLC/preparation/cluster/run_l2_mmseqs95.py)

機器可讀的流程檔見：

- [PDBBIND_CLUSTERING_CHAIN_20260413.yaml](/home/husrcf/Code/ProtBind/PLC/DOCs/clustering/PDBBIND_CLUSTERING_CHAIN_20260413.yaml)

## 對後續工作的含義

### 1. 研究面

目前最值得優先研究的不是「還要不要 dedup」，而是：

- 哪些任務適合只用 L1 exact group-aware split。
- 哪些任務必須至少升到 L2 95% cluster-aware split。
- 多配體保留後，是否需要 group reweighting 或 per-group cap。

### 2. 工程面

目前應把 L1/L2 視為固定前處理資產，而不是 loader 內臨時計算：

- L1 定義 exact protein context。
- L2 定義高相似序列家族。
- 後續 L3 再定義 train/val/test 的低同源切分。

### 3. 資料風險

若直接用 raw entries random split：

- 同一 exact protein context 跨 split 的風險很高。
- 即使 exact 分開，95% sequence-level leakage 依然非常嚴重。

因此，至少需要在 split 層面使用 `L1` 或 `L2` 的 group-aware 約束。

## 建議下一步

目前若繼續調研，建議聚焦三個問題：

1. 在 PLC 任務設定下，validation/test 應以 L1 還是 L2 作為最低隔離單位。
2. 組內多 ligand 是否應在 train 端全保留，還是做 group-cap / resampling。
3. L3 `30%` clustering 完成後，是否要把 L1/L2/L3 全部寫進單一最終 split manifest。
