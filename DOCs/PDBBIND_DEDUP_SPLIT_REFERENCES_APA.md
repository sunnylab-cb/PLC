# PDBbind 去重與分割依據文獻整理

更新日期：2026-04-13

## 目的

本文件整理 PLC 後續處理「單蛋白多配體、重複蛋白、train/validation/test leakage 控制」時所依據的主要文獻與官方實作來源，並統一使用 APA 格式落盤。

本文件涵蓋三類依據：

- 基礎資料集文獻：PDBbind、CrossDocked2020。
- 分割與泛化實務：LP-PDBBind、Pocket2Mol、TargetDiff。
- 聚類與結構比對工具：MMseqs2、ProBiS。

## 與 PLC 目前策略直接相關的結論

### 1. 不應使用 random split 作為主要評估切分

LP-PDBBind 明確指出，PDBbind 上的 random split 會導致 protein 與 ligand 高相似樣本在不同 split 間交叉出現，形成嚴重資料洩漏，因此改用 protein 與 ligand 相似度受控的分割流程（Li et al., 2026）。

這直接支持 PLC 不應再把同蛋白或高同源蛋白的樣本任意散到不同 split。

### 2. 同一蛋白對多個配體通常不是先刪掉，而是保留並做 group-aware split

CrossDocked2020 是 `pocket-ligand` pair 資料，天然允許同一 pocket 對多個 ligand；其官方資料構建與後續工作並未要求先把同蛋白多配體壓成單樣本，而是透過 pocket clustering 來控制 split（Francoeur et al., 2020）。

Pocket2Mol 與 TargetDiff 進一步沿用 CrossDocked 的固定 split 檔，而不是重新做 random split（Guan et al., 2023; Peng et al., 2022; guanjq, n.d.-a; pengxingang, n.d.-a）。

基於這些來源，對 PLC 而言，較合理的做法不是「先把同蛋白多配體全部刪掉」，而是「先形成 protein/pocket group，再確保整組只落在同一 split」。

### 3. PDBbind 相關 benchmark 常以 protein cluster 或 family-aware 設計代表集合

LP-PDBBind 回顧的 PDBbind core-set 構建方式不是單純 random sampling，而是基於 protein cluster/family 的代表樣本挑選。這說明在結構導向的 binding affinity 評估裡，protein-aware grouping 是主流做法，而非單純以 complex 為獨立 IID 樣本（Li et al., 2026）。

### 4. 若要做近似序列併群，MMseqs2 是合理的 Level 2/Level 3 工具

MMseqs2 是針對大規模蛋白序列搜尋與聚類的標準工具之一，適合作為：

- Level 2：代表序列的高相似度併群，例如 95% identity。
- Level 3：低相似度 cluster-based split，例如 30% identity。

這一點由 MMseqs2 原始論文支持（Steinegger & Söding, 2017）。

### 5. 若後續需要 pocket-level 結構聚類，ProBiS 是可引用的結構比對基礎

CrossDocked2020 的 pocket clustering 使用 ProBiS 進行 binding-site 結構相似性比較（Francoeur et al., 2020）。因此，如果 PLC 日後需要從 exact sequence grouping 擴展到 pocket structure-aware grouping，ProBiS 是有明確先例可依循的工具基礎（Konc & Janežič, 2010）。

## 對 PLC 的具體落地含義

### A. Level 1 exact grouping

文獻沒有直接規定 `contact_unique_chain_signature` 這個鍵，但它與上述實務方向一致：

- 它以會與 ligand 接觸的 protein context 為核心。
- 它比「整個 complex 完全相同」更接近 pocket-based 任務的資訊單位。
- 它保留同蛋白多配體於同一群，方便後續做 group-aware split。

這一點是「根據文獻方向 + 本地資料分析」得到的工程決策，不是單一論文逐字規範。

### B. Level 2/Level 3 clustering

推薦流程：

1. 用 `contact_unique_chain_signature` 建立 Level 1 群組。
2. 只對 Level 1 代表序列跑 MMseqs2 95%。
3. 再對 Level 2 代表序列跑 MMseqs2 30%。
4. split 以群組或 cluster 為單位，而非以單一 entry 為單位。

### C. representative entry 的用途

代表樣本較適合用於：

- MMseqs2 聚類輸入。
- split 建構。
- benchmark/ablation 報告中的代表複合體。

不建議把 representative-only 當成預設 training 去重策略。文獻與官方實作更常見的是「保留組內多 ligand 樣本，但避免跨 split 洩漏」。

## 證據對照表

| 主張 | 主要支持來源 | 對 PLC 的含義 |
|---|---|---|
| PDBbind 上 random split 容易造成 leakage | Li et al., 2026; THGLab, n.d. | 必須做 group-aware split |
| 同 pocket/target 多 ligand 可保留 | Francoeur et al., 2020; Peng et al., 2022; Guan et al., 2023 | 不要在 preprocessing 一開始就粗暴刪除多配體樣本 |
| pocket-based 工作常採固定 pocket-aware split | Francoeur et al., 2020; pengxingang, n.d.-a; guanjq, n.d.-a | split 的基本單位應至少是 pocket/target group |
| sequence clustering 可用於去重與 split 控制 | Steinegger & Söding, 2017 | Level 2/3 用 MMseqs2 合理 |
| pocket 結構相似度可用於更進一步的 structure-aware grouping | Konc & Janežič, 2010; Francoeur et al., 2020 | 若 exact + sequence clustering 不夠，再考慮 ProBiS 路線 |

## APA 參考文獻

Francoeur, P. G., Masuda, T., Sunseri, J., Jia, A., Iovanisci, R. B., Snyder, I., & Koes, D. R. (2020). Three-dimensional convolutional neural networks and a cross-docked data set for structure-based drug design. *Journal of Chemical Information and Modeling, 60*(9), 4200-4215. https://doi.org/10.1021/acs.jcim.0c00411

Guan, J., Qian, W. W., Peng, X., Su, Y., Peng, J., & Ma, J. (2023). *3D equivariant diffusion for target-aware molecule generation and affinity prediction* [arXiv preprint arXiv:2303.03543]. arXiv. https://doi.org/10.48550/arXiv.2303.03543

guanjq. (n.d.-a). *targetdiff* README [Computer software documentation]. GitHub. Retrieved April 13, 2026, from https://github.com/guanjq/targetdiff/blob/main/README.md

guanjq. (n.d.-b). *targetdiff* [Computer software]. GitHub. Retrieved April 13, 2026, from https://github.com/guanjq/targetdiff

Konc, J., & Janežič, D. (2010). ProBiS algorithm for detection of structurally similar protein binding sites by local structural alignment. *Bioinformatics, 26*(9), 1160-1168. https://doi.org/10.1093/bioinformatics/btq100

Li, J., Guan, X., Zhang, O., Sun, K., Wang, Y., Bagni, D., & Head-Gordon, T. (2026). Leak proof PDBBind: A reorganized data set of protein-ligand complexes for more generalizable binding affinity prediction. *The Journal of Physical Chemistry B, 130*(2), 730-740. https://doi.org/10.1021/acs.jpcb.5c08598

Liu, Z., Li, Y., Han, L., Li, J., Liu, J., Zhao, Z., Nie, W., Liu, Y., & Wang, R. (2015). PDB-wide collection of binding data: Current status of the PDBbind database. *Bioinformatics, 31*(3), 405-412. https://doi.org/10.1093/bioinformatics/btu626

Peng, X., Luo, S., Guan, J., Xie, Q., Peng, J., & Ma, J. (2022). *Pocket2Mol: Efficient molecular sampling based on 3D protein pockets* [arXiv preprint arXiv:2205.07249]. arXiv. https://doi.org/10.48550/arXiv.2205.07249

pengxingang. (n.d.-a). *Pocket2Mol data README* [Computer software documentation]. GitHub. Retrieved April 13, 2026, from https://github.com/pengxingang/Pocket2Mol/blob/main/data/README.md

pengxingang. (n.d.-b). *Pocket2Mol* [Computer software]. GitHub. Retrieved April 13, 2026, from https://github.com/pengxingang/Pocket2Mol

Steinegger, M., & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology, 35*(11), 1026-1028. https://doi.org/10.1038/nbt.3988

THGLab. (n.d.). *LP-PDBBind* [Computer software]. GitHub. Retrieved April 13, 2026, from https://github.com/THGLab/LP-PDBBind

Wang, R., Fang, X., Lu, Y., Yang, C.-Y., & Wang, S. (2005). The PDBbind database: Methodologies and updates. *Journal of Medicinal Chemistry, 48*(12), 4111-4119. https://doi.org/10.1021/jm048957q

## 備註

- 本文件中的「PLC 建議流程」屬工程綜合判斷，來源是上列文獻與官方實作，而不是某一篇單一論文直接規定。
- 本文件不包含本地資料統計結果；本地 exact grouping 與 `contact_unique_chain_signature` 分析另由內部腳本與輸出檔支撐。
