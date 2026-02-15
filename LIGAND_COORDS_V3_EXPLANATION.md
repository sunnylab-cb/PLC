# Ligand V3 坐标字段说明（提取流程与下游使用）

本文针对 `preparation/ligand/extract_ligand_features_v3.py` 及其依赖流程，解释 `ligand_features_v3/*.pt` 中各类坐标字段的来源、差异、下游用途，以及当前数据中的缺失/不一致情况。

---

## 1. 结论先看（TL;DR）

- `ligand_features_v3` 的主坐标字段是 **`coords`**（当前坐标，按当前原子顺序，通常包含氢）。
- 训练数据集与 VDW 距离计算脚本都默认使用 **`coords`**，不是 `coords_original`。
- `coords_original` 是“原始实验构象的重原子坐标（仅 heavy atoms）”，主要用于保真/回溯，不直接用于当前训练主流程。
- `coords_original_with_h` 是可选字段，目前覆盖率很低（仅少量文件存在），不能作为全量流程依赖字段。

---

## 2. 实际提取链路（V3 -> V2）

## 2.1 顶层脚本（V3）

`extract_ligand_features_v3.py` 的核心逻辑：

1. 先调用 `ligand_features_v2.extract_ligand_features_from_file()` 产出 V2 特征与坐标集合；
2. 再单独加载分子，提取 45 维全局特征（global tag）；
3. 将 V2 的 35 维原子特征 + 45 维全局特征广播拼接，得到 80 维 `atom_features`；
4. 将 V2 返回的坐标字段原样写入 `.pt`（并在可用时附加 `coords_original`/`coords_original_with_h`）。

> 注意：脚本头部有“64维”旧描述，但代码实际是 `80维`（35+45）。

## 2.2 坐标真正生成处（V2）

关键在 `ligand_features_v2.py`：

- 若输入分子有 3D 构象：先缓存原始 conformer 坐标；
- 若原始文件不含氢：执行 `Chem.AddHs(..., addCoords=True)`，并把重原子坐标恢复为原始值；
- 柔性分析使用深拷贝分子 `mol_copy` 做构象系综，不直接改主分子的当前坐标（有原始 3D 时）；
- 最终 `coords` 来自主分子当前 conformer（`mol.GetConformer(0)`）。

因此，`coords` 的本质是“当前工作分子坐标”（可能是原始 + 加氢补点后的结果）。

---

## 3. 各坐标字段的精确定义

| 字段名 | 维度 | 原子集合 | 是否必有 | 坐标来源 | 典型用途 |
|---|---:|---|---|---|---|
| `coords` | `[N_atoms, 3]` | 当前分子全部原子（通常含氢） | 是 | 当前 `mol` 的 conformer(0) | 主流程几何输入、VDW 距离计算 |
| `coords_heavy_current` | `[N_heavy, 3]` | 当前分子重原子 | 是 | 从 `coords` 过滤 heavy atoms | heavy-only 分析/排错 |
| `coords_original` | `[N_heavy, 3]` | 原始文件重原子 | 是（在当前数据中） | 提取前原始 conformer 坐标（heavy） | 保留实验结构、与当前坐标比对 |
| `coords_original_with_h` | `[N_original, 3]` | 原始文件中的全部原子（若原文件本就有氢才“含氢”） | 否 | 原始 conformer 全原子拷贝 | 历史兼容/补充信息 |

### 关键差异

1. **`coords` vs `coords_original`**
   - `coords`：当前分子全原子坐标（训练/距离计算实际用）。
   - `coords_original`：原始构象的重原子快照（不含加氢后新增 H）。

2. **`coords_heavy_current` vs `coords_original`**
   - 两者都是 heavy-only，但前者来自“当前分子”，后者来自“原始输入”。
   - 若加氢流程中恢复了重原子坐标，两者通常接近；但语义仍不同。

3. **`coords_original_with_h` 与名字语义不完全一致**
   - 该字段保存的是“原始文件所有原子坐标”，并非保证“补氢后的完整坐标”。
   - 若原始文件本身无氢，则该字段即使存在，也可能只有 heavy atoms。

---

## 4. 下游流程实际使用了哪个坐标？

## 4.1 训练数据集（`src/data/pocket_dataset_v6.py`）

- 默认取 `ligand_data['coords']` 作为 `ligand_coords`。
- 若长度与 `atom_features` 不一致，才 fallback 到 `coords_original` 并截断特征。

## 4.2 VDW 距离构建（`scripts/calculate_distances_vdw.py`）

- 直接读取 `ligand_data['coords']` 参与蛋白-配体距离计算。
- 因此当前标签体系也与 `coords` 对齐。

结论：**当前主流程坐标基准是 `coords`。**

---

## 5. 对现有 `data/ligand_features_v3` 的实测统计

对 `data/ligand_features_v3` 全量 `19037` 个 `*_ligand.pt` 文件统计结果：

- `coords`：`19037/19037`
- `coords_heavy_current`：`19037/19037`
- `coords_original`：`19037/19037`
- `coords_original_with_h`：`6/19037`

### 缺失与不一致

1. `coords_original_with_h` 大规模缺失（`19031` 个文件缺失）
   - 说明它不是当前提取产物中的稳定必备字段。

2. 在存在该字段的 6 个文件中，发现 1 个长度不一致样例：
   - `data/ligand_features_v3/1981-2000/1jmi/1jmi_ligand.pt`
   - `coords_original_with_h.shape[0] = 20`，`coords.shape[0] = 31`
   - 这也印证了该字段并不总表示“当前全原子（含补氢）坐标”。

3. 其余一致性检查
   - `num_atoms == coords.shape[0]`：全量通过
   - `num_heavy_atoms == coords_heavy_current.shape[0]`：全量通过
   - `num_heavy_atoms == coords_original.shape[0]`：全量通过

---

## 6. 目前信息缺口（建议你在报告中明确）

建议在报告中明确以下“语义缺口”，避免审稿/复现实验误解：

1. **缺少显式坐标来源标记**
   - `.pt` 中没有 `coords_source`（例如 `experimental` / `experimental+H` / `generated`）。

2. **缺少原子映射字段**
   - 没有保存 heavy atom 在 full-atom 中的 index 映射，跨字段对齐主要依赖“构建时顺序一致”的隐式约定。

3. **`coords_original_with_h` 非全量字段且语义不稳定**
   - 名称可能让人误以为“一定是带氢原始坐标”；实际并不总是如此。

---

## 7. 建议的规范化命名（可选改进）

如果后续你要清理格式，建议使用更明确字段名：

- `coords_current_all`（替代 `coords`）
- `coords_original_heavy`（替代 `coords_original`）
- `coords_current_heavy`（替代 `coords_heavy_current`）
- `coords_original_all`（替代 `coords_original_with_h`）
- `coords_source`（新增，记录来源）
- `heavy_atom_indices_in_all`（新增，显式索引映射）

---

## 8. 本文对应的关键代码文件

- `preparation/ligand/extract_ligand_features_v3.py`
- `preparation/ligand/ligand_features_v2.py`
- `src/data/pocket_dataset_v6.py`
- `scripts/calculate_distances_vdw.py`
- `scripts/add_h_coords_to_ligand_features.py`

