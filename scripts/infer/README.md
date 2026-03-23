# V7 模型推理脚本使用指南

## 功能特性

1. **单样本/批量推理** - 对新数据进行预测
2. **评估模式** - 计算完整的评估指标
3. **灵活的 Similarity 分组评估** - 可选的 high/medium/low similarity 分组统计
   - 如果提供 `--split_file`，则计算分组指标
   - 如果不提供，则只显示全局指标
4. **全局推理模式** - 自动遍历 test set，按性能分类保存结果
   - 难例: F1/AP/IoU < 0.2
   - 一般: 0.2 <= F1/AP/IoU < 0.4
   - 优秀: F1/AP/IoU >= 0.4
5. **可视化分析** - 自动生成性能分析图表
   - F1/AP/IoU 分布直方图
   - 准确率分布
   - F1 vs AP 散点图
   - Per-class F1 对比箱线图
6. **详细统计信息** - 终端输出详细的性能统计
   - 整体性能统计（均值、标准差、范围、中位数、四分位数）
   - Per-class 详细指标
   - Top/Bottom 样本分析

## 使用方法

### 1. 单样本推理

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_id 1a2b
```

### 2. 批量推理

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_list test_ids.txt
```

### 3. 保存预测结果

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_list test_ids.txt \
    --output predictions/
```

### 4. 评估模式（需要标签数据）- 仅全局指标

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_list test_ids.txt \
    --evaluate
```

**输出示例（无 similarity 分组）：**
```
================================================================================
评估结果
================================================================================

整体指标:
  准确率: 0.8523
  Contact (c0) - Precision: 0.7512, Recall: 0.8034, F1: 0.7764, AP: 0.8234
  Interface (c1) - Precision: 0.6823, Recall: 0.7145, F1: 0.6980
  Non-binding (c2) - Precision: 0.9234, Recall: 0.9012, F1: 0.9122

================================================================================
```

### 5. Similarity 分组评估（推荐）

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_list test_ids.txt \
    --split_file configs/splits/filtered_splits.txt \
    --evaluate \
    --output results/
```

**输出示例（有 similarity 分组）：**
```
================================================================================
评估结果
================================================================================

整体指标:
  准确率: 0.8523
  Contact (c0) - Precision: 0.7512, Recall: 0.8034, F1: 0.7764, AP: 0.8234
  Interface (c1) - Precision: 0.6823, Recall: 0.7145, F1: 0.6980
  Non-binding (c2) - Precision: 0.9234, Recall: 0.9012, F1: 0.9122

================================================================================
Similarity 分组指标:
================================================================================

HIGH_SIM:
  样本数: 596, 残基数: 123456
  准确率: 0.8823
  Contact (c0) - Precision: 0.8012, Recall: 0.8534, F1: 0.8264, AP: 0.8734

MEDIUM_SIM:
  样本数: 595, 残基数: 118234
  准确率: 0.8523
  Contact (c0) - Precision: 0.7512, Recall: 0.8034, F1: 0.7764, AP: 0.8234

LOW_SIM:
  样本数: 597, 残基数: 121345
  准确率: 0.8223
  Contact (c0) - Precision: 0.7012, Recall: 0.7534, F1: 0.7264, AP: 0.7734

================================================================================
```

### 6. 全局推理模式（自动加载 test set）

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --split_file configs/splits/filtered_splits.txt \
    --global_infer
```

**功能说明：**
- 自动从 split 文件加载所有 test set 样本
- 对每个样本计算详细指标（F1, AP, IoU for all classes）
- 按 Contact class (c0) 的性能自动分类
- 自动创建带时间戳的子目录（如 `outputs/infer_v7_20260220_143052/`）
- 保存 4 个 CSV 文件 + 1 个可视化图表：
  - `hard_cases.csv` - 难例（平均 F1/AP/IoU < 0.2）
  - `medium_cases.csv` - 一般例子（0.2 <= 平均 F1/AP/IoU < 0.4）
  - `good_cases.csv` - 优秀例子（平均 F1/AP/IoU >= 0.4）
  - `all_results.csv` - 所有样本的完整结果
  - `inference_visualization.png` - 性能分析图表（6 个子图）
- 使用 tqdm 显示推理进度条
- 终端输出详细统计信息（均值、标准差、范围、中位数、四分位数等）

**CSV 文件格式：**
```csv
pdb_id,accuracy,n_residues,f1_c0,ap_c0,iou_c0,precision_c0,recall_c0,f1_c1,iou_c1,precision_c1,recall_c1,f1_c2,iou_c2,precision_c2,recall_c2
1a2b,0.8523,245,0.7764,0.8234,0.6345,0.7512,0.8034,0.6980,0.5367,0.6823,0.7145,0.9122,0.8389,0.9234,0.9012
...
```

**输出示例：**
```
输出目录: outputs/infer_v7_20260220_143052

=== 全局推理模式 ===

推理进度: 100%|████████████████████| 1788/1788 [15:23<00:00,  1.94batch/s]

================================================================================
分类统计:
================================================================================
总样本数: 1788
难例 (F1/AP/IoU < 0.2):       156 (  8.7%)
一般 (0.2 <= F1/AP/IoU < 0.4):  523 ( 29.2%)
优秀 (F1/AP/IoU >= 0.4):     1109 ( 62.0%)
================================================================================

保存结果到: outputs/infer_v7_20260220_143052/
  hard_cases.csv: 156 个样本
  medium_cases.csv: 523 个样本
  good_cases.csv: 1109 个样本
  all_results.csv: 1788 个样本

================================================================================
详细统计信息
================================================================================

【整体性能】
  样本数: 1788
  平均准确率: 0.8523 ± 0.0845
  准确率范围: [0.4521, 0.9876]
  中位数准确率: 0.8634

【Contact Class (c0) 详细统计】
  F1        : 0.7764 ± 0.1234  [0.1234, 0.9567]  Median: 0.7856  Q25-Q75: [0.6845, 0.8723]
  AP        : 0.8234 ± 0.1123  [0.2345, 0.9789]  Median: 0.8345  Q25-Q75: [0.7456, 0.9012]
  IoU       : 0.6345 ± 0.1456  [0.0987, 0.9123]  Median: 0.6456  Q25-Q75: [0.5234, 0.7567]
  Precision : 0.7512 ± 0.1345  [0.1456, 0.9678]  Median: 0.7623  Q25-Q75: [0.6534, 0.8456]
  Recall    : 0.8034 ± 0.1234  [0.2345, 0.9789]  Median: 0.8145  Q25-Q75: [0.7123, 0.8867]

【Interface Class (c1) 统计】
  F1        : 0.6980 ± 0.1456  [0.0876, 0.9234]  Median: 0.7089  Q25-Q75: [0.5967, 0.7934]
  IoU       : 0.5367 ± 0.1567  [0.0456, 0.8567]  Median: 0.5478  Q25-Q75: [0.4234, 0.6456]
  ...

【Non-binding Class (c2) 统计】
  F1        : 0.9122 ± 0.0567  [0.6789, 0.9876]  Median: 0.9234  Q25-Q75: [0.8867, 0.9456]
  ...

【Top 10 最佳样本 (按 F1_c0)】
  1. 1abc: F1=0.9567, AP=0.9789, IoU=0.9123, Acc=0.9876
  2. 2def: F1=0.9456, AP=0.9678, IoU=0.8967, Acc=0.9765
  ...

【Bottom 10 最差样本 (按 F1_c0)】
  1. 9xyz: F1=0.1234, AP=0.2345, IoU=0.0987, Acc=0.4521
  2. 8uvw: F1=0.1456, AP=0.2567, IoU=0.1123, Acc=0.4789
  ...

生成可视化图表...
  可视化图表: inference_visualization.png
```

### 7. 禁用可视化（加快速度）

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --split_file configs/splits/filtered_splits.txt \
    --global_infer \
    --no_viz
```

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/pocket_gvp_fusion_v7/best_model.pt \
    --pdb_list test_ids.txt \
    --evaluate \
    --amp
```

## 输出指标说明

### 整体指标（始终显示）
- **Accuracy** - 整体准确率
- **Per-class Precision/Recall/F1** - 每个类别的精确率、召回率、F1 分数
  - Contact (c0) - 接触残基
  - Interface (c1) - 界面残基
  - Non-binding (c2) - 非结合残基
- **AP (Average Precision)** - Contact class 的平均精度

### Similarity 分组指标（仅当提供 --split_file 时显示）
- **high_acc/med_acc/low_acc** - 各分组的准确率
- **high_f1_c0/med_f1_c0/low_f1_c0** - 各分组 Contact class 的 F1 分数
- **high_ap_c0/med_ap_c0/low_ap_c0** - 各分组 Contact class 的 AP 分数
- **n_samples/n_tokens** - 各分组的样本数和残基数

## 灵活性说明

脚本会根据是否提供 `--split_file` 自动调整输出：

1. **不提供 `--split_file`**：只显示全局指标，适合快速评估
2. **提供 `--split_file`**：显示全局指标 + similarity 分组指标，适合详细分析
3. **Split 文件中无 similarity 信息**：自动回退到只显示全局指标

## 输出文件

### 1. NPZ 格式预测结果 (.npz)
```python
{
    'pdb_id': str,
    'predictions': np.ndarray,  # [L] 预测类别
    'probabilities': np.ndarray,  # [L, 3] 类别概率
    'sequence_length': int,
    'n_contact': int,
    'n_interface': int,
    'n_non_binding': int
}
```

### 2. TSV 格式详细预测 (.tsv)
每个残基一行，包含详细的预测信息：

```tsv
residue_id	predicted_class	predicted_label	prob_contact	prob_interface	prob_non_binding	confidence
0	0	Contact	0.8523	0.1234	0.0243	0.8523
1	2	Non-binding	0.0456	0.0789	0.8755	0.8755
2	0	Contact	0.9123	0.0567	0.0310	0.9123
3	1	Interface	0.2345	0.6789	0.0866	0.6789
```

**列说明：**
- `residue_id`: 残基索引（从0开始）
- `predicted_class`: 预测类别 (0=Contact, 1=Interface, 2=Non-binding)
- `predicted_label`: 预测类别名称
- `prob_contact/interface/non_binding`: 各类别概率
- `confidence`: 预测置信度（最大概率值）

### 3. PDB 格式预测口袋 (.pdb)
在输出目录下创建 `pdb/` 子目录，组织结构如下：

```
outputs/infer_v7_20260220_143052/
└── pdb/
    └── 1981-2000/
        └── 1kjb/
            ├── 1kjb_protein.pdb          # 完整蛋白质结构（从数据集复制）
            ├── 1kjb_pocket.pdb           # 真实口袋（从数据集复制）
            ├── 1kjb_ligand.sdf           # 配体结构（从数据集复制）
            └── 1kjb_pocket_predicted.pdb # 预测的口袋残基（新生成）
```

**`{pdb_id}_pocket_predicted.pdb` 说明：**
- 只包含预测为 Contact 或 Interface 类别的残基
- 保留完整的原子坐标信息
- 可以直接在 PyMOL/Chimera 等软件中可视化
- 便于与真实口袋 `{pdb_id}_pocket.pdb` 进行对比

### 4. 评估结果 (evaluation_metrics.json)
```json
{
    "accuracy": 0.85,
    "precision_c0": 0.75,
    "recall_c0": 0.80,
    "f1_c0": 0.77,
    "ap_c0": 0.82,
    "high_acc": 0.88,
    "high_f1_c0": 0.80,
    "high_ap_c0": 0.85,
    "med_acc": 0.85,
    "med_f1_c0": 0.77,
    "med_ap_c0": 0.82,
    "low_acc": 0.82,
    "low_f1_c0": 0.74,
    "low_ap_c0": 0.78
}
```
