# V7 推理脚本使用示例

## 快速开始

### 1. 全局推理模式（推荐用于完整评估）

```bash
# 自动加载 test set，按性能分类保存
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --split_file configs/splits/filtered_splits.txt \
    --global_infer
```

**输出文件：**
- `outputs/infer_v7_20260220_143052/hard_cases.csv` - 难例（F1/AP/IoU < 0.2）
- `outputs/infer_v7_20260220_143052/medium_cases.csv` - 一般例子
- `outputs/infer_v7_20260220_143052/good_cases.csv` - 优秀例子
- `outputs/infer_v7_20260220_143052/all_results.csv` - 所有结果

**进度显示：**
```
推理进度: 100%|████████████████████| 1788/1788 [15:23<00:00,  1.94batch/s]
```

### 2. 评估模式（计算整体指标）

```bash
# 仅全局指标
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_list test_ids.txt \
    --evaluate

# 全局指标 + Similarity 分组
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_list test_ids.txt \
    --split_file configs/splits/filtered_splits.txt \
    --evaluate \
    --output results/
```

### 3. 单样本推理

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_id 1a2b
```

### 4. 批量推理

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_list my_proteins.txt \
    --output predictions/
```

## 性能优化

### 使用 AMP 加速

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --split_file configs/splits/filtered_splits.txt \
    --global_infer \
    --amp
```

### 调整批量大小

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_list test_ids.txt \
    --evaluate \
    --batch_size 4 \
    --amp
```

## 分析难例

### 1. 运行全局推理

```bash
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --split_file configs/splits/filtered_splits.txt \
    --global_infer
```

### 2. 查看难例

```bash
# 进入最新的输出目录
cd outputs/infer_v7_20260220_143052/

# 查看难例列表
head -20 hard_cases.csv

# 统计难例数量
wc -l hard_cases.csv

# 按 F1 排序查看最差的 10 个
sort -t, -k4 -n hard_cases.csv | head -11
```

### 3. 针对难例进行详细分析

```bash
# 提取难例 ID（从最新的输出目录）
tail -n +2 outputs/infer_v7_20260220_143052/hard_cases.csv | cut -d, -f1 > hard_case_ids.txt

# 对难例进行详细推理
python scripts/infer/infer_v7.py \
    --checkpoint experiments/best_model.pt \
    --pdb_list hard_case_ids.txt \
    --output hard_case_predictions/
```

## 常见问题

### Q: 如何只评估 test set？
A: 使用 `--global_infer` 模式，它会自动从 split 文件加载 test set。

### Q: 如何查看每个样本的详细指标？
A: 使用 `--global_infer` 模式，所有样本的详细指标会保存在 `outputs/all_results.csv`。

### Q: 如何自定义输出目录？
A: 使用 `--output` 参数指定输出目录。

### Q: 如何加速推理？
A: 使用 `--amp` 启用自动混合精度，并适当增加 `--batch_size`。
