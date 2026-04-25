# 数据集划分流程文档

## 📋 概述

使用MMseqs2进行严格的序列去冗余，确保训练/验证/测试集之间没有高度相似的蛋白质序列。

### 核心策略
- **序列一致性阈值**: 30% (严格去冗余)
- **比对覆盖率**: 80%
- **数据集比例**: 8:1:1 (Train:Val:Test)
- **划分原则**: 基于簇的完整划分（同簇样本不分离）

---

## 🔄 完整流程

### 步骤1：提取蛋白质序列

**脚本**: `dataset_split_step1_extract_sequences.py`

**功能**:
- 从19,037个PDB文件中提取蛋白质序列
- 转换为FASTA格式
- 检测完全相同的序列
- 生成序列映射文件

**输出**:
```
dataset_split/
├── dataset_proteins.fasta        # FASTA格式序列文件
└── dataset_proteins.mapping.txt  # PDB ID到序列的映射
```

**运行**:
```bash
python3 dataset_split_step1_extract_sequences.py
```

**预计时间**: ~2-3分钟（19,037个PDB文件）

---

### 步骤2：MMseqs2聚类

**工具**: MMseqs2

**参数说明**:
```bash
mmseqs easy-cluster \
    dataset_proteins.fasta \
    cluster_res \
    tmp_dir \
    --min-seq-id 0.3    # 30%序列一致性阈值
    -c 0.8              # 80%比对覆盖率
    --cov-mode 1        # 覆盖率计算模式（基于靶标和查询）
    --threads 16        # 使用16线程
    --cluster-mode 0    # 聚类模式（Set-Cover）
    --cluster-steps 3   # 聚类步骤数
```

**输出**:
```
dataset_split/
├── cluster_res_cluster.tsv       # 聚类结果（重要！）
├── cluster_res_rep_seq.fasta     # 代表序列
└── cluster_res_all_seqs.fasta    # 所有序列
```

**cluster_res_cluster.tsv格式**:
```
cluster_rep_1    member_1
cluster_rep_1    member_2
cluster_rep_2    member_3
...
```

**预计时间**: ~5-10分钟（取决于CPU）

---

### 步骤3：解析聚类并划分数据集

**脚本**: `dataset_split_step3_parse_and_split.py`

**功能**:
1. 解析MMseqs2聚类结果
2. 使用贪心算法划分簇到Train/Val/Test
3. 确保同簇样本不分离
4. 最终比例接近8:1:1

**贪心算法**:
```python
# 伪代码
1. 按簇大小（样本数）从大到小排序
2. for each cluster:
     计算每个集合的缺口 = 目标数量 - 当前数量
     将整个簇分配给缺口最大的集合
3. 输出最终划分
```

**输出**:
```
dataset_split/
├── train_ids.txt              # 训练集PDB ID列表
├── val_ids.txt                # 验证集PDB ID列表
├── test_ids.txt               # 测试集PDB ID列表
├── dataset_splits.json        # 完整划分信息（包含文件路径）
└── split_statistics.json      # 统计信息
```

**运行**:
```bash
python3 dataset_split_step3_parse_and_split.py
```

**预计时间**: <1分钟

---

## 🚀 一键运行

使用主脚本运行完整流程：

```bash
bash dataset_split_pipeline.sh
```

该脚本会：
1. 检查依赖（MMseqs2）
2. 依次运行步骤1-3
3. 提供交互式确认（避免重复计算）
4. 输出最终统计信息

---

## 📊 输出文件说明

### 1. train_ids.txt / val_ids.txt / test_ids.txt

纯文本文件，每行一个PDB ID：
```
1jmi
1ibg
1qka
...
```

**用途**: 直接用于数据加载器过滤样本

### 2. dataset_splits.json

JSON格式，包含完整信息：
```json
{
  "train": [
    {
      "pdb_id": "1jmi",
      "protein_pdb": "1981-2000/1jmi/1jmi_protein.pdb",
      "ligand_sdf": "1981-2000/1jmi/1jmi_ligand.sdf"
    },
    ...
  ],
  "val": [...],
  "test": [...]
}
```

**用途**:
- 快速构建数据加载器
- 包含相对路径，便于移植

### 3. split_statistics.json

统计信息：
```json
{
  "total_samples": 19037,
  "train": {
    "count": 15230,
    "ratio": 0.80
  },
  "val": {
    "count": 1904,
    "ratio": 0.10
  },
  "test": {
    "count": 1903,
    "ratio": 0.10
  }
}
```

**用途**: 验证划分比例

---

## 🔍 质量检查

### 1. 检查序列一致性

验证Train/Val/Test之间没有高度相似的序列：

```python
# 使用MMseqs2搜索
mmseqs easy-search \
    val_sequences.fasta \
    train_sequences.fasta \
    search_results.tsv \
    tmp_dir \
    --min-seq-id 0.3
```

**预期结果**: 应该没有匹配（或极少匹配）

### 2. 检查簇完整性

验证同簇样本没有被分离：

```python
import json

# 加载聚类结果
clusters = parse_mmseqs_clusters("cluster_res_cluster.tsv")

# 加载划分结果
with open("dataset_splits.json") as f:
    splits = json.load(f)

# 检查每个簇
for cluster_rep, members in clusters.items():
    # 检查所有成员是否在同一集合
    train_count = sum(1 for m in members if m in splits['train'])
    val_count = sum(1 for m in members if m in splits['val'])
    test_count = sum(1 for m in members if m in splits['test'])

    # 应该只有一个计数>0
    assert sum([train_count>0, val_count>0, test_count>0]) == 1
```

### 3. 检查比例偏差

```python
import json

with open("split_statistics.json") as f:
    stats = json.load(f)

# 检查偏差
target_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
for split_name, target_ratio in target_ratios.items():
    actual_ratio = stats[split_name]['ratio']
    deviation = abs(actual_ratio - target_ratio)
    print(f"{split_name}: {actual_ratio:.2%} (目标: {target_ratio:.0%}, 偏差: {deviation:.2%})")
```

**可接受偏差**: ±2%

---

## 💡 使用示例

### PyTorch DataLoader

```python
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class ProteinLigandDataset(Dataset):
    def __init__(self, data_dir, split='train', splits_file='dataset_splits.json'):
        with open(splits_file) as f:
            splits = json.load(f)

        self.data_dir = Path(data_dir)
        self.samples = splits[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载蛋白质和配体
        protein_path = self.data_dir / sample['protein_pdb']
        ligand_path = self.data_dir / sample['ligand_sdf']

        # ... 加载和处理数据 ...

        return protein_features, ligand_features, label

# 使用
train_dataset = ProteinLigandDataset('/path/to/New', split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 简单过滤

```python
# 读取ID列表
with open('train_ids.txt') as f:
    train_ids = set(line.strip() for line in f)

# 过滤数据
for pdb_file in all_pdb_files:
    pdb_id = pdb_file.stem.replace('_protein', '')
    if pdb_id in train_ids:
        # 处理训练样本
        ...
```

---

## ⚠️ 注意事项

### 1. 序列一致性阈值选择

- **30%**: 严格去冗余（推荐用于最终评估）
- **40%**: 中等去冗余（可用于初步实验）
- **50%**: 宽松去冗余（不推荐）

### 2. 比对覆盖率

- **80%**: 确保比对覆盖大部分序列
- 防止局部短序列的错误匹配

### 3. 簇大小不均

- 某些热门靶点可能有几百个结构
- 冷门靶点可能只有1个结构
- 贪心算法会自动平衡

### 4. 完全相同的序列

- 数据集中可能存在完全相同的序列（不同配体）
- 这些会被聚类到同一簇
- 确保它们在同一集合中

---

## 🔧 故障排除

### MMseqs2未安装

```bash
# Conda安装
conda install -c bioconda mmseqs2

# 或手动安装
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

### Biotite未安装

```bash
conda install -c conda-forge biotite
```

### 内存不足

如果MMseqs2内存不足，可以：
1. 减少线程数: `--threads 8`
2. 使用磁盘模式: `--split-memory-limit 32G`

### 聚类时间过长

- 使用更多线程: `--threads 32`
- 减少聚类步骤: `--cluster-steps 1`

---

## 📚 参考资料

- [MMseqs2文档](https://github.com/soedinglab/MMseqs2)
- [序列去冗余最佳实践](https://www.nature.com/articles/s41592-019-0437-4)
- [AlphaFold数据集划分](https://github.com/deepmind/alphafold)

---

## ✅ 检查清单

- [ ] 步骤1完成：FASTA文件生成
- [ ] 步骤2完成：MMseqs2聚类
- [ ] 步骤3完成：数据集划分
- [ ] 验证比例接近8:1:1
- [ ] 检查簇完整性
- [ ] 验证序列去冗余
- [ ] 测试数据加载器

---

## 📞 支持

如有问题，请检查：
1. 所有依赖是否正确安装
2. 文件路径是否正确
3. MMseqs2版本是否兼容（推荐 ≥13.0）
