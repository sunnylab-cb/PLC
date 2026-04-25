#!/bin/bash
# -*- coding: utf-8 -*-
# 数据集划分完整流程
# 使用MMseqs2进行严格的序列去冗余（30%一致性）
# 最终比例：Train 80% : Val 10% : Test 10%

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "蛋白质-配体数据集划分流程"
echo "======================================================================"
echo ""
echo "策略："
echo "  - 序列一致性阈值: 30%"
echo "  - 比对覆盖率: 80%"
echo "  - 数据集比例: 8:1:1 (Train:Val:Test)"
echo "  - 基于簇的严格划分（同簇样本不分离）"
echo ""

# 配置路径
DATA_DIR="/home/husrcf/Code/ProtBind/New"
OUTPUT_DIR="/home/husrcf/Code/ProtBind/dataset_split"
FASTA_FILE="$OUTPUT_DIR/dataset_proteins.fasta"
CLUSTER_PREFIX="$OUTPUT_DIR/cluster_res"
TMP_DIR="$OUTPUT_DIR/tmp_mmseqs"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

echo "======================================================================"
echo "步骤1：提取蛋白质序列"
echo "======================================================================"
echo ""

if [ -f "$FASTA_FILE" ]; then
    echo "⚠️  FASTA文件已存在: $FASTA_FILE"
    read -p "是否重新提取？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 dataset_split_step1_extract_sequences.py
    else
        echo "跳过步骤1，使用现有FASTA文件"
    fi
else
    python3 dataset_split_step1_extract_sequences.py
fi

echo ""
echo "======================================================================"
echo "步骤2：MMseqs2聚类"
echo "======================================================================"
echo ""

# 检查MMseqs2是否安装
if ! command -v mmseqs &> /dev/null; then
    echo "❌ MMseqs2未安装"
    echo ""
    echo "安装方法："
    echo "  conda install -c bioconda mmseqs2"
    echo "  或"
    echo "  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
    echo "  tar xvfz mmseqs-linux-avx2.tar.gz"
    echo "  export PATH=\$(pwd)/mmseqs/bin/:\$PATH"
    exit 1
fi

echo "MMseqs2版本:"
mmseqs version
echo ""

# 检查是否已经运行过聚类
if [ -f "${CLUSTER_PREFIX}_cluster.tsv" ]; then
    echo "⚠️  聚类结果已存在: ${CLUSTER_PREFIX}_cluster.tsv"
    read -p "是否重新聚类？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过步骤2，使用现有聚类结果"
        SKIP_CLUSTERING=true
    fi
fi

if [ "$SKIP_CLUSTERING" != "true" ]; then
    echo "开始聚类..."
    echo "  输入: $FASTA_FILE"
    echo "  输出: ${CLUSTER_PREFIX}"
    echo "  参数:"
    echo "    --min-seq-id 0.3  (30%序列一致性)"
    echo "    -c 0.8            (80%比对覆盖率)"
    echo "    --cov-mode 1      (覆盖率计算模式)"
    echo "    --threads 16      (使用16线程)"
    echo ""

    # 运行MMseqs2聚类
    mmseqs easy-cluster \
        "$FASTA_FILE" \
        "$CLUSTER_PREFIX" \
        "$TMP_DIR" \
        --min-seq-id 0.3 \
        -c 0.8 \
        --cov-mode 1 \
        --threads 16 \
        --cluster-mode 0 \
        --cluster-steps 3

    echo ""
    echo "✅ 聚类完成"
fi

echo ""
echo "======================================================================"
echo "步骤3：解析聚类结果并划分数据集"
echo "======================================================================"
echo ""

python3 dataset_split_step3_parse_and_split.py

echo ""
echo "======================================================================"
echo "✅ 数据集划分完成！"
echo "======================================================================"
echo ""
echo "输出文件:"
echo "  - train_ids.txt: 训练集PDB ID列表"
echo "  - val_ids.txt: 验证集PDB ID列表"
echo "  - test_ids.txt: 测试集PDB ID列表"
echo "  - dataset_splits.json: 完整划分信息（包含文件路径）"
echo "  - split_statistics.json: 统计信息"
echo ""
echo "所有文件位于: $OUTPUT_DIR"
echo ""
echo "下一步："
echo "  1. 检查split_statistics.json验证比例"
echo "  2. 使用train/val/test_ids.txt进行模型训练"
echo ""
