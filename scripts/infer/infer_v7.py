#!/usr/bin/env python3
"""
推理脚本 for Pocket-GVP-Fusion v7

功能:
1. 单样本/批量推理
2. 评估模式 (计算 Accuracy, Precision, Recall, F1, AP)
3. Similarity 分组评估 (high/medium/low similarity)
4. 全局推理模式 (自动遍历 test set，按性能分类保存)

使用方法:
    # 单个样本推理
    python infer_v7.py --checkpoint path/to/best_model.pt --pdb_id 1a2b

    # 批量推理
    python infer_v7.py --checkpoint path/to/best_model.pt --pdb_list test_ids.txt

    # 保存预测结果
    python infer_v7.py --checkpoint path/to/best_model.pt --pdb_id 1a2b --output predictions/

    # 评估模式 (需要标签数据)
    python infer_v7.py --checkpoint path/to/best_model.pt --pdb_list test_ids.txt --evaluate

    # 评估模式 + Similarity 分组
    python infer_v7.py --checkpoint path/to/best_model.pt \
        --pdb_list test_ids.txt \
        --split_file configs/splits/filtered_splits.txt \
        --evaluate \
        --output results/

    # 全局推理模式 (自动加载 test set，按性能分类)
    python infer_v7.py --checkpoint path/to/best_model.pt \
        --split_file configs/splits/filtered_splits.txt \
        --global_infer
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import csv
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.pocket_dataset_v6 import PocketDatasetV6
from src.data.collate import pocket_collate_fn, prepare_model_inputs
from src.models.pocket_gvp_fusion_v7 import PocketGVPFusionV7

# Similarity groups
SIM_GROUPS = ("high_sim", "medium_sim", "low_sim")


def load_similarity_groups(split_file: str) -> Dict[str, str]:
    """
    从 split 文件加载 similarity 分组信息

    文件格式: split\tpdb_id\tsim_group
    例如: test\t1a2b\thigh_sim
    """
    sim_groups = {}

    if not Path(split_file).exists():
        print(f"警告: Split 文件不存在: {split_file}")
        return sim_groups

    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                split, pdb_id, sim_group = parts[0], parts[1], parts[2]
                if sim_group in SIM_GROUPS:
                    sim_groups[pdb_id] = sim_group

    return sim_groups


def load_test_ids_from_split(split_file: str) -> List[str]:
    """
    从 split 文件加载 test set 的 PDB IDs

    文件格式: split\tpdb_id\t[sim_group]
    """
    test_ids = []

    if not Path(split_file).exists():
        print(f"错误: Split 文件不存在: {split_file}")
        return test_ids

    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                split, pdb_id = parts[0], parts[1]
                if split == 'test':
                    test_ids.append(pdb_id)

    return test_ids


def compute_ap_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 Average Precision (AP) 分数

    Args:
        probs: [N] 正类概率
        labels: [N] 二值标签 (0/1)

    Returns:
        AP 分数
    """
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, probs))
    except ImportError:
        print("警告: sklearn 未安装，无法计算 AP")
        return 0.0


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """加载模型检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config')
    if config is None:
        raise ValueError("检查点中未找到配置信息")

    return checkpoint, config


def create_model_from_checkpoint(
    checkpoint: Dict,
    config: Dict,
    device: torch.device
) -> torch.nn.Module:
    """从检查点创建模型"""
    model_cfg = config['model']

    print("创建 V7 模型...")
    model = PocketGVPFusionV7(model_cfg)
    model = model.to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}")

    return model


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    sample: Dict,
    device: torch.device,
    use_amp: bool = False
) -> Dict:
    """对单个样本进行推理"""
    # 准备输入
    inputs = prepare_model_inputs(sample, device)

    # 前向传播
    with torch.amp.autocast("cuda", enabled=use_amp):
        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_scalar=inputs['protein_scalar'],
            protein_vectors=inputs['protein_vectors'],
            protein_coords=inputs['protein_coords'],
            protein_mask=inputs['protein_mask'],
            ligand_x=inputs['ligand_x'],
            ligand_pos=inputs['ligand_pos'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_to_protein=inputs['ligand_to_protein'],
            ligand_mask=inputs['ligand_mask']
        )

    logits = outputs['logits']  # [B, L, 3]
    probs = F.softmax(logits, dim=-1)  # [B, L, 3]
    preds = logits.argmax(dim=-1)  # [B, L]

    # 应用 mask
    mask = sample['protein_mask'].to(device)

    # 获取 PDB ID（兼容不同的键名）
    if 'pdb_ids' in sample:
        pdb_id = sample['pdb_ids'][0] if isinstance(sample['pdb_ids'], list) else sample['pdb_ids']
    elif 'pdb_id' in sample:
        pdb_id = sample['pdb_id'][0] if isinstance(sample['pdb_id'], list) else sample['pdb_id']
    else:
        pdb_id = 'unknown'

    result = {
        'pdb_id': pdb_id,
        'predictions': preds[0][mask[0]].cpu().numpy(),  # 只保留有效残基
        'probabilities': probs[0][mask[0]].cpu().numpy(),  # [L_valid, 3]
        'sequence_length': mask[0].sum().item(),
        'n_contact': (preds[0][mask[0]] == 0).sum().item(),
        'n_interface': (preds[0][mask[0]] == 1).sum().item(),
        'n_non_binding': (preds[0][mask[0]] == 2).sum().item()
    }

    # 如果有真实标签，也添加到结果中
    if 'labels' in sample:
        labels = sample['labels'].to(device)
        result['true_labels'] = labels[0][mask[0]].cpu().numpy()

    return result


def find_pdb_path(pdb_id: str, pdb_root: str = '/home/husrcf/Code/ProtBind/New') -> Optional[Path]:
    """
    查找 PDB ID 对应的目录路径

    Args:
        pdb_id: PDB ID (如 '1kjb')
        pdb_root: PDB 数据根目录

    Returns:
        PDB 目录路径，如果找不到返回 None
    """
    pdb_root = Path(pdb_root)

    # 如果是相对路径，转换为绝对路径
    if not pdb_root.is_absolute():
        pdb_root = Path.cwd() / pdb_root

    # 检查目录是否存在
    if not pdb_root.exists():
        print(f"  调试: PDB 根目录不存在: {pdb_root}")
        return None

    # 调试信息
    # print(f"  调试: 在 {pdb_root} 中查找 {pdb_id}")

    # 遍历年份目录
    for year_dir in sorted(pdb_root.iterdir()):
        if year_dir.is_dir():
            pdb_dir = year_dir / pdb_id.lower()
            if pdb_dir.exists():
                # print(f"  调试: 找到 {pdb_dir}")
                return pdb_dir

    # 如果没找到，打印调试信息
    print(f"  调试: 未在以下目录找到 {pdb_id}:")
    for year_dir in sorted(pdb_root.iterdir()):
        if year_dir.is_dir():
            print(f"    - {year_dir}")

    return None


def save_predictions_tsv(result: Dict, output_dir: Path, true_labels: Optional[np.ndarray] = None):
    """
    保存预测结果为 TSV 格式

    Args:
        result: 预测结果字典
        output_dir: 输出目录
        true_labels: 真实标签（可选，用于评估模式）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = result['pdb_id']
    predictions = result['predictions']
    probabilities = result['probabilities']

    # 类别名称映射
    class_names = {0: 'Contact', 1: 'Interface', 2: 'Non-binding'}

    # 创建 TSV 文件
    tsv_file = output_dir / f"{pdb_id}_predictions.tsv"

    with open(tsv_file, 'w') as f:
        # 写入表头
        if true_labels is not None:
            f.write("residue_id\tpredicted_class\tpredicted_label\ttrue_class\ttrue_label\t"
                   "correct\tprob_contact\tprob_interface\tprob_non_binding\tconfidence\n")
        else:
            f.write("residue_id\tpredicted_class\tpredicted_label\t"
                   "prob_contact\tprob_interface\tprob_non_binding\tconfidence\n")

        # 写入每个残基的预测
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = probs.max()

            if true_labels is not None:
                true_class = true_labels[i]
                correct = (pred == true_class)
                f.write(f"{i}\t{pred}\t{class_names[pred]}\t{true_class}\t{class_names[true_class]}\t"
                       f"{correct}\t{probs[0]:.4f}\t{probs[1]:.4f}\t{probs[2]:.4f}\t{confidence:.4f}\n")
            else:
                f.write(f"{i}\t{pred}\t{class_names[pred]}\t"
                       f"{probs[0]:.4f}\t{probs[1]:.4f}\t{probs[2]:.4f}\t{confidence:.4f}\n")

    return tsv_file


def create_predicted_pocket_pdb(
    pdb_id: str,
    predictions: np.ndarray,
    output_dir: Path,
    pdb_root: str = '/home/husrcf/Code/ProtBind/New'
) -> Optional[Path]:
    """
    根据预测结果创建 pocket PDB 文件

    Args:
        pdb_id: PDB ID
        predictions: 预测类别数组 [L]
        output_dir: 输出目录
        pdb_root: PDB 数据根目录

    Returns:
        生成的 PDB 文件路径，如果失败返回 None
    """
    # 查找原始 PDB 目录
    pdb_dir = find_pdb_path(pdb_id, pdb_root)
    if pdb_dir is None:
        print(f"  警告: 未找到 PDB {pdb_id} 的目录")
        return None

    # 读取原始 protein PDB 文件
    protein_pdb = pdb_dir / f"{pdb_id}_protein.pdb"
    if not protein_pdb.exists():
        print(f"  警告: 未找到 {protein_pdb}")
        return None

    # 确定年份目录
    year_dir = pdb_dir.parent.name

    # 创建输出目录结构: outputs/infer_xx/pdb/1981-2000/1kjb/
    pdb_output_dir = output_dir / 'pdb' / year_dir / pdb_id
    pdb_output_dir.mkdir(parents=True, exist_ok=True)

    # 复制原始文件到输出目录
    import shutil

    # 复制 protein.pdb
    shutil.copy2(protein_pdb, pdb_output_dir / f"{pdb_id}_protein.pdb")

    # 复制 pocket.pdb (如果存在)
    pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
    if pocket_pdb.exists():
        shutil.copy2(pocket_pdb, pdb_output_dir / f"{pdb_id}_pocket.pdb")

    # 复制 ligand.sdf (如果存在)
    ligand_sdf = pdb_dir / f"{pdb_id}_ligand.sdf"
    if ligand_sdf.exists():
        shutil.copy2(ligand_sdf, pdb_output_dir / f"{pdb_id}_ligand.sdf")

    # 读取 protein PDB 文件并提取残基信息
    residues = []
    current_residue = None
    residue_atoms = []

    with open(protein_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # 提取残基序号
                res_seq = int(line[22:26].strip())
                chain_id = line[21]

                if current_residue != (chain_id, res_seq):
                    if current_residue is not None:
                        residues.append((current_residue, residue_atoms))
                    current_residue = (chain_id, res_seq)
                    residue_atoms = []

                residue_atoms.append(line)

        # 添加最后一个残基
        if current_residue is not None:
            residues.append((current_residue, residue_atoms))

    # 根据预测结果筛选 pocket 残基 (Contact + Interface)
    predicted_pocket_pdb = pdb_output_dir / f"{pdb_id}_pocket_predicted.pdb"

    with open(predicted_pocket_pdb, 'w') as f:
        # 写入 PDB 头部
        f.write(f"HEADER    {pdb_id.upper()}_POCKET_PREDICTED\n")
        f.write(f"COMPND    {pdb_id.upper()}_POCKET_PREDICTED\n")
        f.write(f"REMARK    Generated by V7 inference script\n")
        f.write(f"REMARK    Predicted pocket residues (Contact + Interface classes)\n")

        atom_counter = 1
        for idx, ((chain_id, res_seq), atoms) in enumerate(residues):
            # 检查索引是否在预测范围内
            if idx < len(predictions):
                pred_class = predictions[idx]

                # 只保留 Contact (0) 和 Interface (1) 残基
                if pred_class in [0, 1]:
                    for atom_line in atoms:
                        # 更新原子序号
                        new_line = f"{atom_line[:6]}{atom_counter:5d}{atom_line[11:]}"
                        f.write(new_line)
                        atom_counter += 1

        f.write("END\n")

    return predicted_pocket_pdb


def save_predictions(result: Dict, output_dir: Path, pdb_root: str = '/home/husrcf/Code/ProtBind/New', true_labels: Optional[np.ndarray] = None):
    """
    保存预测结果（NPZ + TSV + PDB）

    Args:
        result: 预测结果字典
        output_dir: 输出目录
        pdb_root: PDB 数据根目录
        true_labels: 真实标签（可选，用于评估模式）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = result['pdb_id']

    # 1. ��存 NPZ 格式（原有格式）
    output_file = output_dir / f"{pdb_id}_predictions.npz"
    npz_data = {
        'pdb_id': pdb_id,
        'predictions': result['predictions'],
        'probabilities': result['probabilities'],
        'sequence_length': result['sequence_length'],
        'n_contact': result['n_contact'],
        'n_interface': result['n_interface'],
        'n_non_binding': result['n_non_binding']
    }

    # 如果有真实标签，也保存到 NPZ
    if true_labels is not None:
        npz_data['true_labels'] = true_labels

    np.savez(output_file, **npz_data)

    # 2. 保存 TSV 格式（传递真实标签）
    tsv_file = save_predictions_tsv(result, output_dir, true_labels)
    print(f"✓ 保存预测结果: {output_file.name}, {tsv_file.name}")

    # 3. 创建预测 pocket PDB 文件
    try:
        predicted_pdb = create_predicted_pocket_pdb(
            pdb_id,
            result['predictions'],
            output_dir,
            pdb_root
        )
        if predicted_pdb:
            print(f"✓ 生成预测 pocket PDB: pdb/{predicted_pdb.parent.parent.name}/{pdb_id}/{predicted_pdb.name}")
    except Exception as e:
        print(f"  警告: 生成 PDB 文件失败: {e}")


def print_prediction_summary(result: Dict):
    """打印预测摘要"""
    pdb_id = result['pdb_id']
    seq_len = result['sequence_length']
    n_contact = result['n_contact']
    n_interface = result['n_interface']
    n_non_binding = result['n_non_binding']

    print(f"\n{'='*60}")
    print(f"PDB ID: {pdb_id}")
    print(f"序列长度: {seq_len}")
    print(f"{'='*60}")
    print(f"预测统计:")
    print(f"  接触残基 (Contact):      {n_contact:4d} ({n_contact/seq_len*100:5.1f}%)")
    print(f"  界面残基 (Interface):    {n_interface:4d} ({n_interface/seq_len*100:5.1f}%)")
    print(f"  非结合残基 (Non-binding): {n_non_binding:4d} ({n_non_binding/seq_len*100:5.1f}%)")
    print(f"{'='*60}\n")


@torch.no_grad()
def evaluate_with_labels(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    sim_groups: Optional[Dict[str, str]] = None,
    use_amp: bool = False
) -> Dict:
    """
    对带标签的数据集进行完整评估，包括 similarity 分组指标

    Returns:
        包含整体指标和 similarity 分组指标的字典
    """
    model.eval()

    # 整体累积器
    all_preds = []
    all_labels = []
    all_probs = []
    all_pdb_ids = []

    # Similarity 分组累积器
    sim_accumulators = {
        g: {
            'preds': [],
            'labels': [],
            'probs': [],
            'n_samples': 0,
            'n_tokens': 0,
            'n_correct': 0,
            'tp_c0': 0,
            'fp_c0': 0,
            'fn_c0': 0,
        }
        for g in SIM_GROUPS
    }

    print("\n开始评估...")
    for batch in tqdm(dataloader, desc="评估进度", unit="batch"):
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)
        batch_pdb_ids = batch.get('pdb_ids', batch.get('pdb_id', []))

        # 前向传播
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_scalar=inputs['protein_scalar'],
                protein_vectors=inputs['protein_vectors'],
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # 整体累积
        for b in range(len(batch_pdb_ids)):
            mask_b = mask[b]
            preds_b = preds[b][mask_b].cpu().numpy()
            labels_b = labels[b][mask_b].cpu().numpy()
            probs_b = probs[b][mask_b].cpu().numpy()

            all_preds.extend(preds_b)
            all_labels.extend(labels_b)
            all_probs.extend(probs_b)
            all_pdb_ids.append(batch_pdb_ids[b])

            # Similarity 分组统计
            if sim_groups:
                pdb_id = batch_pdb_ids[b]
                group = sim_groups.get(pdb_id)

                if group and group in sim_accumulators:
                    stat = sim_accumulators[group]
                    stat['n_samples'] += 1
                    stat['n_tokens'] += len(preds_b)
                    stat['n_correct'] += int((preds_b == labels_b).sum())

                    stat['preds'].extend(preds_b)
                    stat['labels'].extend(labels_b)
                    stat['probs'].extend(probs_b)

                    # Contact class (class 0) 的 TP/FP/FN
                    tp = int(((preds_b == 0) & (labels_b == 0)).sum())
                    fp = int(((preds_b == 0) & (labels_b != 0)).sum())
                    fn = int(((preds_b != 0) & (labels_b == 0)).sum())
                    stat['tp_c0'] += tp
                    stat['fp_c0'] += fp
                    stat['fn_c0'] += fn

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算整体指标
    metrics = compute_overall_metrics(all_preds, all_labels, all_probs)

    # 计算 similarity 分组指标（仅当有 sim_groups 且至少有一个分组有数据时）
    has_sim_data = False
    if sim_groups:
        for group in SIM_GROUPS:
            stat = sim_accumulators[group]
            if stat['n_tokens'] > 0:
                has_sim_data = True
                group_preds = np.array(stat['preds'])
                group_labels = np.array(stat['labels'])
                group_probs = np.array(stat['probs'])

                short_name = group.replace('_sim', '')

                # 准确率
                metrics[f'{short_name}_acc'] = stat['n_correct'] / stat['n_tokens']
                metrics[f'{short_name}_n_samples'] = stat['n_samples']
                metrics[f'{short_name}_n_tokens'] = stat['n_tokens']

                # Contact class (class 0) 的 Precision/Recall/F1
                precision = stat['tp_c0'] / (stat['tp_c0'] + stat['fp_c0'] + 1e-8)
                recall = stat['tp_c0'] / (stat['tp_c0'] + stat['fn_c0'] + 1e-8)
                f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

                metrics[f'{short_name}_precision_c0'] = precision
                metrics[f'{short_name}_recall_c0'] = recall
                metrics[f'{short_name}_f1_c0'] = f1

                # AP (Average Precision) for contact class
                # 使用 contact class 的概率 (probs[:, 0])
                contact_probs = group_probs[:, 0]
                contact_labels = (group_labels == 0).astype(int)
                ap = compute_ap_score(contact_probs, contact_labels)
                metrics[f'{short_name}_ap_c0'] = ap

    # 标记是否有 similarity 分组数据
    metrics['_has_sim_groups'] = has_sim_data

    return metrics


def compute_overall_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict:
    """计算整体指标"""
    metrics = {}

    # 准确率
    metrics['accuracy'] = float((preds == labels).mean())

    # Per-class 指标
    for c in range(3):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f'precision_c{c}'] = float(precision)
        metrics[f'recall_c{c}'] = float(recall)
        metrics[f'f1_c{c}'] = float(f1)

    # Contact class AP
    contact_probs = probs[:, 0]
    contact_labels = (labels == 0).astype(int)
    metrics['ap_c0'] = compute_ap_score(contact_probs, contact_labels)

    return metrics


def create_visualizations(output_dir: Path, sample_results: List[Dict]):
    """
    创建可视化图表

    Args:
        output_dir: 输出目录
        sample_results: 样本结果列表
    """
    if not sample_results:
        return

    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # 创建 DataFrame
    df = pd.DataFrame(sample_results)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Inference Results Analysis', fontsize=16, fontweight='bold')

    # 1. F1 Score 分布 (Contact class)
    axes[0, 0].hist(df['f1_c0'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['f1_c0'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["f1_c0"].mean():.3f}')
    axes[0, 0].axvline(df['f1_c0'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["f1_c0"].median():.3f}')
    axes[0, 0].set_xlabel('F1 Score (Contact)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('F1 Score Distribution (Contact Class)', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. AP Score 分布 (Contact class)
    axes[0, 1].hist(df['ap_c0'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['ap_c0'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["ap_c0"].mean():.3f}')
    axes[0, 1].axvline(df['ap_c0'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["ap_c0"].median():.3f}')
    axes[0, 1].set_xlabel('AP Score (Contact)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('AP Score Distribution (Contact Class)', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. IoU 分布 (Contact class)
    axes[0, 2].hist(df['iou_c0'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(df['iou_c0'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["iou_c0"].mean():.3f}')
    axes[0, 2].axvline(df['iou_c0'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["iou_c0"].median():.3f}')
    axes[0, 2].set_xlabel('IoU (Contact)', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('IoU Distribution (Contact Class)', fontsize=13, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Accuracy 分布
    axes[1, 0].hist(df['accuracy'], bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(df['accuracy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["accuracy"].mean():.3f}')
    axes[1, 0].axvline(df['accuracy'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["accuracy"].median():.3f}')
    axes[1, 0].set_xlabel('Accuracy', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Overall Accuracy Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. F1 vs AP 散点图 (Contact class)
    scatter = axes[1, 1].scatter(df['f1_c0'], df['ap_c0'], c=df['accuracy'], cmap='viridis', alpha=0.6, s=30)
    axes[1, 1].set_xlabel('F1 Score (Contact)', fontsize=12)
    axes[1, 1].set_ylabel('AP Score (Contact)', fontsize=12)
    axes[1, 1].set_title('F1 vs AP (colored by Accuracy)', fontsize=13, fontweight='bold')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Accuracy', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Per-class F1 对比箱线图
    f1_data = [df['f1_c0'], df['f1_c1'], df['f1_c2']]
    bp = axes[1, 2].boxplot(f1_data, labels=['Contact', 'Interface', 'Non-binding'],
                             patch_artist=True, showmeans=True)
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 2].set_ylabel('F1 Score', fontsize=12)
    axes[1, 2].set_title('F1 Score Comparison Across Classes', fontsize=13, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = output_dir / 'inference_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  可视化图表: inference_visualization.png")


def print_detailed_statistics(sample_results: List[Dict]):
    """
    打印详细的统计信息

    Args:
        sample_results: 样本结果列表
    """
    if not sample_results:
        return

    df = pd.DataFrame(sample_results)

    print(f"\n{'='*80}")
    print("详细统计信息")
    print(f"{'='*80}\n")

    # 整体统计
    print("【整体性能】")
    print(f"  样本数: {len(df)}")
    print(f"  平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  准确率范围: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
    print(f"  中位数准确率: {df['accuracy'].median():.4f}")
    print()

    # Contact class 详细统计
    print("【Contact Class (c0) 详细统计】")
    metrics_c0 = ['f1_c0', 'ap_c0', 'iou_c0', 'precision_c0', 'recall_c0']
    metric_names = ['F1', 'AP', 'IoU', 'Precision', 'Recall']

    for metric, name in zip(metrics_c0, metric_names):
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        median_val = df[metric].median()
        q25 = df[metric].quantile(0.25)
        q75 = df[metric].quantile(0.75)

        print(f"  {name:10s}: {mean_val:.4f} ± {std_val:.4f}  "
              f"[{min_val:.4f}, {max_val:.4f}]  "
              f"Median: {median_val:.4f}  "
              f"Q25-Q75: [{q25:.4f}, {q75:.4f}]")
    print()

    # Interface class 统计
    print("【Interface Class (c1) 统计】")
    metrics_c1 = ['f1_c1', 'iou_c1', 'precision_c1', 'recall_c1']
    metric_names = ['F1', 'IoU', 'Precision', 'Recall']

    for metric, name in zip(metrics_c1, metric_names):
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"  {name:10s}: {mean_val:.4f} ± {std_val:.4f}")
    print()

    # Non-binding class 统计
    print("【Non-binding Class (c2) 统计】")
    metrics_c2 = ['f1_c2', 'iou_c2', 'precision_c2', 'recall_c2']
    metric_names = ['F1', 'IoU', 'Precision', 'Recall']

    for metric, name in zip(metrics_c2, metric_names):
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"  {name:10s}: {mean_val:.4f} ± {std_val:.4f}")
    print()

    # 性能分布统计
    print("【性能分布】")
    avg_score = (df['f1_c0'] + df['ap_c0'] + df['iou_c0']) / 3.0
    excellent = (avg_score >= 0.4).sum()
    medium = ((avg_score >= 0.2) & (avg_score < 0.4)).sum()
    hard = (avg_score < 0.2).sum()

    print(f"  优秀 (≥0.4):  {excellent:4d} ({excellent/len(df)*100:5.1f}%)")
    print(f"  一般 (0.2-0.4): {medium:4d} ({medium/len(df)*100:5.1f}%)")
    print(f"  难例 (<0.2):   {hard:4d} ({hard/len(df)*100:5.1f}%)")
    print()

    # Top 10 和 Bottom 10
    print("【Top 10 样本 (按 Contact F1 排序)】")
    top10 = df.nlargest(10, 'f1_c0')[['pdb_id', 'f1_c0', 'ap_c0', 'iou_c0', 'accuracy']]
    for idx, row in top10.iterrows():
        print(f"  {row['pdb_id']:6s}: F1={row['f1_c0']:.4f}, AP={row['ap_c0']:.4f}, "
              f"IoU={row['iou_c0']:.4f}, Acc={row['accuracy']:.4f}")
    print()

    print("【Bottom 10 样本 (按 Contact F1 排序)】")
    bottom10 = df.nsmallest(10, 'f1_c0')[['pdb_id', 'f1_c0', 'ap_c0', 'iou_c0', 'accuracy']]
    for idx, row in bottom10.iterrows():
        print(f"  {row['pdb_id']:6s}: F1={row['f1_c0']:.4f}, AP={row['ap_c0']:.4f}, "
              f"IoU={row['iou_c0']:.4f}, Acc={row['accuracy']:.4f}")
    print()

    print(f"{'='*80}\n")


def print_evaluation_summary(metrics: Dict, has_sim_groups: bool = False):
    """
    打印评估摘要

    Args:
        metrics: 评估指标字典
        has_sim_groups: 是否有 similarity 分组数据
    """
    print(f"\n{'='*80}")
    print("评估结果")
    print(f"{'='*80}")

    # 整体指标
    print("\n整体指标:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  Contact (c0) - Precision: {metrics['precision_c0']:.4f}, Recall: {metrics['recall_c0']:.4f}, F1: {metrics['f1_c0']:.4f}, AP: {metrics['ap_c0']:.4f}")
    print(f"  Interface (c1) - Precision: {metrics['precision_c1']:.4f}, Recall: {metrics['recall_c1']:.4f}, F1: {metrics['f1_c1']:.4f}")
    print(f"  Non-binding (c2) - Precision: {metrics['precision_c2']:.4f}, Recall: {metrics['recall_c2']:.4f}, F1: {metrics['f1_c2']:.4f}")

    # Similarity 分组指标（仅当有数据时显示）
    if has_sim_groups:
        print(f"\n{'='*80}")
        print("Similarity 分组指标:")
        print(f"{'='*80}")

        for group in SIM_GROUPS:
            short_name = group.replace('_sim', '')
            # 只显示有数据的分组
            if f'{short_name}_n_samples' in metrics and metrics[f'{short_name}_n_samples'] > 0:
                print(f"\n{group.upper()}:")
                print(f"  样本数: {metrics[f'{short_name}_n_samples']:.0f}, 残基数: {metrics[f'{short_name}_n_tokens']:.0f}")
                print(f"  准确率: {metrics[f'{short_name}_acc']:.4f}")
                print(f"  Contact (c0) - Precision: {metrics[f'{short_name}_precision_c0']:.4f}, "
                      f"Recall: {metrics[f'{short_name}_recall_c0']:.4f}, "
                      f"F1: {metrics[f'{short_name}_f1_c0']:.4f}, "
                      f"AP: {metrics[f'{short_name}_ap_c0']:.4f}")

    print(f"\n{'='*80}\n")


@torch.no_grad()
def evaluate_and_categorize(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = False,
    no_viz: bool = False
) -> Dict:
    """
    全局推理模式：评估所有样本并按性能分类保存

    按 Contact class (c0) 的 F1/AP/IoU 分为三类：
    - 难例: F1/AP/IoU < 0.2
    - 一般: 0.2 <= F1/AP/IoU < 0.4
    - 优秀: F1/AP/IoU >= 0.4

    保存 CSV 文件包含: pdb_id, f1_c0, ap_c0, iou_c0, 以及所有类别的指标
    """
    model.eval()

    # 存储每个样本的详细指标
    sample_results = []

    print("\n开始全局推理...")
    for batch in tqdm(dataloader, desc="推理进度", unit="batch"):
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['protein_mask'].to(device)
        batch_pdb_ids = batch.get('pdb_ids', batch.get('pdb_id', []))

        # 前向传播
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                esm_embeddings=inputs['esm_embeddings'],
                protein_scalar=inputs['protein_scalar'],
                protein_vectors=inputs['protein_vectors'],
                protein_coords=inputs['protein_coords'],
                protein_mask=inputs['protein_mask'],
                ligand_x=inputs['ligand_x'],
                ligand_pos=inputs['ligand_pos'],
                ligand_edge_index=inputs['ligand_edge_index'],
                ligand_edge_attr=inputs['ligand_edge_attr'],
                ligand_batch=inputs['ligand_batch'],
                ligand_to_protein=inputs['ligand_to_protein'],
                ligand_mask=inputs['ligand_mask']
            )

        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # 逐样本计算指标
        for b in range(len(batch_pdb_ids)):
            mask_b = mask[b]
            preds_b = preds[b][mask_b].cpu().numpy()
            labels_b = labels[b][mask_b].cpu().numpy()
            probs_b = probs[b][mask_b].cpu().numpy()
            pdb_id = batch_pdb_ids[b]

            # 计算每个类别的指标
            sample_metrics = {'pdb_id': pdb_id}

            for c in range(3):
                tp = ((preds_b == c) & (labels_b == c)).sum()
                fp = ((preds_b == c) & (labels_b != c)).sum()
                fn = ((preds_b != c) & (labels_b == c)).sum()
                tn = ((preds_b != c) & (labels_b != c)).sum()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

                # IoU (Intersection over Union)
                iou = tp / (tp + fp + fn + 1e-8)

                sample_metrics[f'precision_c{c}'] = float(precision)
                sample_metrics[f'recall_c{c}'] = float(recall)
                sample_metrics[f'f1_c{c}'] = float(f1)
                sample_metrics[f'iou_c{c}'] = float(iou)

                # AP for contact class
                if c == 0:
                    contact_probs = probs_b[:, 0]
                    contact_labels = (labels_b == 0).astype(int)
                    ap = compute_ap_score(contact_probs, contact_labels)
                    sample_metrics[f'ap_c{c}'] = float(ap)

            # 准确率
            sample_metrics['accuracy'] = float((preds_b == labels_b).mean())
            sample_metrics['n_residues'] = len(preds_b)

            sample_results.append(sample_metrics)

    # 按 Contact class (c0) 的性能分类
    hard_cases = []
    medium_cases = []
    good_cases = []

    for result in sample_results:
        # 使用 F1, AP, IoU 的平均值作为判断标准
        f1_c0 = result['f1_c0']
        ap_c0 = result['ap_c0']
        iou_c0 = result['iou_c0']
        avg_score = (f1_c0 + ap_c0 + iou_c0) / 3.0

        if avg_score < 0.2:
            hard_cases.append(result)
        elif avg_score < 0.4:
            medium_cases.append(result)
        else:
            good_cases.append(result)

    # 保存 CSV 文件
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(cases: List[Dict], filename: str):
        if not cases:
            print(f"  {filename}: 0 个样本")
            return

        csv_path = output_dir / filename
        fieldnames = ['pdb_id', 'accuracy', 'n_residues',
                      'f1_c0', 'ap_c0', 'iou_c0', 'precision_c0', 'recall_c0',
                      'f1_c1', 'iou_c1', 'precision_c1', 'recall_c1',
                      'f1_c2', 'iou_c2', 'precision_c2', 'recall_c2']

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in cases:
                writer.writerow(case)

        print(f"  {filename}: {len(cases)} 个样本")

    print(f"\n保存结果到: {output_dir}")
    save_csv(hard_cases, 'hard_cases.csv')
    save_csv(medium_cases, 'medium_cases.csv')
    save_csv(good_cases, 'good_cases.csv')

    # 保存所有结果
    save_csv(sample_results, 'all_results.csv')

    # 打印详细统计信息
    print_detailed_statistics(sample_results)

    # 创建可视化（如果未禁用）
    if not no_viz:
        print(f"\n生成可视化图表...")
        try:
            create_visualizations(output_dir, sample_results)
        except Exception as e:
            print(f"  警告: 可视化生成失败: {e}")
            print(f"  提示: 可以使用 --no_viz 跳过可视化")

    # 统计信息
    summary = {
        'total_samples': len(sample_results),
        'hard_cases': len(hard_cases),
        'medium_cases': len(medium_cases),
        'good_cases': len(good_cases),
        'hard_ratio': len(hard_cases) / len(sample_results) if sample_results else 0,
        'medium_ratio': len(medium_cases) / len(sample_results) if sample_results else 0,
        'good_ratio': len(good_cases) / len(sample_results) if sample_results else 0,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='V7 模型推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--pdb_id', type=str, default=None,
                        help='单个 PDB ID')
    parser.add_argument('--pdb_list', type=str, default=None,
                        help='PDB ID 列表文件 (每行一个 ID)')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Split 文件路径 (包含 similarity 分组信息)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据根目录（用于加载特征）')
    parser.add_argument('--pdb_root', type=str, default='/home/husrcf/Code/ProtBind/New',
                        help='PDB 数据根目录（用于生成 PDB 格式输出）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (保存预测结果)')
    parser.add_argument('--amp', action='store_true',
                        help='使用自动混合精度')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--evaluate', action='store_true',
                        help='评估模式 (需要标签数据，计算指标)')
    parser.add_argument('--global_infer', action='store_true',
                        help='全局推理模式 (自动加载 test set，按性能分类保存)')
    parser.add_argument('--no_viz', action='store_true',
                        help='禁用可视化（加快推理速度）')

    args = parser.parse_args()

    # 全局推理模式：自动加载 test set
    if args.global_infer:
        if not args.split_file:
            parser.error("全局推理模式需要指定 --split_file 来加载 test set")
        if args.pdb_id or args.pdb_list:
            print("警告: 全局推理模式将忽略 --pdb_id 和 --pdb_list，自动加载 test set")
        # 强制设置输出目录
        if not args.output:
            args.output = 'outputs'
    else:
        # 非全局推理模式：检查输入
        if args.pdb_id is None and args.pdb_list is None:
            parser.error("必须指定 --pdb_id 或 --pdb_list（或使用 --global_infer 模式）")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载检查点
    checkpoint, config = load_checkpoint(args.checkpoint, device)

    # 创建模型
    model = create_model_from_checkpoint(checkpoint, config, device)

    # 获取 PDB ID 列表
    if args.global_infer:
        # 全局推理模式：从 split 文件加载 test set
        pdb_ids = load_test_ids_from_split(args.split_file)
        if not pdb_ids:
            print("错误: 未能从 split 文件加载 test set")
            return
        print(f"✓ 从 split 文件加载了 {len(pdb_ids)} 个 test 样本")
    elif args.pdb_id:
        pdb_ids = [args.pdb_id]
    else:
        with open(args.pdb_list, 'r') as f:
            pdb_ids = [line.strip() for line in f if line.strip()]

    print(f"\n待推理样本数: {len(pdb_ids)}")

    # 加载 similarity 分组 (如果提供)
    sim_groups = None
    if args.split_file:
        sim_groups = load_similarity_groups(args.split_file)
        if sim_groups:
            print(f"✓ 加载了 {len(sim_groups)} 个样本的 similarity 分组信息")
        else:
            print(f"⚠ Split 文件中未找到 similarity 分组信息，将只计算全局指标")
    elif args.evaluate and not args.global_infer:
        print(f"ℹ 未提供 --split_file，将只计算全局指标（不包含 similarity 分组）")

    # 数据配置
    data_cfg = config['data']
    ligand_features_dir = data_cfg.get('ligand_features_dir', 'ligand_features_v3')
    protein_features_dir = data_cfg.get('protein_features_dir', 'complete_features')
    distances_file = data_cfg.get('distances_file')

    # 创建数据集
    dataset = PocketDatasetV6(
        data_root=args.data_root,
        pdb_ids=pdb_ids,
        max_seq_len=data_cfg['max_seq_len'],
        max_atoms=data_cfg['max_atoms'],
        distances_file=distances_file,
        ligand_features_dir=ligand_features_dir,
        protein_features_dir=protein_features_dir,
        split="test",
        verbose=True
    )

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pocket_collate_fn
    )

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = None

    # 全局推理模式：创建带时间戳的子目录
    if args.global_infer:
        if output_dir is None:
            output_dir = Path('outputs')

        # 创建带时间戳的子目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir / f'infer_v7_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n输出目录: {output_dir}")
        print("\n=== 全局推理模式 ===")
        summary = evaluate_and_categorize(
            model, dataloader, device,
            output_dir=output_dir,
            use_amp=args.amp,
            no_viz=args.no_viz
        )

        # 打印统计信息
        print(f"\n{'='*80}")
        print("分类统计:")
        print(f"{'='*80}")
        print(f"总样本数: {summary['total_samples']}")
        print(f"难例 (F1/AP/IoU < 0.2):     {summary['hard_cases']:4d} ({summary['hard_ratio']*100:5.1f}%)")
        print(f"一般 (0.2 <= F1/AP/IoU < 0.4): {summary['medium_cases']:4d} ({summary['medium_ratio']*100:5.1f}%)")
        print(f"优秀 (F1/AP/IoU >= 0.4):    {summary['good_cases']:4d} ({summary['good_ratio']*100:5.1f}%)")
        print(f"{'='*80}\n")

    # 评估模式
    elif args.evaluate:
        print("\n=== 评估模式 ===")
        metrics = evaluate_with_labels(
            model, dataloader, device,
            sim_groups=sim_groups,
            use_amp=args.amp
        )

        # 从 metrics 中提取是否有 similarity 分组数据
        has_sim_groups = metrics.pop('_has_sim_groups', False)
        print_evaluation_summary(metrics, has_sim_groups)

        # 保存评估结果
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = output_dir / 'evaluation_metrics.json'
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✓ 评估结果已保存到: {metrics_file}")

    # 推理模式
    else:
        print(f"\n开始推理...")
        for batch in tqdm(dataloader, desc="推理进度", unit="batch"):
            result = predict_single(model, batch, device, use_amp=args.amp)

            # 打印摘要
            print_prediction_summary(result)

            # 保存结果
            if output_dir:
                # 如果结果中有真实标签，传递给 save_predictions
                true_labels = result.get('true_labels', None)
                save_predictions(result, output_dir, pdb_root=args.pdb_root, true_labels=true_labels)

    print("完成!")


if __name__ == '__main__':
    main()
