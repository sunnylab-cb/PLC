#!/usr/bin/env python3
"""
测试数据加载器的对齐逻辑

检查项：
1. 特征维度是否正确
2. 序列长度是否对齐
3. mask是否正确应用
4. 标签是否正确加载
5. 配体特征是否正确
6. batch collation是否正确
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.pocket_dataset_v3 import PocketDatasetV3, create_data_splits_v3
from src.data.collate import pocket_collate_fn, prepare_model_inputs

def test_single_sample():
    """测试单个样本的加载"""
    print("=" * 60)
    print("测试 1: 单个样本加载")
    print("=" * 60)

    data_root = "data"

    # 创建数据集
    dataset = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=1000,
        max_atoms=100,
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    print(f"✓ 数据集大小: {len(dataset)} 个样本")

    # 加载第一个样本
    sample = dataset[0]

    print(f"\n样本键: {sample.keys()}")
    print(f"\n蛋白质特征:")
    print(f"  - ESM embeddings: {sample['esm'].shape}")
    print(f"  - Protein scalar: {sample['protein_scalar'].shape}")
    print(f"  - GVP vectors: {sample['gvp_vectors'].shape}")
    print(f"  - GVP coords: {sample['gvp_coords'].shape}")

    print(f"\n配体特征:")
    print(f"  - Ligand features: {sample['ligand_features'].shape}")
    print(f"  - Ligand coords: {sample['ligand_coords'].shape}")

    print(f"\n标签:")
    print(f"  - Labels: {sample['labels'].shape}")
    print(f"  - Label distribution: {torch.bincount(sample['labels'])}")

    # 检查对齐
    seq_len = sample['n_residues']
    print(f"\n对齐检查:")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - ESM embeddings 长度: {sample['esm'].shape[0]}")
    print(f"  - Protein scalar 长度: {sample['protein_scalar'].shape[0]}")
    print(f"  - Labels 长度: {sample['labels'].shape[0]}")

    # 检查维度
    assert sample['esm'].shape[1] == 1280, f"ESM维度错误: {sample['esm'].shape[1]}, 期望1280"
    assert sample['protein_scalar'].shape[1] == 82, f"Protein scalar维度错误: {sample['protein_scalar'].shape[1]}, 期望82"
    assert sample['gvp_vectors'].shape[1] == 4, f"GVP vectors维度错误: {sample['gvp_vectors'].shape[1]}, 期望4"
    assert sample['gvp_vectors'].shape[2] == 3, f"GVP vectors最后维度错误: {sample['gvp_vectors'].shape[2]}, 期望3"
    assert sample['ligand_features'].shape[1] == 35, f"Ligand特征维度错误: {sample['ligand_features'].shape[1]}, 期望35"

    print(f"\n✓ 所有维度检查通过!")

    return sample


def test_batch_collation():
    """测试batch collation"""
    print("\n" + "=" * 60)
    print("测试 2: Batch Collation")
    print("=" * 60)

    data_root = "data"

    # 创建数据集
    dataset = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=1000,
        max_atoms=100,
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    # 创建dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=pocket_collate_fn,
        num_workers=0
    )

    # 获取一个batch
    batch = next(iter(dataloader))

    print(f"\nBatch键: {batch.keys()}")
    print(f"\nBatch形状:")
    print(f"  - ESM embeddings: {batch['esm_embeddings'].shape}")
    print(f"  - Protein scalar: {batch['protein_scalar'].shape}")
    print(f"  - Protein vectors: {batch['protein_vectors'].shape}")
    print(f"  - Protein coords: {batch['protein_coords'].shape}")
    print(f"  - Protein mask: {batch['protein_mask'].shape}")
    print(f"  - Labels: {batch['labels'].shape}")

    print(f"\n配体特征 (padding格式):")
    print(f"  - Ligand features: {batch['ligand_features'].shape}")
    print(f"  - Ligand coords: {batch['ligand_coords'].shape}")
    print(f"  - Ligand mask: {batch['ligand_mask'].shape}")

    # 检查batch维度
    B = batch['protein_mask'].shape[0]
    L = batch['protein_mask'].shape[1]
    M = batch['ligand_mask'].shape[1]

    print(f"\nBatch统计:")
    print(f"  - Batch size (B): {B}")
    print(f"  - Max sequence length (L): {L}")
    print(f"  - Max ligand atoms (M): {M}")

    # 检查每个样本的有效长度
    print(f"\n每个样本的有效长度:")
    for i in range(B):
        protein_len = batch['protein_mask'][i].sum().item()
        ligand_len = batch['ligand_mask'][i].sum().item()
        print(f"  样本 {i}: 蛋白质={protein_len}, 配体={ligand_len}")

    # 检查标签分布
    print(f"\nBatch标签分布:")
    valid_labels = batch['labels'][batch['protein_mask']]
    label_counts = torch.bincount(valid_labels, minlength=3)
    print(f"  - Class 0 (Contact): {label_counts[0]} ({label_counts[0]/valid_labels.numel()*100:.2f}%)")
    print(f"  - Class 1 (Near): {label_counts[1]} ({label_counts[1]/valid_labels.numel()*100:.2f}%)")
    print(f"  - Class 2 (Far): {label_counts[2]} ({label_counts[2]/valid_labels.numel()*100:.2f}%)")

    # 测试图构建
    print(f"\n测试配体图构建...")
    from src.data.collate import build_ligand_graphs
    batch_with_graphs = build_ligand_graphs(batch)

    print(f"  - Ligand x: {batch_with_graphs['ligand_x'].shape}")
    print(f"  - Ligand edge_index: {batch_with_graphs['ligand_edge_index'].shape}")
    print(f"  - Ligand edge_attr: {batch_with_graphs['ligand_edge_attr'].shape}")
    print(f"  - Ligand batch: {batch_with_graphs['ligand_batch'].shape}")

    print(f"\n✓ Batch collation检查通过!")

    return batch


def test_model_input_preparation():
    """测试模型输入准备"""
    print("\n" + "=" * 60)
    print("测试 3: 模型输入准备")
    print("=" * 60)

    data_root = "data"

    # 创建数据集和dataloader
    dataset = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=1000,
        max_atoms=100,
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=pocket_collate_fn,
        num_workers=0
    )

    batch = next(iter(dataloader))

    # 准备模型输入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = prepare_model_inputs(batch, device)

    print(f"\n模型输入键: {inputs.keys()}")
    print(f"\n模型输入形状:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}, device={value.device}, dtype={value.dtype}")

    # 检查所有张量都在正确的设备上
    print(f"\n设备检查:")
    all_on_device = all(
        v.device.type == device.type
        for v in inputs.values()
        if isinstance(v, torch.Tensor)
    )
    print(f"  - 所有张量在 {device}: {'✓' if all_on_device else '✗'}")

    # 检查数据类型
    print(f"\nDtype检查:")
    print(f"  - ESM embeddings: {inputs['esm_embeddings'].dtype}")
    print(f"  - Protein scalar: {inputs['protein_scalar'].dtype}")
    print(f"  - Protein vectors: {inputs['protein_vectors'].dtype}")
    print(f"  - Ligand x: {inputs['ligand_x'].dtype}")
    print(f"  - Ligand edge_index: {inputs['ligand_edge_index'].dtype}")

    print(f"\n✓ 模型输入准备检查通过!")

    return inputs


def test_alignment_consistency():
    """测试对齐一致性"""
    print("\n" + "=" * 60)
    print("测试 4: 对齐一致性检查")
    print("=" * 60)

    data_root = "data"

    dataset = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=1000,
        max_atoms=100,
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    # 测试多个样本
    n_samples = min(10, len(dataset))
    print(f"\n测试 {n_samples} 个样本的对齐一致性...")

    # 检查对齐
    print(f"\n对齐检查:")
    for i in range(n_samples):
        sample = dataset[i]

        # 检查所有蛋白质特征的序列长度是否一致
        esm_len = sample['esm'].shape[0]
        scalar_len = sample['protein_scalar'].shape[0]
        vectors_len = sample['gvp_vectors'].shape[0]
        coords_len = sample['gvp_coords'].shape[0]
        labels_len = sample['labels'].shape[0]

        assert esm_len == scalar_len == vectors_len == coords_len == labels_len, \
            f"样本 {i}: 序列长度不一致! ESM={esm_len}, scalar={scalar_len}, vectors={vectors_len}, coords={coords_len}, labels={labels_len}"

        # 检查标签值范围
        assert sample['labels'].min() >= 0 and sample['labels'].max() <= 2, \
            f"样本 {i}: 标签值超出范围 [0, 2]"

        # 检查配体特征
        ligand_len = sample['ligand_features'].shape[0]
        ligand_coords_len = sample['ligand_coords'].shape[0]

        assert ligand_len == ligand_coords_len, \
            f"样本 {i}: 配体长度不一致! features={ligand_len}, coords={ligand_coords_len}"

        if i == 0:
            print(f"\n样本 {i} 详细信息:")
            print(f"  - 序列长度: {esm_len}")
            print(f"  - 配体原子数: {ligand_len}")

    print(f"\n✓ 所有 {n_samples} 个样本的对齐一致性检查通过!")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试 5: 边界情况")
    print("=" * 60)

    data_root = "data"

    dataset = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=1000,
        max_atoms=100,
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    # 查找最短和最长的序列
    print("\n扫描数据集以查找边界情况...")

    min_protein_len = float('inf')
    max_protein_len = 0
    min_ligand_len = float('inf')
    max_ligand_len = 0

    n_samples = min(100, len(dataset))

    for i in range(n_samples):
        sample = dataset[i]
        protein_len = sample['n_residues']
        ligand_len = sample['n_atoms']

        min_protein_len = min(min_protein_len, protein_len)
        max_protein_len = max(max_protein_len, protein_len)
        min_ligand_len = min(min_ligand_len, ligand_len)
        max_ligand_len = max(max_ligand_len, ligand_len)

    print(f"\n序列长度统计 (前{n_samples}个样本):")
    print(f"  - 蛋白质: min={min_protein_len}, max={max_protein_len}")
    print(f"  - 配体: min={min_ligand_len}, max={max_ligand_len}")

    # 测试截断情况
    print(f"\n测试截断逻辑...")
    dataset_short = PocketDatasetV3(
        data_root=data_root,
        max_seq_len=100,  # 短序列长度
        max_atoms=20,     # 少量原子
        distances_file="data/residue_ligand_distances_v2.pt"
    )

    sample = dataset_short[0]
    print(f"  - 截断后蛋白质长度: {sample['n_residues']} (max=100)")
    print(f"  - 截断后配体长度: {sample['n_atoms']} (max=20)")

    assert sample['n_residues'] <= 100, "蛋白质序列未正确截断"
    assert sample['n_atoms'] <= 20, "配体未正确截断"

    print(f"\n✓ 边界情况检查通过!")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("数据加载器对齐测试")
    print("=" * 60)

    try:
        # 测试1: 单个样本
        sample = test_single_sample()

        # 测试2: Batch collation
        batch = test_batch_collation()

        # 测试3: 模型输入准备
        inputs = test_model_input_preparation()

        # 测试4: 对齐一致性
        test_alignment_consistency()

        # 测试5: 边界情况
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
