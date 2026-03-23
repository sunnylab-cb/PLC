#!/usr/bin/env python3
"""
测试 V8 模型和数据集设置是否正确

验证：
1. V8 模型能否正确初始化
2. V7 数据集能否正确加载 V3 特征
3. V8 collate 函数能否正确处理批次
4. 模型前向传播能否正常运行
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pocket_gvp_fusion_v8 import PocketGVPFusionV8
from src.data.pocket_dataset_v7 import PocketDatasetV7
from src.data.collate import pocket_collate_fn_v8, prepare_model_inputs_v8


def test_model_initialization():
    """测试 V8 模型初始化"""
    print("=" * 60)
    print("测试 1: V8 模型初始化")
    print("=" * 60)

    config = {
        'd_esm': 1280,
        'd_phys': 62,
        'd_spatial_scalar': 72,
        'd_phys_encoded': 128,
        'd_gvp_internal': 256,
        'd_model': 512,
        'd_ligand_atom': 96,
        'n_gvp_vectors': 10,
        'n_gvp_vector_hidden': 20,
        'n_gvp_layers': 3,
        'n_transformer_layers': 4,
        'n_ligand_layers': 3,
        'n_heads': 8,
        'n_cross_attention_layers': 1,
        'dropout': 0.1,
        'use_gated_attn': True,
        'use_swiglu': True,
        'gvp_radius': 10.0,
        'gvp_k_neighbors': 20,
        'esm_dropout_prob': 0.3,
        'physchem_dropout_prob': 0.3,
        'use_gradient_checkpointing': False,
        'n_classes': 3,
        'edge_dim': 6,
    }

    try:
        model = PocketGVPFusionV8(config)
        print("✅ 模型初始化成功")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")

        return model
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise


def test_dataset_loading():
    """测试 V7 数据集加载"""
    print("\n" + "=" * 60)
    print("测试 2: V7 数据集加载 (V3 特征)")
    print("=" * 60)

    # 检查数据目录是否存在
    data_root = Path("data")
    v3_features_dir = data_root / "complete_features_v3"

    if not v3_features_dir.exists():
        print(f"⚠️  V3 特征目录不存在: {v3_features_dir}")
        print("   请先运行特征提取脚本:")
        print("   python preparation/protein/extract_complete_features_v3.py \\")
        print("     --input data/processed/ \\")
        print("     --output data/complete_features_v3/ \\")
        print("     --workers 32")
        return None

    try:
        # 尝试加载数据集（只加载前 5 个样本用于测试）
        dataset = PocketDatasetV7(
            data_root=str(data_root),
            pdb_ids=None,  # 加载所有可用样本
            max_seq_len=1500,
            max_atoms=100,
            distances_file=str(data_root / "residue_ligand_distances_vdw.pt"),
            ligand_features_dir="ligand_features_v4",
            protein_features_dir="complete_features_v3",
            split="test",
            verbose=True
        )

        print(f"✅ 数据集加载成功")
        print(f"   样本数量: {len(dataset)}")

        if len(dataset) > 0:
            # 测试加载第一个样本
            sample = dataset[0]
            print(f"\n   第一个样本:")
            print(f"   - PDB ID: {sample['pdb_id']}")
            print(f"   - 残基数: {sample['n_residues']}")
            print(f"   - 原子数: {sample['n_atoms']}")
            print(f"   - ESM 特征: {sample['esm'].shape}")
            print(f"   - 物理化学特征: {sample['protein_physchem'].shape}")
            print(f"   - 空间标量特征: {sample['protein_spatial_scalar'].shape}")
            print(f"   - 空间矢量特征: {sample['protein_spatial_vector'].shape}")
            print(f"   - GVP 矢量: {sample['gvp_vectors'].shape}")
            print(f"   - 配体特征: {sample['ligand_features'].shape}")

            return dataset
        else:
            print("⚠️  数据集为空")
            return None

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_collate_and_forward(model, dataset):
    """测试 collate 函数和模型前向传播"""
    if dataset is None or len(dataset) == 0:
        print("\n⚠️  跳过测试 3 和 4 (数据集不可用)")
        return

    print("\n" + "=" * 60)
    print("测试 3: V8 Collate 函数")
    print("=" * 60)

    try:
        # 创建一个小批次
        batch_size = min(2, len(dataset))
        batch = [dataset[i] for i in range(batch_size)]

        # 使用 V8 collate 函数
        collated = pocket_collate_fn_v8(batch)

        print(f"✅ Collate 函数成功")
        print(f"   批次大小: {batch_size}")
        print(f"   ESM 特征: {collated['esm_embeddings'].shape}")
        print(f"   物理化学特征: {collated['protein_physchem'].shape}")
        print(f"   空间标量特征: {collated['protein_spatial_scalar'].shape}")
        print(f"   空间矢量特征: {collated['protein_spatial_vector'].shape}")
        print(f"   GVP 矢量: {collated['protein_vectors'].shape}")
        print(f"   配体特征: {collated['ligand_features'].shape}")

    except Exception as e:
        print(f"❌ Collate 函数失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("测试 4: 模型前向传播")
    print("=" * 60)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   使用设备: {device}")

        model = model.to(device)
        model.eval()

        # 准备模型输入
        inputs = prepare_model_inputs_v8(collated, device)

        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ 前向传播成功")
        print(f"   输出 logits: {outputs['logits'].shape}")
        print(f"   蛋白质特征: {outputs['h_prot_global'].shape}")
        print(f"   融合特征: {outputs['h_fused'].shape}")
        print(f"   配体特征: {outputs['h_ligand_fused'].shape}")

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 60)
    print("V8 模型和数据集设置测试")
    print("=" * 60)

    # 测试 1: 模型初始化
    model = test_model_initialization()

    # 测试 2: 数据集加载
    dataset = test_dataset_loading()

    # 测试 3 & 4: Collate 和前向传播
    test_collate_and_forward(model, dataset)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
