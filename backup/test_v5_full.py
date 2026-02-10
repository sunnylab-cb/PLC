#!/usr/bin/env python3
"""
完整测试 V5 训练脚本的所有功能

测试内容:
1. 数据加载 (VDW距离标签 + 原始标签)
2. 模型创建
3. 训练一个batch
4. Checkpoint保存和加载
"""

import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset_v5 import PocketDatasetV5
from src.data.collate import pocket_collate_fn, prepare_model_inputs
from src.models.pocket_gvp_fusion import PocketGVPFusion
from torch.utils.data import DataLoader, Subset


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*60)
    print("测试 1: 数据集加载")
    print("="*60)

    try:
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        print(f"✓ 数据集大小: {len(dataset)} 个复合物")

        # 测试加载一个样本
        sample = dataset[0]
        print(f"✓ 样本加载成功:")
        print(f"  - PDB ID: {sample['pdb_id']}")
        print(f"  - ESM shape: {sample['esm'].shape}")
        print(f"  - Protein scalar shape: {sample['protein_scalar'].shape}")
        print(f"  - Ligand features shape: {sample['ligand_features'].shape}")
        print(f"  - Labels (VDW) shape: {sample['labels'].shape}")

        if 'labels_original' in sample:
            print(f"  - Labels (original) shape: {sample['labels_original'].shape}")

            # 比较两种标签的差异
            vdw_binding = (sample['labels'] == 0).sum().item()
            orig_binding = (sample['labels_original'] == 0).sum().item()
            print(f"  - VDW结合位点数: {vdw_binding}")
            print(f"  - 原始结合位点数: {orig_binding}")
            print(f"  - 差异: {vdw_binding - orig_binding}")
        else:
            print(f"  ⚠️  未加载原始标签")

        return True
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试DataLoader"""
    print("\n" + "="*60)
    print("测试 2: DataLoader批处理")
    print("="*60)

    try:
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        # 只取前10个样本测试
        test_dataset = Subset(dataset, range(min(10, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"✓ Batch加载成功:")
        print(f"  - Batch size: {batch['esm_embeddings'].shape[0]}")
        print(f"  - ESM shape: {batch['esm_embeddings'].shape}")
        print(f"  - Protein scalar shape: {batch['protein_scalar'].shape}")
        print(f"  - Ligand features shape: {batch['ligand_features'].shape}")
        print(f"  - Labels shape: {batch['labels'].shape}")
        print(f"  - Mask shape: {batch['mask'].shape}")

        if 'labels_original' in batch:
            print(f"  - Labels original shape: {batch['labels_original'].shape}")

        return True
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("测试 3: 模型创建")
    print("="*60)

    try:
        config = load_config('configs/pocket_gvp_v5.yaml')
        model = PocketGVPFusion(config['model'])

        print(f"✓ 模型创建成功")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")

        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试 4: 前向传播")
    print("="*60)

    try:
        config = load_config('configs/pocket_gvp_v5.yaml')
        model = PocketGVPFusion(config['model'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # 加载数据
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        test_dataset = Subset(dataset, range(min(2, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        inputs = prepare_model_inputs(batch, device)

        # 前向传播
        with torch.no_grad():
            logits = model(**inputs)

        print(f"✓ 前向传播成功")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "="*60)
    print("测试 5: 训练步骤 (backward + optimizer)")
    print("="*60)

    try:
        config = load_config('configs/pocket_gvp_v5.yaml')
        model = PocketGVPFusion(config['model'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()

        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 加载数据
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        test_dataset = Subset(dataset, range(min(2, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)

        # 前向传播
        logits = model(**inputs)

        # 计算loss
        import torch.nn.functional as F
        loss = F.cross_entropy(
            logits.view(-1, 3),
            labels.view(-1),
            reduction='none'
        )
        loss = (loss * mask.view(-1)).sum() / mask.sum()

        print(f"✓ Loss计算成功: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        print(f"✓ 反向传播成功")
        print(f"  - 梯度范数: {total_norm:.4f}")

        # 优化器步骤
        optimizer.step()
        print(f"✓ 优化器步骤成功")

        return True
    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试checkpoint保存和加载"""
    print("\n" + "="*60)
    print("测试 6: Checkpoint保存和加载")
    print("="*60)

    try:
        config = load_config('configs/pocket_gvp_v5.yaml')
        model = PocketGVPFusion(config['model'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # 保存checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': 0.5,
            'patience_counter': 2,
            'train_metrics': {'loss': 0.6, 'accuracy': 0.8},
            'val_metrics': {'loss': 0.5, 'accuracy': 0.85}
        }

        test_ckpt_path = Path('test_checkpoint.pt')
        torch.save(checkpoint, test_ckpt_path)
        print(f"✓ Checkpoint保存成功: {test_ckpt_path}")

        # 加载checkpoint
        loaded_ckpt = torch.load(test_ckpt_path, map_location='cpu')

        # 创建新模型并加载
        model2 = PocketGVPFusion(config['model'])
        model2.load_state_dict(loaded_ckpt['model_state_dict'])
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        optimizer2.load_state_dict(loaded_ckpt['optimizer_state_dict'])

        print(f"✓ Checkpoint加载成功")
        print(f"  - Epoch: {loaded_ckpt['epoch']}")
        print(f"  - Best val loss: {loaded_ckpt['best_val_loss']}")
        print(f"  - Patience counter: {loaded_ckpt['patience_counter']}")
        print(f"  - Learning rate: {optimizer2.param_groups[0]['lr']:.6f}")

        # 清理
        test_ckpt_path.unlink()
        print(f"✓ 测试文件已清理")

        return True
    except Exception as e:
        print(f"✗ Checkpoint测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("V5 完整功能测试")
    print("="*60)

    results = {}

    # 运行测试
    results['数据集加载'] = test_dataset_loading()
    results['DataLoader'] = test_dataloader()
    results['模型创建'] = test_model_creation()
    results['前向传播'] = test_forward_pass()
    results['训练步骤'] = test_training_step()
    results['Checkpoint'] = test_checkpoint_save_load()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！V5可以开始训练了。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == '__main__':
    exit(main())


def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*60)
    print("测试 1: 数据集加载")
    print("="*60)

    try:
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        print(f"✓ 数据集大小: {len(dataset)} 个复合物")

        # 测试加载一个样本
        sample = dataset[0]
        print(f"✓ 样本加载成功:")
        print(f"  - PDB ID: {sample['pdb_id']}")
        print(f"  - ESM shape: {sample['esm'].shape}")
        print(f"  - Protein scalar shape: {sample['protein_scalar'].shape}")
        print(f"  - Ligand features shape: {sample['ligand_features'].shape}")
        print(f"  - Labels (VDW) shape: {sample['labels'].shape}")

        if 'labels_original' in sample:
            print(f"  - Labels (original) shape: {sample['labels_original'].shape}")

            # 比较两种标签的差异
            vdw_binding = (sample['labels'] == 0).sum().item()
            orig_binding = (sample['labels_original'] == 0).sum().item()
            print(f"  - VDW结合位点数: {vdw_binding}")
            print(f"  - 原始结合位点数: {orig_binding}")
            print(f"  - 差异: {vdw_binding - orig_binding}")
        else:
            print(f"  ⚠️  未加载原始标签")

        return True
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试DataLoader"""
    print("\n" + "="*60)
    print("测试 2: DataLoader批处理")
    print("="*60)

    try:
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        # 只取前10个样本测试
        from torch.utils.data import Subset
        test_dataset = Subset(dataset, range(min(10, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"✓ Batch加载成功:")
        print(f"  - Batch size: {len(batch['pdb_id'])}")
        print(f"  - ESM shape: {batch['esm_embeddings'].shape}")
        print(f"  - Protein scalar shape: {batch['protein_scalar'].shape}")
        print(f"  - Ligand features shape: {batch['ligand_features'].shape}")
        print(f"  - Labels shape: {batch['labels'].shape}")
        print(f"  - Mask shape: {batch['mask'].shape}")

        if 'labels_original' in batch:
            print(f"  - Labels original shape: {batch['labels_original'].shape}")

        return True
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("测试 3: 模型创建")
    print("="*60)

    try:
        from src.models.pocket_gvp_fusion import PocketGVPFusion

        model = PocketGVPFusion(
            d_esm=1280,
            d_phys=82,
            d_gvp_internal=256,
            d_model=512,
            d_ligand_atom=80,
            n_gvp_vectors=4,
            n_gvp_vector_hidden=16,
            n_gvp_layers=3,
            n_transformer_layers=4,
            n_ligand_layers=3,
            n_heads=8,
            n_cross_attention_layers=4,
            dropout=0.1,
            use_gated_attn=True,
            use_swiglu=True,
            gvp_radius=10.0,
            gvp_k_neighbors=20,
            esm_dropout_prob=0.5,
            n_classes=3,
            edge_dim=6
        )

        print(f"✓ 模型创建成功")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")

        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试 4: 前向传播")
    print("="*60)

    try:
        from src.models.pocket_gvp_fusion import PocketGVPFusion
        from src.data.collate import prepare_model_inputs

        # 创建模型
        model = PocketGVPFusion(
            d_esm=1280, d_phys=82, d_gvp_internal=256, d_model=512,
            d_ligand_atom=80, n_gvp_vectors=4, n_gvp_vector_hidden=16,
            n_gvp_layers=3, n_transformer_layers=4, n_ligand_layers=3,
            n_heads=8, n_cross_attention_layers=4, dropout=0.1,
            use_gated_attn=True, use_swiglu=True, gvp_radius=10.0,
            gvp_k_neighbors=20, esm_dropout_prob=0.5, n_classes=3, edge_dim=6
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # 加载数据
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        from torch.utils.data import Subset
        test_dataset = Subset(dataset, range(min(2, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        inputs = prepare_model_inputs(batch, device)

        # 前向传播
        with torch.no_grad():
            logits = model(**inputs)

        print(f"✓ 前向传播成功")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "="*60)
    print("测试 5: 训练步骤 (backward + optimizer)")
    print("="*60)

    try:
        from src.models.pocket_gvp_fusion import PocketGVPFusion
        from src.data.collate import prepare_model_inputs
        import torch.nn.functional as F

        # 创建模型
        model = PocketGVPFusion(
            d_esm=1280, d_phys=82, d_gvp_internal=256, d_model=512,
            d_ligand_atom=80, n_gvp_vectors=4, n_gvp_vector_hidden=16,
            n_gvp_layers=3, n_transformer_layers=4, n_ligand_layers=3,
            n_heads=8, n_cross_attention_layers=4, dropout=0.1,
            use_gated_attn=True, use_swiglu=True, gvp_radius=10.0,
            gvp_k_neighbors=20, esm_dropout_prob=0.5, n_classes=3, edge_dim=6
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()

        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 加载数据
        dataset = PocketDatasetV5(
            data_root='data',
            pdb_ids=None,
            max_seq_len=1500,
            max_atoms=100,
            distances_file='data/residue_ligand_distances_vdw.pt',
            distances_file_original='data/residue_ligand_distances_poc_style.pt',
            ligand_features_dir='ligand_features_v3'
        )

        from torch.utils.data import Subset
        test_dataset = Subset(dataset, range(min(2, len(dataset))))

        dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pocket_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))
        inputs = prepare_model_inputs(batch, device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)

        # 前向传播
        logits = model(**inputs)

        # 计算loss
        loss = F.cross_entropy(
            logits.view(-1, 3),
            labels.view(-1),
            reduction='none'
        )
        loss = (loss * mask.view(-1)).sum() / mask.sum()

        print(f"✓ Loss计算成功: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        print(f"✓ 反向传播成功")
        print(f"  - 梯度范数: {total_norm:.4f}")

        # 优化器步骤
        optimizer.step()
        print(f"✓ 优化器步骤成功")

        return True
    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试checkpoint保存和加载"""
    print("\n" + "="*60)
    print("测试 6: Checkpoint保存和加载")
    print("="*60)

    try:
        from src.models.pocket_gvp_fusion import PocketGVPFusion

        # 创建模型
        model = PocketGVPFusion(
            d_esm=1280, d_phys=82, d_gvp_internal=256, d_model=512,
            d_ligand_atom=80, n_gvp_vectors=4, n_gvp_vector_hidden=16,
            n_gvp_layers=3, n_transformer_layers=4, n_ligand_layers=3,
            n_heads=8, n_cross_attention_layers=4, dropout=0.1,
            use_gated_attn=True, use_swiglu=True, gvp_radius=10.0,
            gvp_k_neighbors=20, esm_dropout_prob=0.5, n_classes=3, edge_dim=6
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # 保存checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': 0.5,
            'patience_counter': 2,
            'train_metrics': {'loss': 0.6, 'accuracy': 0.8},
            'val_metrics': {'loss': 0.5, 'accuracy': 0.85}
        }

        test_ckpt_path = Path('test_checkpoint.pt')
        torch.save(checkpoint, test_ckpt_path)
        print(f"✓ Checkpoint保存成功: {test_ckpt_path}")

        # 加载checkpoint
        loaded_ckpt = torch.load(test_ckpt_path, map_location='cpu')

        # 创建新模型并加载
        model2 = PocketGVPFusion(
            d_esm=1280, d_phys=82, d_gvp_internal=256, d_model=512,
            d_ligand_atom=80, n_gvp_vectors=4, n_gvp_vector_hidden=16,
            n_gvp_layers=3, n_transformer_layers=4, n_ligand_layers=3,
            n_heads=8, n_cross_attention_layers=4, dropout=0.1,
            use_gated_attn=True, use_swiglu=True, gvp_radius=10.0,
            gvp_k_neighbors=20, esm_dropout_prob=0.5, n_classes=3, edge_dim=6
        )

        model2.load_state_dict(loaded_ckpt['model_state_dict'])
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        optimizer2.load_state_dict(loaded_ckpt['optimizer_state_dict'])

        print(f"✓ Checkpoint加载成功")
        print(f"  - Epoch: {loaded_ckpt['epoch']}")
        print(f"  - Best val loss: {loaded_ckpt['best_val_loss']}")
        print(f"  - Patience counter: {loaded_ckpt['patience_counter']}")
        print(f"  - Learning rate: {optimizer2.param_groups[0]['lr']:.6f}")

        # 清理
        test_ckpt_path.unlink()
        print(f"✓ 测试文件已清理")

        return True
    except Exception as e:
        print(f"✗ Checkpoint测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("V5 完整功能测试")
    print("="*60)

    results = {}

    # 运行测试
    results['数据集加载'] = test_dataset_loading()
    results['DataLoader'] = test_dataloader()
    results['模型创建'] = test_model_creation()
    results['前向传播'] = test_forward_pass()
    results['训练步骤'] = test_training_step()
    results['Checkpoint'] = test_checkpoint_save_load()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！V5可以开始训练了。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == '__main__':
    exit(main())
