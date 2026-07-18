#!/usr/bin/env python3
"""
推理脚本 for Pocket-GVP-Fusion V10m3

功能:
1. 单样本/批量推理
2. 评估模式 (计算 Accuracy, Precision, Recall, F1, AP, IoU)
3. 全局推理模式 (自动遍历 test set，按性能分类保存 CSV)
4. 预测 pocket PDB 文件导出

标签定义 (d_center, 重原子中心距离):
    Contact    (c0): < 5.0 Å
    Near       (c1): 5.0 – 8.0 Å
    Far        (c2): > 8.0 Å

使用方法:
    # 单个样本推理
    python scripts/infer/infer_v10m3.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --pdb_id 1a2b

    # 批量推理 (从文件读取 PDB ID)
    python scripts/infer/infer_v10m3.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --pdb_list test_ids.txt --output results/

    # 评估模式 (计算完整指标)
    python scripts/infer/infer_v10m3.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --split_file configs/splits/splits.txt \\
        --evaluate --output results/

    # 全局推理模式 (test set, 按性能分类)
    python scripts/infer/infer_v10m3.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --split_file configs/splits/splits.txt \\
        --global_infer --output results/
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.pocket_dataset import load_splits
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v10
from src.models.pocket_gvp_fusion_v10m3 import PocketGVPFusionV10m3

# V10m3 label scheme
CLASS_NAMES = {0: 'Contact', 1: 'Near', 2: 'Far'}
LABEL_SCHEME = "d_center (Contact<5Å, Near 5-8Å, Far>8Å)"


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """加载检查点，返回 (checkpoint_dict, config)。"""
    print(f"加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config')
    if config is None:
        raise ValueError("检查点中未找到 config，请确认使用 V10m3 训练脚本保存的检查点")
    return ckpt, config


def create_model_from_checkpoint(
    ckpt: Dict, config: Dict, device: torch.device
) -> torch.nn.Module:
    """从检查点恢复 V10m3 模型。"""
    model_cfg = config['model']
    print("创建 V10m3 模型...")
    model = PocketGVPFusionV10m3(model_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Dataset / split helpers
# ---------------------------------------------------------------------------

def load_test_ids_from_split(split_file: str) -> List[str]:
    """从 splits.txt 中读取 test 集 PDB ID。"""
    test_ids = []
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] == 'test':
                test_ids.append(parts[1])
    return test_ids


def build_dataset(
    pdb_ids: List[str],
    config: Dict,
    data_root: str = 'data'
) -> PocketDatasetV8:
    """构建 PocketDatasetV8（推理/评估，use_nma=False）。"""
    data_cfg = config['data']
    return PocketDatasetV8(
        data_root=data_root,
        pdb_ids=pdb_ids,
        max_seq_len=data_cfg.get('max_seq_len', 1500),
        max_atoms=data_cfg.get('max_atoms', 128),
        distances_file=data_cfg.get('distances_file'),
        distances_file_original=None,
        esm_features_dir=data_cfg.get('esm_features_dir', 'esm2_latents'),
        esm_alignment_features_dir=data_cfg.get('esm_alignment_features_dir'),
        esm3_features_dir=data_cfg.get('esm3_features_dir'),
        ligand_features_dir=data_cfg.get('ligand_features_dir', 'ligand_features_v5'),
        protein_features_dir=data_cfg.get('protein_features_dir', 'complete_features_v4'),
        nma_features_dir=data_cfg.get('nma_features_dir', 'complete_features_v4_nma'),
        use_nma=False,
        label_contact_thresh=data_cfg.get('label_contact_thresh', 5.0),
        label_near_thresh=data_cfg.get('label_near_thresh', 8.0),
        split='test',
        verbose=True,
    )


# ---------------------------------------------------------------------------
# AP helper
# ---------------------------------------------------------------------------

def compute_ap(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, probs))
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    preds: np.ndarray, labels: np.ndarray, probs: np.ndarray
) -> Dict:
    """计算单样本的 Precision/Recall/F1/IoU/AP (per-class) + 准确率。"""
    metrics = {'accuracy': float((preds == labels).mean())}
    for c in range(3):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        metrics[f'precision_c{c}'] = float(precision)
        metrics[f'recall_c{c}']    = float(recall)
        metrics[f'f1_c{c}']        = float(f1)
        metrics[f'iou_c{c}']       = float(iou)
        if c == 0:
            metrics['ap_c0'] = compute_ap(probs[:, 0], (labels == 0).astype(int))
    return metrics


def compute_global_metrics(
    all_preds: np.ndarray, all_labels: np.ndarray, all_probs: np.ndarray
) -> Dict:
    """在拼接后的全量残基上计算全局指标。"""
    metrics = {'accuracy': float((all_preds == all_labels).mean())}
    for c in range(3):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        metrics[f'precision_c{c}'] = float(precision)
        metrics[f'recall_c{c}']    = float(recall)
        metrics[f'f1_c{c}']        = float(f1)
        metrics[f'iou_c{c}']       = float(iou)
    metrics['ap_c0'] = compute_ap(all_probs[:, 0], (all_labels == 0).astype(int))
    return metrics


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
    use_amp: bool = False,
) -> tuple:
    """对一个 batch 推理，返回 (logits, probs, preds)，均为 cpu tensor。"""
    inputs = prepare_model_inputs_v10(batch, device, non_blocking=False)
    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_physchem=inputs['protein_physchem'],
            protein_spatial_scalar=inputs['protein_spatial_scalar'],
            protein_spatial_vector=inputs['protein_spatial_vector'],
            protein_gvp_vectors=inputs['protein_gvp_vectors'],
            protein_coords=inputs['protein_coords'],
            protein_mask=inputs['protein_mask'],
            ligand_x=inputs['ligand_x'],
            ligand_pos=inputs['ligand_pos'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_to_protein=inputs['ligand_to_protein'],
            ligand_mask=inputs['ligand_mask'],
        )
    logits = outputs['logits'].cpu()
    probs  = F.softmax(logits, dim=-1)
    preds  = logits.argmax(dim=-1)
    return logits, probs, preds


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_prediction_summary(
    pdb_id: str,
    preds: np.ndarray,
    probs: np.ndarray,
    labels: Optional[np.ndarray] = None,
):
    seq_len = len(preds)
    print(f"\n{'='*60}")
    print(f"PDB ID: {pdb_id}  序列长度: {seq_len}")
    print(f"标签体系: {LABEL_SCHEME}")
    print(f"{'='*60}")
    for c, name in CLASS_NAMES.items():
        n = (preds == c).sum()
        print(f"  {name:8s} (c{c}): {n:4d} ({n/seq_len*100:5.1f}%)")
    if labels is not None:
        acc = (preds == labels).mean()
        print(f"\n  准确率: {acc:.4f}")
    print(f"{'='*60}\n")


def save_tsv(
    pdb_id: str,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    labels: Optional[np.ndarray] = None,
) -> Path:
    """保存逐残基预测为 TSV 格式。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{pdb_id}_predictions.tsv"
    with open(tsv_path, 'w') as f:
        header = "residue_idx\tpred_class\tpred_label\tprob_contact\tprob_near\tprob_far\tconfidence"
        if labels is not None:
            header += "\ttrue_class\ttrue_label\tcorrect"
        f.write(header + '\n')
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            conf = prob.max()
            row = f"{i}\t{pred}\t{CLASS_NAMES[pred]}\t{prob[0]:.4f}\t{prob[1]:.4f}\t{prob[2]:.4f}\t{conf:.4f}"
            if labels is not None:
                tl = labels[i]
                row += f"\t{tl}\t{CLASS_NAMES[tl]}\t{int(pred==tl)}"
            f.write(row + '\n')
    return tsv_path


def save_npz(
    pdb_id: str,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    labels: Optional[np.ndarray] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{pdb_id}_predictions.npz"
    data = dict(
        pdb_id=pdb_id,
        predictions=preds,
        probabilities=probs,
        label_scheme=LABEL_SCHEME,
    )
    if labels is not None:
        data['true_labels'] = labels
    np.savez(npz_path, **data)
    return npz_path


def export_pocket_pdb(
    pdb_id: str,
    preds: np.ndarray,
    output_dir: Path,
    pdb_root: str = '/home/husrcf/Code/ProtBind/New',
    contact_only: bool = False,
) -> Optional[Path]:
    """根据预测结果导出 pocket PDB（Contact + Near，或只导出 Contact）。"""
    pdb_root_p = Path(pdb_root)
    pdb_dir = None
    for year_dir in sorted(pdb_root_p.iterdir()):
        if year_dir.is_dir():
            candidate = year_dir / pdb_id.lower()
            if candidate.exists():
                pdb_dir = candidate
                break
    if pdb_dir is None:
        print(f"  警告: 未找到 {pdb_id} 的 PDB 目录")
        return None

    protein_pdb = pdb_dir / f"{pdb_id}_protein.pdb"
    if not protein_pdb.exists():
        print(f"  警告: 未找到 {protein_pdb}")
        return None

    # parse residue order
    residue_atoms: List[List[str]] = []
    current_key = None
    buf: List[str] = []
    with open(protein_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                key = (line[21], line[22:26].strip())
                if key != current_key:
                    if current_key is not None:
                        residue_atoms.append(buf)
                    current_key = key
                    buf = []
                buf.append(line)
    if buf:
        residue_atoms.append(buf)

    # pocket = Contact(0) + Near(1), or Contact(0) only
    keep_classes = {0} if contact_only else {0, 1}
    year_name = pdb_dir.parent.name
    pdb_out_dir = output_dir / 'pdb' / year_name / pdb_id
    pdb_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = 'contact_only' if contact_only else 'pocket'
    out_pdb = pdb_out_dir / f"{pdb_id}_predicted_{suffix}.pdb"
    with open(out_pdb, 'w') as f:
        f.write(f"HEADER    {pdb_id.upper()} PREDICTED {suffix.upper()}\n")
        f.write(f"REMARK    V10m3 inference | {LABEL_SCHEME}\n")
        atom_idx = 1
        for res_idx, atoms in enumerate(residue_atoms):
            if res_idx < len(preds) and preds[res_idx] in keep_classes:
                for atom_line in atoms:
                    f.write(f"{atom_line[:6]}{atom_idx:5d}{atom_line[11:]}")
                    atom_idx += 1
        f.write("END\n")
    return out_pdb


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict:
    """评估模式：在整个数据集上计算全局指标，同时打印 per-sample 统计。"""
    model.eval()
    all_preds_list, all_labels_list, all_probs_list = [], [], []
    sample_results = []

    for batch in tqdm(dataloader, desc="评估", unit="batch"):
        labels_batch = batch['labels']
        mask_batch   = batch['protein_mask']
        pdb_ids      = batch.get('pdb_ids', [])
        _, probs, preds = predict_batch(model, batch, device, use_amp)

        for b in range(labels_batch.shape[0]):
            mask_b   = mask_batch[b].bool()
            preds_b  = preds[b][mask_b].numpy()
            labels_b = labels_batch[b][mask_b].numpy()
            probs_b  = probs[b][mask_b].numpy()
            pdb_id   = pdb_ids[b] if b < len(pdb_ids) else f'sample_{b}'

            all_preds_list.append(preds_b)
            all_labels_list.append(labels_b)
            all_probs_list.append(probs_b)

            sm = compute_sample_metrics(preds_b, labels_b, probs_b)
            sm['pdb_id'] = pdb_id
            sm['n_residues'] = len(preds_b)
            sample_results.append(sm)

    all_preds  = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    all_probs  = np.concatenate(all_probs_list)
    global_metrics = compute_global_metrics(all_preds, all_labels, all_probs)
    global_metrics['n_samples'] = len(sample_results)

    # Print summary
    print(f"\n{'='*70}")
    print("全局评估指标")
    print(f"{'='*70}")
    print(f"样本数: {global_metrics['n_samples']}  准确率: {global_metrics['accuracy']:.4f}")
    for c, name in CLASS_NAMES.items():
        print(f"  {name:8s} (c{c}) P={global_metrics[f'precision_c{c}']:.4f}  "
              f"R={global_metrics[f'recall_c{c}']:.4f}  "
              f"F1={global_metrics[f'f1_c{c}']:.4f}  "
              f"IoU={global_metrics[f'iou_c{c}']:.4f}"
              + (f"  AP={global_metrics['ap_c0']:.4f}" if c == 0 else ""))
    print(f"{'='*70}\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'global_metrics.json', 'w') as f:
            json.dump(global_metrics, f, indent=2)

        _save_per_sample_csv(sample_results, output_dir / 'per_sample_metrics.csv')
        print(f"结果已保存到: {output_dir}")

    return global_metrics


# ---------------------------------------------------------------------------
# Global infer mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_global_infer(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = False,
    pdb_root: str = '/home/husrcf/Code/ProtBind/New',
    export_pdb: bool = False,
) -> Dict:
    """全局推理模式：对 test set 推理，按 Contact F1 分类保存 CSV。"""
    model.eval()
    sample_results = []

    for batch in tqdm(dataloader, desc="全局推理", unit="batch"):
        labels_batch = batch.get('labels')
        mask_batch   = batch['protein_mask']
        pdb_ids      = batch.get('pdb_ids', [])
        _, probs, preds = predict_batch(model, batch, device, use_amp)

        for b in range(mask_batch.shape[0]):
            mask_b  = mask_batch[b].bool()
            preds_b = preds[b][mask_b].numpy()
            probs_b = probs[b][mask_b].numpy()
            pdb_id  = pdb_ids[b] if b < len(pdb_ids) else f'sample_{b}'

            result = {
                'pdb_id': pdb_id,
                'n_residues': len(preds_b),
                'n_contact': int((preds_b == 0).sum()),
                'n_near':    int((preds_b == 1).sum()),
                'n_far':     int((preds_b == 2).sum()),
            }

            if labels_batch is not None:
                labels_b = labels_batch[b][mask_b].numpy()
                sm = compute_sample_metrics(preds_b, labels_b, probs_b)
                result.update(sm)
            else:
                result['accuracy'] = float('nan')
                result['f1_c0'] = float('nan')
                result['ap_c0'] = float('nan')
                result['iou_c0'] = float('nan')

            sample_results.append(result)

            if export_pdb:
                try:
                    export_pocket_pdb(pdb_id, preds_b, output_dir, pdb_root)
                except Exception as e:
                    print(f"  警告: {pdb_id} PDB 导出失败: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify by average(f1_c0, ap_c0, iou_c0)
    hard, medium, good, unknown = [], [], [], []
    for r in sample_results:
        f1, ap, iou = r.get('f1_c0', float('nan')), r.get('ap_c0', float('nan')), r.get('iou_c0', float('nan'))
        if any(np.isnan(v) for v in [f1, ap, iou]):
            unknown.append(r)
        else:
            avg = (f1 + ap + iou) / 3.0
            if avg >= 0.4:
                good.append(r)
            elif avg >= 0.2:
                medium.append(r)
            else:
                hard.append(r)

    _save_per_sample_csv(sample_results, output_dir / 'all_results.csv')
    _save_per_sample_csv(good,    output_dir / 'good_cases.csv')
    _save_per_sample_csv(medium,  output_dir / 'medium_cases.csv')
    _save_per_sample_csv(hard,    output_dir / 'hard_cases.csv')
    if unknown:
        _save_per_sample_csv(unknown, output_dir / 'no_labels.csv')

    summary = {
        'total': len(sample_results),
        'good':   len(good),
        'medium': len(medium),
        'hard':   len(hard),
    }

    print(f"\n{'='*60}")
    print("全局推理完成")
    print(f"{'='*60}")
    print(f"总样本数: {summary['total']}")
    if good or medium or hard:
        print(f"  优秀 (avg>=0.4): {summary['good']:4d} ({summary['good']/max(len(sample_results),1)*100:.1f}%)")
        print(f"  一般 (0.2-0.4): {summary['medium']:4d} ({summary['medium']/max(len(sample_results),1)*100:.1f}%)")
        print(f"  难例 (<0.2):    {summary['hard']:4d}  ({summary['hard']/max(len(sample_results),1)*100:.1f}%)")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*60}\n")

    return summary


def _save_per_sample_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    fields = ['pdb_id', 'n_residues', 'accuracy',
              'f1_c0', 'ap_c0', 'iou_c0', 'precision_c0', 'recall_c0',
              'f1_c1', 'iou_c1', 'precision_c1', 'recall_c1',
              'f1_c2', 'iou_c2', 'precision_c2', 'recall_c2',
              'n_contact', 'n_near', 'n_far']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='V10m3 推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径 (best_model.pt)')
    parser.add_argument('--pdb_id',     type=str, default=None, help='单个 PDB ID')
    parser.add_argument('--pdb_list',   type=str, default=None, help='PDB ID 列表文件 (每行一个)')
    parser.add_argument('--split_file', type=str, default=None, help='splits.txt，用于 evaluate/global_infer 模式')
    parser.add_argument('--data_root',  type=str, default='data', help='数据根目录')
    parser.add_argument('--pdb_root',   type=str, default='/home/husrcf/Code/ProtBind/New',
                        help='原始 PDB 根目录（用于导出 pocket PDB）')
    parser.add_argument('--output',     type=str, default=None, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4, help='推理 batch size')
    parser.add_argument('--num_workers',type=int, default=4, help='DataLoader workers')
    parser.add_argument('--amp',        action='store_true', help='使用 AMP（需要 GPU）')
    parser.add_argument('--export_pdb', action='store_true', help='导出预测 pocket PDB 文件')
    # 模式
    parser.add_argument('--evaluate',    action='store_true', help='评估模式（需要有标签）')
    parser.add_argument('--global_infer',action='store_true', help='全局推理：对 test set 推理并分类保存')
    args = parser.parse_args()

    if not args.global_infer and args.pdb_id is None and args.pdb_list is None:
        parser.error("请指定 --pdb_id、--pdb_list 或使用 --global_infer")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    ckpt, config = load_checkpoint(args.checkpoint, device)
    model = create_model_from_checkpoint(ckpt, config, device)

    # Determine PDB IDs
    if args.global_infer:
        if not args.split_file:
            parser.error("--global_infer 需要 --split_file")
        pdb_ids = load_test_ids_from_split(args.split_file)
        print(f"从 split 文件加载 {len(pdb_ids)} 个 test 样本")
    elif args.pdb_id:
        pdb_ids = [args.pdb_id]
    else:
        with open(args.pdb_list, 'r') as f:
            pdb_ids = [line.strip() for line in f if line.strip()]

    print(f"待处理样本数: {len(pdb_ids)}")

    dataset = build_dataset(pdb_ids, config, data_root=args.data_root)
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据路径和特征文件")
        sys.exit(1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=(device.type == 'cuda'),
    )

    # Output directory
    output_dir = Path(args.output) if args.output else None

    # ---- Global infer mode ----
    if args.global_infer:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = (output_dir or Path('outputs')) / f'infer_v10m3_{ts}'
        run_global_infer(model, dataloader, device, out, args.amp, args.pdb_root, args.export_pdb)
        return

    # ---- Evaluate mode ----
    if args.evaluate:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = (output_dir or Path('outputs')) / f'eval_v10m3_{ts}'
        run_evaluate(model, dataloader, device, args.amp, out)
        return

    # ---- Single / batch inference mode ----
    print("\n开始推理...")
    for batch in tqdm(dataloader, desc="推理", unit="batch"):
        mask_batch   = batch['protein_mask']
        pdb_ids_b    = batch.get('pdb_ids', [])
        labels_batch = batch.get('labels')
        _, probs, preds = predict_batch(model, batch, device, args.amp)

        for b in range(mask_batch.shape[0]):
            mask_b  = mask_batch[b].bool()
            preds_b = preds[b][mask_b].numpy()
            probs_b = probs[b][mask_b].numpy()
            pdb_id  = pdb_ids_b[b] if b < len(pdb_ids_b) else f'sample_{b}'
            labels_b = labels_batch[b][mask_b].numpy() if labels_batch is not None else None

            print_prediction_summary(pdb_id, preds_b, probs_b, labels_b)

            if output_dir:
                save_tsv(pdb_id, preds_b, probs_b, output_dir, labels_b)
                save_npz(pdb_id, preds_b, probs_b, output_dir, labels_b)
                if args.export_pdb:
                    try:
                        pdb_out = export_pocket_pdb(pdb_id, preds_b, output_dir, args.pdb_root)
                        if pdb_out:
                            print(f"  pocket PDB: {pdb_out}")
                    except Exception as e:
                        print(f"  警告: PDB 导出失败: {e}")

    print("完成!")


if __name__ == '__main__':
    main()
