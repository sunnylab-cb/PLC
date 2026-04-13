#!/usr/bin/env python3
"""
推理脚本 for Pocket-GVP-Fusion V10m3 — COACH420 benchmark

COACH420 数据目录结构:
    /home/husrcf/Code/ProtBind/coach420/
        protein/   {chain_id}.pt          (v4 protein features)
        esm2/      {chain_id}.pt          (ESM-2 embeddings)
        ligand/    {pdb_stem}_{lig}_{num}.pt
        residue_ligand_distances_dcenter.pt   (420 samples, key = pdb_stem)

标签定义 (d_center):
    Contact (c0): < 5.0 Å
    Near    (c1): 5.0 – 8.0 Å
    Far     (c2): > 8.0 Å

使用方法:
    python scripts/infer/infer_v10m3_coach420.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --output results/coach420/

    # 不计算指标（无标签模式）
    python scripts/infer/infer_v10m3_coach420.py \\
        --checkpoint outputs/xxx/best_model.pt \\
        --no_labels --output results/coach420/
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v10
from src.models.pocket_gvp_fusion_v10m3 import PocketGVPFusionV10m3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COACH420_ROOT = Path('/home/husrcf/Code/ProtBind/coach420')
CLASS_NAMES   = {0: 'Contact', 1: 'Near', 2: 'Far'}
LABEL_SCHEME  = "d_center (Contact<5Å, Near 5-8Å, Far>8Å)"

LABEL_CONTACT_THRESH = 5.0
LABEL_NEAR_THRESH    = 8.0

MAX_SEQ_LEN = 1500
MAX_ATOMS   = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Coach420Dataset(Dataset):
    """
    Dataset for COACH420 benchmark.

    Each item corresponds to one PDB complex (key = pdb_stem).
    protein 和 esm2 按 pdb_stem 查找；
    ligand 会聚合同一 pdb_stem 下的全部 ligand `.pt`。
    """

    def __init__(
        self,
        coach420_root: Path,
        pair_keys: List[str],
        label_contact_thresh: float = LABEL_CONTACT_THRESH,
        label_near_thresh: float = LABEL_NEAR_THRESH,
        use_labels: bool = True,
    ):
        self.root   = Path(coach420_root)
        self.keys   = pair_keys
        self.ct     = label_contact_thresh
        self.nt     = label_near_thresh
        self.use_labels = use_labels

        dist_path = self.root / 'residue_ligand_distances_dcenter.pt'
        if use_labels and dist_path.exists():
            self.dist_dict = torch.load(dist_path, weights_only=False)
        else:
            self.dist_dict = {}

        self.protein_dir = self.root / 'protein'
        self.esm_dir     = self.root / 'esm2'
        self.ligand_dir  = self.root / 'ligand'

    def __len__(self):
        return len(self.keys)

    def _distances_to_labels(self, distances: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(distances, 2, dtype=torch.long)
        labels[distances < self.ct] = 0
        labels[(distances >= self.ct) & (distances < self.nt)] = 1
        return labels

    def _compute_gvp_vectors(self, all_atom_coords, ca_coords: torch.Tensor) -> torch.Tensor:
        n = len(all_atom_coords)
        vecs = torch.zeros(n, 4, 3)
        for i, atoms in enumerate(all_atom_coords):
            if atoms is None or len(atoms) == 0:
                continue
            if not isinstance(atoms, torch.Tensor):
                atoms = torch.tensor(atoms, dtype=torch.float32)
            ca = ca_coords[i]
            vecs[i, 0] = atoms.mean(dim=0) - ca
            for j in range(min(3, len(atoms))):
                vecs[i, j + 1] = atoms[j] - ca
        return vecs

    def _find_ligand_files(self, pdb_id: str) -> List[Path]:
        direct = self.ligand_dir / f'{pdb_id}.pt'
        if direct.exists():
            return [direct]
        return sorted(self.ligand_dir.glob(f'{pdb_id}_*.pt'))

    def _merge_ligand_data(self, pdb_id: str) -> Dict[str, torch.Tensor]:
        ligand_files = self._find_ligand_files(pdb_id)
        if not ligand_files:
            raise FileNotFoundError(f'找不到 {pdb_id} 的 ligand 特征文件')

        feat_blocks = []
        coord_blocks = []
        edge_index_blocks = []
        edge_attr_blocks = []
        atom_offset = 0

        for path in ligand_files:
            lig_data = torch.load(path, weights_only=False)
            atom_features = lig_data['atom_features'].float()
            coords = lig_data['coords'].float()
            feat_blocks.append(atom_features)
            coord_blocks.append(coords)

            edge_index = lig_data.get('edge_index')
            edge_attr = lig_data.get('edge_attr')
            if edge_index is not None and edge_attr is not None and edge_index.numel() > 0:
                edge_index_blocks.append(edge_index.long() + atom_offset)
                edge_attr_blocks.append(edge_attr.float())
            atom_offset += atom_features.shape[0]

        atom_features = torch.cat(feat_blocks, dim=0)
        coords = torch.cat(coord_blocks, dim=0)
        if edge_index_blocks:
            edge_index = torch.cat(edge_index_blocks, dim=1)
            edge_attr = torch.cat(edge_attr_blocks, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float32)

        return {
            'atom_features': atom_features,
            'coords': coords,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        }

    def __getitem__(self, idx: int) -> Dict:
        key = self.keys[idx]
        pdb_id = key

        # Load protein
        prot_data = torch.load(self.protein_dir / f'{pdb_id}.pt', weights_only=False)
        esm_data  = torch.load(self.esm_dir     / f'{pdb_id}.pt', weights_only=False)
        lig_data  = self._merge_ligand_data(pdb_id)

        # Protein features
        protein_physchem        = prot_data['physchem_features']         # [N, 62]
        protein_spatial_scalar  = prot_data['spatial_scalar_features']   # [N, 89]
        protein_spatial_vector  = prot_data['spatial_vector_features']   # [N, 8, 3]
        ca_coords               = prot_data['ca_coords']                  # [N, 3]
        all_atom_coords         = prot_data['all_atom_coords']
        gvp_vectors             = self._compute_gvp_vectors(all_atom_coords, ca_coords)  # [N, 4, 3]

        esm = esm_data['embeddings']  # [N, 1280]

        # Ligand features
        lig_feats      = lig_data['atom_features']   # [M, 109]
        lig_coords     = lig_data['coords']           # [M, 3]
        lig_edge_index = lig_data.get('edge_index')
        lig_edge_attr  = lig_data.get('edge_attr')

        n_residues = protein_physchem.shape[0]
        n_atoms    = lig_feats.shape[0]

        # Labels
        if self.use_labels and key in self.dist_dict:
            dist_data = self.dist_dict[key]
            distances = dist_data['distances'] if isinstance(dist_data, dict) else dist_data
            labels = self._distances_to_labels(distances)
            # Align length
            if len(labels) < n_residues:
                pad = torch.full((n_residues - len(labels),), 2, dtype=torch.long)
                labels = torch.cat([labels, pad])
            else:
                labels = labels[:n_residues]
        else:
            labels = torch.full((n_residues,), 2, dtype=torch.long)

        # Truncate
        if n_residues > MAX_SEQ_LEN:
            protein_physchem       = protein_physchem[:MAX_SEQ_LEN]
            protein_spatial_scalar = protein_spatial_scalar[:MAX_SEQ_LEN]
            protein_spatial_vector = protein_spatial_vector[:MAX_SEQ_LEN]
            ca_coords              = ca_coords[:MAX_SEQ_LEN]
            gvp_vectors            = gvp_vectors[:MAX_SEQ_LEN]
            esm                    = esm[:MAX_SEQ_LEN]
            labels                 = labels[:MAX_SEQ_LEN]
            n_residues             = MAX_SEQ_LEN

        if n_atoms > MAX_ATOMS:
            lig_feats  = lig_feats[:MAX_ATOMS]
            lig_coords = lig_coords[:MAX_ATOMS]
            n_atoms    = MAX_ATOMS

        # Edge index/attr
        if lig_edge_index is None or lig_edge_attr is None:
            lig_edge_index = torch.zeros((2, 0), dtype=torch.long)
            lig_edge_attr  = torch.zeros((0, 6),  dtype=torch.float32)
        else:
            lig_edge_index = lig_edge_index.long()
            lig_edge_attr  = lig_edge_attr.float()
            mask = (lig_edge_index[0] < n_atoms) & (lig_edge_index[1] < n_atoms)
            lig_edge_index = lig_edge_index[:, mask]
            lig_edge_attr  = lig_edge_attr[mask]

        return {
            'pdb_id':                key,
            'esm':                   esm.float(),
            'protein_physchem':      protein_physchem.float(),
            'protein_spatial_scalar':protein_spatial_scalar.float(),
            'protein_spatial_vector':protein_spatial_vector.float(),
            'gvp_coords':            ca_coords.float(),
            'gvp_vectors':           gvp_vectors.float(),
            'ligand_features':       lig_feats.float(),
            'ligand_coords':         lig_coords.float(),
            'ligand_edge_index':     lig_edge_index,
            'ligand_edge_attr':      lig_edge_attr,
            'labels':                labels.long(),
            'n_residues':            n_residues,
            'n_atoms':               n_atoms,
            # CA coords stored for DCC computation (already truncated to n_residues)
            'ca_coords_raw':         ca_coords.float(),
        }


# ---------------------------------------------------------------------------
# Checkpoint / model
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device):
    print(f"加载检查点: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config')
    if config is None:
        raise ValueError("检查点中未找到 config")
    return ckpt, config


def create_model(ckpt: Dict, config: Dict, device: torch.device) -> torch.nn.Module:
    model = PocketGVPFusionV10m3(config['model'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n:,}")
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ap(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        if labels.sum() == 0:
            return float('nan')
        return float(average_precision_score(labels, probs))
    except Exception:
        return float('nan')


def compute_sample_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    ca_coords: np.ndarray,          # [N, 3] CA coordinates of valid residues
    dcc_thresh: float = 4.0,
) -> Dict:
    """
    Compute per-sample metrics including DCC.

    DCC (Distance to Closest Center):
        pred_center  = mean CA of predicted Contact residues (top-1 if none predicted)
        true_center  = mean CA of true Contact residues
        dcc          = L2 distance between the two centers
        dcc_success  = dcc < dcc_thresh (default 4 Å)

    If no true Contact residues exist, dcc = nan.
    If no predicted Contact residues exist, fall back to the single residue
    with highest Contact probability as pred_center (Top-1 DCC).
    """
    m: Dict = {'accuracy': float((preds == labels).mean())}

    # Per-class P/R/F1/IoU/Dice (Dice == F1 for binary segmentation)
    for c in range(3):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        tn = int(((preds != c) & (labels != c)).sum())
        pr  = tp / (tp + fp + 1e-8)
        rc  = tp / (tp + fn + 1e-8)
        f1  = 2 * pr * rc / (pr + rc + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        # MCC (binary: c vs rest)
        mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        m[f'precision_c{c}'] = float(pr)
        m[f'recall_c{c}']    = float(rc)
        m[f'f1_c{c}']        = float(f1)
        m[f'dice_c{c}']      = float(f1)   # Dice == F1
        m[f'iou_c{c}']       = float(iou)
        m[f'mcc_c{c}']       = float(mcc)
        if c == 0:
            m['ap_c0'] = _ap(probs[:, 0], (labels == 0).astype(int))

    # Binary binding F1: (c0+c1) vs c2
    true_binding = (labels <= 1)   # c0 + c1 = binding
    pred_binding = (preds  <= 1)
    bin_tp = int(( pred_binding &  true_binding).sum())
    bin_fp = int(( pred_binding & ~true_binding).sum())
    bin_fn = int((~pred_binding &  true_binding).sum())
    bin_pr = bin_tp / (bin_tp + bin_fp + 1e-8)
    bin_rc = bin_tp / (bin_tp + bin_fn + 1e-8)
    m['f1_binary'] = float(2 * bin_pr * bin_rc / (bin_pr + bin_rc + 1e-8))

    # DCC
    true_contact_mask = (labels == 0)
    pred_contact_mask = (preds  == 0)

    if true_contact_mask.sum() == 0:
        m['dcc'] = float('nan')
        m['dcc_success'] = float('nan')
    else:
        true_center = ca_coords[true_contact_mask].mean(axis=0)  # [3]
        if pred_contact_mask.sum() > 0:
            pred_center = ca_coords[pred_contact_mask].mean(axis=0)
        else:
            # Top-1 DCC: residue with highest Contact probability
            top1_idx = int(np.argmax(probs[:, 0]))
            pred_center = ca_coords[top1_idx]
        dcc = float(np.linalg.norm(pred_center - true_center))
        m['dcc'] = dcc
        m['dcc_success'] = float(dcc < dcc_thresh)

    return m


def aggregate_metrics(sample_results: List[Dict]) -> Dict:
    """
    Aggregate per-sample metrics into dataset-level summary.

    AP(filt): AP computed only on samples that have at least one true Contact residue.
    DCC Top-1 / DCC Success: mean over samples with valid (non-nan) dcc.
    All other metrics: simple mean over all samples.
    """
    n = len(sample_results)
    if n == 0:
        return {}

    def _nanmean(key):
        vals = [r[key] for r in sample_results if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    agg: Dict = {
        'N_evaluated': n,
        # DCC
        'DCC_Top1':    _nanmean('dcc'),
        'DCC_Success': _nanmean('dcc_success'),
        # Contact class
        'Mean_IoU':    _nanmean('iou_c0'),
        'Mean_Prec':   _nanmean('precision_c0'),
        'Mean_Rec':    _nanmean('recall_c0'),
        'Mean_F1':     _nanmean('f1_c0'),
        'Mean_Dice':   _nanmean('dice_c0'),
        'Mean_MCC':    _nanmean('mcc_c0'),
        'Mean_AP':     _nanmean('ap_c0'),
        # Binary binding F1: (c0+c1) vs c2, per-protein mean
        'Mean_F1_binary': _nanmean('f1_binary'),
    }

    # AP(filt): only samples with at least one true contact
    ap_filt_vals = [
        r['ap_c0'] for r in sample_results
        if 'ap_c0' in r
        and not (isinstance(r['ap_c0'], float) and math.isnan(r['ap_c0']))
        and r.get('n_true_contact', 1) > 0   # fallback: always include
    ]
    # Re-filter properly using labels stored in each row
    ap_filt_vals2 = [
        r['ap_c0'] for r in sample_results
        if 'ap_c0' in r
        and not (isinstance(r['ap_c0'], float) and math.isnan(r['ap_c0']))
    ]
    agg['Mean_AP_filt'] = float(np.mean(ap_filt_vals2)) if ap_filt_vals2 else float('nan')

    return agg


def print_benchmark_summary(agg: Dict):
    w = 65
    print(f"\n{'='*w}")
    print(f"  COACH420 Benchmark Summary   (N={agg.get('N_evaluated', '?')})")
    print(f"{'='*w}")
    print(f"  {'DCC Top-1':<20s}  {agg.get('DCC_Top1', float('nan')):.4f} Å")
    print(f"  {'DCC Success (<4Å)':<20s}  {agg.get('DCC_Success', float('nan')):.4f}")
    print(f"  {'-'*50}")
    print(f"  {'Mean IoU':<20s}  {agg.get('Mean_IoU', float('nan')):.4f}")
    print(f"  {'Mean Prec':<20s}  {agg.get('Mean_Prec', float('nan')):.4f}")
    print(f"  {'Mean Rec':<20s}  {agg.get('Mean_Rec', float('nan')):.4f}")
    print(f"  {'Mean F1':<20s}  {agg.get('Mean_F1', float('nan')):.4f}")
    print(f"  {'Mean F1 (binary)':<20s}  {agg.get('Mean_F1_binary', float('nan')):.4f}  [c0+c1 vs c2]")
    print(f"  {'Mean Dice':<20s}  {agg.get('Mean_Dice', float('nan')):.4f}")
    print(f"  {'Mean MCC':<20s}  {agg.get('Mean_MCC', float('nan')):.4f}")
    print(f"  {'Mean AP':<20s}  {agg.get('Mean_AP', float('nan')):.4f}")
    print(f"  {'Mean AP(filt)':<20s}  {agg.get('Mean_AP_filt', float('nan')):.4f}")
    print(f"  {'N Evaluated':<20s}  {agg.get('N_evaluated', '?')}")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model, batch, device, use_amp=False):
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
    return probs, preds


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    fields = [
        'pdb_id', 'n_residues', 'accuracy',
        'dcc', 'dcc_success',
        'f1_c0', 'dice_c0', 'ap_c0', 'iou_c0', 'precision_c0', 'recall_c0', 'mcc_c0',
        'f1_c1', 'iou_c1', 'precision_c1', 'recall_c1',
        'f1_c2', 'iou_c2', 'precision_c2', 'recall_c2',
        'n_contact', 'n_near', 'n_far',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='V10m3 × COACH420 推理脚本')
    parser.add_argument('--checkpoint',    required=True,          help='检查点路径 (best_model.pt)')
    parser.add_argument('--coach420_root', default=str(COACH420_ROOT), help='COACH420 根目录')
    parser.add_argument('--output',        default=None,           help='输出目录')
    parser.add_argument('--batch_size',    type=int, default=4)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--amp',           action='store_true',    help='使用 AMP')
    parser.add_argument('--no_labels',     action='store_true',    help='不使用距离标签（纯推理模式）')
    parser.add_argument('--contact_thresh',type=float, default=LABEL_CONTACT_THRESH)
    parser.add_argument('--near_thresh',   type=float, default=LABEL_NEAR_THRESH)
    parser.add_argument('--dcc_thresh',    type=float, default=4.0, help='DCC success 距离阈值 (Å)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    coach_root = Path(args.coach420_root)
    ckpt, config = load_checkpoint(args.checkpoint, device)
    model = create_model(ckpt, config, device)

    dist_path = coach_root / 'residue_ligand_distances_dcenter.pt'
    dist_dict = torch.load(dist_path, weights_only=False)
    all_keys  = sorted(dist_dict.keys())
    print(f"COACH420 总样本数: {len(all_keys)}")

    use_labels = not args.no_labels
    dataset = Coach420Dataset(
        coach420_root=coach_root,
        pair_keys=all_keys,
        label_contact_thresh=args.contact_thresh,
        label_near_thresh=args.near_thresh,
        use_labels=use_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=(device.type == 'cuda'),
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output) if args.output else Path('results') / f'coach420_v10m3_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference loop ----
    sample_results: List[Dict] = []
    # We need per-sample CA coords for DCC; store them keyed by batch position.
    # pocket_collate_fn_v10 pads 'gvp_coords' → shape [B, L, 3]; we recover
    # valid residues via protein_mask.

    for batch in tqdm(loader, desc='COACH420 推理', unit='batch'):
        labels_batch  = batch['labels']        # [B, L]
        mask_batch    = batch['protein_mask']  # [B, L]
        coords_batch  = batch.get('protein_coords')  # [B, L, 3]  (gvp_coords after collate)
        pdb_ids       = batch.get('pdb_ids', [])

        probs, preds = predict_batch(model, batch, device, args.amp)

        for b in range(mask_batch.shape[0]):
            mask_b   = mask_batch[b].bool()
            preds_b  = preds[b][mask_b].numpy()
            probs_b  = probs[b][mask_b].numpy()
            labels_b = labels_batch[b][mask_b].numpy()
            key      = pdb_ids[b] if b < len(pdb_ids) else f'sample_{b}'

            # CA coordinates of valid residues (for DCC)
            if coords_batch is not None:
                ca_b = coords_batch[b][mask_b].numpy()   # [N_valid, 3]
            else:
                ca_b = np.zeros((mask_b.sum().item(), 3))

            row: Dict = {
                'pdb_id':     key,
                'n_residues': int(mask_b.sum()),
                'n_contact':  int((preds_b == 0).sum()),
                'n_near':     int((preds_b == 1).sum()),
                'n_far':      int((preds_b == 2).sum()),
                'n_true_contact': int((labels_b == 0).sum()),
            }

            if use_labels:
                sm = compute_sample_metrics(
                    preds_b, labels_b, probs_b, ca_b, dcc_thresh=args.dcc_thresh
                )
                row.update(sm)
            else:
                row.update({
                    'accuracy': float('nan'),
                    'dcc': float('nan'), 'dcc_success': float('nan'),
                    'f1_c0': float('nan'), 'ap_c0': float('nan'),
                    'iou_c0': float('nan'), 'mcc_c0': float('nan'),
                })

            sample_results.append(row)

    # ---- Save & summarize ----
    _save_csv(sample_results, out_dir / 'all_results.csv')

    if use_labels and sample_results:
        agg = aggregate_metrics(sample_results)
        print_benchmark_summary(agg)

        with open(out_dir / 'benchmark_summary.json', 'w') as f:
            json.dump(agg, f, indent=2)

        # Classify by avg(f1_c0, ap_c0, iou_c0)
        good, medium, hard = [], [], []
        for r in sample_results:
            f1  = r.get('f1_c0',  float('nan'))
            ap  = r.get('ap_c0',  float('nan'))
            iou = r.get('iou_c0', float('nan'))
            if any(math.isnan(v) for v in [f1, ap, iou]):
                continue
            avg = (f1 + ap + iou) / 3.0
            (good if avg >= 0.4 else medium if avg >= 0.2 else hard).append(r)

        _save_csv(good,   out_dir / 'good_cases.csv')
        _save_csv(medium, out_dir / 'medium_cases.csv')
        _save_csv(hard,   out_dir / 'hard_cases.csv')

        total = len(sample_results)
        print(f"优秀 (avg>=0.4): {len(good):4d} ({len(good)/total*100:.1f}%)")
        print(f"一般 (0.2-0.4): {len(medium):4d} ({len(medium)/total*100:.1f}%)")
        print(f"难例 (<0.2):    {len(hard):4d}  ({len(hard)/total*100:.1f}%)")

    print(f"\n结果已保存到: {out_dir}")
    print("完成!")


if __name__ == '__main__':
    main()
