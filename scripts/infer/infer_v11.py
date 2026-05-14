#!/usr/bin/env python3
"""
推理脚本 for Pocket-GVP-Fusion V11 / V11m1 / V11m2

支持 HOLO4K 和 COACH420 兩個 benchmark，通過 --set 旗標切換。
模型版本（V11 / V11m1 / V11m2）從 checkpoint 自動偵測。

目錄結構（兩個 benchmark 相同）:
    {benchmark_root}/
        protein/   {pdb_id}.pt          (v4 protein features)
        esm2/      {pdb_id}.pt          (ESM-2 embeddings)
        ligand/    {pdb_id}.pt  或  {pdb_id}_{lig}_{num}.pt
        residue_ligand_distances_dcenter.pt

標籤定義 (d_min):
    Contact (c0): < 5.0 Å
    Near    (c1): 5.0 – 8.0 Å
    Far     (c2): > 8.0 Å

配體消融模式 (--ligand_mode):
    real    — 正常推理（default）
    empty   — 0 個原子；global token 退化為全零向量
              → 測試「純蛋白質預測」下限
    zero    — 保留 graph 拓撲，但所有原子特徵/座標歸零
              → 去掉化學/3D 信息，保留分子大小/連通性先驗
    random  — 隨機 N(0,1) 特徵 + 隨機座標
              → 測試模型對亂碼配體的魯棒性
    shuffle — 將每個蛋白質的配體替換為另一個隨機蛋白質的配體
              → 同分佈但配對錯誤；若結果與 real 相近則模型忽略配體

使用方法:
    # 正常推理（HOLO4K）
    conda run -n AIAA python scripts/infer/infer_v11.py \\
        --checkpoint outputs/pocket_gvp_fusion_v11m2_xxx/best_model.pt \\
        --set holo4k --output results/holo4k_real/

    # 配體消融：0 原子
    conda run -n AIAA python scripts/infer/infer_v11.py \\
        --checkpoint outputs/.../best_model.pt \\
        --set holo4k --ligand_mode empty --output results/holo4k_empty/

    # 一次性跑所有消融模式（for 迴圈）
    for mode in real empty zero random shuffle; do
        conda run -n AIAA python scripts/infer/infer_v11.py \\
            --checkpoint outputs/.../best_model.pt \\
            --set coach420 --ligand_mode $mode \\
            --output results/ablation_$mode/
    done
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11

# ---------------------------------------------------------------------------
# Default benchmark roots
# ---------------------------------------------------------------------------

BENCHMARK_ROOTS = {
    'holo4k':   Path('/home/husrcf/Code/ProtBind/holo4k'),
    'coach420': Path('/home/husrcf/Code/ProtBind/coach420'),
}

CLASS_NAMES  = {0: 'Contact', 1: 'Near', 2: 'Far'}
LABEL_SCHEME = "d_center (Contact<5Å, Near 5-8Å, Far>8Å)"

LABEL_CONTACT_THRESH = 5.0
LABEL_NEAR_THRESH    = 8.0

MAX_SEQ_LEN = 1800
MAX_ATOMS   = 128


# ---------------------------------------------------------------------------
# Unified benchmark dataset (HOLO4K + COACH420)
# ---------------------------------------------------------------------------

LIGAND_MODES = ('real', 'empty', 'zero', 'random', 'shuffle')

# Canonical ligand atom feature dimension (ligand_features_v5)
LIGAND_FEAT_DIM = 109


class BenchmarkDataset(Dataset):
    """
    通用 benchmark dataset，支援 HOLO4K 和 COACH420。

    目錄結構:
        root/protein/{pdb_id}.pt
        root/esm2/{pdb_id}.pt
        root/ligand/{pdb_id}.pt  （或 {pdb_id}_{lig}_{num}.pt）
        root/residue_ligand_distances_dcenter.pt

    ligand_mode:
        'real'    — 正常，使用真實配體特徵
        'empty'   — 0 個原子，global token 退化為全零
        'zero'    — 保留拓撲，所有特徵/座標歸零
        'random'  — 隨機 N(0,1) 特徵 + 隨機座標
        'shuffle' — 配體來自另一隨機蛋白質（配對錯誤）
    """

    def __init__(
        self,
        root: Path,
        pair_keys: List[str],
        label_contact_thresh: float = LABEL_CONTACT_THRESH,
        label_near_thresh: float = LABEL_NEAR_THRESH,
        use_labels: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
        max_atoms: int = MAX_ATOMS,
        ligand_mode: str = 'real',
        shuffle_seed: int = 42,
    ):
        assert ligand_mode in LIGAND_MODES, f"ligand_mode 必須是 {LIGAND_MODES} 之一"
        self.root        = Path(root)
        self.keys        = pair_keys
        self.ct          = label_contact_thresh
        self.nt          = label_near_thresh
        self.use_labels  = use_labels
        self.max_seq_len = max_seq_len
        self.max_atoms   = max_atoms
        self.ligand_mode = ligand_mode

        self.protein_dir = self.root / 'protein'
        self.esm_dir     = self.root / 'esm2'
        self.ligand_dir  = self.root / 'ligand'

        dist_path = self.root / 'residue_ligand_distances_dcenter.pt'
        if use_labels and dist_path.exists():
            self.dist_dict = torch.load(dist_path, weights_only=False)
        else:
            self.dist_dict = {}

        # shuffle mode: pre-compute a fixed permutation of keys so that
        # protein[i] is paired with ligand of keys[shuffled_idx[i]] ≠ keys[i]
        if ligand_mode == 'shuffle':
            rng = np.random.default_rng(shuffle_seed)
            n   = len(pair_keys)
            perm = rng.permutation(n)
            # Ensure no sample is paired with itself
            for i in range(n):
                if perm[i] == i:
                    swap = (i + 1) % n
                    perm[i], perm[swap] = perm[swap], perm[i]
            self._shuffle_idx = perm.tolist()
            print(f"[BenchmarkDataset] ligand_mode=shuffle  seed={shuffle_seed}  "
                  f"self-pair check: {sum(perm[i]==i for i in range(n))} self-pairs")
        else:
            self._shuffle_idx = None

    def __len__(self) -> int:
        return len(self.keys)

    # ---- helpers ----

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

    def _merge_ligand_data(self, pdb_id: str) -> Dict:
        ligand_files = self._find_ligand_files(pdb_id)
        if not ligand_files:
            raise FileNotFoundError(f'找不到 {pdb_id} 的 ligand 特征文件')

        feat_list, coord_list, ei_list, ea_list = [], [], [], []
        atom_offset = 0

        for path in ligand_files:
            lig = torch.load(path, weights_only=False)
            feats  = lig['atom_features'].float()
            coords = lig['coords'].float()
            feat_list.append(feats)
            coord_list.append(coords)

            ei = lig.get('edge_index')
            ea = lig.get('edge_attr')
            if ei is not None and ea is not None and ei.numel() > 0:
                ei_list.append(ei.long() + atom_offset)
                ea_list.append(ea.float())
            atom_offset += feats.shape[0]

        atom_features = torch.cat(feat_list,  dim=0)
        coords        = torch.cat(coord_list, dim=0)
        if ei_list:
            edge_index = torch.cat(ei_list, dim=1)
            edge_attr  = torch.cat(ea_list, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 6), dtype=torch.float32)

        return {
            'atom_features': atom_features,
            'coords':        coords,
            'edge_index':    edge_index,
            'edge_attr':     edge_attr,
        }

    # ---- ligand ablation helpers ----

    def _ablate_ligand(
        self,
        lig_feats: torch.Tensor,
        lig_coords: torch.Tensor,
        lig_edge_index: torch.Tensor,
        lig_edge_attr: torch.Tensor,
    ):
        """
        Apply ligand_mode ablation to real ligand tensors.

        Returns (lig_feats, lig_coords, lig_edge_index, lig_edge_attr).
        Call AFTER loading real data but BEFORE size truncation.
        """
        mode = self.ligand_mode
        n    = lig_feats.shape[0]   # number of real atoms
        d    = lig_feats.shape[1]   # atom feature dim

        if mode == 'real':
            pass  # no change

        elif mode == 'empty':
            # Discard all atoms → model sees an empty ligand.
            # The dynamic global token (masked mean pool) collapses to zeros.
            lig_feats      = torch.zeros((0, d),    dtype=torch.float32)
            lig_coords     = torch.zeros((0, 3),    dtype=torch.float32)
            lig_edge_index = torch.zeros((2, 0),    dtype=torch.long)
            lig_edge_attr  = torch.zeros((0, 6),    dtype=torch.float32)

        elif mode == 'zero':
            # Keep graph topology; zero out all features and 3D coords.
            lig_feats  = torch.zeros_like(lig_feats)
            lig_coords = torch.zeros_like(lig_coords)
            # edge_attr contains chemical bond features → also zero out
            lig_edge_attr = torch.zeros_like(lig_edge_attr)

        elif mode == 'random':
            # Replace features with N(0,1) noise; coords from N(0,10) Å
            lig_feats  = torch.randn(n, d)
            lig_coords = torch.randn(n, 3) * 10.0
            lig_edge_attr = torch.randn_like(lig_edge_attr)

        # 'shuffle' is handled in __getitem__ by loading a different ligand file.
        return lig_feats, lig_coords, lig_edge_index, lig_edge_attr

    # ---- __getitem__ ----

    def __getitem__(self, idx: int) -> Dict:
        key    = self.keys[idx]
        pdb_id = key

        prot_data = torch.load(self.protein_dir / f'{pdb_id}.pt', weights_only=False)
        esm_data  = torch.load(self.esm_dir     / f'{pdb_id}.pt', weights_only=False)

        # For 'shuffle' mode: load ligand from a different (permuted) protein
        if self.ligand_mode == 'shuffle':
            src_key = self.keys[self._shuffle_idx[idx]]
            lig_data = self._merge_ligand_data(src_key)
        else:
            lig_data = self._merge_ligand_data(pdb_id)

        # Protein features
        protein_physchem       = prot_data['physchem_features']       # [N, 62]
        protein_spatial_scalar = prot_data['spatial_scalar_features'] # [N, 89]
        protein_spatial_vector = prot_data['spatial_vector_features'] # [N, 8, 3]
        ca_coords              = prot_data['ca_coords']                # [N, 3]
        gvp_vectors = self._compute_gvp_vectors(
            prot_data['all_atom_coords'], ca_coords
        )  # [N, 4, 3]

        esm = esm_data['embeddings']  # [N, 1280]

        # Chain IDs → integer indices
        raw_chain_ids = prot_data.get('chain_ids', None)
        if raw_chain_ids is not None and len(raw_chain_ids) > 0:
            unique_chains = sorted(set(raw_chain_ids))
            c2i = {c: i for i, c in enumerate(unique_chains)}
            chain_ids_int = torch.tensor(
                [c2i[c] for c in raw_chain_ids], dtype=torch.long
            )
        else:
            chain_ids_int = torch.zeros(protein_physchem.shape[0], dtype=torch.long)

        # Raw ligand tensors
        lig_feats      = lig_data['atom_features']
        lig_coords     = lig_data['coords']
        lig_edge_index = lig_data['edge_index']
        lig_edge_attr  = lig_data['edge_attr']

        # ── Apply ligand ablation (before truncation) ──────────────────────
        lig_feats, lig_coords, lig_edge_index, lig_edge_attr = self._ablate_ligand(
            lig_feats, lig_coords, lig_edge_index, lig_edge_attr
        )

        n_residues = protein_physchem.shape[0]
        n_atoms    = lig_feats.shape[0]

        # Labels (always from correct protein, regardless of ligand_mode)
        if self.use_labels and key in self.dist_dict:
            dist_data = self.dist_dict[key]
            distances = dist_data['distances'] if isinstance(dist_data, dict) else dist_data
            labels = self._distances_to_labels(distances)
            if len(labels) < n_residues:
                pad    = torch.full((n_residues - len(labels),), 2, dtype=torch.long)
                labels = torch.cat([labels, pad])
            else:
                labels = labels[:n_residues]
        else:
            labels = torch.full((n_residues,), 2, dtype=torch.long)

        # Truncate protein
        if n_residues > self.max_seq_len:
            protein_physchem       = protein_physchem[:self.max_seq_len]
            protein_spatial_scalar = protein_spatial_scalar[:self.max_seq_len]
            protein_spatial_vector = protein_spatial_vector[:self.max_seq_len]
            ca_coords              = ca_coords[:self.max_seq_len]
            gvp_vectors            = gvp_vectors[:self.max_seq_len]
            esm                    = esm[:self.max_seq_len]
            labels                 = labels[:self.max_seq_len]
            chain_ids_int          = chain_ids_int[:self.max_seq_len]
            n_residues             = self.max_seq_len

        # Truncate ligand
        if n_atoms > self.max_atoms:
            lig_feats  = lig_feats[:self.max_atoms]
            lig_coords = lig_coords[:self.max_atoms]
            n_atoms    = self.max_atoms

        # Sanitize edge indices after truncation
        if lig_edge_index.numel() > 0:
            edge_mask      = (lig_edge_index[0] < n_atoms) & (lig_edge_index[1] < n_atoms)
            lig_edge_index = lig_edge_index[:, edge_mask]
            lig_edge_attr  = lig_edge_attr[edge_mask]

        return {
            'pdb_id':                key,
            'esm':                   esm.float(),
            'protein_physchem':      protein_physchem.float(),
            'protein_spatial_scalar':protein_spatial_scalar.float(),
            'protein_spatial_vector':protein_spatial_vector.float(),
            'gvp_coords':            ca_coords.float(),
            'gvp_vectors':           gvp_vectors.float(),
            'chain_ids':             chain_ids_int,
            'ligand_features':       lig_feats.float(),
            'ligand_coords':         lig_coords.float(),
            'ligand_edge_index':     lig_edge_index,
            'ligand_edge_attr':      lig_edge_attr,
            'labels':                labels.long(),
            'n_residues':            n_residues,
            'n_atoms':               n_atoms,
        }


# ---------------------------------------------------------------------------
# Checkpoint / model — auto-detect V11 / V11m1 / V11m2
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device):
    print(f"加載 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('config') is None:
        raise ValueError("checkpoint 中未找到 config")
    return ckpt


def _detect_version(ckpt: Dict) -> str:
    """
    從 checkpoint state_dict 和 config 自動偵測模型版本。

    偵測邏輯:
      - lig_gate_proj.weight in state   → V11m1  (V11m1 新增配體 gate)
      - ligand_self_attn.dist_bias_modules.* in state
        且 config exp name 含 v11m2      → V11m2
      - 否則                             → V11
    """
    state   = ckpt['model_state_dict']
    cfg     = ckpt.get('config', {})
    exp_name = cfg.get('experiment', {}).get('name', '').lower()

    if 'lig_gate_proj.weight' in state:
        return 'v11m1'
    if any(k.startswith('ligand_self_attn.dist_bias_modules') for k in state):
        return 'v11m2'
    if 'v11m2' in exp_name:
        return 'v11m2'
    if 'v11m1' in exp_name:
        return 'v11m1'
    return 'v11'


def create_model(ckpt: Dict, device: torch.device):
    config       = ckpt['config']
    model_config = dict(config['model'])
    version      = _detect_version(ckpt)

    if version == 'v11m2':
        from src.models.pocket_gvp_fusion_v11m2 import PocketGVPFusionV11m2 as ModelClass
    elif version == 'v11m1':
        from src.models.pocket_gvp_fusion_v11m1 import PocketGVPFusionV11m1 as ModelClass
    else:
        from src.models.pocket_gvp_fusion_v11 import PocketGVPFusionV11 as ModelClass

    print(f"偵測到模型版本: {version.upper()}  →  {ModelClass.__name__}")

    model = ModelClass(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數: {n_params:,}")
    return model, version


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
    ca_coords: np.ndarray,
    dcc_thresh: float = 4.0,
) -> Dict:
    """
    Per-sample 指標：Precision / Recall / F1 / IoU / MCC / AP / DCC。

    DCC (Distance to Closest Center):
        pred_center = mean CA of predicted Contact residues
                      (若無預測 Contact，取最高 Contact 機率的殘基)
        true_center = mean CA of true Contact residues
        dcc_success = dcc < dcc_thresh (預設 4 Å)
    """
    m: Dict = {'accuracy': float((preds == labels).mean())}

    for c in range(3):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        tn = int(((preds != c) & (labels != c)).sum())
        pr  = tp / (tp + fp + 1e-8)
        rc  = tp / (tp + fn + 1e-8)
        f1  = 2 * pr * rc / (pr + rc + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0.0
        m[f'precision_c{c}'] = float(pr)
        m[f'recall_c{c}']    = float(rc)
        m[f'f1_c{c}']        = float(f1)
        m[f'dice_c{c}']      = float(f1)
        m[f'iou_c{c}']       = float(iou)
        m[f'mcc_c{c}']       = float(mcc)
        if c == 0:
            m['ap_c0'] = _ap(probs[:, 0], (labels == 0).astype(int))

    # Binary binding F1: (c0+c1) vs c2
    true_b = (labels <= 1)
    pred_b = (preds  <= 1)
    btp    = int(( pred_b &  true_b).sum())
    bfp    = int(( pred_b & ~true_b).sum())
    bfn    = int((~pred_b &  true_b).sum())
    bpr    = btp / (btp + bfp + 1e-8)
    brc    = btp / (btp + bfn + 1e-8)
    m['f1_binary'] = float(2 * bpr * brc / (bpr + brc + 1e-8))

    # DCC
    true_contact = (labels == 0)
    pred_contact = (preds  == 0)
    if true_contact.sum() == 0:
        m['dcc'] = float('nan')
        m['dcc_success'] = float('nan')
    else:
        true_center = ca_coords[true_contact].mean(axis=0)
        if pred_contact.sum() > 0:
            pred_center = ca_coords[pred_contact].mean(axis=0)
        else:
            top1 = int(np.argmax(probs[:, 0]))
            pred_center = ca_coords[top1]
        dcc = float(np.linalg.norm(pred_center - true_center))
        m['dcc']         = dcc
        m['dcc_success'] = float(dcc < dcc_thresh)

    return m


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Dataset-level 指標聚合（nanmean）。"""
    n = len(results)
    if n == 0:
        return {}

    def _nm(key):
        vals = [r[key] for r in results
                if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    agg = {
        'N_evaluated':    n,
        'DCC_Top1':       _nm('dcc'),
        'DCC_Success':    _nm('dcc_success'),
        'Mean_IoU':       _nm('iou_c0'),
        'Mean_Prec':      _nm('precision_c0'),
        'Mean_Rec':       _nm('recall_c0'),
        'Mean_F1':        _nm('f1_c0'),
        'Mean_Dice':      _nm('dice_c0'),
        'Mean_MCC':       _nm('mcc_c0'),
        'Mean_AP':        _nm('ap_c0'),
        'Mean_F1_binary': _nm('f1_binary'),
    }

    ap_filt = [r['ap_c0'] for r in results
               if 'ap_c0' in r and not (isinstance(r['ap_c0'], float) and math.isnan(r['ap_c0']))]
    agg['Mean_AP_filt'] = float(np.mean(ap_filt)) if ap_filt else float('nan')

    return agg


def print_summary(agg: Dict, benchmark: str, version: str):
    w = 65
    print(f"\n{'='*w}")
    print(f"  {benchmark.upper()} Benchmark  —  V11 Family: {version.upper()}  (N={agg.get('N_evaluated','?')})")
    print(f"{'='*w}")
    print(f"  {'DCC Top-1 (Å)':<22s}  {agg.get('DCC_Top1', float('nan')):.4f}")
    print(f"  {'DCC Success (<4Å)':<22s}  {agg.get('DCC_Success', float('nan')):.4f}")
    print(f"  {'-'*52}")
    print(f"  {'Mean IoU':<22s}  {agg.get('Mean_IoU',  float('nan')):.4f}")
    print(f"  {'Mean Precision':<22s}  {agg.get('Mean_Prec', float('nan')):.4f}")
    print(f"  {'Mean Recall':<22s}  {agg.get('Mean_Rec',  float('nan')):.4f}")
    print(f"  {'Mean F1 (Contact)':<22s}  {agg.get('Mean_F1',   float('nan')):.4f}")
    print(f"  {'Mean F1 (Binary)':<22s}  {agg.get('Mean_F1_binary', float('nan')):.4f}  [c0+c1 vs c2]")
    print(f"  {'Mean Dice':<22s}  {agg.get('Mean_Dice', float('nan')):.4f}")
    print(f"  {'Mean MCC':<22s}  {agg.get('Mean_MCC',  float('nan')):.4f}")
    print(f"  {'Mean AP':<22s}  {agg.get('Mean_AP',   float('nan')):.4f}")
    print(f"  {'Mean AP (filt)':<22s}  {agg.get('Mean_AP_filt', float('nan')):.4f}")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model, batch: Dict, device: torch.device, use_amp: bool = False):
    inputs = prepare_model_inputs_v11(batch, device, non_blocking=False)
    # autocast device_type must match actual device; AMP is only meaningful on CUDA/XPU
    _amp_enabled = use_amp and device.type != 'cpu'
    with torch.amp.autocast(device.type, enabled=_amp_enabled):
        outputs = model(
            esm_embeddings=inputs['esm_embeddings'],
            protein_physchem=inputs['protein_physchem'],
            protein_spatial_scalar=inputs['protein_spatial_scalar'],
            protein_spatial_vector=inputs['protein_spatial_vector'],
            protein_gvp_vectors=inputs['protein_gvp_vectors'],
            protein_coords=inputs['protein_coords'],
            protein_mask=inputs['protein_mask'],
            chain_ids=inputs.get('chain_ids'),
            ligand_x=inputs['ligand_x'],
            ligand_pos=inputs['ligand_pos'],
            ligand_edge_index=inputs['ligand_edge_index'],
            ligand_edge_attr=inputs['ligand_edge_attr'],
            ligand_batch=inputs['ligand_batch'],
            ligand_mask=inputs['ligand_mask'],
        )
    logits = outputs['logits'].cpu()
    probs  = F.softmax(logits, dim=-1)
    preds  = logits.argmax(dim=-1)
    return probs, preds


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'pdb_id', 'n_residues', 'accuracy',
    'dcc', 'dcc_success',
    'f1_c0', 'dice_c0', 'ap_c0', 'iou_c0', 'precision_c0', 'recall_c0', 'mcc_c0',
    'f1_c1', 'iou_c1', 'precision_c1', 'recall_c1',
    'f1_c2', 'iou_c2', 'precision_c2', 'recall_c2',
    'f1_binary',
    'n_contact', 'n_near', 'n_far',
]


def _save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='V11 系列推理腳本 (HOLO4K / COACH420)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint',    required=True,
                        help='checkpoint 路徑 (best_model.pt)')
    parser.add_argument('--set',           required=True,
                        choices=['holo4k', 'coach420'],
                        help='benchmark 數據集選擇')
    parser.add_argument('--root',          default=None,
                        help='benchmark 根目錄（預設自動使用内建路徑）')
    parser.add_argument('--output',        default=None,
                        help='輸出目錄（預設 results/{set}_{version}_{ts}）')
    parser.add_argument('--batch_size',    type=int,   default=4)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--amp',           action='store_true',
                        help='使用 AMP（cuda only，cpu 下自動忽略）')
    parser.add_argument('--cpu',           action='store_true',
                        help='強制使用 CPU 推理（即使 CUDA 可用）')
    parser.add_argument('--no_labels',     action='store_true',
                        help='無標籤純推理模式（跳過指標計算）')
    parser.add_argument('--contact_thresh',type=float, default=LABEL_CONTACT_THRESH,
                        help='Contact 標籤閾值 (Å)')
    parser.add_argument('--near_thresh',   type=float, default=LABEL_NEAR_THRESH,
                        help='Near 標籤閾值 (Å)')
    parser.add_argument('--dcc_thresh',    type=float, default=4.0,
                        help='DCC success 距離閾值 (Å)')
    parser.add_argument('--max_seq_len',   type=int,   default=MAX_SEQ_LEN)
    parser.add_argument('--max_atoms',     type=int,   default=MAX_ATOMS)
    parser.add_argument('--ligand_mode',   default='real', choices=list(LIGAND_MODES),
                        help=(
                            '配體消融模式: '
                            'real=正常; '
                            'empty=0原子(純蛋白質基準); '
                            'zero=特徵歸零; '
                            'random=隨機雜訊; '
                            'shuffle=配對錯誤'
                        ))
    parser.add_argument('--shuffle_seed',  type=int,   default=42,
                        help='shuffle 模式的隨機種子')
    args = parser.parse_args()

    # Device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備: {device}")

    # Benchmark root
    benchmark_root = Path(args.root) if args.root else BENCHMARK_ROOTS[args.set]
    if not benchmark_root.exists():
        print(f"[錯誤] benchmark 根目錄不存在: {benchmark_root}")
        sys.exit(1)
    print(f"Benchmark: {args.set.upper()}  →  {benchmark_root}")

    # Load checkpoint & model
    ckpt              = load_checkpoint(args.checkpoint, device)
    model, version    = create_model(ckpt, device)

    # Build sample keys from distance file
    dist_path = benchmark_root / 'residue_ligand_distances_dcenter.pt'
    if not dist_path.exists():
        print(f"[錯誤] 找不到距離標籤文件: {dist_path}")
        sys.exit(1)
    dist_dict = torch.load(dist_path, weights_only=False)
    all_keys  = sorted(dist_dict.keys())
    print(f"{args.set.upper()} 總樣本數: {len(all_keys)}")

    # Dataset + DataLoader
    use_labels  = not args.no_labels
    ligand_mode = args.ligand_mode
    if ligand_mode != 'real':
        print(f"[消融] ligand_mode = {ligand_mode.upper()}")

    dataset = BenchmarkDataset(
        root=benchmark_root,
        pair_keys=all_keys,
        label_contact_thresh=args.contact_thresh,
        label_near_thresh=args.near_thresh,
        use_labels=use_labels,
        max_seq_len=args.max_seq_len,
        max_atoms=args.max_atoms,
        ligand_mode=ligand_mode,
        shuffle_seed=args.shuffle_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=(device.type == 'cuda'),
    )

    # Output directory — include ligand_mode in default name
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = (
        Path(args.output)
        if args.output
        else Path('results') / f'{args.set}_{version}_lig{ligand_mode}_{ts}'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"結果將儲存至: {out_dir}")

    # ── Inference loop ────────────────────────────────────────────────────────
    sample_results: List[Dict] = []

    for batch in tqdm(loader, desc=f'{args.set.upper()} 推理', unit='batch'):
        labels_b = batch['labels']       # [B, L]
        mask_b   = batch['protein_mask'] # [B, L]
        coords_b = batch.get('protein_coords')  # [B, L, 3]
        pdb_ids  = batch.get('pdb_ids', [])

        probs, preds = predict_batch(model, batch, device, args.amp)

        for b in range(mask_b.shape[0]):
            valid    = mask_b[b].bool()
            preds_np = preds[b][valid].numpy()
            probs_np = probs[b][valid].numpy()
            labs_np  = labels_b[b][valid].numpy()
            key      = pdb_ids[b] if b < len(pdb_ids) else f'sample_{b}'

            ca_np = (
                coords_b[b][valid].numpy()
                if coords_b is not None
                else np.zeros((int(valid.sum()), 3))
            )

            row: Dict = {
                'pdb_id':     key,
                'n_residues': int(valid.sum()),
                'n_contact':  int((preds_np == 0).sum()),
                'n_near':     int((preds_np == 1).sum()),
                'n_far':      int((preds_np == 2).sum()),
                'n_true_contact': int((labs_np == 0).sum()),
            }

            if use_labels:
                row.update(
                    compute_sample_metrics(
                        preds_np, labs_np, probs_np, ca_np,
                        dcc_thresh=args.dcc_thresh,
                    )
                )
            else:
                row.update({'accuracy': float('nan'), 'dcc': float('nan'),
                            'dcc_success': float('nan'), 'f1_c0': float('nan'),
                            'ap_c0': float('nan'), 'iou_c0': float('nan'),
                            'mcc_c0': float('nan')})

            sample_results.append(row)

    # ── Save & summarize ─────────────────────────────────────────────────────
    _save_csv(sample_results, out_dir / 'all_results.csv')

    if use_labels and sample_results:
        agg = aggregate_metrics(sample_results)
        print_summary(agg, benchmark=args.set, version=version)

        with open(out_dir / 'benchmark_summary.json', 'w') as f:
            json.dump(agg, f, indent=2)

        # Difficulty tiering by avg(f1_c0, ap_c0, iou_c0)
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
        print(f"優秀 (avg>=0.4): {len(good):4d}  ({len(good)/total*100:.1f}%)")
        print(f"一般 (0.2-0.4): {len(medium):4d}  ({len(medium)/total*100:.1f}%)")
        print(f"難例 (<0.2):    {len(hard):4d}  ({len(hard)/total*100:.1f}%)")

    # Save metadata
    meta = {
        'benchmark':       args.set,
        'model_version':   version,
        'checkpoint':      str(args.checkpoint),
        'benchmark_root':  str(benchmark_root),
        'ligand_mode':     ligand_mode,
        'shuffle_seed':    args.shuffle_seed if ligand_mode == 'shuffle' else None,
        'n_samples':       len(sample_results),
        'contact_thresh':  args.contact_thresh,
        'near_thresh':     args.near_thresh,
        'dcc_thresh':      args.dcc_thresh,
        'timestamp':       ts,
    }
    with open(out_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n全部結果已儲存至: {out_dir}")
    print("完成！")


if __name__ == '__main__':
    main()
