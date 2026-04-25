#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip Dataset Phase 1 Quality Control Scanner
===============================================

掃描 BioLip 資料集中每個 bundle 的品質，不做任何修改，只統計。

檢查項目：
  [Protein]
    - 實際殘基數 vs protein_meta.json 中的 sequence_length
    - 缺失殘基率 (missing_ratio)
    - 非標準殘基種類
    - Backbone 不完整殘基數 (缺 N/CA/C/O 任一)
    - 鏈斷裂 (CA-CA > 5 Å)
    - 是否含 H 原子

  [Ligand]
    - Heavy atom 數 vs pair_meta.json 中的 ligand_heavy_atom_count
    - 是否含 H 原子
    - 3D 座標是否有效 (Z 不全為 0)

輸出：
    {output_dir}/qc_protein.csv       — 每個 bundle 一行
    {output_dir}/qc_ligand.csv        — 每個 pair 一行
    {output_dir}/qc_summary.json      — 整體統計
    {output_dir}/recommended_discard.txt  — 建議捨棄的 bundle

使用方式:
    python preparation/biolip_qc_phase1.py \\
        --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
        --output /home/husrcf/Code/ProtBind/BioLip/DOCs/qc_phase1 \\
        --workers 16

    # 快速測試前 100 個 bundle
    python preparation/biolip_qc_phase1.py \\
        --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
        --output /tmp/qc_test \\
        --max_bundles 100 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 常數 ─────────────────────────────────────────────────────────────────────

STD_AA = frozenset(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE '
    'LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
)
BACKBONE_ATOMS = frozenset(['N', 'CA', 'C', 'O'])
CA_CA_BREAK_THRESH = 5.0   # Å，超過此距離視為鏈斷裂
MISSING_RATIO_DISCARD = 0.20  # 缺失 > 20% → 建議捨棄
HEAVY_COUNT_TOL = 0          # ligand heavy atom 容差 (0 = 必須精確吻合)


# ── Protein 解析 (gemmi) ──────────────────────────────────────────────────────

def _parse_protein_chain(cif_path: Path, chain_id: str) -> Dict:
    """
    用 gemmi 解析指定 chain 的蛋白質品質指標。
    回傳 dict，若發生例外則含 'error' key。
    """
    try:
        import gemmi
    except ImportError:
        return {'error': 'gemmi not available'}

    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as e:
        return {'error': f'read_structure failed: {e}'}

    model = st[0]
    try:
        chain = model[chain_id]
    except Exception:
        # 有些 CIF 的 chain ID 不符，嘗試第一個 polymer chain
        available = [c.name for c in model]
        return {'error': f'chain {chain_id!r} not found; available={available}'}

    residues = list(chain)
    if not residues:
        return {'error': 'empty chain'}

    # polymer_residues：所有含 CA 的殘基（含 modified AA，如 MSE/TPO/SEP）
    # 用於 CA 連續性和骨架完整性，避免把已修飾殘基誤算為缺失。
    polymer_residues = [r for r in residues if r.find_atom('CA', '*') is not None]

    std_residues  = [r for r in residues if r.name in STD_AA]
    nonstd_res    = [r for r in residues if r.name not in STD_AA and
                     r.find_atom('CA', '*') is not None]   # 只計有 CA 的非標準殘基
    nonstd_types  = sorted(set(r.name for r in nonstd_res))

    # ── backbone 完整性（在所有 polymer 殘基上檢查）─────────────────────────
    n_backbone_incomplete = 0
    for r in polymer_residues:
        atom_names = {a.name for a in r}
        if not BACKBONE_ATOMS.issubset(atom_names):
            n_backbone_incomplete += 1

    # ── CA-CA 鏈斷裂（在所有 polymer 殘基上檢查）─────────────────────────
    # 只有在 seqid 連續（差值 ≤ 3）且 CA-CA > 5Å 才算真正的結構斷裂。
    # seqid 差值 > 3 表示是已知的缺失區段（missing loop），由 missing_ratio 涵蓋。
    #
    # 插入碼處理：使用 r.seqid.num（gemmi 原生 int）避免 "100A" 解析失敗。
    # 若相鄰兩殘基中任一帶有插入碼（icode != ' '），代表它們在結構上是
    # 連續插入殘基，視為 seqid_gap=1。
    ca_list: List[Tuple] = []
    for r in polymer_residues:
        ca = r.find_atom('CA', '*')
        if ca:
            seqnum = r.seqid.num                    # 純整數，不含插入碼
            icode  = r.seqid.icode.strip()          # '' 或 'A'/'B'/...
            ca_list.append((seqnum, icode, ca.pos))

    chain_breaks = 0
    for i in range(1, len(ca_list)):
        sn_prev, ic_prev, pos_prev = ca_list[i - 1]
        sn_curr, ic_curr, pos_curr = ca_list[i]
        d = pos_curr.dist(pos_prev)

        if d <= CA_CA_BREAK_THRESH:
            continue

        if sn_prev == sn_curr:
            # 同一 base seqnum 的不同插入字母（100A→100B）→ 連續，gap=1
            seqid_gap = 1
        elif not ic_prev and ic_curr:
            # 進入插入區塊（100→100A）→ 同位置，視為 gap=1
            seqid_gap = 1
        else:
            # 其他情況（含 100A→102）→ 用實際 seqnum 差值
            seqid_gap = sn_curr - sn_prev

        # 只有真正相鄰（gap == 1）且 CA-CA 過大才算結構斷裂。
        # gap 2-3 代表有 1-2 個缺失殘基，CA 距離本就會拉長，
        # 這已由 missing_ratio 涵蓋，不應重複計入 chain_break。
        if seqid_gap == 1:
            chain_breaks += 1

    # ── H 原子檢查 ────────────────────────────────────────────────────────
    try:
        import gemmi as _gemmi
        H_elem = _gemmi.Element('H')
        has_h = any(
            any(a.element == H_elem for a in r)
            for r in residues
        )
    except Exception:
        has_h = False

    return {
        'n_residues_actual':  len(residues),
        'n_polymer_residues': len(polymer_residues),   # CA 齊全（含 modified AA）
        'n_std_residues':     len(std_residues),
        'n_nonstd_residues':  len(nonstd_res),
        'nonstd_types':       ','.join(nonstd_types) if nonstd_types else '',
        'n_backbone_incomplete': n_backbone_incomplete,
        'n_chain_breaks':     chain_breaks,
        'has_h_protein':      int(has_h),
        'error':              '',
    }


# ── Ligand 解析 (純文字解析 SDF，不需 RDKit) ──────────────────────────────────

def _parse_sdf_stats(sdf_path: Path) -> Dict:
    """
    輕量解析 SDF 檔案，回傳 heavy/H atom 數和 3D 有效性。
    支援 V2000 格式（BioLip 通常輸出此格式）。
    """
    try:
        lines = sdf_path.read_text(errors='replace').splitlines()
    except Exception as e:
        return {'error': str(e)}

    # 找 counts line (第 4 行，index 3)
    # 若有多個 molecule block，取第一個
    counts_idx = None
    for i, line in enumerate(lines):
        if 'V2000' in line or 'V3000' in line:
            counts_idx = i
            break

    if counts_idx is None:
        return {'error': 'no V2000/V3000 counts line found'}

    is_v3 = 'V3000' in lines[counts_idx]

    heavy_atoms = 0
    h_atoms     = 0
    z_coords    = []

    if not is_v3:
        # V2000: counts line 格式 "aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv"
        # n_atoms 在前 3 chars
        try:
            n_atoms = int(lines[counts_idx][:3])
        except ValueError:
            return {'error': f'cannot parse n_atoms from: {lines[counts_idx]!r}'}

        atom_start = counts_idx + 1
        for j in range(n_atoms):
            idx = atom_start + j
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if len(parts) < 4:
                continue
            try:
                z_coords.append(float(parts[2]))
            except ValueError:
                pass
            element = parts[3].strip().upper()
            if element == 'H':
                h_atoms += 1
            else:
                heavy_atoms += 1
    else:
        # V3000: 找 BEGIN ATOM block
        in_atom_block = False
        for line in lines:
            stripped = line.strip()
            if 'BEGIN ATOM' in stripped:
                in_atom_block = True
                continue
            if 'END ATOM' in stripped:
                in_atom_block = False
                continue
            if in_atom_block:
                # M  V30 index type x y z aamap
                # cols: 0=M 1=V30 2=index 3=type 4=x 5=y 6=z
                parts = stripped.split()
                if len(parts) < 7:
                    continue
                element = parts[3].strip().upper()
                try:
                    z_coords.append(float(parts[6]))
                except ValueError:
                    pass
                if element == 'H':
                    h_atoms += 1
                else:
                    heavy_atoms += 1

    valid_3d = bool(z_coords) and not all(abs(z) < 1e-6 for z in z_coords)

    return {
        'actual_heavy':  heavy_atoms,
        'h_atoms_lig':   h_atoms,
        'valid_3d':      int(valid_3d),
        'error':         '',
    }


def _check_pair(pair_path: Path, bundle_name: str, pair_dir: str) -> Dict:
    """
    檢查單個 pair 的配體品質。
    """
    base = {
        'bundle_name': bundle_name,
        'pair_dir':    pair_dir,
        'ligand_id':   '',
        'expected_heavy': -1,
        'actual_heavy':   -1,
        'heavy_match':    -1,
        'h_atoms_lig':    -1,
        'valid_3d':       -1,
        'sdf_found':      0,
        'error':          '',
    }

    meta_path = pair_path / 'pair_meta.json'
    if not meta_path.exists():
        base['error'] = 'pair_meta.json missing'
        return base

    try:
        pair_meta = json.loads(meta_path.read_text())
    except Exception as e:
        base['error'] = f'pair_meta.json parse error: {e}'
        return base

    ligand_id      = pair_meta.get('ligand_id', '')
    expected_heavy = pair_meta.get('ligand_heavy_atom_count', -1)
    sdf_file       = pair_meta.get('ligand_pose_sdf_file', '')

    base['ligand_id']       = ligand_id
    base['expected_heavy']  = expected_heavy

    sdf_path = pair_path / sdf_file if sdf_file else None
    if sdf_path is None or not sdf_path.exists():
        # fallback: search for any sdf
        sdfs = list(pair_path.glob('*_ligand.sdf'))
        sdf_path = sdfs[0] if sdfs else None

    if sdf_path is None or not sdf_path.exists():
        base['error'] = 'SDF not found'
        return base

    base['sdf_found'] = 1
    stats = _parse_sdf_stats(sdf_path)

    if stats.get('error'):
        base['error'] = stats['error']
        return base

    actual_heavy = stats['actual_heavy']
    base['actual_heavy']  = actual_heavy
    base['h_atoms_lig']   = stats['h_atoms_lig']
    base['valid_3d']      = stats['valid_3d']
    # -1 = unknown（expected_heavy 缺失），0 = 不符，1 = 符合
    if expected_heavy >= 0:
        base['heavy_match'] = int(abs(actual_heavy - expected_heavy) <= HEAVY_COUNT_TOL)
    # else: 保持初始值 -1（不計入 mismatch 統計）
    return base


# ── 主要 worker 函式 ──────────────────────────────────────────────────────────

def _worker(args: Tuple) -> Tuple[Optional[Dict], List[Dict]]:
    """
    處理一個 bundle。回傳 (protein_row, [ligand_rows])。
    """
    bundle_name, biolip_root_str = args
    biolip_root = Path(biolip_root_str)
    bundle_dir  = biolip_root / 'data' / bundle_name

    prot_row: Dict = {
        'bundle_name':           bundle_name,
        'pdb_id':                '',
        'protein_chain':         '',
        'seq_len_from_seq':      -1,
        'n_residues_actual':     -1,
        'n_polymer_residues':    -1,
        'n_std_residues':        -1,
        'n_nonstd_residues':     -1,
        'nonstd_types':          '',
        'missing_ratio':         float('nan'),
        'n_backbone_incomplete': -1,
        'n_chain_breaks':        -1,
        'has_h_protein':         -1,
        'cif_found':             0,
        'error':                 '',
        'pair_index_missing':    0,   # pair_index.json 檔案不存在
        'pair_index_error':      '',  # pair_index.json 解析/處理失敗
    }

    # ── 讀 protein_meta.json ──────────────────────────────────────────────
    meta_path = bundle_dir / 'protein_meta.json'
    if not meta_path.exists():
        prot_row['error'] = 'protein_meta.json missing'
        return prot_row, []

    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        prot_row['error'] = f'protein_meta.json parse error: {e}'
        return prot_row, []

    pdb_id        = meta.get('pdb_id', '')
    protein_chain = meta.get('protein_chain', '')

    # sequence_length 欄位不可靠（混用 UniProt 全長 / 片段長度 / 錯誤值）
    # 統一以 sequence 字串長度作為唯一參考
    sequence_str     = meta.get('sequence') or ''
    seq_len_from_seq = len(sequence_str) if sequence_str else -1

    prot_row['pdb_id']           = pdb_id
    prot_row['protein_chain']    = protein_chain
    prot_row['seq_len_from_seq'] = seq_len_from_seq

    # ── 找 CIF ───────────────────────────────────────────────────────────
    cif_path = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
    if not cif_path.exists():
        prot_row['error'] = 'receptor_full.cif not found'
    else:
        prot_row['cif_found'] = 1
        stats = _parse_protein_chain(cif_path, protein_chain)

        if stats.get('error'):
            prot_row['error'] = stats['error']
        else:
            prot_row.update({
                'n_residues_actual':     stats['n_residues_actual'],
                'n_polymer_residues':    stats['n_polymer_residues'],
                'n_std_residues':        stats['n_std_residues'],
                'n_nonstd_residues':     stats['n_nonstd_residues'],
                'nonstd_types':          stats['nonstd_types'],
                'n_backbone_incomplete': stats['n_backbone_incomplete'],
                'n_chain_breaks':        stats['n_chain_breaks'],
                'has_h_protein':         stats['has_h_protein'],
            })
            if seq_len_from_seq > 0:
                # 用 n_polymer_residues（含 modified AA）避免把 MSE/TPO 等算成缺失
                actual_polymer = stats['n_polymer_residues']
                prot_row['missing_ratio'] = round(
                    max(0, seq_len_from_seq - actual_polymer) / seq_len_from_seq, 4
                )

    # ── 讀 pair_index.json → 逐 pair 檢查配體 ────────────────────────────
    lig_rows: List[Dict] = []
    pair_idx_path = bundle_dir / 'pair_index.json'
    if not pair_idx_path.exists():
        # 檔案不存在：記旗標，不做 ligand QC（此 bundle 無任何 lig_rows）
        prot_row['pair_index_missing'] = 1
    else:
        try:
            pair_idx = json.loads(pair_idx_path.read_text())
            pair_root_rel = pair_idx.get('pair_root', '')
            # pair_root 為相對於 bundle_dir 的路徑
            pair_root = (bundle_dir / pair_root_rel).resolve()
            for pair_dir in pair_idx.get('pair_dirs', []):
                pair_path = pair_root / pair_dir
                lig_row = _check_pair(pair_path, bundle_name, pair_dir)
                lig_rows.append(lig_row)
        except Exception as e:
            # pair_index 解析/處理失敗，不污染 protein error 欄位
            prot_row['pair_index_error'] = str(e)

    return prot_row, lig_rows


# ── CSV / JSON 輸出 ───────────────────────────────────────────────────────────

PROT_FIELDS = [
    'bundle_name', 'pdb_id', 'protein_chain',
    'seq_len_from_seq', 'n_residues_actual', 'n_polymer_residues',
    'n_std_residues', 'n_nonstd_residues', 'nonstd_types', 'missing_ratio',
    'n_backbone_incomplete', 'n_chain_breaks', 'has_h_protein',
    'cif_found', 'error', 'pair_index_missing', 'pair_index_error',
]

LIG_FIELDS = [
    'bundle_name', 'pair_dir', 'ligand_id',
    'expected_heavy', 'actual_heavy', 'heavy_match',
    'h_atoms_lig', 'valid_3d', 'sdf_found', 'error',
]


def _nan_safe(v):
    if isinstance(v, float) and math.isnan(v):
        return ''
    return v


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BioLip Phase 1 QC Scanner'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip 資料集根目錄')
    parser.add_argument('--output', required=True,
                        help='輸出目錄')
    parser.add_argument('--workers', type=int, default=None,
                        help='並行 worker 數量（預設：CPU 核心數 - 1）')
    parser.add_argument('--maxtasksperchild', type=int, default=None,
                        help='每個 worker 處理幾個 bundle 後重啟以釋放記憶體（預設不限制）')
    parser.add_argument('--max_bundles', type=int, default=None,
                        help='最多處理幾個 bundle（用於快速測試）')
    parser.add_argument('--bundle_list_file', type=str, default=None,
                        help='只掃描此文字檔中列出的 bundle（每行一個名稱），用於補掃缺失 bundle')
    parser.add_argument('--append', action='store_true',
                        help='Append 模式：結果追加到現有 CSV 而非覆蓋（補掃缺失時使用）')
    parser.add_argument('--chunksize', type=int, default=64,
                        help='multiprocessing chunksize（預設 64）')
    parser.add_argument('--missing_ratio_thresh', type=float,
                        default=MISSING_RATIO_DISCARD,
                        help='建議捨棄的缺失殘基率閾值（預設 0.20）')
    args = parser.parse_args()

    biolip_root = Path(args.biolip_root)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = biolip_root / 'data'

    if args.bundle_list_file:
        # 補掃模式：只處理指定清單中的 bundle
        bundle_names = [
            line.strip() for line in Path(args.bundle_list_file).read_text().splitlines()
            if line.strip()
        ]
        print(f"補掃模式：從 {args.bundle_list_file} 讀取 {len(bundle_names)} 個 bundle")
    else:
        bundle_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    if args.max_bundles:
        bundle_names = bundle_names[:args.max_bundles]

    total = len(bundle_names)
    print(f"總 bundle 數: {total:,}")

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"Worker 數量: {n_workers}")
    print(f"輸出目錄: {output_dir}")

    tasks = [(name, str(biolip_root)) for name in bundle_names]

    prot_csv_path = output_dir / 'qc_protein.csv'
    lig_csv_path  = output_dir / 'qc_ligand.csv'

    # 統計變數
    n_done                = 0
    n_prot_error          = 0
    n_cif_missing         = 0
    n_chain_break         = 0
    n_discard             = 0
    n_pair_index_missing  = 0
    n_pair_index_error    = 0
    n_lig_total           = 0
    n_lig_no_h            = 0
    n_lig_mismatch        = 0
    nonstd_counter: Dict[str, int] = {}
    missing_ratio_sum = 0.0
    missing_ratio_cnt = 0
    discard_list: List[str] = []

    t0 = time.time()

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if args.append:
        # Append 模式：直接 append 到現有 CSV（不需要 header）
        prot_out_f = open(prot_csv_path, 'a', newline='')
        lig_out_f  = open(lig_csv_path,  'a', newline='')
        pw = csv.DictWriter(prot_out_f, fieldnames=PROT_FIELDS, extrasaction='ignore')
        lw = csv.DictWriter(lig_out_f,  fieldnames=LIG_FIELDS,  extrasaction='ignore')
        prot_tmp = None
        lig_tmp  = None
    else:
        # 先寫到 .tmp 檔案，掃描全部完成後再 atomic rename，
        # 防止中途 abort 破壞舊的掃描結果。
        prot_tmp = Path(tempfile.mktemp(dir=output_dir, prefix='qc_protein_', suffix='.tmp'))
        lig_tmp  = Path(tempfile.mktemp(dir=output_dir, prefix='qc_ligand_',  suffix='.tmp'))
        prot_out_f = open(prot_tmp, 'w', newline='')
        lig_out_f  = open(lig_tmp,  'w', newline='')
        pw = csv.DictWriter(prot_out_f, fieldnames=PROT_FIELDS, extrasaction='ignore')
        lw = csv.DictWriter(lig_out_f,  fieldnames=LIG_FIELDS,  extrasaction='ignore')
        pw.writeheader()
        lw.writeheader()

    with prot_out_f, lig_out_f:
        ctx = mp.get_context('fork')

        def _run():
            nonlocal n_done, n_prot_error, n_cif_missing, n_chain_break
            nonlocal n_discard, n_pair_index_missing, n_pair_index_error
            nonlocal n_lig_total, n_lig_no_h, n_lig_mismatch
            nonlocal missing_ratio_sum, missing_ratio_cnt

            with ctx.Pool(processes=n_workers,
                          maxtasksperchild=args.maxtasksperchild) as pool:
                it = pool.imap_unordered(
                    _worker, tasks, chunksize=args.chunksize
                )
                if use_tqdm:
                    it = tqdm(it, total=total, desc='QC Scan', unit='bundle',
                              dynamic_ncols=True)

                for prot_row, lig_rows in it:
                    # ── protein 統計 ──────────────────────────────────────
                    n_done += 1

                    if not prot_row['cif_found']:
                        n_cif_missing += 1
                    elif prot_row['error']:
                        # 純 protein/gemmi 解析錯誤（不含 pair_index 失敗）
                        n_prot_error += 1
                    if prot_row.get('pair_index_missing'):
                        n_pair_index_missing += 1
                    if prot_row.get('pair_index_error'):
                        n_pair_index_error += 1
                    if prot_row.get('n_chain_breaks', 0) and prot_row['n_chain_breaks'] > 0:
                        n_chain_break += 1
                    mr = prot_row.get('missing_ratio', float('nan'))
                    if isinstance(mr, float) and not math.isnan(mr):
                        missing_ratio_sum += mr
                        missing_ratio_cnt += 1

                    has_break = bool(prot_row.get('n_chain_breaks', 0) and
                                     prot_row['n_chain_breaks'] > 0)
                    mr_over   = isinstance(mr, float) and not math.isnan(mr) and mr > args.missing_ratio_thresh
                    if mr_over or has_break:
                        n_discard += 1
                        discard_list.append(prot_row['bundle_name'])

                    for t in (prot_row.get('nonstd_types') or '').split(','):
                        t = t.strip()
                        if t:
                            nonstd_counter[t] = nonstd_counter.get(t, 0) + 1

                    # write row (nan → '')
                    pw.writerow({k: _nan_safe(prot_row.get(k, '')) for k in PROT_FIELDS})

                    # ── ligand 統計 ───────────────────────────────────────
                    for lr in lig_rows:
                        n_lig_total += 1
                        if lr.get('h_atoms_lig', -1) == 0:
                            n_lig_no_h += 1
                        if lr.get('heavy_match', -1) == 0:
                            n_lig_mismatch += 1
                        lw.writerow({k: _nan_safe(lr.get(k, '')) for k in LIG_FIELDS})

                    if not use_tqdm and n_done % 5000 == 0:
                        elapsed = time.time() - t0
                        rate = n_done / max(elapsed, 1)
                        eta  = (total - n_done) / max(rate, 1)
                        print(
                            f"[{n_done:>7,}/{total:,}] "
                            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                            f"errors={n_prot_error}  discard={n_discard}",
                            flush=True,
                        )

        _run()

    # ── atomic rename：只有掃描完成才覆蓋正式輸出（非 append 模式）────────
    if prot_tmp is not None:
        prot_tmp.replace(prot_csv_path)
        lig_tmp.replace(lig_csv_path)

    elapsed = time.time() - t0

    # ── 輸出 discard list ──────────────────────────────────────────────────
    discard_path = output_dir / 'recommended_discard.txt'
    discard_path.write_text('\n'.join(sorted(set(discard_list))) + '\n')

    # ── 輸出 summary JSON ──────────────────────────────────────────────────
    top_nonstd = sorted(nonstd_counter.items(), key=lambda x: -x[1])[:30]
    summary = {
        'scan_date':           time.strftime('%Y-%m-%d %H:%M:%S'),
        'biolip_root':         str(biolip_root),
        'total_bundles':       total,
        'bundles_scanned':     n_done,
        'elapsed_seconds':     round(elapsed, 1),
        'protein': {
            'cif_missing':            n_cif_missing,
            'parse_error':            n_prot_error,
            'pair_index_missing':     n_pair_index_missing,
            'pair_index_error':       n_pair_index_error,
            'has_chain_break':        n_chain_break,
            'mean_missing_ratio':     round(missing_ratio_sum / missing_ratio_cnt, 4)
                                      if missing_ratio_cnt > 0 else None,
            'recommended_discard':    n_discard,
            'discard_threshold':      args.missing_ratio_thresh,
            'top_nonstd_residues':    [{'name': n, 'count': c} for n, c in top_nonstd],
        },
        'ligand': {
            'total_pairs':       n_lig_total,
            'no_h_atoms':        n_lig_no_h,
            'heavy_mismatch':    n_lig_mismatch,
            'pct_no_h':          round(n_lig_no_h / max(n_lig_total, 1) * 100, 2),
            'pct_heavy_mismatch':round(n_lig_mismatch / max(n_lig_total, 1) * 100, 2),
        },
    }

    summary_path = output_dir / 'qc_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── 列印摘要 ──────────────────────────────────────────────────────────
    w = 60
    print(f"\n{'='*w}")
    print(f"  BioLip QC Phase 1 完成   ({elapsed:.1f}s)")
    print(f"{'='*w}")
    print(f"  掃描 bundle 數      : {n_done:>10,}")
    print(f"  [Protein]")
    print(f"    CIF 缺失          : {n_cif_missing:>10,}")
    print(f"    解析錯誤 (protein): {n_prot_error:>10,}")
    print(f"    pair_index 缺失   : {n_pair_index_missing:>10,}  (pair_index.json 不存在)")
    print(f"    pair_index 錯誤   : {n_pair_index_error:>10,}  (pair_index.json 解析失敗)")
    print(f"    有鏈斷裂          : {n_chain_break:>10,}")
    if missing_ratio_cnt > 0:
        print(f"    平均缺失殘基率    : {missing_ratio_sum/missing_ratio_cnt:>10.4f}")
    print(f"    建議捨棄          : {n_discard:>10,}  (missing_ratio>{args.missing_ratio_thresh:.2f} 或有鏈斷裂)")
    print(f"  [Ligand]")
    print(f"    總 pair 數        : {n_lig_total:>10,}")
    print(f"    無 H 原子         : {n_lig_no_h:>10,}  ({n_lig_no_h/max(n_lig_total,1)*100:.1f}%)")
    print(f"    Heavy atom 不符   : {n_lig_mismatch:>10,}  ({n_lig_mismatch/max(n_lig_total,1)*100:.1f}%)")
    print(f"  Top 非標準殘基: {top_nonstd[:5]}")
    print(f"{'='*w}")
    print(f"\n  輸出:")
    print(f"    {prot_csv_path}")
    print(f"    {lig_csv_path}")
    print(f"    {summary_path}")
    print(f"    {discard_path}")


if __name__ == '__main__':
    main()
