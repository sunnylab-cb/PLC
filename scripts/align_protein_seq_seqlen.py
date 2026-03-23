#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只比较序列长度和氨基酸序列字符串，不比较 res_id。
用于诊断 feat vs esm 的序列是否一致。

用法：
  python scripts/align_protein_seq_seqlen.py \
      --feat-dir data/complete_features_v3 \
      --esm-dir  data/esm2_latents \
      --output   results/seqlen_failures.csv \
      [--max 50]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


AA_3TO1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
    'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
    'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
    'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}

def seq_from_feat(pt: dict) -> str:
    """从 extract_complete_features_v3 .pt 提取氨基酸序列（从 residue_names 转换）"""
    if 'sequence' in pt and pt['sequence']:
        return pt['sequence']
    names = pt.get('residue_names', [])
    return ''.join(AA_3TO1.get(str(n).strip(), 'X') for n in names)


def seq_from_esm(pt: dict) -> str:
    """从 esm2_extract_latents .pt 提取氨基酸序列"""
    # 优先用 sequence 字段，其次从 residue_name_1 拼
    if 'sequence' in pt:
        return pt['sequence']
    names = pt.get('residue_name_1', [])
    return ''.join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-dir', required=True)
    parser.add_argument('--esm-dir',  required=True)
    parser.add_argument('--output',   default='results/seqlen_failures.csv')
    parser.add_argument('--max',      type=int, default=None)
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    esm_dir  = Path(args.esm_dir)

    feat_files = sorted(feat_dir.rglob('*_protein.pt'))
    if args.max:
        feat_files = feat_files[:args.max]

    if not feat_files:
        print(f"[ERROR] 未找到文件: {feat_dir}")
        sys.exit(1)

    total = ok = fail_missing = fail_len = fail_seq = 0
    csv_rows = []

    for feat_path in feat_files:
        stem   = feat_path.stem
        pdb_id = stem[:-8] if stem.endswith('_protein') else stem

        esm_candidates = [
            esm_dir / feat_path.relative_to(feat_dir),
            esm_dir / f"{pdb_id}_protein.pt",
        ]
        esm_path = next((p for p in esm_candidates if p.exists()), None)
        total += 1

        if esm_path is None:
            fail_missing += 1
            csv_rows.append({'pdb_id': pdb_id, 'n_feat': '', 'n_esm': '',
                             'failure_type': 'esm_missing', 'seq_feat': '', 'seq_esm': '', 'diff': ''})
            continue

        try:
            feat_pt = torch.load(feat_path, weights_only=False)
            esm_pt  = torch.load(esm_path,  weights_only=False)
        except Exception as e:
            fail_missing += 1
            csv_rows.append({'pdb_id': pdb_id, 'n_feat': '', 'n_esm': '',
                             'failure_type': 'load_error', 'seq_feat': '', 'seq_esm': '', 'diff': str(e)})
            continue

        seq_feat = seq_from_feat(feat_pt)
        seq_esm  = seq_from_esm(esm_pt)
        n_feat   = len(seq_feat)
        n_esm    = len(seq_esm)

        if n_feat == n_esm and seq_feat == seq_esm:
            ok += 1
            continue

        # 找第一个不同的位置
        first_diff = ''
        if n_feat == n_esm:
            fail_seq += 1
            ftype = 'seq_mismatch'
            for i, (a, b) in enumerate(zip(seq_feat, seq_esm)):
                if a != b:
                    first_diff = f"pos={i} feat={a} esm={b}"
                    break
        else:
            fail_len += 1
            ftype = 'len_mismatch'
            first_diff = f"feat={n_feat} esm={n_esm} diff={n_feat - n_esm:+d}"

        print(f"[FAIL] {pdb_id}: {ftype}  {first_diff}")
        csv_rows.append({
            'pdb_id':      pdb_id,
            'n_feat':      n_feat,
            'n_esm':       n_esm,
            'failure_type': ftype,
            'seq_feat':    seq_feat[:60] + ('...' if len(seq_feat) > 60 else ''),
            'seq_esm':     seq_esm[:60]  + ('...' if len(seq_esm)  > 60 else ''),
            'diff':        first_diff,
        })

    # 写 CSV
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pdb_id','n_feat','n_esm','failure_type','seq_feat','seq_esm','diff'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n结果写入: {out_csv}  ({len(csv_rows)} 条失败)")
    print("=" * 50)
    print(f"总计:          {total}")
    print(f"  序列一致:    {ok}")
    print(f"  ESM缺失:     {fail_missing}")
    print(f"  长度不一致:  {fail_len}")
    print(f"  序列不一致:  {fail_seq}")


if __name__ == '__main__':
    main()
