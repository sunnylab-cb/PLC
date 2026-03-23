#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对齐三个来源的蛋白质残基序列：
  1. extract_complete_features_v3  → *_protein.pt  (physchem/spatial features)
  2. esm2_extract_latents          → *_protein.pt  (ESM2 embeddings)
  3. calculate_distances_vdw       → residue_ligand_distances_vdw.pt (per-residue distances)

对齐 key：(chain_id, res_id, ins_code)
输出：每个 pdb_id 的对齐状态报告，以及不一致的详细信息。

用法：
  python scripts/align_protein_seq.py \
      --feat-dir   data/complete_features_v3 \
      --esm-dir    data/esm2_latents \
      --dist-file  data/residue_ligand_distances_vdw.pt \
      --pdb-root   /home/husrcf/Code/ProtBind/New \
      [--max 20]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch


# ─────────────────────────────────────────────
# 工具：从各来源提取残基 key 列表
# ─────────────────────────────────────────────

ResKey = Tuple[str, int, str]  # (chain_id, res_id, ins_code)


def keys_from_feat(pt: dict) -> List[ResKey]:
    """从 extract_complete_features_v3 输出的 .pt 提取 key 列表"""
    chains = pt['chain_ids']
    indices = pt['residue_indices']
    ins_codes = pt.get('insertion_codes', [''] * len(indices))
    return [(str(c), int(r), str(i).strip()) for c, r, i in zip(chains, indices, ins_codes)]


def keys_from_esm(pt: dict) -> List[ResKey]:
    """从 esm2_extract_latents 输出的 .pt 提取 key 列表"""
    chains = pt['chain_id']
    indices = pt['residue_index']
    ins_codes = pt.get('insertion_code', [''] * len(indices))
    return [(str(c), int(r), str(i).strip()) for c, r, i in zip(chains, indices, ins_codes)]


def keys_from_dist(pt: dict, pdb_id: str) -> Optional[List[ResKey]]:
    """
    从 calculate_distances_vdw 输出的全局 .pt 提取指定 pdb_id 的 key 列表。
    距离文件只存了坐标顺序（无 key），所以只能比较长度。
    返回 None 表示该 pdb_id 不在距离文件中。
    """
    if pdb_id not in pt:
        return None
    entry = pt[pdb_id]
    n = int(entry.get('n_total', len(entry.get('distances', []))))
    # 距离文件无 key 信息，返回长度占位
    return n


# ─────────────────────────────────────────────
# 对齐检查
# ─────────────────────────────────────────────

def check_alignment(
    pdb_id: str,
    feat_pt: dict,
    esm_pt: dict,
    dist_pt: Optional[dict],
) -> dict:
    feat_keys = keys_from_feat(feat_pt)
    esm_keys  = keys_from_esm(esm_pt)

    result = {
        'pdb_id': pdb_id,
        'n_feat': len(feat_keys),
        'n_esm':  len(esm_keys),
        'n_dist': None,
        'feat_esm_match': False,
        'feat_dist_len_match': None,
        'mismatches': [],
    }

    # 1. feat vs esm：key 级别对齐
    if feat_keys == esm_keys:
        result['feat_esm_match'] = True
    else:
        # 找出不一致的位置
        max_len = max(len(feat_keys), len(esm_keys))
        for i in range(max_len):
            fk = feat_keys[i] if i < len(feat_keys) else None
            ek = esm_keys[i]  if i < len(esm_keys)  else None
            if fk != ek:
                result['mismatches'].append({'pos': i, 'feat': fk, 'esm': ek})
                if len(result['mismatches']) >= 10:
                    result['mismatches'].append('...(truncated)')
                    break

    # 2. feat vs dist：只比长度（距离文件无 key）
    if dist_pt is not None:
        n_dist = keys_from_dist(dist_pt, pdb_id)
        if n_dist is not None:
            result['n_dist'] = n_dist
            result['feat_dist_len_match'] = (len(feat_keys) == n_dist)

    return result


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="对齐三个来源的蛋白质残基序列")
    parser.add_argument('--feat-dir',  required=True, help='extract_complete_features_v3 输出目录')
    parser.add_argument('--esm-dir',   required=True, help='esm2_extract_latents 输出目录')
    parser.add_argument('--dist-file', default=None,  help='residue_ligand_distances_vdw.pt 路径（可选）')
    parser.add_argument('--output',    default='alignment_failures.csv', help='失败记录输出 CSV 路径')
    parser.add_argument('--max',       type=int, default=None, help='最多处理多少个样本')
    parser.add_argument('--verbose',   action='store_true', help='打印每个样本的详细信息')
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    esm_dir  = Path(args.esm_dir)

    # 收集 feat .pt 文件
    feat_files = sorted(feat_dir.rglob('*_protein.pt'))
    if args.max:
        feat_files = feat_files[:args.max]

    if not feat_files:
        print(f"[ERROR] 在 {feat_dir} 下未找到 *_protein.pt 文件")
        sys.exit(1)

    # 加载距离文件（可选）
    dist_pt = None
    if args.dist_file:
        dist_path = Path(args.dist_file)
        if dist_path.exists():
            dist_pt = torch.load(dist_path, weights_only=False)
            print(f"已加载距离文件：{dist_path}，共 {len(dist_pt)} 条记录")
        else:
            print(f"[WARN] 距离文件不存在：{dist_path}")

    # 统计
    total = 0
    ok = 0
    fail_esm_missing = 0
    fail_mismatch = 0
    fail_dist_len = 0
    mismatch_details = []
    csv_rows = []  # 用于写入 CSV

    for feat_path in feat_files:
        # 推断 pdb_id
        stem = feat_path.stem  # e.g. 1a0q_protein
        pdb_id = stem[:-8] if stem.endswith('_protein') else stem

        # 找对应 ESM .pt
        # 支持扁平目录或 year/pdb_id 子目录两种结构
        esm_candidates = [
            esm_dir / feat_path.relative_to(feat_dir),
            esm_dir / f"{pdb_id}_protein.pt",
        ]
        esm_path = next((p for p in esm_candidates if p.exists()), None)

        total += 1

        if esm_path is None:
            fail_esm_missing += 1
            if args.verbose:
                print(f"[MISS] {pdb_id}: ESM 文件不存在")
            csv_rows.append({
                'pdb_id': pdb_id,
                'feat_path': str(feat_path),
                'esm_path': '',
                'n_feat': '',
                'n_esm': '',
                'n_dist': '',
                'failure_type': 'esm_missing',
                'detail': 'ESM文件不存在',
            })
            continue

        try:
            feat_pt = torch.load(feat_path, weights_only=False)
            esm_pt  = torch.load(esm_path,  weights_only=False)
        except Exception as e:
            fail_esm_missing += 1
            print(f"[ERR]  {pdb_id}: 加载失败 {e}")
            csv_rows.append({
                'pdb_id': pdb_id,
                'feat_path': str(feat_path),
                'esm_path': str(esm_path),
                'n_feat': '',
                'n_esm': '',
                'n_dist': '',
                'failure_type': 'load_error',
                'detail': str(e),
            })
            continue

        res = check_alignment(pdb_id, feat_pt, esm_pt, dist_pt)

        feat_esm_ok   = res['feat_esm_match']
        dist_len_ok   = res['feat_dist_len_match']  # None = 无距离文件

        if feat_esm_ok and (dist_len_ok is None or dist_len_ok):
            ok += 1
            if args.verbose:
                print(f"[OK]   {pdb_id}: feat={res['n_feat']} esm={res['n_esm']}"
                      + (f" dist={res['n_dist']}" if res['n_dist'] is not None else ""))
        else:
            failure_types = []
            detail_parts = []

            if not feat_esm_ok:
                fail_mismatch += 1
                mismatch_details.append(res)
                failure_types.append('feat_esm_mismatch')
                for m in res['mismatches']:
                    if isinstance(m, str):
                        detail_parts.append(m)
                    else:
                        detail_parts.append(f"pos={m['pos']} feat={m['feat']} esm={m['esm']}")

            if dist_len_ok is False:
                fail_dist_len += 1
                failure_types.append('feat_dist_len_mismatch')
                detail_parts.append(f"feat_len={res['n_feat']} dist_len={res['n_dist']}")

            csv_rows.append({
                'pdb_id': pdb_id,
                'feat_path': str(feat_path),
                'esm_path': str(esm_path),
                'n_feat': res['n_feat'],
                'n_esm': res['n_esm'],
                'n_dist': res['n_dist'] if res['n_dist'] is not None else '',
                'failure_type': '|'.join(failure_types),
                'detail': ' | '.join(detail_parts),
            })

            print(f"[FAIL] {pdb_id}: feat={res['n_feat']} esm={res['n_esm']}"
                  + (f" dist={res['n_dist']}" if res['n_dist'] is not None else "")
                  + (" feat≠esm" if not feat_esm_ok else "")
                  + (" feat≠dist_len" if dist_len_ok is False else ""))

    # 写入 CSV
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['pdb_id', 'feat_path', 'esm_path', 'n_feat', 'n_esm', 'n_dist', 'failure_type', 'detail']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n失败记录已写入: {out_csv}  ({len(csv_rows)} 条)")

    # 汇总
    print("\n" + "=" * 60)
    print(f"总计:            {total}")
    print(f"  对齐成功:      {ok}")
    print(f"  ESM文件缺失:   {fail_esm_missing}")
    print(f"  feat/esm不一致:{fail_mismatch}")
    print(f"  feat/dist长度差:{fail_dist_len}")

    # 打印不一致详情
    if mismatch_details:
        print("\n── 不一致详情（前10个）──")
        for r in mismatch_details[:10]:
            print(f"\n  {r['pdb_id']}  feat={r['n_feat']} esm={r['n_esm']}")
            for m in r['mismatches']:
                if isinstance(m, str):
                    print(f"    {m}")
                else:
                    print(f"    pos={m['pos']:4d}  feat={m['feat']}  esm={m['esm']}")


if __name__ == '__main__':
    main()
