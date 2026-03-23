#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程检查 protein feature / ESM / tag 三方长度一致性。

默认检查：
  - data/complete_features_v3
  - data/esm2_latents
  - data/ligand_features_v4
  - data/residue_ligand_distances_vdw.pt

示例：
  python scripts/check_feature_esm_tag_consistency_mt.py --workers 128
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


YEAR_RANGES = ["1981-2000", "2001-2010", "2011-2019"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多线程检查 feature/esm/tag 三方长度一致性")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="数据根目录")
    parser.add_argument("--protein-features-dir", type=str, default="complete_features_v3", help="蛋白特征目录")
    parser.add_argument("--esm-dir", type=str, default="esm2_latents", help="ESM特征目录")
    parser.add_argument("--ligand-features-dir", type=str, default="ligand_features_v4", help="配体特征目录（仅用于筛选可训练样本）")
    parser.add_argument("--tag-file", type=str, default="residue_ligand_distances_vdw.pt", help="标签文件名或绝对路径")
    parser.add_argument("--workers", type=int, default=128, help="线程数")
    parser.add_argument("--max", type=int, default=None, help="最多检查样本数（调试用）")
    parser.add_argument("--report-csv", type=Path, default=Path("/tmp/feature_esm_tag_consistency_report.csv"), help="明细CSV输出")
    parser.add_argument("--errors-csv", type=Path, default=Path("/tmp/feature_esm_tag_consistency_errors.csv"), help="错误CSV输出")
    return parser.parse_args()


def resolve_tag_path(data_root: Path, tag_file: str) -> Path:
    p = Path(tag_file)
    return p if p.is_absolute() else (data_root / p)


def get_tag_length(entry) -> Optional[int]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        if "distances_vdw_adjusted" in entry:
            return len(entry["distances_vdw_adjusted"])
        if "distances" in entry:
            return len(entry["distances"])
        if "vdw_distances" in entry:
            return len(entry["vdw_distances"])
        return None
    return len(entry)


def build_candidates(
    data_root: Path,
    protein_features_dir: str,
    esm_dir: str,
    ligand_features_dir: str,
) -> List[Tuple[str, str, Path, Path, Path]]:
    feat_root = data_root / protein_features_dir
    esm_root = data_root / esm_dir
    lig_root = data_root / ligand_features_dir

    candidates: List[Tuple[str, str, Path, Path, Path]] = []
    for year in YEAR_RANGES:
        year_dir = feat_root / year
        if not year_dir.exists():
            continue
        for complex_dir in year_dir.iterdir():
            if not complex_dir.is_dir():
                continue
            pdb_id = complex_dir.name
            feat_file = complex_dir / f"{pdb_id}_protein.pt"
            esm_file = esm_root / year / pdb_id / f"{pdb_id}_protein.pt"
            lig_file = lig_root / year / pdb_id / f"{pdb_id}_ligand.pt"
            if feat_file.exists() and esm_file.exists() and lig_file.exists():
                candidates.append((pdb_id, year, feat_file, esm_file, lig_file))
    return sorted(candidates, key=lambda x: (x[1], x[0]))


def check_one(
    item: Tuple[str, str, Path, Path, Path],
    tag_dict: Dict,
) -> Dict:
    pdb_id, year, feat_file, esm_file, _lig_file = item
    try:
        feat = torch.load(feat_file, map_location="cpu", weights_only=False)
        esm = torch.load(esm_file, map_location="cpu", weights_only=False)

        feat_len = len(feat["ca_coords"])
        esm_len = len(esm["embeddings"])
        tag_len = get_tag_length(tag_dict.get(pdb_id))

        all_equal = tag_len is not None and (feat_len == esm_len == tag_len)
        return {
            "pdb_id": pdb_id,
            "year": year,
            "feat_len": feat_len,
            "esm_len": esm_len,
            "tag_len": tag_len,
            "ok": all_equal,
            "error": "",
        }
    except Exception as exc:
        return {
            "pdb_id": pdb_id,
            "year": year,
            "feat_len": None,
            "esm_len": None,
            "tag_len": None,
            "ok": False,
            "error": repr(exc),
        }


def main() -> None:
    args = parse_args()

    data_root = args.data_root
    tag_path = resolve_tag_path(data_root, args.tag_file)
    if not tag_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {tag_path}")

    print(f"加载标签: {tag_path}")
    tag_dict = torch.load(tag_path, map_location="cpu", weights_only=False)

    print("收集可训练样本（要求 feat/esm/lig 文件同时存在）...")
    candidates = build_candidates(
        data_root=data_root,
        protein_features_dir=args.protein_features_dir,
        esm_dir=args.esm_dir,
        ligand_features_dir=args.ligand_features_dir,
    )

    if args.max is not None:
        candidates = candidates[: args.max]

    print(f"候选样本数: {len(candidates)}")
    print(f"线程数: {args.workers}")

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_one, item, tag_dict) for item in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="checking"):
            results.append(fut.result())

    good = [r for r in results if r["error"] == ""]
    bad = [r for r in results if r["error"] != ""]

    total = len(good)
    missing_tag = sum(1 for r in good if r["tag_len"] is None)
    feat_eq_esm = sum(1 for r in good if r["feat_len"] == r["esm_len"])
    feat_eq_tag = sum(1 for r in good if r["tag_len"] is not None and r["feat_len"] == r["tag_len"])
    esm_eq_tag = sum(1 for r in good if r["tag_len"] is not None and r["esm_len"] == r["tag_len"])
    all_three_equal = sum(1 for r in good if r["ok"])

    print("\n=== OVERALL ===")
    print(f"usable_samples {total}")
    print(f"load_errors {len(bad)}")
    print(f"missing_tag {missing_tag}")
    print(f"feat_eq_esm {feat_eq_esm}")
    print(f"feat_eq_tag {feat_eq_tag}")
    print(f"esm_eq_tag {esm_eq_tag}")
    print(f"all_three_equal {all_three_equal}")
    print(f"any_mismatch {total - all_three_equal}")

    print("\n=== PER_YEAR ===")
    for year in YEAR_RANGES:
        sub = [r for r in good if r["year"] == year]
        if not sub:
            continue
        all_eq = sum(1 for r in sub if r["ok"])
        miss_tag = sum(1 for r in sub if r["tag_len"] is None)
        print(f"{year}\tusable={len(sub)}\tall_three_equal={all_eq}\tmismatch={len(sub)-all_eq}\tmissing_tag={miss_tag}")

    mismatch = [r for r in good if not r["ok"]]
    if mismatch:
        dif = np.array([r["feat_len"] - r["esm_len"] for r in mismatch], dtype=int)
        print("\n=== DELTA(feat-esm) ON MISMATCH ===")
        print(f"median {float(np.median(dif))} p95 {float(np.percentile(dif, 95))} min {int(dif.min())} max {int(dif.max())}")

        print("\n=== TOP20 |feat-esm| ===")
        for r in sorted(mismatch, key=lambda x: abs(x["feat_len"] - x["esm_len"]), reverse=True)[:20]:
            print(
                f"{r['pdb_id']}\t{r['year']}\t"
                f"feat={r['feat_len']}\tesm={r['esm_len']}\ttag={r['tag_len']}\t"
                f"delta_feat_esm={r['feat_len'] - r['esm_len']}"
            )

    # 输出CSV
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "year", "feat_len", "esm_len", "tag_len", "ok", "error"])
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["year"], x["pdb_id"])):
            writer.writerow(r)
    print(f"\n明细已写入: {args.report_csv}")

    with args.errors_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "year", "feat_len", "esm_len", "tag_len", "ok", "error"])
        writer.writeheader()
        for r in sorted(bad, key=lambda x: (x["year"], x["pdb_id"])):
            writer.writerow(r)
    print(f"错误已写入: {args.errors_csv}")


if __name__ == "__main__":
    main()

