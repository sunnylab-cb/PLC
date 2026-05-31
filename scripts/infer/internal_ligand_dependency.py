#!/usr/bin/env python3
"""
Shared utilities for internal ligand-dependency evaluation.

This evaluates the normal PLC test split, not external benchmarks.  The main
ablation modes are:
  real    : original protein-ligand pair
  shuffle : keep protein/labels, replace ligand features/coords/graph/Uni-Mol
  zero    : keep ligand graph shape, zero ligand features/coords/edge attrs/Uni-Mol

Use --ligand_mode source_split to decouple ligand graph features from Uni-Mol:
  shuffle_graph_keep_unimol : shuffled ligand graph, original Uni-Mol
  keep_graph_shuffle_unimol : original ligand graph, shuffled Uni-Mol
  zero_ligand_graph_only    : zero ligand graph tensors, original Uni-Mol
  zero_unimol_only          : original ligand graph, zero Uni-Mol

Use --ligand_mode feature_split to split the ligand V5 atom feature source:
  zero_ligand_global55_only : zero feature dims 54:109 (global45 + shape10)
  zero_ligand_local54_only  : zero feature dims 0:54   (local atom chemistry)

Use --ligand_mode local54_subgroups for mechanism ablations inside local54:
  zero_ligand_local54_base33       : atom type/hybrid/ring/degree/formal charge
  zero_ligand_local54_electronic3  : Gasteiger charge, Crippen LogP, TPSA
  zero_ligand_local54_sasa_hbond4  : atom/lipophilic SASA, HBD, HBA
  zero_ligand_local54_flex_rot5    : RMSF, flexibility, rotatable context
  zero_ligand_local54_metal4       : metal-binding flags
  zero_ligand_local54_stereo5      : chirality + stereochemistry
  zero_ligand_edge_attr_only       : keep topology/coords, zero bond features
  shuffle_ligand_topology_only     : keep atom features, relabel graph topology
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.collate import pocket_collate_fn_v10
from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8


LIGAND_MODES = (
    "real",
    "shuffle",
    "zero",
    "empty",
    "random",
    "shuffle_graph_keep_unimol",
    "keep_graph_shuffle_unimol",
    "zero_ligand_graph_only",
    "zero_unimol_only",
    "zero_ligand_global55_only",
    "zero_ligand_local54_only",
    "zero_ligand_local54_base33",
    "zero_ligand_local54_electronic3",
    "zero_ligand_local54_sasa_hbond4",
    "zero_ligand_local54_flex_rot5",
    "zero_ligand_local54_metal4",
    "zero_ligand_local54_stereo5",
    "zero_ligand_edge_attr_only",
    "shuffle_ligand_topology_only",
)

SOURCE_SPLIT_MODES = (
    "real",
    "shuffle",
    "zero",
    "shuffle_graph_keep_unimol",
    "keep_graph_shuffle_unimol",
    "zero_ligand_graph_only",
    "zero_unimol_only",
)

FEATURE_SPLIT_MODES = (
    "real",
    "zero_ligand_global55_only",
    "zero_ligand_local54_only",
)

LOCAL54_SUBGROUP_MODES = (
    "real",
    "zero_ligand_local54_only",
    "zero_ligand_local54_base33",
    "zero_ligand_local54_electronic3",
    "zero_ligand_local54_sasa_hbond4",
    "zero_ligand_local54_flex_rot5",
    "zero_ligand_local54_metal4",
    "zero_ligand_local54_stereo5",
    "zero_ligand_edge_attr_only",
    "shuffle_ligand_topology_only",
)

LOCAL54_SUBGROUP_RANGES = {
    "zero_ligand_local54_base33": ((0, 33),),
    "zero_ligand_local54_electronic3": ((33, 36),),
    "zero_ligand_local54_sasa_hbond4": ((38, 42),),
    "zero_ligand_local54_flex_rot5": ((36, 38), (44, 47)),
    "zero_ligand_local54_metal4": ((47, 51),),
    "zero_ligand_local54_stereo5": ((42, 44), (51, 54)),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="model checkpoint path")
    parser.add_argument("--config", default=None, help="optional YAML config override")
    parser.add_argument("--output", default=None, help="output directory")
    parser.add_argument("--data_root", default=None, help="override config data.data_root")
    parser.add_argument("--split_file", default=None, help="override config data.split_file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--ligand_mode",
        default="all",
        choices=["all", "source_split", "feature_split", "local54_subgroups", *LIGAND_MODES],
        help=(
            "ablation mode; all runs real/shuffle/zero; source_split decouples "
            "ligand graph vs Uni-Mol; feature_split zeros local54 vs global55 ligand features; "
            "local54_subgroups zeros local54 chemistry subgroups and edge controls"
        ),
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    parser.add_argument("--amp", action="store_true", help="use autocast on CUDA/ROCm")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--unimol_embeddings_dir", default=None)
    parser.add_argument("--no_unimol", action="store_true", help="force zero Uni-Mol")
    parser.add_argument("--num_save_rows", type=int, default=0, help="0 = save all rows")
    parser.add_argument(
        "--class0_logit_bias",
        type=float,
        default=0.0,
        help="additive inference-only bias to class-0/contact logits",
    )


def load_checkpoint(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"checkpoint missing model_state_dict: {path}")
    if ckpt.get("config") is None:
        raise KeyError(f"checkpoint missing config: {path}")
    return ckpt


def load_config_from_checkpoint_or_yaml(ckpt: Dict, config_path: Optional[str]) -> Dict:
    if config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return ckpt["config"]


def _clone_value(v):
    if isinstance(v, torch.Tensor):
        return v.clone()
    return v


def clone_sample(sample: Dict) -> Dict:
    return {k: _clone_value(v) for k, v in sample.items()}


class LigandAblationDataset(Dataset):
    """Wrap PocketDatasetV8 and replace only ligand-side fields."""

    def __init__(self, base: PocketDatasetV8, mode: str, seed: int = 42):
        if mode not in LIGAND_MODES:
            raise ValueError(f"unknown ligand mode: {mode}")
        self.base = base
        self.mode = mode
        self.seed = int(seed)

        n = len(base)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()
        if n > 1:
            # Avoid identity pairs for shuffle.
            for i, p in enumerate(perm):
                if p == i:
                    perm[i] = perm[(i + 1) % n]
        self.perm = perm

    def __len__(self) -> int:
        return len(self.base)

    def _empty_ligand(self, sample: Dict) -> None:
        lig_dim = int(sample["ligand_features"].shape[-1])
        edge_dim = int(sample.get("ligand_edge_attr", torch.zeros(0, 6)).shape[-1])
        sample["ligand_features"] = torch.zeros((0, lig_dim), dtype=torch.float32)
        sample["ligand_coords"] = torch.zeros((0, 3), dtype=torch.float32)
        sample["ligand_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        sample["ligand_edge_attr"] = torch.zeros((0, edge_dim), dtype=torch.float32)
        sample["n_atoms"] = 0

    def _zero_ligand(self, sample: Dict) -> None:
        sample["ligand_features"] = torch.zeros_like(sample["ligand_features"])
        sample["ligand_coords"] = torch.zeros_like(sample["ligand_coords"])
        sample["ligand_edge_attr"] = torch.zeros_like(sample["ligand_edge_attr"])

    def _zero_ligand_feature_slice(self, sample: Dict, start: int, end: int) -> None:
        x = sample["ligand_features"].clone()
        n_feat = int(x.shape[-1])
        s = max(0, min(int(start), n_feat))
        e = max(s, min(int(end), n_feat))
        x[..., s:e] = 0
        sample["ligand_features"] = x

    def _zero_ligand_feature_ranges(self, sample: Dict, ranges: Iterable[tuple[int, int]]) -> None:
        x = sample["ligand_features"].clone()
        n_feat = int(x.shape[-1])
        for start, end in ranges:
            s = max(0, min(int(start), n_feat))
            e = max(s, min(int(end), n_feat))
            x[..., s:e] = 0
        sample["ligand_features"] = x

    def _zero_ligand_edge_attr(self, sample: Dict) -> None:
        sample["ligand_edge_attr"] = torch.zeros_like(sample["ligand_edge_attr"])

    def _shuffle_ligand_topology(self, sample: Dict, idx: int) -> None:
        edge_index = sample["ligand_edge_index"]
        if edge_index.numel() == 0:
            return
        n_atoms_raw = sample.get("n_atoms", sample["ligand_features"].shape[0])
        n_atoms = int(n_atoms_raw.item()) if isinstance(n_atoms_raw, torch.Tensor) else int(n_atoms_raw)
        if n_atoms <= 1:
            return

        g = torch.Generator().manual_seed(_stable_seed(f"{sample.get('pdb_id', idx)}:{idx}", self.seed))
        perm = torch.randperm(n_atoms, generator=g, dtype=edge_index.dtype)
        shuffled = edge_index.clone()
        valid = (shuffled >= 0) & (shuffled < n_atoms)
        shuffled[valid] = perm[shuffled[valid].long()]
        sample["ligand_edge_index"] = shuffled

    def _random_ligand(self, sample: Dict, idx: int) -> None:
        g = torch.Generator().manual_seed(self.seed * 1000003 + int(idx))
        sample["ligand_features"] = torch.randn(
            sample["ligand_features"].shape,
            generator=g,
            dtype=sample["ligand_features"].dtype,
        )
        sample["ligand_coords"] = torch.randn(
            sample["ligand_coords"].shape,
            generator=g,
            dtype=sample["ligand_coords"].dtype,
        )
        sample["ligand_edge_attr"] = torch.randn(
            sample["ligand_edge_attr"].shape,
            generator=g,
            dtype=sample["ligand_edge_attr"].dtype,
        )

    def __getitem__(self, idx: int) -> Dict:
        sample = clone_sample(self.base[idx])
        sample["ligand_source_pdb_id"] = sample["pdb_id"]
        sample["unimol_source_pdb_id"] = sample["pdb_id"]

        if self.mode == "real":
            return sample

        if self.mode in {"shuffle", "shuffle_graph_keep_unimol", "keep_graph_shuffle_unimol"}:
            src = clone_sample(self.base[self.perm[idx]])

        if self.mode == "shuffle":
            for key in (
                "ligand_features",
                "ligand_coords",
                "ligand_edge_index",
                "ligand_edge_attr",
                "n_atoms",
            ):
                sample[key] = src[key]
            sample["ligand_source_pdb_id"] = src["pdb_id"]
            sample["unimol_source_pdb_id"] = src["pdb_id"]
            return sample

        if self.mode == "shuffle_graph_keep_unimol":
            for key in (
                "ligand_features",
                "ligand_coords",
                "ligand_edge_index",
                "ligand_edge_attr",
                "n_atoms",
            ):
                sample[key] = src[key]
            sample["ligand_source_pdb_id"] = src["pdb_id"]
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode == "keep_graph_shuffle_unimol":
            sample["ligand_source_pdb_id"] = sample["pdb_id"]
            sample["unimol_source_pdb_id"] = src["pdb_id"]
            return sample

        if self.mode == "zero":
            self._zero_ligand(sample)
            sample["ligand_source_pdb_id"] = "__ZERO__"
            sample["unimol_source_pdb_id"] = "__ZERO__"
            return sample

        if self.mode == "empty":
            self._empty_ligand(sample)
            sample["ligand_source_pdb_id"] = "__ZERO__"
            sample["unimol_source_pdb_id"] = "__ZERO__"
            return sample

        if self.mode == "random":
            self._random_ligand(sample, idx)
            sample["ligand_source_pdb_id"] = "__RANDOM__"
            sample["unimol_source_pdb_id"] = "__RANDOM__"
            return sample

        if self.mode == "zero_ligand_graph_only":
            self._zero_ligand(sample)
            sample["ligand_source_pdb_id"] = "__ZERO_GRAPH__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode == "zero_unimol_only":
            sample["ligand_source_pdb_id"] = sample["pdb_id"]
            sample["unimol_source_pdb_id"] = "__ZERO__"
            return sample

        if self.mode == "zero_ligand_global55_only":
            self._zero_ligand_feature_slice(sample, 54, 109)
            sample["ligand_source_pdb_id"] = "__ZERO_GLOBAL55__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode == "zero_ligand_local54_only":
            self._zero_ligand_feature_slice(sample, 0, 54)
            sample["ligand_source_pdb_id"] = "__ZERO_LOCAL54__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode in LOCAL54_SUBGROUP_RANGES:
            self._zero_ligand_feature_ranges(sample, LOCAL54_SUBGROUP_RANGES[self.mode])
            sample["ligand_source_pdb_id"] = f"__{self.mode.upper()}__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode == "zero_ligand_edge_attr_only":
            self._zero_ligand_edge_attr(sample)
            sample["ligand_source_pdb_id"] = "__ZERO_EDGE_ATTR__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        if self.mode == "shuffle_ligand_topology_only":
            self._shuffle_ligand_topology(sample, idx)
            sample["ligand_source_pdb_id"] = "__SHUFFLE_TOPOLOGY__"
            sample["unimol_source_pdb_id"] = sample["pdb_id"]
            return sample

        raise AssertionError(self.mode)


def collate_with_ligand_source(batch: List[Dict]) -> Dict:
    out = pocket_collate_fn_v10(batch)
    out["ligand_source_pdb_ids"] = [
        sample.get("ligand_source_pdb_id", sample["pdb_id"]) for sample in batch
    ]
    out["unimol_source_pdb_ids"] = [
        sample.get("unimol_source_pdb_id", sample.get("ligand_source_pdb_id", sample["pdb_id"]))
        for sample in batch
    ]
    return out


def resolve_split_ids(config: Dict, args) -> List[str]:
    data_cfg = config.get("data", {})
    split_file = args.split_file or data_cfg.get("split_file", "configs/splits/splits.txt")
    train_ids, val_ids, test_ids = load_splits(split_file)
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    ids = list(split_map[args.split])
    if args.max_samples and args.max_samples > 0:
        ids = ids[: args.max_samples]
    if not ids:
        raise RuntimeError(f"empty split: {args.split} from {split_file}")
    return ids


def build_internal_dataset(config: Dict, args) -> PocketDatasetV8:
    data_cfg = config.get("data", {})
    pdb_ids = resolve_split_ids(config, args)
    data_root = args.data_root or data_cfg.get("data_root", "data")
    return PocketDatasetV8(
        data_root=data_root,
        pdb_ids=pdb_ids,
        max_seq_len=int(data_cfg.get("max_seq_len", 1800)),
        max_atoms=int(data_cfg.get("max_atoms", 128)),
        distances_file=data_cfg.get("distances_file", "data/residue_ligand_distances_dcenter.pt"),
        ligand_features_dir=data_cfg.get("ligand_features_dir", "ligand_features_v5"),
        protein_features_dir=data_cfg.get("protein_features_dir", "complete_features_v4"),
        nma_features_dir=data_cfg.get("nma_features_dir", "complete_features_v4_nma"),
        nma_n_confs=1,
        nma_max_conf=int(data_cfg.get("nma_max_conf", 9)),
        use_nma=False,
        label_contact_thresh=float(data_cfg.get("label_contact_thresh", 5.0)),
        label_near_thresh=float(data_cfg.get("label_near_thresh", 8.0)),
        use_spatial_clamp=bool(data_cfg.get("use_spatial_clamp", False)),
        spatial_clamp_dims=data_cfg.get("spatial_clamp_dims", None),
        spatial_clamp_range=data_cfg.get("spatial_clamp_range", None),
        split=args.split,
        verbose=True,
    )


def load_unimol_cache(config: Dict, args, d_unimol: int = 512) -> Dict[str, torch.Tensor]:
    if args.no_unimol:
        return {}

    data_cfg = config.get("data", {})
    data_root = Path(args.data_root or data_cfg.get("data_root", "data"))
    unimol_dir = args.unimol_embeddings_dir or data_cfg.get("unimol_embeddings_dir", "unimol_embeddings")
    root = data_root / unimol_dir
    if not root.exists():
        raise RuntimeError(f"Uni-Mol directory not found: {root}")

    cache: Dict[str, torch.Tensor] = {}
    n_bad = 0
    for fpath in root.rglob("*_unimol.pt"):
        pdb_id = fpath.stem.replace("_unimol", "")
        obj = torch.load(fpath, map_location="cpu", weights_only=True)
        embed = obj.get("embedding") if isinstance(obj, dict) else obj
        if not isinstance(embed, torch.Tensor) or tuple(embed.shape) != (d_unimol,):
            n_bad += 1
            continue
        embed = embed.float()
        if not torch.isfinite(embed).all() or embed.abs().sum().item() <= 1e-6:
            n_bad += 1
            continue
        cache[pdb_id] = embed

    if not cache:
        raise RuntimeError(f"no valid Uni-Mol embeddings loaded from {root}")
    print(f"[Uni-Mol] loaded {len(cache)} embeddings from {root}; skipped={n_bad}")
    return cache


def _stable_seed(text: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def batch_unimol(
    source_pdb_ids: List[str],
    cache: Dict[str, torch.Tensor],
    d_unimol: int,
    device: torch.device,
    seed: int,
) -> Optional[torch.Tensor]:
    if not cache:
        return torch.zeros(len(source_pdb_ids), d_unimol, device=device)

    rows = []
    missing = []
    for pid in source_pdb_ids:
        if pid.startswith("__ZERO__"):
            rows.append(torch.zeros(d_unimol))
        elif pid.startswith("__RANDOM__"):
            g = torch.Generator().manual_seed(_stable_seed(pid, seed))
            rows.append(torch.randn(d_unimol, generator=g))
        elif pid in cache:
            rows.append(cache[pid])
        else:
            missing.append(pid)
    if missing:
        raise RuntimeError(
            f"missing Uni-Mol embeddings for {len(missing)} Uni-Mol sources, e.g. {missing[:5]}"
        )
    return torch.stack(rows, dim=0).to(device, non_blocking=True)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    iou = _safe_div(tp, tp + fp + fn)
    return {"precision": p, "recall": r, "f1": f1, "dice": f1, "iou": iou}


def _sample_metrics(pred: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
    out = {"accuracy": float((pred == label).float().mean().item()) if label.numel() else 0.0}
    for c in (0, 1, 2):
        tp = int(((pred == c) & (label == c)).sum().item())
        fp = int(((pred == c) & (label != c)).sum().item())
        fn = int(((pred != c) & (label == c)).sum().item())
        m = _prf(tp, fp, fn)
        out.update({f"{k}_c{c}": v for k, v in m.items()})
    pred_bin = pred != 2
    label_bin = label != 2
    m = _prf(
        int((pred_bin & label_bin).sum().item()),
        int((pred_bin & ~label_bin).sum().item()),
        int((~pred_bin & label_bin).sum().item()),
    )
    out.update({f"{k}_binary": v for k, v in m.items()})
    return out


def _aggregate(rows: List[Dict], global_counts: Dict[str, int]) -> Dict[str, float]:
    out: Dict[str, float] = {"n_samples": len(rows)}
    if not rows:
        return out

    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "n_residues"]
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            out[f"mean_sample_{key}"] = float(sum(vals) / len(vals))

    total = global_counts["total"]
    correct = global_counts["correct"]
    out["global_accuracy"] = _safe_div(correct, total)
    for c in (0, 1, 2):
        m = _prf(global_counts[f"tp{c}"], global_counts[f"fp{c}"], global_counts[f"fn{c}"])
        out.update({f"global_{k}_c{c}": v for k, v in m.items()})
    m = _prf(global_counts["tp_bin"], global_counts["fp_bin"], global_counts["fn_bin"])
    out.update({f"global_{k}_binary": v for k, v in m.items()})
    return out


def _init_counts() -> Dict[str, int]:
    d = {"total": 0, "correct": 0, "tp_bin": 0, "fp_bin": 0, "fn_bin": 0}
    for c in (0, 1, 2):
        d[f"tp{c}"] = d[f"fp{c}"] = d[f"fn{c}"] = 0
    return d


def _update_counts(counts: Dict[str, int], pred: torch.Tensor, label: torch.Tensor) -> None:
    counts["total"] += int(label.numel())
    counts["correct"] += int((pred == label).sum().item())
    for c in (0, 1, 2):
        counts[f"tp{c}"] += int(((pred == c) & (label == c)).sum().item())
        counts[f"fp{c}"] += int(((pred == c) & (label != c)).sum().item())
        counts[f"fn{c}"] += int(((pred != c) & (label == c)).sum().item())
    pred_bin = pred != 2
    label_bin = label != 2
    counts["tp_bin"] += int((pred_bin & label_bin).sum().item())
    counts["fp_bin"] += int((pred_bin & ~label_bin).sum().item())
    counts["fn_bin"] += int((~pred_bin & label_bin).sum().item())


def evaluate_mode(
    *,
    model: torch.nn.Module,
    dataset: PocketDatasetV8,
    args,
    mode: str,
    device: torch.device,
    predict_logits: Callable,
    unimol_cache: Dict[str, torch.Tensor],
    d_unimol: int,
    out_dir: Path,
) -> Dict:
    ablated = LigandAblationDataset(dataset, mode=mode, seed=args.shuffle_seed)
    loader = DataLoader(
        ablated,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_ligand_source,
        pin_memory=(device.type == "cuda"),
    )

    rows: List[Dict] = []
    counts = _init_counts()
    model.eval()

    autocast_enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{mode}", unit="batch"):
            ligand_source_ids = batch["ligand_source_pdb_ids"]
            unimol_source_ids = batch["unimol_source_pdb_ids"]
            ue = batch_unimol(
                unimol_source_ids,
                cache=unimol_cache,
                d_unimol=d_unimol,
                device=device,
                seed=args.shuffle_seed,
            )
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = predict_logits(model, batch, device, ue)
            class0_bias = float(getattr(args, "class0_logit_bias", 0.0) or 0.0)
            if class0_bias:
                logits = logits.clone()
                logits[..., 0] = logits[..., 0] + class0_bias
            pred = logits.argmax(dim=-1).detach().cpu()
            labels = batch["labels"].detach().cpu()
            mask = batch["protein_mask"].bool().detach().cpu()
            pdb_ids = batch["pdb_ids"]

            for b, pdb_id in enumerate(pdb_ids):
                valid = mask[b]
                p = pred[b][valid]
                y = labels[b][valid]
                _update_counts(counts, p, y)
                row = {
                    "pdb_id": pdb_id,
                    "ligand_source_pdb_id": ligand_source_ids[b],
                    "unimol_source_pdb_id": unimol_source_ids[b],
                    "mode": mode,
                    "n_residues": int(valid.sum().item()),
                    "n_true_contact": int((y == 0).sum().item()),
                    "n_pred_contact": int((p == 0).sum().item()),
                }
                row.update(_sample_metrics(p, y))
                rows.append(row)

    summary = _aggregate(rows, counts)
    summary.update({"mode": mode, "n_residues_global": counts["total"]})

    row_limit = None if args.num_save_rows == 0 else args.num_save_rows
    save_rows = rows if row_limit is None else rows[:row_limit]
    if save_rows:
        csv_path = out_dir / f"per_sample_{mode}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(save_rows[0].keys()))
            writer.writeheader()
            writer.writerows(save_rows)
    with open(out_dir / f"summary_{mode}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def expand_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["real", "shuffle", "zero"]
    if mode == "source_split":
        return list(SOURCE_SPLIT_MODES)
    if mode == "feature_split":
        return list(FEATURE_SPLIT_MODES)
    if mode == "local54_subgroups":
        return list(LOCAL54_SUBGROUP_MODES)
    return [mode]


def add_deltas(summaries: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    if "real" not in summaries:
        return {}
    deltas: Dict[str, Dict[str, float]] = {}
    keys = (
        "global_f1_c0",
        "global_dice_c0",
        "global_iou_c0",
        "global_recall_c0",
        "global_precision_c0",
        "mean_sample_f1_c0",
        "mean_sample_iou_c0",
        "global_f1_binary",
    )
    real = summaries["real"]
    for mode, summary in summaries.items():
        if mode == "real":
            continue
        deltas[mode] = {}
        for key in keys:
            if key in real and key in summary:
                deltas[mode][f"delta_{key}"] = float(real[key] - summary[key])
    return deltas


def run_ligand_dependency_eval(
    *,
    args,
    config: Dict,
    model: torch.nn.Module,
    model_tag: str,
    device: torch.device,
    predict_logits: Callable,
    needs_unimol: bool,
    d_unimol: int,
) -> Path:
    dataset = build_internal_dataset(config, args)
    modes = expand_modes(args.ligand_mode)

    unimol_cache: Dict[str, torch.Tensor] = {}
    if needs_unimol and not args.no_unimol:
        unimol_cache = load_unimol_cache(config, args, d_unimol=d_unimol)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path("results") / f"ligdep_{model_tag}_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict] = {}
    for mode in modes:
        summaries[mode] = evaluate_mode(
            model=model,
            dataset=dataset,
            args=args,
            mode=mode,
            device=device,
            predict_logits=predict_logits,
            unimol_cache=unimol_cache,
            d_unimol=d_unimol,
            out_dir=out_dir,
        )
        s = summaries[mode]
        print(
            f"[{mode}] global_f1_c0={s.get('global_f1_c0', 0):.4f} "
            f"recall_c0={s.get('global_recall_c0', 0):.4f} "
            f"precision_c0={s.get('global_precision_c0', 0):.4f} "
            f"mean_sample_f1_c0={s.get('mean_sample_f1_c0', 0):.4f}"
        )

    combined = {
        "model_tag": model_tag,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(dataset),
        "modes": modes,
        "summaries": summaries,
        "deltas_from_real": add_deltas(summaries),
        "args": vars(args),
    }
    with open(out_dir / "ligand_dependency_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\nLigand dependency deltas from real:")
    for mode, delta in combined["deltas_from_real"].items():
        print(f"  {mode}: " + ", ".join(f"{k}={v:.4f}" for k, v in delta.items()))
    print(f"Saved to: {out_dir}")
    return out_dir
