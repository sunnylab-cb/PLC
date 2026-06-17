#!/usr/bin/env python3
"""Run V11-family checkpoints on COACH420/HOLO4K benchmark features.

This runner targets the benchmark assets prepared under ``benchmark/``:

  - protein V4 features:
      benchmark/features_fixed_conservative/protein_v4/{coach420,holo4k}/{id}/{id}_protein.pt
  - ESM2:
      benchmark/esm2/{coach420,holo4k}/{id}.pt
  - ligand V5 features:
      benchmark/ligand_features_v5/{coach420,holo4k}/{id}_{ligand}.pt
  - atom+H no-missing-residue tags:
      benchmark/residue_ligand_tags/atom_h/{dataset}_aggregate_pdb_dmin.pt

The primary benchmark mode is aggregate-by-PDB.  Per-ligand mode is supported
for diagnostics, but aggregate mode should be treated as the main external
benchmark number because COACH420/HOLO4K are pocket-level datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.infer.infer_v11_internal_ligdep import (  # noqa: E402
    create_v11_model,
    model_needs_unimol,
)
from scripts.infer.internal_ligand_dependency import (  # noqa: E402
    load_checkpoint,
    load_config_from_checkpoint_or_yaml,
)
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11  # noqa: E402


CLASS_NAMES = {0: "Contact", 1: "Near", 2: "Far"}
LIGAND_MODES = ("real", "empty", "zero", "shuffle")


def _as_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def _continuous_norm_dims(stats: dict) -> list[int]:
    """Match scripts/apply_feature_normalization.py for ligand V5 features."""
    std = np.asarray(stats["std"])
    mn = np.asarray(stats["min"])
    mx = np.asarray(stats["max"])
    all_int = np.asarray(stats["all_int"])
    rng = mx - mn
    dims = np.where((~all_int) & ((rng > 5.0) | (std > 2.0)))[0]
    return [int(i) for i in dims.tolist()]


def _resolve_feature_stats_path(args, config: Dict) -> Optional[Path]:
    if args.feature_stats:
        path = Path(args.feature_stats)
        return path if path.is_absolute() else Path.cwd() / path

    data_root = Path(config.get("data", {}).get("data_root", "data"))
    candidates = [
        data_root / "stats_output.pt",
        data_root / "stats" / "stats_output.pt",
        Path("data") / "stats_output.pt",
        Path("data") / "stats" / "stats_output.pt",
        args.benchmark_root.parent / "data" / "stats_output.pt",
        args.benchmark_root.parent / "data" / "stats" / "stats_output.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_ligand_norm(stats_path: Optional[Path], enabled: bool) -> Optional[Dict]:
    if not enabled:
        return None
    if stats_path is None or not stats_path.exists():
        raise FileNotFoundError(
            "Ligand feature normalization is enabled, but feature stats were not found. "
            "Pass --feature-stats /path/to/data/stats_output.pt or disable with --no-ligand-normalization."
        )
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if "ligand" not in stats:
        raise KeyError(f"{stats_path} does not contain ligand stats")
    ligand_stats = stats["ligand"]
    return {
        "path": str(stats_path),
        "dims": _continuous_norm_dims(ligand_stats),
        "mean": torch.as_tensor(ligand_stats["mean"], dtype=torch.float32),
        "std": torch.as_tensor(ligand_stats["std"], dtype=torch.float32),
    }


def _normalize_ligand_features(atom_features: torch.Tensor, norm: Optional[Dict]) -> torch.Tensor:
    if norm is None:
        return atom_features.float()
    x = atom_features.float()
    dims = norm["dims"]
    if not dims:
        return x
    if x.shape[-1] != int(norm["mean"].numel()):
        raise ValueError(f"ligand atom feature dim={x.shape[-1]} expected={int(norm['mean'].numel())}")
    idx = torch.tensor(dims, dtype=torch.long)
    out = x.clone()
    out[..., idx] = (out[..., idx] - norm["mean"][idx]) / norm["std"][idx].clamp_min(1e-6)
    return out


def _atom_crop_bounds(n_atoms: int, max_atoms: int, policy: str) -> tuple[int, int]:
    n_atoms = int(n_atoms)
    max_atoms = int(max_atoms)
    if max_atoms < 0:
        return 0, n_atoms
    if max_atoms == 0:
        raise ValueError("max_atoms=0 is ambiguous; use a positive cap or -1 to disable atom truncation")
    if n_atoms <= max_atoms:
        return 0, n_atoms
    policy = str(policy or "head").lower()
    if policy in {"head", "first"}:
        return 0, max_atoms
    if policy in {"alternating_ends", "trim_ends", "center"}:
        excess = n_atoms - max_atoms
        drop_front = (excess + 1) // 2
        return drop_front, drop_front + max_atoms
    raise ValueError(f"unknown atom_truncation_policy={policy!r}")


def _load_keys(path: Path) -> list[str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return sorted(data.keys())


def _distances_to_labels(distances: torch.Tensor, contact: float, near: float) -> torch.Tensor:
    labels = torch.full_like(distances, 2, dtype=torch.long)
    labels[distances < contact] = 0
    labels[(distances >= contact) & (distances < near)] = 1
    return labels


def _compute_gvp_vectors(all_atom_coords, ca_coords: torch.Tensor) -> torch.Tensor:
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


def _chain_ids_to_int(chain_ids: Iterable[str], n: int) -> torch.Tensor:
    chain_ids = list(chain_ids or [])
    if len(chain_ids) < n:
        return torch.zeros(n, dtype=torch.long)
    unique = sorted(set(chain_ids[:n]))
    mapping = {chain: i for i, chain in enumerate(unique)}
    return torch.tensor([mapping[c] for c in chain_ids[:n]], dtype=torch.long)


class BenchmarkV11Dataset(Dataset):
    def __init__(
        self,
        benchmark_root: Path,
        dataset: str,
        tag_file: Path,
        protein_feature_root: Path,
        esm_root: Path,
        ligand_feature_root: Path,
        contact_thresh: float = 5.0,
        near_thresh: float = 8.0,
        max_seq_len: int = 1800,
        max_atoms: int = 128,
        tag_mode: str = "aggregate",
        ligand_mode: str = "real",
        shuffle_seed: int = 42,
        max_samples: int = 0,
        ligand_norm: Optional[Dict] = None,
        spatial_clamp_dims: Optional[List[int]] = None,
        spatial_clamp_range: Optional[List[float]] = None,
        atom_truncation_policy: str = "head",
    ) -> None:
        if tag_mode not in ("aggregate", "per_ligand"):
            raise ValueError(f"unknown tag_mode: {tag_mode}")
        if ligand_mode not in LIGAND_MODES:
            raise ValueError(f"unknown ligand_mode: {ligand_mode}")
        self.benchmark_root = Path(benchmark_root)
        self.dataset = dataset
        self.tag_mode = tag_mode
        self.ligand_mode = ligand_mode
        self.contact_thresh = float(contact_thresh)
        self.near_thresh = float(near_thresh)
        self.max_seq_len = int(max_seq_len)
        self.max_atoms = int(max_atoms)
        self.atom_truncation_policy = str(atom_truncation_policy or "head").lower()
        self.ligand_norm = ligand_norm
        self.spatial_clamp_dims = [int(d) for d in (spatial_clamp_dims or [])]
        clamp_range = spatial_clamp_range or [-5.0, 5.0]
        if len(clamp_range) != 2:
            raise ValueError("spatial_clamp_range must contain [min, max]")
        self.spatial_clamp_min = float(clamp_range[0])
        self.spatial_clamp_max = float(clamp_range[1])
        self.protein_root = protein_feature_root / dataset
        self.esm_root = esm_root / dataset
        self.ligand_root = ligand_feature_root / dataset
        self.dist_dict = torch.load(tag_file, map_location="cpu", weights_only=False)
        self.keys = sorted(self.dist_dict.keys())
        if max_samples and max_samples > 0:
            self.keys = self.keys[: int(max_samples)]

        if ligand_mode == "shuffle":
            rng = np.random.default_rng(shuffle_seed)
            perm = rng.permutation(len(self.keys))
            for i in range(len(perm)):
                if perm[i] == i and len(perm) > 1:
                    j = (i + 1) % len(perm)
                    perm[i], perm[j] = perm[j], perm[i]
            self._shuffle_idx = perm.tolist()
        else:
            self._shuffle_idx = None

    def __len__(self) -> int:
        return len(self.keys)

    def _protein_key(self, sample_key: str) -> str:
        return sample_key.split("_", 1)[0] if self.tag_mode == "per_ligand" else sample_key

    def _protein_path(self, protein_key: str) -> Path:
        candidates = [
            self.protein_root / protein_key / f"{protein_key}_protein.pt",
            self.protein_root / f"{protein_key}.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"protein feature not found for {protein_key}: {candidates[0]}")

    def _esm_path(self, protein_key: str) -> Path:
        path = self.esm_root / f"{protein_key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"ESM feature not found for {protein_key}: {path}")
        return path

    def _ligand_files(self, sample_key: str, protein_key: str) -> list[Path]:
        if self.tag_mode == "per_ligand":
            direct = self.ligand_root / f"{sample_key}.pt"
            return [direct] if direct.exists() else []
        direct = self.ligand_root / f"{protein_key}.pt"
        if direct.exists():
            return [direct]
        return sorted(self.ligand_root.glob(f"{protein_key}_*.pt"))

    def _merge_ligands(self, ligand_files: list[Path]) -> Dict[str, torch.Tensor]:
        if not ligand_files:
            raise FileNotFoundError("no ligand feature files")
        feat_blocks, coord_blocks, ei_blocks, ea_blocks = [], [], [], []
        offset = 0
        for path in ligand_files:
            lig = torch.load(path, map_location="cpu", weights_only=False)
            feats = _normalize_ligand_features(lig["atom_features"], self.ligand_norm)
            coords = lig["coords"].float()
            feat_blocks.append(feats)
            coord_blocks.append(coords)
            edge_index = lig.get("edge_index")
            edge_attr = lig.get("edge_attr")
            if edge_index is not None and edge_attr is not None and edge_index.numel() > 0:
                ei_blocks.append(edge_index.long() + offset)
                ea_blocks.append(edge_attr.float())
            offset += feats.shape[0]
        atom_features = torch.cat(feat_blocks, dim=0)
        coords = torch.cat(coord_blocks, dim=0)
        if ei_blocks:
            edge_index = torch.cat(ei_blocks, dim=1)
            edge_attr = torch.cat(ea_blocks, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        return {
            "atom_features": atom_features,
            "coords": coords,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    def _apply_ligand_mode(self, lig: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mode = self.ligand_mode
        if mode == "real":
            return lig
        x = lig["atom_features"]
        pos = lig["coords"]
        edge_index = lig["edge_index"]
        edge_attr = lig["edge_attr"]
        if mode == "empty":
            return {
                "atom_features": torch.zeros((0, x.shape[-1]), dtype=torch.float32),
                "coords": torch.zeros((0, 3), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, edge_attr.shape[-1] if edge_attr.ndim == 2 else 6), dtype=torch.float32),
            }
        if mode == "zero":
            return {
                "atom_features": torch.zeros_like(x),
                "coords": torch.zeros_like(pos),
                "edge_index": edge_index,
                "edge_attr": torch.zeros_like(edge_attr),
            }
        if mode == "shuffle":
            return lig
        raise ValueError(mode)

    def __getitem__(self, idx: int) -> Dict:
        sample_key = self.keys[idx]
        protein_key = self._protein_key(sample_key)
        ligand_key = sample_key
        if self.ligand_mode == "shuffle":
            ligand_key = self.keys[self._shuffle_idx[idx]]
        ligand_protein_key = self._protein_key(ligand_key)

        prot = torch.load(self._protein_path(protein_key), map_location="cpu", weights_only=False)
        esm = torch.load(self._esm_path(protein_key), map_location="cpu", weights_only=False)["embeddings"].float()
        lig = self._merge_ligands(self._ligand_files(ligand_key, ligand_protein_key))
        lig = self._apply_ligand_mode(lig)

        phys = prot["physchem_features"].float()
        scalar = prot["spatial_scalar_features"].float()
        vector = prot["spatial_vector_features"].float()
        ca = prot["ca_coords"].float()
        gvp_vectors = _compute_gvp_vectors(prot["all_atom_coords"], ca)
        chain_ids = _chain_ids_to_int(prot.get("chain_ids", []), phys.shape[0])

        dist_data = self.dist_dict[sample_key]
        distances = dist_data["distances"].float() if isinstance(dist_data, dict) else dist_data.float()
        labels = _distances_to_labels(distances, self.contact_thresh, self.near_thresh)

        lengths = {
            "protein": int(phys.shape[0]),
            "esm": int(esm.shape[0]),
            "labels": int(labels.shape[0]),
            "ligand_atoms_raw": int(lig["atom_features"].shape[0]),
        }
        n_res = min(lengths["protein"], lengths["esm"], lengths["labels"], self.max_seq_len)

        phys = phys[:n_res]
        scalar = scalar[:n_res]
        if self.spatial_clamp_dims:
            max_dim = max(self.spatial_clamp_dims)
            if scalar.shape[-1] <= max_dim:
                raise ValueError(
                    f"spatial_clamp_dims={self.spatial_clamp_dims} exceed scalar dim={scalar.shape[-1]}"
                )
            scalar = scalar.clone()
            scalar[..., self.spatial_clamp_dims] = scalar[..., self.spatial_clamp_dims].clamp(
                self.spatial_clamp_min, self.spatial_clamp_max
            )
        vector = vector[:n_res]
        ca = ca[:n_res]
        gvp_vectors = gvp_vectors[:n_res]
        chain_ids = chain_ids[:n_res]
        esm = esm[:n_res]
        labels = labels[:n_res]
        distances = distances[:n_res]

        ligand_center = (
            lig["coords"].mean(dim=0)
            if lig["coords"].numel() > 0
            else torch.zeros(3, dtype=torch.float32)
        )
        raw_n_atoms = int(lig["atom_features"].shape[0])
        atom_start, atom_end = _atom_crop_bounds(raw_n_atoms, self.max_atoms, self.atom_truncation_policy)
        n_atoms = atom_end - atom_start
        lig_feats = lig["atom_features"][atom_start:atom_end]
        lig_coords = lig["coords"][atom_start:atom_end]
        lig_edge_index = lig["edge_index"].long()
        lig_edge_attr = lig["edge_attr"].float()
        if lig_edge_index.numel() > 0:
            mask = (
                (lig_edge_index[0] >= atom_start)
                & (lig_edge_index[0] < atom_end)
                & (lig_edge_index[1] >= atom_start)
                & (lig_edge_index[1] < atom_end)
            )
            lig_edge_index = lig_edge_index[:, mask] - atom_start
            lig_edge_attr = lig_edge_attr[mask]

        return {
            "pdb_id": sample_key,
            "protein_key": protein_key,
            "ligand_key": ligand_key,
            "esm": esm,
            "protein_physchem": phys,
            "protein_spatial_scalar": scalar,
            "protein_spatial_vector": vector,
            "gvp_coords": ca,
            "gvp_vectors": gvp_vectors,
            "chain_ids": chain_ids,
            "ligand_features": lig_feats,
            "ligand_coords": lig_coords,
            "ligand_edge_index": lig_edge_index,
            "ligand_edge_attr": lig_edge_attr,
            "labels": labels.long(),
            "raw_distances": distances,
            "n_residues": n_res,
            "n_atoms": n_atoms,
            "ligand_center": ligand_center.float(),
            "length_info": lengths,
        }


def benchmark_collate(batch: List[Dict]) -> Dict:
    result = pocket_collate_fn_v10(batch)
    result["protein_keys"] = [sample["protein_key"] for sample in batch]
    result["ligand_keys"] = [sample["ligand_key"] for sample in batch]
    result["ligand_centers"] = torch.stack([sample["ligand_center"] for sample in batch], dim=0)
    result["length_info"] = [sample["length_info"] for sample in batch]
    return result


def _safe_ap(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        if labels.sum() == 0:
            return float("nan")
        return float(average_precision_score(labels, probs))
    except Exception:
        return float("nan")


def compute_sample_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    ca_coords: np.ndarray,
    ligand_center: np.ndarray,
    dcc_thresh: float = 4.0,
) -> Dict:
    row: Dict[str, float] = {"accuracy": float((preds == labels).mean())}
    for c in range(3):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        tn = int(((preds != c) & (labels != c)).sum())
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
        row[f"precision_c{c}"] = float(pr)
        row[f"recall_c{c}"] = float(rc)
        row[f"f1_c{c}"] = float(f1)
        row[f"iou_c{c}"] = float(iou)
        row[f"mcc_c{c}"] = float(mcc)
    row["ap_c0"] = _safe_ap(probs[:, 0], (labels == 0).astype(int))

    true_binding = labels <= 1
    pred_binding = preds <= 1
    tp = int((pred_binding & true_binding).sum())
    fp = int((pred_binding & ~true_binding).sum())
    fn = int((~pred_binding & true_binding).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    row["f1_binary"] = float(2 * pr * rc / (pr + rc + 1e-8))

    pred_contact = preds == 0
    if pred_contact.sum() > 0:
        pred_center = ca_coords[pred_contact].mean(axis=0)
    else:
        pred_center = ca_coords[int(np.argmax(probs[:, 0]))]
    row["pred_contact_center_x"] = float(pred_center[0])
    row["pred_contact_center_y"] = float(pred_center[1])
    row["pred_contact_center_z"] = float(pred_center[2])
    dcc_lig = float(np.linalg.norm(pred_center - ligand_center))
    row["dcc_ligand"] = dcc_lig
    row["dcc_ligand_success"] = float(dcc_lig < dcc_thresh)

    true_contact = labels == 0
    if true_contact.sum() > 0:
        true_center = ca_coords[true_contact].mean(axis=0)
        row["true_contact_center_x"] = float(true_center[0])
        row["true_contact_center_y"] = float(true_center[1])
        row["true_contact_center_z"] = float(true_center[2])
        dcc_contact = float(np.linalg.norm(pred_center - true_center))
        row["dcc_contact_center"] = dcc_contact
        row["dcc_contact_success"] = float(dcc_contact < dcc_thresh)
    else:
        row["true_contact_center_x"] = float("nan")
        row["true_contact_center_y"] = float("nan")
        row["true_contact_center_z"] = float("nan")
        row["dcc_contact_center"] = float("nan")
        row["dcc_contact_success"] = float("nan")

    top1_idx = int(np.argmax(probs[:, 0]))
    row["top1_ligand_dist"] = float(np.linalg.norm(ca_coords[top1_idx] - ligand_center))
    row["top1_ligand_success"] = float(row["top1_ligand_dist"] < dcc_thresh)
    return row


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not math.isnan(float(value))


def _mean_numeric(rows: Sequence[Dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and _is_finite_number(row[key])]
    return float(np.mean(vals)) if vals else float("nan")


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    vals = np.asarray([float(v) for v in values if _is_finite_number(v)], dtype=np.float64)
    if vals.size == 0:
        return {"min": float("nan"), "p10": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }


METRIC_KEYS = [
    "f1_c0",
    "precision_c0",
    "recall_c0",
    "iou_c0",
    "ap_c0",
    "mcc_c0",
    "f1_binary",
    "dcc_ligand",
    "dcc_ligand_success",
    "dcc_contact_center",
    "dcc_contact_success",
    "top1_ligand_dist",
    "top1_ligand_success",
    "n_pred_contact",
    "n_true_contact",
    "pdr_union_success",
    "pdr_union_dcc",
    "laa_success",
    "laa_rank",
    "laa_margin",
    "bp_f1_c0",
    "bp_precision_c0",
    "bp_iou_c0",
]

LOWER_IS_BETTER_METRICS = {
    "dcc_ligand",
    "dcc_contact_center",
    "top1_ligand_dist",
    "pdr_union_dcc",
    "laa_rank",
}


def _canonical_metric_name(key: str) -> str:
    names = {
        "f1_c0": "Mean_F1_c0",
        "precision_c0": "Mean_Prec_c0",
        "recall_c0": "Mean_Rec_c0",
        "iou_c0": "Mean_IoU_c0",
        "ap_c0": "Mean_AP_c0",
        "mcc_c0": "Mean_MCC_c0",
        "f1_binary": "Mean_F1_binary",
        "dcc_ligand": "DCC_Ligand",
        "dcc_ligand_success": "DCC_Ligand_Success",
        "dcc_contact_center": "DCC_ContactCenter",
        "dcc_contact_success": "DCC_ContactCenter_Success",
        "top1_ligand_dist": "Top1_Ligand_Dist",
        "top1_ligand_success": "Top1_Ligand_Success",
        "n_pred_contact": "Mean_pred_contact_count",
        "n_true_contact": "Mean_true_contact_count",
        "pdr_union_success": "PDR",
        "pdr_union_dcc": "PDR_Union_DCC",
        "laa_success": "LAA",
        "laa_rank": "LAA_Rank",
        "laa_margin": "LAA_Margin",
        "bp_f1_c0": "BP_F1_c0",
        "bp_precision_c0": "BP_Prec_c0",
        "bp_iou_c0": "BP_IoU_c0",
    }
    return names.get(key, key)


def _group_rows_by_key(rows: Sequence[Dict], key: str) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def _mean_over_proteins(rows: Sequence[Dict], metric_key: str, reducer: str) -> float:
    protein_groups = _group_rows_by_key(rows, "protein_key")
    vals = []
    for protein_rows in protein_groups.values():
        local_vals = [float(row[metric_key]) for row in protein_rows if metric_key in row and _is_finite_number(row[metric_key])]
        if not local_vals:
            continue
        if reducer == "mean":
            vals.append(float(np.mean(local_vals)))
        elif reducer == "max":
            # Keep the user-facing name protein_mean_of_ligand_max for
            # compatibility, but use best-of-ligands semantics for distances.
            if metric_key in LOWER_IS_BETTER_METRICS:
                vals.append(float(np.min(local_vals)))
            else:
                vals.append(float(np.max(local_vals)))
        else:
            raise ValueError(reducer)
    return float(np.mean(vals)) if vals else float("nan")


def _metric_block(rows: Sequence[Dict], reducer: str = "sample") -> Dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        if reducer == "sample":
            value = _mean_numeric(rows, key)
        else:
            value = _mean_over_proteins(rows, key, reducer)
        out[_canonical_metric_name(key)] = value
    return out


def _key_alignment_summary(rows: Sequence[Dict], tag_mode: str) -> Dict:
    by_protein = _group_rows_by_key(rows, "protein_key")
    ligand_keys = [str(row.get("ligand_key", "")) for row in rows]
    pdb_ids = [str(row.get("pdb_id", "")) for row in rows]
    shuffled = sum(1 for row in rows if row.get("pdb_id") != row.get("ligand_key"))
    ligand_counts = [len({str(row.get("ligand_key", "")) for row in protein_rows}) for protein_rows in by_protein.values()]
    return {
        "tag_mode": tag_mode,
        "n_samples": len(rows),
        "n_unique_pdb_ids": len(set(pdb_ids)),
        "n_unique_protein_keys": len(by_protein),
        "n_unique_ligand_keys": len(set(ligand_keys)),
        "n_rows_with_ligand_key_mismatch": int(shuffled),
        "ligands_per_protein": _quantiles(ligand_counts),
    }


def _contact_count_summary(rows: Sequence[Dict]) -> Dict:
    return {
        "true_contact_count": _quantiles([row.get("n_true_contact", float("nan")) for row in rows]),
        "pred_contact_count": _quantiles([row.get("n_pred_contact", float("nan")) for row in rows]),
        "n_zero_true_contact": int(sum(1 for row in rows if int(row.get("n_true_contact", -1)) == 0)),
        "n_zero_pred_contact": int(sum(1 for row in rows if int(row.get("n_pred_contact", -1)) == 0)),
        "residue_count": _quantiles([row.get("n_residues", float("nan")) for row in rows]),
        "ligand_atom_count_after_crop": _quantiles([row.get("n_atoms", float("nan")) for row in rows]),
        "ligand_atom_count_raw": _quantiles([row.get("ligand_atoms_raw", float("nan")) for row in rows]),
    }


def _center_from_row(row: Dict, prefix: str) -> Optional[np.ndarray]:
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    if not all(key in row and _is_finite_number(row[key]) for key in keys):
        return None
    return np.asarray([float(row[key]) for key in keys], dtype=np.float64)


def enrich_rows_with_interpretability_metrics(rows: List[Dict], dcc_thresh: float) -> None:
    """Add PDR/LAA/BP diagnostics in-place.

    PDR uses the protein-level union of per-ligand true pocket centers: it asks
    whether the predicted contact center hits any real pocket for the protein.
    LAA is defined only for proteins with at least two ligand pockets and asks
    whether the current ligand's pocket is the nearest true pocket to the
    prediction. BP is the contact-boundary quality conditioned on correct
    current-ligand pocket localization.
    """
    by_protein = _group_rows_by_key(rows, "protein_key")
    for protein_rows in by_protein.values():
        true_centers = []
        for row in protein_rows:
            center = _center_from_row(row, "true_contact_center")
            if center is not None:
                true_centers.append(center)
        true_centers_arr = np.stack(true_centers, axis=0) if true_centers else None

        for row in protein_rows:
            pred_center = _center_from_row(row, "pred_contact_center")
            true_center = _center_from_row(row, "true_contact_center")
            if pred_center is not None and true_centers_arr is not None:
                dists = np.linalg.norm(true_centers_arr - pred_center[None, :], axis=1)
                row["pdr_union_dcc"] = float(np.min(dists))
                row["pdr_union_success"] = float(row["pdr_union_dcc"] < dcc_thresh)
            else:
                row["pdr_union_dcc"] = float("nan")
                row["pdr_union_success"] = float("nan")

            row["laa_success"] = float("nan")
            row["laa_rank"] = float("nan")
            row["laa_margin"] = float("nan")
            if pred_center is not None and true_center is not None and true_centers_arr is not None and len(true_centers_arr) >= 2:
                dists = np.linalg.norm(true_centers_arr - pred_center[None, :], axis=1)
                current_dist = float(np.linalg.norm(true_center - pred_center))
                rank = int((dists < current_dist - 1e-8).sum() + 1)
                row["laa_rank"] = float(rank)
                row["laa_success"] = float(rank == 1 and current_dist < dcc_thresh)
                if len(dists) >= 2:
                    sorted_dists = np.sort(dists)
                    row["laa_margin"] = float(sorted_dists[1] - sorted_dists[0])

            detected_current = bool(_is_finite_number(row.get("dcc_contact_success", float("nan"))) and float(row["dcc_contact_success"]) >= 0.5)
            row["bp_f1_c0"] = float(row["f1_c0"]) if detected_current else float("nan")
            row["bp_precision_c0"] = float(row["precision_c0"]) if detected_current else float("nan")
            row["bp_iou_c0"] = float(row["iou_c0"]) if detected_current else float("nan")


def aggregate_metrics(rows: List[Dict], tag_mode: str = "aggregate") -> Dict:
    def mean(key: str) -> float:
        return _mean_numeric(rows, key)

    legacy = {
        "N_evaluated": len(rows),
        "Mean_F1_c0": mean("f1_c0"),
        "Mean_Prec_c0": mean("precision_c0"),
        "Mean_Rec_c0": mean("recall_c0"),
        "Mean_IoU_c0": mean("iou_c0"),
        "Mean_AP_c0": mean("ap_c0"),
        "Mean_MCC_c0": mean("mcc_c0"),
        "Mean_F1_binary": mean("f1_binary"),
        "DCC_Ligand": mean("dcc_ligand"),
        "DCC_Ligand_Success": mean("dcc_ligand_success"),
        "DCC_ContactCenter": mean("dcc_contact_center"),
        "DCC_ContactCenter_Success": mean("dcc_contact_success"),
        "Top1_Ligand_Dist": mean("top1_ligand_dist"),
        "Top1_Ligand_Success": mean("top1_ligand_success"),
        "Mean_pred_contact_count": mean("n_pred_contact"),
        "Mean_true_contact_count": mean("n_true_contact"),
    }
    summary = dict(legacy)
    summary.update({
        "sample_mean": _metric_block(rows, "sample"),
        "protein_mean_of_ligand_mean": _metric_block(rows, "mean"),
        "protein_mean_of_ligand_max": _metric_block(rows, "max"),
        "key_alignment": _key_alignment_summary(rows, tag_mode),
        "contact_count_distribution": _contact_count_summary(rows),
        "metric_definitions": {
            "sample_mean": "Mean over every row in the benchmark CSV.",
            "protein_mean_of_ligand_mean": "Mean over proteins after averaging ligand rows within each protein.",
            "protein_mean_of_ligand_max": "Mean over proteins after taking the best ligand row within each protein for each metric; distance-like metrics use min distance.",
            "PDR": "Pocket Detection Rate: predicted contact center is within dcc_thresh of any true ligand-pocket center for the same protein.",
            "LAA": "Ligand Assignment Accuracy: for proteins with >=2 ligand-pocket centers, the current ligand pocket is the nearest true pocket to the predicted center and is within dcc_thresh.",
            "BP": "Boundary Precision: contact boundary F1/precision/IoU averaged only over rows whose predicted center localizes the current ligand pocket within dcc_thresh.",
        },
    })
    return summary


CSV_FIELDS = [
    "dataset",
    "pdb_id",
    "protein_key",
    "ligand_key",
    "n_residues",
    "n_atoms",
    "n_pred_contact",
    "n_true_contact",
    "accuracy",
    "precision_c0",
    "recall_c0",
    "f1_c0",
    "iou_c0",
    "ap_c0",
    "mcc_c0",
    "precision_c1",
    "recall_c1",
    "f1_c1",
    "iou_c1",
    "precision_c2",
    "recall_c2",
    "f1_c2",
    "iou_c2",
    "f1_binary",
    "dcc_ligand",
    "dcc_ligand_success",
    "dcc_contact_center",
    "dcc_contact_success",
    "pdr_union_dcc",
    "pdr_union_success",
    "laa_success",
    "laa_rank",
    "laa_margin",
    "bp_f1_c0",
    "bp_precision_c0",
    "bp_iou_c0",
    "top1_ligand_dist",
    "top1_ligand_success",
    "pred_contact_center_x",
    "pred_contact_center_y",
    "pred_contact_center_z",
    "true_contact_center_x",
    "true_contact_center_y",
    "true_contact_center_z",
    "len_protein",
    "len_esm",
    "len_labels",
    "ligand_atoms_raw",
]


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_dataset(
    args,
    dataset_name: str,
    model,
    model_version: str,
    device: torch.device,
    config: Dict,
    out_dir: Path,
    ligand_norm: Optional[Dict] = None,
    spatial_clamp_dims: Optional[List[int]] = None,
    spatial_clamp_range: Optional[List[float]] = None,
    atom_truncation_policy: str = "head",
) -> Dict:
    tag_name = (
        f"{dataset_name}_aggregate_pdb_dmin.pt"
        if args.tag_mode == "aggregate"
        else f"{dataset_name}_per_ligand_dmin.pt"
    )
    tag_file = _as_path(args.benchmark_root, args.tag_root) / tag_name
    ds = BenchmarkV11Dataset(
        benchmark_root=args.benchmark_root,
        dataset=dataset_name,
        tag_file=tag_file,
        protein_feature_root=_as_path(args.benchmark_root, args.protein_feature_root),
        esm_root=_as_path(args.benchmark_root, args.esm_root),
        ligand_feature_root=_as_path(args.benchmark_root, args.ligand_feature_root),
        contact_thresh=args.contact_thresh,
        near_thresh=args.near_thresh,
        max_seq_len=args.max_seq_len,
        max_atoms=args.max_atoms,
        tag_mode=args.tag_mode,
        ligand_mode=args.ligand_mode,
        shuffle_seed=args.shuffle_seed,
        max_samples=args.max_samples,
        ligand_norm=ligand_norm,
        spatial_clamp_dims=spatial_clamp_dims,
        spatial_clamp_range=spatial_clamp_range,
        atom_truncation_policy=atom_truncation_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=benchmark_collate,
        pin_memory=(device.type == "cuda" and not args.no_pin_memory),
    )
    needs_unimol = model_needs_unimol(model)
    d_unimol = int(getattr(model, "d_unimol", config.get("model", {}).get("d_unimol", 512)))
    rows: List[Dict] = []
    amp_enabled = bool(args.amp and device.type != "cpu")
    for batch in tqdm(loader, desc=f"{dataset_name} {args.tag_mode}/{args.ligand_mode}", unit="batch"):
        inputs = prepare_model_inputs_v11(batch, device, non_blocking=True)
        if needs_unimol:
            inputs["unimol_embed"] = torch.zeros(
                batch["protein_mask"].shape[0], d_unimol, device=device, dtype=torch.float32
            )
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if args.class0_logit_bias:
            logits[..., 0] = logits[..., 0] + float(args.class0_logit_bias)
        probs = F.softmax(logits.float(), dim=-1).cpu()
        preds = probs.argmax(dim=-1)
        labels_b = batch["labels"]
        mask_b = batch["protein_mask"]
        coords_b = batch["protein_coords"]
        centers_b = batch["ligand_centers"]
        for b, sample_key in enumerate(batch["pdb_ids"]):
            valid = mask_b[b].bool()
            probs_np = probs[b][valid].numpy()
            preds_np = preds[b][valid].numpy()
            labels_np = labels_b[b][valid].numpy()
            ca_np = coords_b[b][valid].numpy()
            center_np = centers_b[b].numpy()
            metrics = compute_sample_metrics(
                preds_np,
                labels_np,
                probs_np,
                ca_np,
                center_np,
                dcc_thresh=args.dcc_thresh,
            )
            length_info = batch["length_info"][b]
            row = {
                "dataset": dataset_name,
                "pdb_id": sample_key,
                "protein_key": batch["protein_keys"][b],
                "ligand_key": batch["ligand_keys"][b],
                "n_residues": int(valid.sum()),
                "n_atoms": int(batch["n_atoms"][b]),
                "n_pred_contact": int((preds_np == 0).sum()),
                "n_true_contact": int((labels_np == 0).sum()),
                "len_protein": length_info["protein"],
                "len_esm": length_info["esm"],
                "len_labels": length_info["labels"],
                "ligand_atoms_raw": length_info["ligand_atoms_raw"],
            }
            row.update(metrics)
            rows.append(row)
    enrich_rows_with_interpretability_metrics(rows, dcc_thresh=args.dcc_thresh)
    summary = aggregate_metrics(rows, tag_mode=args.tag_mode)
    save_csv(rows, out_dir / f"{dataset_name}_all_results.csv")
    with (out_dir / f"{dataset_name}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\n[{dataset_name}] {model_version} {args.tag_mode}/{args.ligand_mode}")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--datasets", default="coach420,holo4k")
    parser.add_argument("--protein-feature-root", default="features_fixed_conservative/protein_v4")
    parser.add_argument("--esm-root", default="esm2")
    parser.add_argument("--ligand-feature-root", default="ligand_features_v5")
    parser.add_argument("--tag-root", default="residue_ligand_tags/atom_h")
    parser.add_argument("--tag-mode", default="aggregate", choices=["aggregate", "per_ligand"])
    parser.add_argument("--ligand-mode", default="real", choices=LIGAND_MODES)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1800)
    parser.add_argument("--max-atoms", type=int, default=128, help="Maximum ligand atoms; use -1 to disable atom truncation")
    parser.add_argument("--atom-truncation-policy", default="head", choices=["head", "first", "alternating_ends", "trim_ends", "center"])
    parser.add_argument("--contact-thresh", type=float, default=5.0)
    parser.add_argument("--near-thresh", type=float, default=8.0)
    parser.add_argument("--dcc-thresh", type=float, default=4.0)
    parser.add_argument("--class0-logit-bias", type=float, default=0.0)
    parser.add_argument("--feature-stats", default=None, help="Training feature stats for ligand V5 normalization")
    parser.add_argument("--no-ligand-normalization", action="store_true")
    parser.add_argument("--no-spatial-clamp", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = load_config_from_checkpoint_or_yaml(ckpt, args.config)
    stats_path = _resolve_feature_stats_path(args, config)
    ligand_norm = _load_ligand_norm(stats_path, enabled=not args.no_ligand_normalization)
    data_cfg = config.get("data", {})
    spatial_clamp_dims = [] if args.no_spatial_clamp or not data_cfg.get("use_spatial_clamp", False) else data_cfg.get("spatial_clamp_dims", [])
    spatial_clamp_range = data_cfg.get("spatial_clamp_range", [-5.0, 5.0])
    if ligand_norm is not None:
        print(f"Ligand normalization: {ligand_norm['path']} dims={len(ligand_norm['dims'])}")
    else:
        print("Ligand normalization: disabled")
    print(f"Spatial clamp: dims={spatial_clamp_dims} range={spatial_clamp_range}")
    model, version = create_v11_model(ckpt, config, device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path("results") / f"benchmark_{version}_{args.tag_mode}_{args.ligand_mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    all_summaries = {}
    for dataset_name in datasets:
        all_summaries[dataset_name] = run_dataset(
            args,
            dataset_name,
            model,
            version,
            device,
            config,
            out_dir,
            ligand_norm=ligand_norm,
            spatial_clamp_dims=spatial_clamp_dims,
            spatial_clamp_range=spatial_clamp_range,
            atom_truncation_policy=args.atom_truncation_policy,
        )

    meta = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config else None,
        "benchmark_root": str(args.benchmark_root),
        "model_version": version,
        "datasets": datasets,
        "tag_mode": args.tag_mode,
        "ligand_mode": args.ligand_mode,
        "class0_logit_bias": args.class0_logit_bias,
        "feature_stats": str(stats_path) if stats_path else None,
        "ligand_normalization": None if ligand_norm is None else {"path": ligand_norm["path"], "n_dims": len(ligand_norm["dims"]), "dims": ligand_norm["dims"]},
        "spatial_clamp_dims": spatial_clamp_dims,
        "spatial_clamp_range": spatial_clamp_range,
        "protein_feature_root": args.protein_feature_root,
        "esm_root": args.esm_root,
        "ligand_feature_root": args.ligand_feature_root,
        "tag_root": args.tag_root,
        "max_seq_len": args.max_seq_len,
        "max_atoms": args.max_atoms,
        "atom_truncation_policy": args.atom_truncation_policy,
        "timestamp": ts,
        "summaries": all_summaries,
    }
    with (out_dir / "benchmark_summary_all.json").open("w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"\nSaved benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()
