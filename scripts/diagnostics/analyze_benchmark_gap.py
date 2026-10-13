#!/usr/bin/env python3
"""Diagnose COACH420 vs HOLO4K benchmark gaps.

This script is intentionally read-only.  It consumes the benchmark result CSVs
produced by scripts/infer/infer_v11_benchmark.py and compares sample-level
metrics, truncation rates, label priors, and feature numeric ranges against a
sample of the training data.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch


NUMERIC_COLUMNS = {
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
    "top1_ligand_dist",
    "top1_ligand_success",
    "len_protein",
    "len_esm",
    "len_labels",
    "ligand_atoms_raw",
}


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return float(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))


def _quantiles(vals: list[float], probs=(0.1, 0.5, 0.9, 0.99)) -> dict[str, float]:
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return {f"p{int(p * 100)}": float("nan") for p in probs}
    out = {}
    for p in probs:
        idx = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
        out[f"p{int(p * 100)}"] = float(vals[idx])
    return out


def _summarize_values(vals: list[float]) -> dict[str, float]:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "std": _std(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        **_quantiles(vals),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in NUMERIC_COLUMNS:
                    row[key] = float(value) if value not in ("", "nan", "NaN") else float("nan")
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "f1_c0",
        "precision_c0",
        "recall_c0",
        "ap_c0",
        "dcc_ligand",
        "dcc_contact_center",
        "top1_ligand_dist",
        "n_residues",
        "ligand_atoms_raw",
        "n_true_contact",
        "n_pred_contact",
    ]
    out: dict[str, Any] = {"N": len(rows)}
    for key in keys:
        out[key] = _summarize_values([float(r[key]) for r in rows])
    out["true_contact_fraction"] = _summarize_values(
        [float(r["n_true_contact"]) / max(1.0, float(r["n_residues"])) for r in rows]
    )
    out["pred_contact_fraction"] = _summarize_values(
        [float(r["n_pred_contact"]) / max(1.0, float(r["n_residues"])) for r in rows]
    )
    out["pred_true_count_ratio"] = _summarize_values(
        [float(r["n_pred_contact"]) / max(1.0, float(r["n_true_contact"])) for r in rows]
    )
    return out


def _bin_rows(rows: list[dict[str, Any]], key: str, bins: list[tuple[float, float, str]]) -> dict[str, Any]:
    out = {}
    for lo, hi, label in bins:
        part = [r for r in rows if lo <= float(r[key]) < hi]
        out[label] = _metric_summary(part) if part else {"N": 0}
    return out


def _flag_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[Any]] = {}
    for row in rows:
        flags: list[str] = []
        if int(row["len_protein"]) != int(row["len_labels"]):
            flags.append("protein_label_len_mismatch")
        if int(row["len_protein"]) != int(row["len_esm"]):
            flags.append("protein_esm_len_mismatch")
        if int(row["len_protein"]) > int(row["n_residues"]):
            flags.append("effective_residue_truncation")
        if int(row["len_protein"]) > 1800:
            flags.append("max_seq_cap_1800")
        if int(row["ligand_atoms_raw"]) > int(row["n_atoms"]):
            flags.append("ligand_atom_cap_128")
        if not flags:
            counts["clean"] += 1
        for flag in flags:
            counts[flag] += 1
            examples.setdefault(flag, [])
            if len(examples[flag]) < 8:
                examples[flag].append(
                    {
                        "pdb_id": row["pdb_id"],
                        "len_protein": int(row["len_protein"]),
                        "len_esm": int(row["len_esm"]),
                        "len_labels": int(row["len_labels"]),
                        "n_residues": int(row["n_residues"]),
                        "ligand_atoms_raw": int(row["ligand_atoms_raw"]),
                        "n_atoms": int(row["n_atoms"]),
                        "f1_c0": float(row["f1_c0"]),
                    }
                )
    return {"counts": dict(counts), "examples": examples}


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str = "f1_c0") -> float:
    pairs = [
        (float(r[x_key]), float(r[y_key]))
        for r in rows
        if not math.isnan(float(r[x_key])) and not math.isnan(float(r[y_key]))
    ]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = _mean(list(xs)), _mean(list(ys))
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return float(num / den) if den > 0 else float("nan")


class Stat:
    def __init__(self, reservoir: int) -> None:
        self.reservoir = reservoir
        self.n = 0
        self.total = 0.0
        self.sq_total = 0.0
        self.min_val = float("inf")
        self.max_val = -float("inf")
        self.nan = 0
        self.inf = 0
        self.abs_sample: list[float] = []

    def add(self, value: torch.Tensor | None) -> None:
        if value is None:
            return
        tensor = value.detach().float().flatten()
        if tensor.numel() == 0:
            return
        self.nan += int(torch.isnan(tensor).sum().item())
        self.inf += int(torch.isinf(tensor).sum().item())
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            return
        self.n += int(finite.numel())
        self.total += float(finite.sum().item())
        self.sq_total += float((finite * finite).sum().item())
        self.min_val = min(self.min_val, float(finite.min().item()))
        self.max_val = max(self.max_val, float(finite.max().item()))
        vals = finite.abs()
        if vals.numel() > 20_000:
            idx = torch.linspace(0, vals.numel() - 1, 20_000).long()
            vals = vals[idx]
        self.abs_sample.extend(float(v) for v in vals.cpu().tolist())
        if len(self.abs_sample) > self.reservoir:
            step = max(2, math.ceil(len(self.abs_sample) / self.reservoir))
            self.abs_sample = self.abs_sample[::step][: self.reservoir]

    def to_dict(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "nan": self.nan, "inf": self.inf}
        mean = self.total / self.n
        var = max(0.0, self.sq_total / self.n - mean * mean)
        abs_vals = sorted(self.abs_sample)

        def q(prob: float) -> float:
            if not abs_vals:
                return float("nan")
            idx = min(len(abs_vals) - 1, max(0, int(round(prob * (len(abs_vals) - 1)))))
            return float(abs_vals[idx])

        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min_val,
            "max": self.max_val,
            "abs_p95": q(0.95),
            "abs_p99": q(0.99),
            "abs_p999": q(0.999),
            "nan": self.nan,
            "inf": self.inf,
        }


def _load_pt(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _first_match(pattern: str) -> str | None:
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _make_feature_stats(names: list[str], reservoir: int) -> dict[str, Any]:
    stat_keys = [
        "protein_len",
        "prot_phys",
        "prot_scalar",
        "prot_raw_scalar",
        "prot_vec_norm",
        "esm",
        "lig_atom_count",
        "lig_local54",
        "lig_global55",
        "lig_coord",
        "edge_attr",
    ]
    return {
        name: {key: Stat(reservoir) for key in stat_keys} | {"errors": []}
        for name in names
    }


def _update_feature_group(
    stats: dict[str, Any],
    name: str,
    prot_path: str | None,
    esm_path: str | None,
    ligand_paths: list[str],
) -> None:
    if prot_path:
        try:
            prot = _load_pt(prot_path)
            stats[name]["protein_len"].add(
                torch.tensor([prot.get("n_residues", prot["physchem_features"].shape[0])])
            )
            stats[name]["prot_phys"].add(prot.get("physchem_features"))
            stats[name]["prot_scalar"].add(prot.get("spatial_scalar_features"))
            stats[name]["prot_raw_scalar"].add(prot.get("spatial_scalar_raw_features"))
            vec = prot.get("spatial_vector_features")
            if vec is not None:
                stats[name]["prot_vec_norm"].add(torch.linalg.norm(vec.float(), dim=-1))
        except Exception as exc:  # noqa: BLE001
            stats[name]["errors"].append(["protein", prot_path, str(exc)])
    if esm_path:
        try:
            esm = _load_pt(esm_path)
            emb = esm["embeddings"] if isinstance(esm, dict) and "embeddings" in esm else esm
            stats[name]["esm"].add(emb)
        except Exception as exc:  # noqa: BLE001
            stats[name]["errors"].append(["esm", esm_path, str(exc)])
    for ligand_path in ligand_paths:
        try:
            lig = _load_pt(ligand_path)
            atom = lig["atom_features"].float()
            stats[name]["lig_atom_count"].add(torch.tensor([atom.shape[0]], dtype=torch.float32))
            stats[name]["lig_local54"].add(atom[:, :54])
            stats[name]["lig_global55"].add(atom[:, 54:109])
            stats[name]["lig_coord"].add(lig.get("coords"))
            stats[name]["edge_attr"].add(lig.get("edge_attr"))
        except Exception as exc:  # noqa: BLE001
            stats[name]["errors"].append(["ligand", ligand_path, str(exc)])


def _training_ids(split_file: Path, max_samples: int, rng: random.Random) -> list[str]:
    ids = []
    with split_file.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "train":
                ids.append(parts[1])
    if max_samples and len(ids) > max_samples:
        ids = rng.sample(ids, max_samples)
    return ids


def _feature_stats(args: argparse.Namespace, rows_by_dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rng = random.Random(args.seed)
    stats = _make_feature_stats(["train", "coach420", "holo4k"], args.reservoir)

    for pid in _training_ids(args.split_file, args.max_train_features, rng):
        prot = _first_match(str(args.train_protein_root / "*" / pid / f"{pid}_protein_conf1.pt"))
        if prot is None:
            prot = _first_match(str(args.train_protein_root / "*" / pid / f"{pid}_protein_conf*.pt"))
        esm = _first_match(str(args.train_esm_root / "*" / pid / f"{pid}_protein.pt"))
        lig = _first_match(str(args.train_ligand_root / "*" / pid / f"{pid}_ligand.pt"))
        _update_feature_group(stats, "train", prot, esm, [lig] if lig else [])

    for dataset in ("coach420", "holo4k"):
        rows = list(rows_by_dataset.get(dataset, []))
        if dataset == "holo4k" and args.max_holo_features and len(rows) > args.max_holo_features:
            rows = rng.sample(rows, args.max_holo_features)
        if dataset == "coach420" and args.max_coach_features and len(rows) > args.max_coach_features:
            rows = rng.sample(rows, args.max_coach_features)
        for row in rows:
            pid = row["protein_key"]
            prot = args.benchmark_root / "features_fixed_conservative" / "protein_v4" / dataset / pid / f"{pid}_protein.pt"
            esm = args.benchmark_root / "esm2" / dataset / f"{pid}.pt"
            ligand_paths = sorted(
                glob.glob(str(args.benchmark_root / "ligand_features_v5" / dataset / f"{pid}_*.pt"))
            )
            _update_feature_group(
                stats,
                dataset,
                str(prot) if prot.exists() else None,
                str(esm) if esm.exists() else None,
                ligand_paths,
            )

    out: dict[str, Any] = {}
    for name, group in stats.items():
        out[name] = {
            key: value.to_dict() for key, value in group.items() if key != "errors"
        }
        out[name]["n_errors"] = len(group["errors"])
        out[name]["sample_errors"] = group["errors"][:10]
    return out


def _checkpoint_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    ckpt = _load_pt(path)
    cfg = ckpt.get("config", {})
    state = ckpt.get("model_state_dict", {})
    probes = [
        "local54_edge_context_",
        "local54_compat_",
        "local54_interaction_",
        "local_evidence_",
        "unimol_memory_",
        "local54_group_adapter.",
    ]
    return {
        "checkpoint": str(path),
        "ckpt_keys": sorted(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "best_val_f1_c0": ckpt.get("best_val_f1_c0"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "experiment_name": cfg.get("experiment", {}).get("name"),
        "version_suffix": cfg.get("experiment", {}).get("version_suffix"),
        "probe_counts": {probe: sum(1 for key in state if key.startswith(probe)) for probe in probes},
        "relevant_model_flags": {
            key: value
            for key, value in cfg.get("model", {}).items()
            if any(token in key.lower() for token in ("v11", "v12", "local54", "compat", "edge", "unimol", "group"))
        },
    }


def _build_report(results: dict[str, Any]) -> str:
    lines = ["# Benchmark Gap Diagnostic", ""]
    ckpt = results.get("checkpoint", {})
    if ckpt:
        lines += [
            "## Checkpoint",
            f"- checkpoint: `{ckpt.get('checkpoint')}`",
            f"- experiment/version: `{ckpt.get('experiment_name')}` / `{ckpt.get('version_suffix')}`",
            f"- epoch/best_val_f1_c0: `{ckpt.get('epoch')}` / `{ckpt.get('best_val_f1_c0')}`",
            f"- probe_counts: `{ckpt.get('probe_counts')}`",
            "",
        ]
    lines.append("## Overall")
    for dataset in ("coach420", "holo4k"):
        overall = results["datasets"][dataset]["overall"]
        lines.append(
            "- "
            f"{dataset}: N={overall['N']}, "
            f"F1={overall['f1_c0']['mean']:.4f}, "
            f"P/R={overall['precision_c0']['mean']:.4f}/{overall['recall_c0']['mean']:.4f}, "
            f"DCC_lig={overall['dcc_ligand']['mean']:.2f}, "
            f"n_res={overall['n_residues']['mean']:.1f}, "
            f"lig_atoms_raw={overall['ligand_atoms_raw']['mean']:.1f}"
        )
    lines += ["", "## Key Stratification"]
    for bin_name in ("n_residues", "ligand_atoms_raw", "true_contacts"):
        lines.append(f"### {bin_name}")
        for label in results["datasets"]["holo4k"]["bins"][bin_name]:
            coach = results["datasets"]["coach420"]["bins"][bin_name].get(label, {"N": 0})
            holo = results["datasets"]["holo4k"]["bins"][bin_name].get(label, {"N": 0})
            cf1 = coach.get("f1_c0", {}).get("mean", float("nan")) if coach.get("N", 0) else float("nan")
            hf1 = holo.get("f1_c0", {}).get("mean", float("nan")) if holo.get("N", 0) else float("nan")
            lines.append(f"- {label}: coach N={coach.get('N', 0)} F1={cf1:.4f}; holo N={holo.get('N', 0)} F1={hf1:.4f}")
    lines += ["", "## Feature Range Snapshot"]
    for group in ("train", "coach420", "holo4k"):
        feat = results.get("feature_stats", {}).get(group, {})
        lines.append(f"### {group}")
        for key in ("protein_len", "prot_scalar", "esm", "lig_atom_count", "lig_local54", "lig_global55", "edge_attr"):
            stat = feat.get(key, {})
            if stat.get("n", 0):
                lines.append(
                    "- "
                    f"{key}: mean={stat.get('mean', float('nan')):.4g}, "
                    f"std={stat.get('std', float('nan')):.4g}, "
                    f"abs_p99={stat.get('abs_p99', float('nan')):.4g}, "
                    f"min/max={stat.get('min', float('nan')):.4g}/{stat.get('max', float('nan')):.4g}, "
                    f"nan/inf={stat.get('nan', 0)}/{stat.get('inf', 0)}"
                )
        if feat.get("n_errors"):
            lines.append(f"- errors: {feat['n_errors']} sample={feat.get('sample_errors', [])[:2]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/splits/splits.txt"))
    parser.add_argument("--train-protein-root", type=Path, default=Path("data/complete_features_v4_nma"))
    parser.add_argument("--train-esm-root", type=Path, default=Path("data/esm2_latents"))
    parser.add_argument("--train-ligand-root", type=Path, default=Path("data/ligand_features_v5"))
    parser.add_argument("--max-train-features", type=int, default=1000)
    parser.add_argument("--max-coach-features", type=int, default=0)
    parser.add_argument("--max-holo-features", type=int, default=1000)
    parser.add_argument("--reservoir", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows_by_dataset = {
        "coach420": _read_rows(args.result_dir / "coach420_all_results.csv"),
        "holo4k": _read_rows(args.result_dir / "holo4k_all_results.csv"),
    }
    bins = {
        "n_residues": [(0, 200, "<200"), (200, 400, "200-399"), (400, 800, "400-799"), (800, 1200, "800-1199"), (1200, 1801, "1200-1800")],
        "ligand_atoms_raw": [(0, 32, "<32"), (32, 64, "32-63"), (64, 128, "64-127"), (128, 256, "128-255"), (256, 1e9, ">=256")],
        "true_contacts": [(0, 20, "<20"), (20, 40, "20-39"), (40, 80, "40-79"), (80, 160, "80-159"), (160, 1e9, ">=160")],
    }
    results: dict[str, Any] = {
        "checkpoint": _checkpoint_info(args.checkpoint),
        "datasets": {},
    }
    for dataset, rows in rows_by_dataset.items():
        results["datasets"][dataset] = {
            "overall": _metric_summary(rows),
            "flags": _flag_summary(rows),
            "bins": {
                name: _bin_rows(rows, key if key != "true_contacts" else "n_true_contact", spec)
                for name, (key, spec) in {
                    "n_residues": ("n_residues", bins["n_residues"]),
                    "ligand_atoms_raw": ("ligand_atoms_raw", bins["ligand_atoms_raw"]),
                    "true_contacts": ("n_true_contact", bins["true_contacts"]),
                }.items()
            },
            "correlations_with_f1": {
                key: _correlation(rows, key)
                for key in ("n_residues", "ligand_atoms_raw", "n_true_contact", "n_pred_contact", "dcc_ligand")
            },
        }
    results["feature_stats"] = _feature_stats(args, rows_by_dataset)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    args.output_md.write_text(_build_report(results))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
