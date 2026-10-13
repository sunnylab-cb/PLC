#!/usr/bin/env python3
"""Generate assets for the PLC project internal report."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
DATA = REPORT / "data"
REMOTE_SUM = DATA / "remote_summaries"


def ensure_dirs() -> None:
    for path in (REPORT, FIG, DATA):
        path.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def count_files(pattern: str) -> int:
    return len(list(ROOT.glob(pattern)))


def collect_dataset_qc() -> dict[str, object]:
    summary = {
        "fixed_pdb": {
            "coach420": count_files("benchmark/p2rank-datasets-fixed_hres/coach420/*.pdb"),
            "holo4k": count_files("benchmark/p2rank-datasets-fixed_hres/holo4k/*.pdb"),
        },
        "esm3_raw": {
            "coach420": count_files("benchmark/esm3_latents/coach420/*.pt"),
            "holo4k": count_files("benchmark/esm3_latents/holo4k/*.pt"),
        },
        "esm3_norm": {
            "coach420": count_files("benchmark/esm3_latents_norm/coach420/*.pt"),
            "holo4k": count_files("benchmark/esm3_latents_norm/holo4k/*.pt"),
        },
        "stats_files": {
            "protein_ligand_stats": str(ROOT / "data/stats/stats_output.pt"),
            "esm3_stats": str(ROOT / "data/stats/esm3_stats_output.pt"),
        },
        "notes": [
            "COACH420/HOLO4K PDBs were processed with conservative fixed-heavy-residue structures for feature extraction.",
            "ESM3 benchmark features were extracted locally and normalized with the training-set ESM3 statistics.",
            "Benchmark inference uses per-ligand tags and max_atoms=-1 for the current no-atom-cap evaluation.",
        ],
    }
    write_json(DATA / "dataset_qc_summary.json", summary)
    return summary


def parse_ligdep() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    core_path = ROOT / "tmp/v11m6_v11m6n_ligdep_core_ledger.csv"
    split_path = ROOT / "tmp/v11m6_v11m6n_ligdep_split_ledger.csv"
    core_rows: list[dict[str, object]] = []
    for row in load_csv(core_path):
        if row.get("split") != "test":
            continue
        core_rows.append(
            {
                "run": row["run"],
                "n_samples": int(as_float(row.get("n_samples"), 0.0)),
                "real": as_float(row["real"]),
                "shuffle": as_float(row["shuffle"]),
                "zero": as_float(row["zero"]),
                "real_shuffle_gap": as_float(row["real_shuffle_gap"]),
                "real_zero_gap": as_float(row["real_zero_gap"]),
                "precision": as_float(row["precision_real"]),
                "recall": as_float(row["recall_real"]),
            }
        )
    split_pivot: dict[tuple[str, str], dict[str, object]] = {}
    for row in load_csv(split_path):
        if row.get("split") != "test":
            continue
        key = (row.get("source", ""), row.get("run", ""))
        out = split_pivot.setdefault(
            key,
            {
                "source": row.get("source", ""),
                "run": row.get("run", ""),
                "real": as_float(row.get("real")),
                "zero_graph_drop": float("nan"),
                "zero_local54_drop": float("nan"),
                "zero_global55_drop": float("nan"),
                "zero_unimol_drop": float("nan"),
            },
        )
        mode = row.get("mode")
        if mode == "zero_graph":
            out["zero_graph_drop"] = as_float(row.get("drop"))
        elif mode in {"zero_ligand_local54_only", "zero_local54"}:
            out["zero_local54_drop"] = as_float(row.get("drop"))
        elif mode in {"zero_ligand_global55_only", "zero_global55"}:
            out["zero_global55_drop"] = as_float(row.get("drop"))
        elif mode == "zero_unimol":
            out["zero_unimol_drop"] = as_float(row.get("drop"))
    split_rows = list(split_pivot.values())
    write_csv(
        DATA / "ligand_dependency_core.csv",
        core_rows,
        ["run", "n_samples", "real", "shuffle", "zero", "real_shuffle_gap", "real_zero_gap", "precision", "recall"],
    )
    write_csv(
        DATA / "ligand_source_drops.csv",
        split_rows,
        ["source", "run", "real", "zero_graph_drop", "zero_local54_drop", "zero_global55_drop", "zero_unimol_drop"],
    )
    return core_rows, split_rows


def parse_training_summary() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows = load_csv(ROOT / "tmp/v11m6_v11m6n_training_ledger.csv")
    scale_rows = []
    for row in train_rows:
        variant = row.get("variant", "")
        if variant in {"grad035_no_gate", "grad05_no_gate", "grad055_no_gate", "grad0575_no_gate", "grad060_no_gate", "grad0625_no_gate", "grad065_no_gate", "grad070_no_gate"}:
            m = re.search(r"grad0?([0-9]+)", variant)
            if m:
                raw = variant.split("_", 1)[0].replace("grad", "")
                scale = float(raw) / (10 ** max(1, len(raw) - 1))
                scale_rows.append({"variant": variant, "scale": scale, "best_f1": as_float(row.get("best_f1")), "best_epoch": as_float(row.get("best_epoch"))})
    # One run can appear in multiple ledgers; keep max best_f1 per variant.
    best_by_variant: dict[str, dict[str, object]] = {}
    for row in scale_rows:
        key = str(row["variant"])
        if key not in best_by_variant or as_float(row["best_f1"]) > as_float(best_by_variant[key]["best_f1"]):
            best_by_variant[key] = row
    scale_rows = sorted(best_by_variant.values(), key=lambda r: as_float(r["scale"]))
    seed_rows = load_csv(ROOT / "tmp/v11m6_v11m6n_seed_stats.csv")
    seed_summary = [
        {
            "set": r["set"],
            "n": int(float(r["n"])),
            "mean_best": as_float(r["mean_best"]),
            "sample_std": as_float(r["sample_std"]),
            "min_best": as_float(r["min_best"]),
            "max_best": as_float(r["max_best"]),
        }
        for r in seed_rows
    ]
    write_csv(DATA / "grad_scale_probe.csv", scale_rows, ["variant", "scale", "best_f1", "best_epoch"])
    write_csv(DATA / "seed_summary.csv", seed_summary, ["set", "n", "mean_best", "sample_std", "min_best", "max_best"])
    return scale_rows, seed_summary


def read_benchmark_block(path: Path, dataset: str) -> dict[str, float] | None:
    f = path / f"{dataset}_summary.json"
    if not f.exists():
        return None
    obj = json.loads(f.read_text())
    sample = obj.get("sample_mean") or obj
    protein_mean = obj.get("protein_mean_of_ligand_mean") or {}
    protein_max = obj.get("protein_mean_of_ligand_max") or {}
    out = {
        "sample_f1": as_float(sample.get("Mean_F1_c0")),
        "sample_precision": as_float(sample.get("Mean_Prec_c0")),
        "sample_recall": as_float(sample.get("Mean_Rec_c0")),
        "sample_pdr": as_float(sample.get("PDR")),
        "sample_laa": as_float(sample.get("LAA")),
        "sample_bp_f1": as_float(sample.get("BP_F1_c0")),
        "protein_mean_f1": as_float(protein_mean.get("Mean_F1_c0")),
        "protein_mean_pdr": as_float(protein_mean.get("PDR")),
        "protein_mean_laa": as_float(protein_mean.get("LAA")),
        "protein_max_f1": as_float(protein_max.get("Mean_F1_c0")),
        "protein_max_pdr": as_float(protein_max.get("PDR")),
        "protein_max_laa": as_float(protein_max.get("LAA")),
    }
    return out


def collect_esm3_benchmarks() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for d in sorted(REMOTE_SUM.glob("*")):
        if not d.is_dir():
            continue
        name = d.name
        if "fullfinetune" in name:
            family = "ESM3 global prior fullfinetune"
        elif "scratch" in name:
            family = "ESM3 global prior scratch"
        else:
            continue
        seed_m = re.search(r"seed(\d+)", name)
        seed = int(seed_m.group(1)) if seed_m else -1
        # Prefer the combined seed-benchmark job over the earlier duplicate scratch-only job.
        priority = 3 if name.startswith("esm3_seed_benchmark") else 2 if name.startswith("esm3_key_benchmark") else 1
        for dataset in ("coach420", "holo4k"):
            vals = read_benchmark_block(d, dataset)
            if not vals:
                continue
            row = {"family": family, "seed": seed, "dataset": dataset, "priority": priority, "dir": name}
            row.update(vals)
            rows.append(row)

    chosen: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["seed"]), str(row["dataset"]))
        if key not in chosen or int(row["priority"]) > int(chosen[key]["priority"]):
            chosen[key] = row
    rows = sorted(chosen.values(), key=lambda r: (str(r["dataset"]), str(r["family"]), int(r["seed"])))
    fields = [
        "family",
        "seed",
        "dataset",
        "sample_f1",
        "sample_precision",
        "sample_recall",
        "sample_pdr",
        "sample_laa",
        "sample_bp_f1",
        "protein_mean_f1",
        "protein_mean_pdr",
        "protein_mean_laa",
        "protein_max_f1",
        "protein_max_pdr",
        "protein_max_laa",
        "priority",
        "dir",
    ]
    write_csv(DATA / "esm3_benchmark_seed_rows.csv", rows, fields)

    stats: list[dict[str, object]] = []
    for dataset in ("coach420", "holo4k"):
        for family in sorted({str(r["family"]) for r in rows}):
            subset = [r for r in rows if r["dataset"] == dataset and r["family"] == family]
            if not subset:
                continue
            out: dict[str, object] = {"dataset": dataset, "family": family, "n": len(subset)}
            for metric in [
                "sample_f1",
                "sample_pdr",
                "sample_laa",
                "sample_bp_f1",
                "protein_mean_f1",
                "protein_max_f1",
            ]:
                mean, std = mean_std([as_float(r.get(metric)) for r in subset])
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            stats.append(out)
    write_csv(
        DATA / "esm3_benchmark_seed_stats.csv",
        stats,
        [
            "dataset",
            "family",
            "n",
            "sample_f1_mean",
            "sample_f1_std",
            "sample_pdr_mean",
            "sample_pdr_std",
            "sample_laa_mean",
            "sample_laa_std",
            "sample_bp_f1_mean",
            "sample_bp_f1_std",
            "protein_mean_f1_mean",
            "protein_mean_f1_std",
            "protein_max_f1_mean",
            "protein_max_f1_std",
        ],
    )
    return rows, stats


def plot_qc_pipeline(qc: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.axis("off")
    steps = [
        ("P2Rank benchmark inputs", "COACH420: 420 proteins\nHOLO4K: 4009 proteins", "#e9f2ff"),
        ("PDB repair", "heavy-residue fixed PDBs\nseparate H/full-fix path for tags", "#eef8ea"),
        ("Protein features", "Protein V4 complete features\n+ ESM2 / optional ESM3 caches", "#fff3db"),
        ("Ligand features", "Ligand V5 per atom:\nlocal54 + global45 + shape10", "#f6ecff"),
        ("Per-ligand tags", "current ligand contact/near labels\n5 A contact + 8 A near", "#fff0e8"),
        ("Benchmark inference", "per-ligand rows, max_atoms=-1\nsample/protein aggregation", "#ecfbff"),
    ]
    positions = [
        (0.05, 0.62),
        (0.38, 0.62),
        (0.71, 0.62),
        (0.05, 0.18),
        (0.38, 0.18),
        (0.71, 0.18),
    ]
    box_w, box_h = 0.24, 0.25
    for i, ((title, body, color), (x, y)) in enumerate(zip(steps, positions)):
        patch = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.018,rounding_size=0.025", fc=color, ec="#333333", lw=1)
        ax.add_patch(patch)
        ax.text(x + box_w / 2, y + box_h * 0.68, title, ha="center", va="center", weight="bold", fontsize=8.4)
        ax.text(x + box_w / 2, y + box_h * 0.36, body, ha="center", va="center", fontsize=7.5)
        if i in {0, 1, 3, 4}:
            x2, y2 = positions[i + 1]
            ax.add_patch(FancyArrowPatch((x + box_w + 0.015, y + box_h / 2), (x2 - 0.015, y2 + box_h / 2), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#333333"))
    # The benchmark consumes both protein features and per-ligand labels.
    ax.add_patch(FancyArrowPatch((0.83, 0.61), (0.83, 0.45), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#333333"))
    ax.add_patch(FancyArrowPatch((0.17, 0.61), (0.17, 0.45), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#333333", linestyle="--"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig_qc_pipeline")


def collect_feature_engineering_summary() -> dict[str, object]:
    summary = {
        "protein_v4_complete": {
            "physchem_features": "62D per residue",
            "spatial_scalar_features": "89D per residue, train-stat normalized when stats are supplied",
            "spatial_vector_features": "8 x 3 directional vectors per residue",
            "metadata": "residue names/indices, chain IDs, CA coordinates, all-atom metadata, sequence, feature_version",
        },
        "ligand_v5": {
            "atom_features": "109D per atom = 54D local atom chemistry + 45D global molecular tag + 10D shape",
            "graph": "edge_index plus 6D edge_attr for single/double/triple/aromatic/conjugated/ring bonds",
            "coordinates": "current coordinates, heavy-atom coordinates, and descriptor/geometry side channels",
        },
        "current_model_use": {
            "mainline": "V11m13-style stable protein trunk + protected late local ligand delta",
            "effective_sources": "source ablations show local54 and ligand graph dominate global55/UniMol in the current heads",
            "latest_options": "local54 group adapter/KV interaction, residue-chem compatibility, edge-conditioned KV residual, weak ESM3 global prior",
        },
    }
    write_json(DATA / "feature_engineering_summary.json", summary)
    return summary


def plot_feature_engineering() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    ax.axis("off")
    columns = [
        (
            "Protein complete V4",
            [
                ("62D physchem:", "AA/v2 descriptors, sequence,\nhbond, residue ID, SS3, class"),
                ("89D spatial scalars:", "SASA/surface/geometry,\npocketness, torsion, packing,\ncharge/hydrophobicity, metal"),
                ("8x3 vectors:", "sidechain, aromatic, H-bond,\ncavity directions"),
            ],
            "#e9f2ff",
        ),
        (
            "Ligand V5",
            [
                ("109D per atom:", "local54 + global45 + shape10"),
                ("local54 chemistry:", "atom identity/topology,\ncharge, LogP/TPSA, RMSF,\nSASA, HBD/HBA, rotatable,\nmetal, stereo"),
                ("graph/geometry:", "6D bond edge_attr + coords"),
            ],
            "#f6ecff",
        ),
        (
            "Current integration",
            [
                ("Protein trunk:", "V4 features + ESM2,\noptional weak ESM3 prior"),
                ("Ligand path:", "late local_delta K/V from\nlocal54 and ligand graph"),
                ("V11m13 options:", "local54 group adapter,\nresidue-chem compat,\nedge-conditioned K/V"),
            ],
            "#eef8ea",
        ),
    ]
    xs = [0.04, 0.365, 0.69]
    for x, (title, lines, color) in zip(xs, columns):
        patch = FancyBboxPatch((x, 0.06), 0.27, 0.84, boxstyle="round,pad=0.02,rounding_size=0.025", fc=color, ec="#333333", lw=1.0)
        ax.add_patch(patch)
        ax.text(x + 0.135, 0.84, title, ha="center", va="center", weight="bold", fontsize=9.2)
        y_pairs = [0.71, 0.47, 0.22]
        for (header, body), y in zip(lines, y_pairs):
            ax.text(x + 0.035, y, header, ha="left", va="top", fontsize=8.2, weight="bold")
            ax.text(x + 0.035, y - 0.075, body, ha="left", va="top", fontsize=7.5)
    ax.text(0.5, 0.97, "Latest feature-engineering layout used by PLC models", ha="center", va="top", weight="bold", fontsize=10)
    save(fig, "fig_feature_engineering")


def plot_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str, dashed: bool = False) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            fc=color,
            ec="#333333",
            lw=1.0,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", weight="bold", fontsize=7.4)
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center", fontsize=6.15, linespacing=1.08)

    def arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False, rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color="#333333",
                linestyle="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    # Protein lane.
    box(
        0.02,
        0.72,
        0.17,
        0.17,
        "Protein input",
        "complete V4\n(physchem62, scalar89,\nvector8x3, coords/meta)",
        "#e9f2ff",
    )
    box(
        0.23,
        0.72,
        0.16,
        0.17,
        "Protein trunk",
        "V4 projection + GVP/\nresidue self-attn\nstable pocket prior",
        "#e9f2ff",
    )
    box(
        0.43,
        0.72,
        0.18,
        0.17,
        "PLM prior",
        "ESM2(final / MLC-wsum /\nMLC-concat)\nESM3(none / weak global prior)",
        "#fff3db",
    )
    box(
        0.64,
        0.72,
        0.17,
        0.17,
        "Protein hidden",
        "h_prot\nresidue decision space\nprotected floor",
        "#e9f2ff",
    )

    # Ligand lane.
    box(
        0.02,
        0.43,
        0.17,
        0.18,
        "Ligand input",
        "V5 atom109\n(local54 + global55;\nglobal55=global45+shape10)\nedge_attr6 + coords",
        "#f6ecff",
    )
    box(
        0.23,
        0.43,
        0.16,
        0.18,
        "Ligand encoder",
        "graph/GNN + atom self-attn\n+ ligand/global token\nh_lig atoms/context",
        "#f6ecff",
    )
    box(
        0.43,
        0.43,
        0.18,
        0.18,
        "Local evidence",
        "local54 groups\n(base33 anchor;\nelectronic/sasa/flex evidence)\n+ edge context",
        "#f6ecff",
    )

    # Fusion lane.
    box(
        0.64,
        0.43,
        0.17,
        0.18,
        "Fusion block",
        "cross module\n(V11 bi-cross / V12 cross-only /\nV12 co-attn block)",
        "#eef8ea",
    )
    box(
        0.64,
        0.245,
        0.15,
        0.155,
        "Base logits",
        "classifier(h_prot_fused)\nprotein-side pocket prior",
        "#ecfbff",
    )
    box(
        0.84,
        0.43,
        0.15,
        0.21,
        "Late local delta",
        "Q=residue; K/V=(lig hidden,\nlocal54, group K/V,\nedge K/V)\nprotein grad protection\n(detach / full / fractional g~0.6)",
        "#ffecec",
    )
    box(
        0.84,
        0.23,
        0.15,
        0.15,
        "Delta head",
        "lig_read + residue hidden\n(+ compat/value enrich)\n-> bounded delta logits",
        "#ffecec",
    )
    box(
        0.64,
        0.075,
        0.35,
        0.115,
        "Logit sum and output",
        "base logits + bounded delta logits (+ optional center aux) -> residue class scores",
        "#f0fff4",
    )

    # Optional V13 probes as side annotation.
    box(
        0.43,
        0.245,
        0.17,
        0.155,
        "V13 probes",
        "optional score/value edits\n(res-atom bias, triangle,\nligand slots, Q/K reparam,\nvalue enrichment)",
        "#f7fafc",
        dashed=True,
    )

    # Arrows: protein path.
    arrow(0.19, 0.805, 0.23, 0.805)
    arrow(0.39, 0.805, 0.43, 0.805)
    arrow(0.61, 0.805, 0.64, 0.805)
    # Ligand path and local evidence.
    arrow(0.19, 0.52, 0.23, 0.52)
    arrow(0.39, 0.52, 0.43, 0.52)
    arrow(0.61, 0.52, 0.64, 0.52)
    arrow(0.81, 0.52, 0.84, 0.525)
    # Cross/fusion consumes both protein and ligand.
    arrow(0.725, 0.72, 0.725, 0.61)
    arrow(0.725, 0.43, 0.715, 0.40)
    # Local delta and output.
    arrow(0.915, 0.43, 0.915, 0.37)
    arrow(0.915, 0.23, 0.84, 0.19)
    arrow(0.715, 0.245, 0.715, 0.19)
    # Optional probes connect to local-delta path.
    arrow(0.60, 0.395, 0.84, 0.435, dashed=True)

    ax.text(0.02, 0.93, "Mainline: stable protein trunk + cross/co-attn fusion + late ligand-conditioned local delta", ha="left", va="center", fontsize=8.5, weight="bold")
    ax.text(0.02, 0.015, "Empirical reading: local54/ligand graph drive residue localization; ESM3 is used as a weak protein prior; V13 scalar probes are diagnostic, not current mainline.", ha="left", va="bottom", fontsize=6.8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1)
    save(fig, "fig_model_architecture")


def plot_grad_and_ligdep(core_rows: list[dict[str, object]], scale_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    if scale_rows:
        xs = [as_float(r["scale"]) for r in scale_rows]
        ys = [as_float(r["best_f1"]) for r in scale_rows]
        ax.plot(xs, ys, marker="o", color="#2b6cb0")
        for x, y, r in zip(xs, ys, scale_rows):
            ax.text(x, y + 0.0005, str(r["variant"]).replace("_no_gate", ""), ha="center", fontsize=6.5, rotation=25)
    ax.set_xlabel("protein gradient scale")
    ax.set_ylabel("validation best F1")
    ax.set_title("V11m6n grad-scale probe")

    ax = axes[1]
    keep = [r for r in core_rows if r["run"] in {"main", "no_detach_protein", "grad05_no_gate", "grad060_no_gate"}]
    x = list(range(len(keep)))
    width = 0.24
    ax.bar([i - width for i in x], [as_float(r["real"]) for r in keep], width, label="real", color="#2f855a")
    ax.bar(x, [as_float(r["shuffle"]) for r in keep], width, label="shuffle", color="#dd6b20")
    ax.bar([i + width for i in x], [as_float(r["zero"]) for r in keep], width, label="zero", color="#718096")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["run"]).replace("_", "\n") for r in keep], fontsize=7)
    ax.set_ylabel("test F1")
    ax.set_title("Ligand-dependency sanity checks")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_grad_ligdep")


def plot_source_drops(split_rows: list[dict[str, object]]) -> None:
    preferred = [r for r in split_rows if r["run"] in {"grad05_no_gate", "grad060_no_gate", "no_detach_protein", "no_gate"}]
    labels = [str(r["run"]).replace("_", "\n") for r in preferred]
    metrics = [
        ("zero_graph_drop", "graph", "#2b6cb0"),
        ("zero_local54_drop", "local54", "#2f855a"),
        ("zero_global55_drop", "global55", "#dd6b20"),
        ("zero_unimol_drop", "UniMol", "#805ad5"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = list(range(len(preferred)))
    width = 0.18
    for j, (key, label, color) in enumerate(metrics):
        ax.bar([i + (j - 1.5) * width for i in x], [as_float(r[key]) for r in preferred], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("F1 drop vs real")
    ax.set_title("Feature-source ablation: local54 and graph dominate")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    save(fig, "fig_source_drops")


def plot_esm3_benchmark(stats: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    metrics = [("sample_f1", "Sample F1"), ("sample_pdr", "PDR"), ("sample_laa", "LAA")]
    families = ["ESM3 global prior fullfinetune", "ESM3 global prior scratch"]
    colors = ["#718096", "#2b6cb0"]
    for ax, (metric, title) in zip(axes, metrics):
        datasets = ["coach420", "holo4k"]
        xbase = [0, 1]
        width = 0.32
        for j, fam in enumerate(families):
            vals = []
            errs = []
            for ds in datasets:
                row = next((r for r in stats if r["dataset"] == ds and r["family"] == fam), None)
                vals.append(as_float(row.get(f"{metric}_mean")) if row else float("nan"))
                errs.append(as_float(row.get(f"{metric}_std")) if row else 0.0)
            ax.bar([x + (j - 0.5) * width for x in xbase], vals, width, yerr=errs, capsize=2, label=fam.replace("ESM3 global prior ", ""), color=colors[j])
        ax.set_xticks(xbase)
        ax.set_xticklabels(["COACH420", "HOLO4K"])
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig_esm3_benchmark")


def plot_holo_gap() -> None:
    # Values from tmp/holo4k_coach420_gap_analysis.md. They are copied to data below.
    ligand_rows = [
        {"dataset": "COACH420", "bin": "1", "n": 298, "f1": 0.548, "recall": 0.521, "dcc": 6.15},
        {"dataset": "HOLO4K", "bin": "1", "n": 1411, "f1": 0.561, "recall": 0.527, "dcc": 6.18},
        {"dataset": "COACH420", "bin": "2", "n": 76, "f1": 0.433, "recall": 0.355, "dcc": 7.38},
        {"dataset": "HOLO4K", "bin": "2", "n": 1213, "f1": 0.353, "recall": 0.266, "dcc": 13.27},
        {"dataset": "COACH420", "bin": "3-4", "n": 33, "f1": 0.398, "recall": 0.309, "dcc": 7.04},
        {"dataset": "HOLO4K", "bin": "3-4", "n": 773, "f1": 0.266, "recall": 0.182, "dcc": 17.69},
        {"dataset": "HOLO4K", "bin": "5-9", "n": 402, "f1": 0.193, "recall": 0.123, "dcc": 20.90},
        {"dataset": "HOLO4K", "bin": ">=10", "n": 209, "f1": 0.136, "recall": 0.086, "dcc": 20.60},
    ]
    length_rows = [
        {"dataset": "COACH420", "bin": "200-399", "n": 237, "f1": 0.550, "dcc": 5.74, "dcc_success": 0.468},
        {"dataset": "HOLO4K", "bin": "200-399", "n": 1591, "f1": 0.529, "dcc": 6.93, "dcc_success": 0.414},
        {"dataset": "COACH420", "bin": "400-799", "n": 68, "f1": 0.380, "dcc": 10.02, "dcc_success": 0.147},
        {"dataset": "HOLO4K", "bin": "400-799", "n": 1226, "f1": 0.282, "dcc": 15.87, "dcc_success": 0.076},
        {"dataset": "HOLO4K", "bin": ">=800", "n": 702, "f1": 0.131, "dcc": 25.64, "dcc_success": 0.010},
    ]
    write_csv(DATA / "holo_gap_ligand_count_bins.csv", ligand_rows, ["dataset", "bin", "n", "f1", "recall", "dcc"])
    write_csv(DATA / "holo_gap_length_bins.csv", length_rows, ["dataset", "bin", "n", "f1", "dcc", "dcc_success"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    bins = ["1", "2", "3-4", "5-9", ">=10"]
    colors = {"COACH420": "#2f855a", "HOLO4K": "#2b6cb0"}
    for ds in ["COACH420", "HOLO4K"]:
        vals = [next((r["f1"] for r in ligand_rows if r["dataset"] == ds and r["bin"] == b), float("nan")) for b in bins]
        axes[0].plot(bins, vals, marker="o", label=ds, color=colors[ds])
    axes[0].set_xlabel("ligands per protein key")
    axes[0].set_ylabel("mean F1")
    axes[0].set_title("HOLO4K gap is multi-ligand driven")
    axes[0].legend(frameon=False)

    bins2 = ["200-399", "400-799", ">=800"]
    for ds in ["COACH420", "HOLO4K"]:
        vals = [next((r["f1"] for r in length_rows if r["dataset"] == ds and r["bin"] == b), float("nan")) for b in bins2]
        axes[1].plot(bins2, vals, marker="o", label=ds, color=colors[ds])
    axes[1].set_xlabel("protein length bin")
    axes[1].set_ylabel("mean F1")
    axes[1].set_title("Large HOLO4K proteins are harder")
    axes[1].legend(frameon=False)
    save(fig, "fig_holo_gap")


def copy_supporting_docs() -> None:
    for src, dst in [
        (ROOT / "tmp/holo4k_coach420_gap_analysis.md", DATA / "holo4k_coach420_gap_analysis.md"),
        (ROOT / "tmp/v11m6_v11m6n_experiment_ledger_20260531.md", DATA / "v11m6_v11m6n_experiment_ledger_20260531.md"),
        (ROOT / "tmp/v13_ligdep_9926224_analysis.md", DATA / "v13_ligdep_9926224_analysis.md"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)


def main() -> None:
    ensure_dirs()
    style()
    qc = collect_dataset_qc()
    core_rows, split_rows = parse_ligdep()
    scale_rows, _seed_rows = parse_training_summary()
    _bench_rows, bench_stats = collect_esm3_benchmarks()
    collect_feature_engineering_summary()
    copy_supporting_docs()
    plot_qc_pipeline(qc)
    plot_feature_engineering()
    plot_model_architecture()
    plot_grad_and_ligdep(core_rows, scale_rows)
    plot_source_drops(split_rows)
    plot_esm3_benchmark(bench_stats)
    plot_holo_gap()
    print(f"Generated PLC report assets under {REPORT}")


if __name__ == "__main__":
    main()
