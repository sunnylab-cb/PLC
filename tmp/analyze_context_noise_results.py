"""Reproducible read-only TB/config analysis of the six original-model trials."""

import json
import math
from pathlib import Path
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.audit_libre_transfer_logs import summarize_events

OUT = ROOT / "tmp/context_noise_20261013"
BASE = ROOT / "tmp/libre_transfer_12964718/baseline"
KEYS = {
    "f1": "Val/f1_c0", "precision": "Val/precision_c0", "recall": "Val/recall_c0",
    "auprc": "Val/auprc_c0_global", "auroc": "Val/auroc_c0_global",
    "f1_at_half": "Val/f1_c0_at_0_5_global", "mcc_at_half": "Val/mcc_c0_at_0_5_global",
    "f1_at_gate": "Val/f1_c0_at_best_gate_global", "gate": "Val/best_gate_c0",
    "mcc_at_gate": "Val/mcc_c0_at_best_gate_global",
    "train_loss": "Train/epoch_loss", "val_loss": "Val/epoch_loss",
    "train_focal": "Train/epoch_focal_loss", "val_focal": "Val/focal_loss",
    "train_tversky": "Train/epoch_dice_loss", "val_tversky": "Val/dice_loss",
    "cnc_clip_fraction": "CNC/epoch_clip_fraction", "lr": "Train/learning_rate",
}
for fraction in (20, 10, 5):
    for metric in ("precision", "recall", "hit_rate"):
        KEYS[f"top_l{fraction}_{metric}"] = f"Val/top_l{fraction}_{metric}_c0_macro"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for key, val in value.items()
                for k, v in flatten(val, f"{prefix}.{key}" if prefix else key).items()}
    return {prefix: value}


base_config = flatten(yaml.safe_load((BASE / "config.yaml").read_text()))
report = {"scope": "Clean strict VAL; one seed44; no noise-model TEST evaluation.",
          "epoch_numbering": "one-based; stored TB step + 1", "runs": {}}
for run in [BASE] + sorted((OUT / "outputs").iterdir()):
    name = "baseline" if run == BASE else run.name.split("_scratch_seed44_")[0]
    config = flatten(yaml.safe_load((run / "config.yaml").read_text()))
    differences = {k: {"baseline": base_config.get(k), "run": config.get(k)}
                   for k in sorted(set(base_config) | set(config))
                   if base_config.get(k) != config.get(k)}
    reader = EventAccumulator(str(run / "runs"), size_guidance={"scalars": 0})
    reader.Reload()
    scalar_report, maps = {}, {}
    for tag in reader.Tags()["scalars"]:
        events = reader.Scalars(tag)
        summary = summarize_events(events)
        maps[tag] = dict(summary["series"])
        values = np.array(list(maps[tag].values()), dtype=float)
        summary.update(p50=float(np.quantile(values, .5)), p95=float(np.quantile(values, .95)))
        summary.pop("series")
        scalar_report[tag] = summary
    steps = sorted(maps["Val/f1_c0"])
    assert steps == list(range(35)), (name, steps)
    curve = [dict(epoch=step + 1, **{k: maps.get(tag, {}).get(step) for k, tag in KEYS.items()})
             for step in steps]
    best = max(curve, key=lambda row: row["f1"])
    noise = {tag: scalar_report[tag] for tag in scalar_report
             if tag.startswith("ContextNoise/")}
    weighted = {}
    for c in range(3):
        count = maps.get(f"ContextNoise/true_class{c}_count", {})
        for field in ("protection_mean", "amplitude_mean"):
            means = maps.get(f"ContextNoise/true_class{c}_{field}", {})
            denominator = sum(count[s] for s in means)
            if denominator:
                weighted[f"class{c}_{field}"] = sum(means[s] * count[s] for s in means) / denominator
                weighted[f"class{c}_logged_residue_observations"] = denominator
    noise["count_weighted_logged_classes"] = weighted
    report["runs"][name] = dict(
        path=str(run), config_differences=differences, best_f1=best,
        best_auprc=max(curve, key=lambda row: row["auprc"]),
        best_auroc=max(curve, key=lambda row: row["auroc"]),
        min_val_loss=min(curve, key=lambda row: row["val_loss"]),
        last=curve[-1], tail5_f1=float(np.mean([x["f1"] for x in curve[-5:]])),
        curve=curve, scalar_summaries=scalar_report, noise=noise,
    )

baseline = report["runs"]["baseline"]
for name, result in report["runs"].items():
    result["best_f1_difference_vs_baseline"] = result["best_f1"]["f1"] - baseline["best_f1"]["f1"]
    result["same_epoch_differences"] = [
        dict(epoch=a["epoch"], **{k: a[k] - b[k] for k in ("f1", "auprc", "precision", "recall")})
        for a, b in zip(result["curve"], baseline["curve"])
    ]

report["paired_comparisons"] = {}
for sigma in ("05", "10", "20"):
    a, b = [report["runs"][f"gfgf_{mode}_noise{sigma}"] for mode in ("background", "random")]
    report["paired_comparisons"][sigma] = dict(
        best_f1_background_minus_random=a["best_f1"]["f1"]-b["best_f1"]["f1"],
        tail5_f1_background_minus_random=a["tail5_f1"]-b["tail5_f1"],
        epoch21_35_mean_difference=float(np.mean([x["f1"]-y["f1"] for x, y in zip(a["curve"][20:], b["curve"][20:])])),
        same_epoch_background_wins=sum(x["f1"] > y["f1"] for x, y in zip(a["curve"], b["curve"])),
    )

(OUT / "tb_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print("name | best F1 | epoch | delta | P | R | AP@F1 | lastF1 | tail5 | minLossEpoch")
for name, r in report["runs"].items():
    b = r["best_f1"]
    print(f'{name} | {b["f1"]:.6f} | {b["epoch"]} | {r["best_f1_difference_vs_baseline"]:+.6f} '
          f'| {b["precision"]:.6f} | {b["recall"]:.6f} | {b["auprc"]:.6f} '
          f'| {r["last"]["f1"]:.6f} | {r["tail5_f1"]:.6f} | {r["min_val_loss"]["epoch"]}')
print("PAIRS", json.dumps(report["paired_comparisons"], indent=2))
print("CONFIG_DIFFS", json.dumps({k:v["config_differences"] for k,v in report["runs"].items()}, indent=2))
