#!/usr/bin/env python3
"""
Sweep inference-only class-0/contact logit bias on val, then evaluate the
selected bias on test real/shuffle/zero.

Purpose:
  - Recover precision/recall balance without retraining.
  - Check whether a fixed calibration bias preserves ligand-dependency gaps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.infer.infer_v11_internal_ligdep import (  # noqa: E402
    create_v11_model,
    model_needs_unimol,
    predict_v11_logits,
)
from scripts.infer.internal_ligand_dependency import (  # noqa: E402
    build_internal_dataset,
    evaluate_mode,
    load_checkpoint,
    load_config_from_checkpoint_or_yaml,
    load_unimol_cache,
)


def _parse_biases(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("--biases parsed to an empty list")
    return values


def _safe_name(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _metric(summary: Dict, key: str) -> float:
    value = float(summary.get(key, float("nan")))
    return value if math.isfinite(value) else -float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Val-select class0 logit bias, then test real/shuffle/zero",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--unimol_embeddings_dir", default=None)
    parser.add_argument("--no_unimol", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_save_rows", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--biases",
        default=(
            "-0.80,-0.75,-0.70,-0.65,-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.30,"
            "-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30,"
            "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"
        ),
        help="comma-separated class0 logit biases to sweep on val",
    )
    parser.add_argument("--select_metric", default="global_f1_c0")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = load_config_from_checkpoint_or_yaml(ckpt, args.config)
    model, version = create_v11_model(ckpt, config, device)
    needs_unimol = model_needs_unimol(model)
    d_unimol = int(getattr(model, "d_unimol", config.get("model", {}).get("d_unimol", 512)))
    unimol_cache = load_unimol_cache(config, args, d_unimol=d_unimol) if needs_unimol and not args.no_unimol else {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path("results") / f"class0_bias_{version}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_args = copy.copy(args)
    val_args.split = "val"
    val_dataset = build_internal_dataset(config, val_args)

    sweep = []
    for bias in _parse_biases(args.biases):
        val_args.class0_logit_bias = bias
        bias_dir = out_dir / "val_sweep" / f"bias_{_safe_name(bias)}"
        bias_dir.mkdir(parents=True, exist_ok=True)
        summary = evaluate_mode(
            model=model,
            dataset=val_dataset,
            args=val_args,
            mode="real",
            device=device,
            predict_logits=predict_v11_logits,
            unimol_cache=unimol_cache,
            d_unimol=d_unimol,
            out_dir=bias_dir,
        )
        row = {
            "bias": bias,
            "split": "val",
            "mode": "real",
            "select_metric": args.select_metric,
            "select_value": _metric(summary, args.select_metric),
            "global_f1_c0": summary.get("global_f1_c0"),
            "global_precision_c0": summary.get("global_precision_c0"),
            "global_recall_c0": summary.get("global_recall_c0"),
            "global_f1_binary": summary.get("global_f1_binary"),
            "mean_sample_f1_c0": summary.get("mean_sample_f1_c0"),
        }
        sweep.append(row)
        print(
            f"[val bias={bias:+.3f}] {args.select_metric}={row['select_value']:.4f} "
            f"P/R={row['global_precision_c0']:.4f}/{row['global_recall_c0']:.4f}"
        )

    best = max(
        sweep,
        key=lambda r: (
            float(r["select_value"]),
            float(r.get("global_f1_c0") or -float("inf")),
            -abs(float(r["bias"])),
        ),
    )
    best_bias = float(best["bias"])
    print(f"Selected bias={best_bias:+.3f} by val {args.select_metric}={best['select_value']:.4f}")

    test_args = copy.copy(args)
    test_args.split = "test"
    test_args.class0_logit_bias = best_bias
    test_dataset = build_internal_dataset(config, test_args)
    test_summaries: Dict[str, Dict] = {}
    test_dir = out_dir / f"test_selected_bias_{_safe_name(best_bias)}"
    test_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("real", "shuffle", "zero"):
        test_summaries[mode] = evaluate_mode(
            model=model,
            dataset=test_dataset,
            args=test_args,
            mode=mode,
            device=device,
            predict_logits=predict_v11_logits,
            unimol_cache=unimol_cache,
            d_unimol=d_unimol,
            out_dir=test_dir,
        )
        s = test_summaries[mode]
        print(
            f"[test {mode} bias={best_bias:+.3f}] f1={s.get('global_f1_c0', 0):.4f} "
            f"P/R={s.get('global_precision_c0', 0):.4f}/{s.get('global_recall_c0', 0):.4f}"
        )

    real = test_summaries["real"].get("global_f1_c0", 0.0)
    shuffle = test_summaries["shuffle"].get("global_f1_c0", 0.0)
    zero = test_summaries["zero"].get("global_f1_c0", 0.0)
    final_metrics = {
        "selected_bias": best_bias,
        "val_select_metric": args.select_metric,
        "val_select_value": best.get("select_value"),
        "val_global_f1_c0": best.get("global_f1_c0"),
        "val_global_precision_c0": best.get("global_precision_c0"),
        "val_global_recall_c0": best.get("global_recall_c0"),
        "test_real_global_f1_c0": test_summaries["real"].get("global_f1_c0"),
        "test_real_global_precision_c0": test_summaries["real"].get("global_precision_c0"),
        "test_real_global_recall_c0": test_summaries["real"].get("global_recall_c0"),
        "test_shuffle_global_f1_c0": test_summaries["shuffle"].get("global_f1_c0"),
        "test_zero_global_f1_c0": test_summaries["zero"].get("global_f1_c0"),
        "test_real_shuffle_gap": float(real - shuffle),
        "test_real_zero_gap": float(real - zero),
    }
    combined = {
        "model_tag": version,
        "checkpoint": str(args.checkpoint),
        "select_metric": args.select_metric,
        "biases": _parse_biases(args.biases),
        "val_sweep": sweep,
        "selected": best,
        "test_selected_bias": best_bias,
        "test_summaries": test_summaries,
        "test_gaps": {
            "real_shuffle": float(real - shuffle),
            "real_zero": float(real - zero),
        },
        "final_metrics": final_metrics,
        "args": vars(args),
    }
    with open(out_dir / "class0_bias_sweep_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved summary: {out_dir / 'class0_bias_sweep_summary.json'}")


if __name__ == "__main__":
    main()
