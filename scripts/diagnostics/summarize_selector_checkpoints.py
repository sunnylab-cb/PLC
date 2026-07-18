#!/usr/bin/env python3
"""Summarize metric-selector checkpoints without instantiating the model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


METRICS = (
    "loss",
    "f1_c0",
    "dice_c0",
    "precision_c0",
    "recall_c0",
    "auprc_c0_global",
    "auroc_c0_global",
    "f1_c0_at_0_5_global",
    "mcc_c0_at_0_5_global",
    "precision_c0_at_0_5_global",
    "recall_c0_at_0_5_global",
    "best_gate_c0",
    "f1_c0_at_best_gate_global",
    "mcc_c0_at_best_gate_global",
    "precision_c0_at_best_gate_global",
    "recall_c0_at_best_gate_global",
    "top_l20_precision_c0_macro",
    "top_l20_recall_c0_macro",
    "top_l20_hit_rate_c0_macro",
    "top_l10_precision_c0_macro",
    "top_l10_recall_c0_macro",
    "top_l10_hit_rate_c0_macro",
    "top_l5_precision_c0_macro",
    "top_l5_recall_c0_macro",
    "top_l5_hit_rate_c0_macro",
)

CHECKPOINTS = (
    "best_f1_c0_model.pt",
    "best_auprc_c0_model.pt",
    "best_auroc_c0_model.pt",
    "best_dice_c0_model.pt",
)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _parameter_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("model_state_dict", {})
    config = checkpoint.get("config", {}) or {}
    model_config = config.get("model", {}) or {}
    result: dict[str, Any] = {}

    for key in (
        "local_delta_logit_scale",
        "esm3_raw_alpha",
        "local54_interaction_kv_raw_alpha",
        "local54_compat_raw_alpha",
    ):
        raw = _scalar(state.get(key))
        if raw is not None:
            result[f"{key}_raw"] = raw

    raw_scale = result.get("local_delta_logit_scale_raw")
    if raw_scale is not None:
        result["local_delta_logit_scale_effective_tanh"] = math.tanh(raw_scale)
        result["local_delta_trainable_logit_scale"] = bool(
            model_config.get("local_delta_trainable_logit_scale", True)
        )

    alpha_specs = (
        ("esm3_raw_alpha", "esm3_alpha_max"),
        ("local54_interaction_kv_raw_alpha", "local54_interaction_kv_alpha_max"),
        ("local54_compat_raw_alpha", "local54_compat_alpha_max"),
    )
    for parameter, max_key in alpha_specs:
        raw = result.get(f"{parameter}_raw")
        maximum = _scalar(model_config.get(max_key))
        if raw is not None and maximum is not None:
            result[f"{parameter.removesuffix('_raw_alpha')}_effective_alpha"] = (
                maximum * _sigmoid(raw)
            )
    return result


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize(directory: Path) -> dict[str, Any]:
    rows = []
    for name in CHECKPOINTS:
        path = directory / name
        if not path.exists():
            continue
        checkpoint = _load(path)
        metrics = checkpoint.get("val_metrics", {}) or {}
        row = {
            "checkpoint": name,
            "epoch_index": checkpoint.get("epoch"),
            "epoch_display": (
                int(checkpoint["epoch"]) + 1 if checkpoint.get("epoch") is not None else None
            ),
            "selection_metric": checkpoint.get("selection_metric"),
            "val_contact_gate": _scalar(checkpoint.get("val_contact_gate")),
            "metrics": {key: _scalar(metrics.get(key)) for key in METRICS if key in metrics},
            "parameters": _parameter_summary(checkpoint),
        }
        rows.append(row)
        del checkpoint
    return {"directory": str(directory), "checkpoints": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = [summarize(directory) for directory in args.directories]
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
