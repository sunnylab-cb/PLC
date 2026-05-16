#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "Train/epoch_loss",
    "Train/epoch_acc",
    "Train/epoch_focal_loss",
    "Train/learning_rate",
    "Val/epoch_loss",
    "Val/epoch_acc",
    "Val/f1_c0",
    "Val/precision_c0",
    "Val/recall_c0",
    "Val/f1_c1",
    "Val/precision_c1",
    "Val/recall_c1",
    "Val/f1_c2",
    "Val/precision_c2",
    "Val/recall_c2",
    "Val/dice_c0",
    "Val/iou_c0",
]

LAST_TAGS = [
    "Gradients/grad_norm_co_attn_blocks",
    "Gradients/grad_norm_ligand_encoder",
    "Gradients/grad_norm_esm_proj",
    "Gradients/grad_norm_gvp_proj",
    "Gradients/grad_ratio_esm_pathway",
    "Gradients/grad_ratio_gvp_pathway",
    "Health/unimol_global_contribution_ratio",
    "Health/unimol_film_signal_norm",
    "Health/esm_contribution_ratio",
    "Health/h_geo_contribution_ratio",
    "Health/proj_esm_ratio",
    "Health/proj_gvp_ratio",
]


def load(path: Path):
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in ea.Tags().get("scalars", [])}


def main():
    path = Path(sys.argv[1])
    rows = load(path)
    epochs = sorted({step for tag in TAGS for step, _ in rows.get(tag, [])})
    print(path)
    print("\t".join(["epoch"] + TAGS))
    for epoch in epochs:
        vals = []
        for tag in TAGS:
            d = dict(rows.get(tag, []))
            vals.append(f"{d[epoch]:.6g}" if epoch in d else "")
        print("\t".join([str(epoch)] + vals))
    print("\nlast_batch_scalars")
    for tag in LAST_TAGS:
        vals = rows.get(tag, [])
        if vals:
            step, value = vals[-1]
            print(f"{tag}\tstep={step}\tvalue={value:.6g}")


if __name__ == "__main__":
    main()
