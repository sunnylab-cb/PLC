#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "Train/batch_loss",
    "Train/batch_acc",
    "Train/focal_loss",
    "Train/grad_norm",
    "Train/learning_rate_batch",
    "Gradients/grad_norm_co_attn_blocks",
    "Gradients/grad_norm_ligand_encoder",
    "Gradients/grad_norm_esm_proj",
    "Gradients/grad_norm_gvp_proj",
    "Gradients/grad_ratio_esm_pathway",
    "Gradients/grad_ratio_gvp_pathway",
    "Health/unimol_global_contribution_ratio",
    "Health/unimol_global_delta_norm",
    "Health/unimol_film_signal_norm",
    "Health/esm_contribution_ratio",
    "Health/h_geo_contribution_ratio",
    "CNC/batch_clip_fraction",
]


def main():
    event = Path(sys.argv[1])
    ea = EventAccumulator(str(event), size_guidance={"scalars": 0})
    ea.Reload()
    print(event)
    for tag in TAGS:
        if tag not in ea.Tags().get("scalars", []):
            continue
        vals = ea.Scalars(tag)
        print(f"\n[{tag}] n={len(vals)}")
        for e in vals[-8:]:
            print(f"  step={e.step:<6d} value={e.value:.6g}")


if __name__ == "__main__":
    main()
