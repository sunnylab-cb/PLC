#!/usr/bin/env python3
"""Dump V12 co-attn Linear weight norms after initialization."""

from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pocket_gvp_fusion_v12 import PocketGVPFusionV12


def main():
    cfg = yaml.safe_load((ROOT / "configs/pocket_gvp_v12.yaml").read_text())["model"]
    cfg = dict(cfg)
    cfg["n_co_attn_blocks"] = 2
    model = PocketGVPFusionV12(cfg)
    for name, module in model.named_modules():
        if "co_attn_blocks" in name and isinstance(module, torch.nn.Linear):
            w = module.weight.detach()
            is_zero = bool((w == 0).all().item())
            print(
                f"{name:76s} zero={str(is_zero):5s} "
                f"norm={w.float().norm().item():.6e} "
                f"max={w.float().abs().max().item():.6e} shape={tuple(w.shape)}"
            )


if __name__ == "__main__":
    main()
