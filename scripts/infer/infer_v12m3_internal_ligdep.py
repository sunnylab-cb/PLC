#!/usr/bin/env python3
"""
Ligand-dependency evaluation for V12m3 checkpoints on the internal split.

This does not use HOLO4K/COACH420.  It evaluates the configured PLC test split
and compares real vs shuffled/zeroed ligand inputs.  For shuffle mode, Uni-Mol
embeddings are also shuffled with the ligand source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.infer.internal_ligand_dependency import (
    add_common_args,
    load_checkpoint,
    load_config_from_checkpoint_or_yaml,
    resolve_contact_threshold,
    run_ligand_dependency_eval,
)
from src.data.collate import prepare_model_inputs_v12
from src.models.pocket_gvp_fusion_v12m3 import PocketGVPFusionV12M3


def create_v12m3_model(ckpt: Dict, config: Dict, device: torch.device):
    model_cfg = dict(config["model"])
    model_cfg["use_gradient_checkpointing"] = False
    model = PocketGVPFusionV12M3(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded V12m3 -> PocketGVPFusionV12M3; params={n_params:,}")
    return model


def predict_v12m3_logits(model, batch: Dict, device: torch.device, unimol_embed: torch.Tensor):
    inputs = prepare_model_inputs_v12(batch, device, unimol_embed=unimol_embed)
    outputs = model(**inputs)
    return outputs["logits"] if isinstance(outputs, dict) else outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal ligand-dependency evaluation for V12m3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    resolve_contact_threshold(args, ckpt)
    config = load_config_from_checkpoint_or_yaml(ckpt, args.config)
    model = create_v12m3_model(ckpt, config, device)
    d_unimol = int(config.get("model", {}).get("d_unimol", 512))

    run_ligand_dependency_eval(
        args=args,
        config=config,
        model=model,
        model_tag="v12m3",
        device=device,
        predict_logits=predict_v12m3_logits,
        needs_unimol=True,
        d_unimol=d_unimol,
    )


if __name__ == "__main__":
    main()
