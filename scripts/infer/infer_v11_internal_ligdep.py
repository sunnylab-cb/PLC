#!/usr/bin/env python3
"""
Ligand-dependency evaluation for V11-family checkpoints on the internal split.

This does not use HOLO4K/COACH420.  It evaluates the configured PLC test split
and compares real vs shuffled/zeroed ligand inputs.
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
    run_ligand_dependency_eval,
)
from src.data.collate import prepare_model_inputs_v11


def detect_v11_version(ckpt: Dict, config: Dict = None) -> str:
    state = ckpt["model_state_dict"]
    cfg = config or ckpt.get("config", {})
    exp_name = cfg.get("experiment", {}).get("name", "").lower()

    has_global_gate = any(k.startswith("global_delta_gate.") for k in state)
    has_pair_bias = any(".pair_bias." in k for k in state)
    has_group_balance = (
        "ligand_group_log_scales" in state
        or any(k.startswith("ligand_local_adapter.") for k in state)
        or any(k.startswith("ligand_global_adapter.") for k in state)
    )
    has_local_delta = any(k.startswith("local_delta_") for k in state)

    if "v11m6" in exp_name or has_local_delta:
        return "v11m6"
    if "v11m5t" in exp_name or (has_global_gate and (has_pair_bias or has_group_balance)):
        return "v11m5t"
    if "v11m5" in exp_name or has_pair_bias:
        return "v11m5"
    if "v11m4tp" in exp_name or any(k.startswith("use_ligand_input_probe") for k in state):
        return "v11m4tp"
    if "v11m4t" in exp_name or has_global_gate:
        return "v11m4t"
    if "v11m4" in exp_name or any(k.startswith("pair_delta_") for k in state):
        return "v11m4"
    if "v11m3" in exp_name or any(k.startswith("unimol_ligand_proj.") for k in state):
        return "v11m3"
    if "lig_gate_proj.weight" in state:
        return "v11m1"
    if "v11m2" in exp_name or any(k.startswith("ligand_self_attn.dist_bias_modules") for k in state):
        return "v11m2"
    if "v11m1" in exp_name:
        return "v11m1"
    return "v11"


def create_v11_model(ckpt: Dict, config: Dict, device: torch.device):
    version = detect_v11_version(ckpt, config)
    model_cfg = dict(config["model"])
    model_cfg["use_gradient_checkpointing"] = False

    if version == "v11m6":
        from src.models.pocket_gvp_fusion_v11m6 import PocketGVPFusionV11m6 as ModelClass
    elif version == "v11m5t":
        from src.models.pocket_gvp_fusion_v11m5t import PocketGVPFusionV11m5t as ModelClass
    elif version == "v11m5":
        from src.models.pocket_gvp_fusion_v11m5 import PocketGVPFusionV11m5 as ModelClass
    elif version == "v11m4tp":
        from src.models.pocket_gvp_fusion_v11m4tp import PocketGVPFusionV11m4tp as ModelClass
    elif version == "v11m4t":
        from src.models.pocket_gvp_fusion_v11m4t import PocketGVPFusionV11m4t as ModelClass
    elif version == "v11m4":
        from src.models.pocket_gvp_fusion_v11m4 import PocketGVPFusionV11m4 as ModelClass
    elif version == "v11m3":
        from src.models.pocket_gvp_fusion_v11m3 import PocketGVPFusionV11m3 as ModelClass
    elif version == "v11m2":
        from src.models.pocket_gvp_fusion_v11m2 import PocketGVPFusionV11m2 as ModelClass
    elif version == "v11m1":
        from src.models.pocket_gvp_fusion_v11m1 import PocketGVPFusionV11m1 as ModelClass
    else:
        from src.models.pocket_gvp_fusion_v11 import PocketGVPFusionV11 as ModelClass

    model = ModelClass(model_cfg)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Backward compatibility: qk-scale was added after earlier V11m6n checkpoints.
    # Old checkpoints have local_delta_use_qk_norm=false, so the default initialized
    # scale is inert for their forward pass. Keep all other mismatches strict.
    allowed_missing = set()
    if version == "v11m6" and "local_delta_qk_log_scale" not in state:
        allowed_missing.add("local_delta_qk_log_scale")
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch when loading V11 eval model: "
            f"missing={bad_missing[:20]}, unexpected={list(unexpected)[:20]}"
        )
    if missing:
        print(f"Loaded with allowed missing keys: {list(missing)}")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Detected {version.upper()} -> {ModelClass.__name__}; params={n_params:,}")
    return model, version


def predict_v11_logits(model, batch: Dict, device: torch.device, unimol_embed: torch.Tensor):
    inputs = prepare_model_inputs_v11(batch, device)
    if getattr(model, "use_unimol_ligand_input", False):
        inputs["unimol_embed"] = unimol_embed
    outputs = model(**inputs)
    return outputs["logits"] if isinstance(outputs, dict) else outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal ligand-dependency evaluation for V11/V11m1/V11m2/V11m3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = load_config_from_checkpoint_or_yaml(ckpt, args.config)
    model, version = create_v11_model(ckpt, config, device)

    needs_unimol = bool(getattr(model, "use_unimol_ligand_input", False))
    d_unimol = int(getattr(model, "d_unimol", config.get("model", {}).get("d_unimol", 512)))
    run_ligand_dependency_eval(
        args=args,
        config=config,
        model=model,
        model_tag=version,
        device=device,
        predict_logits=predict_v11_logits,
        needs_unimol=needs_unimol,
        d_unimol=d_unimol,
    )


if __name__ == "__main__":
    main()
