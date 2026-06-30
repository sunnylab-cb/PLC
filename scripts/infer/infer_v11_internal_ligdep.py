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


def _as_source_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip().lower() for part in value.split(",") if part.strip()}
    return {str(part).strip().lower() for part in value if str(part).strip()}


def model_needs_unimol(model) -> bool:
    """Return True when any active model path consumes batch Uni-Mol embeddings."""
    cfg = getattr(model, "config", {}) or {}

    if bool(getattr(model, "use_unimol_ligand_input", cfg.get("use_unimol_ligand_input", False))):
        return True

    if bool(getattr(model, "use_ligand_context_prior", cfg.get("use_ligand_context_prior", False))):
        sources = _as_source_set(
            getattr(model, "ligand_context_prior_sources", cfg.get("ligand_context_prior_sources", []))
        )
        if "unimol" in sources:
            return True

    for flag in (
        "use_unimol_memory_kv",
        "use_unimol_group_gate",
        "use_unimol_atom_gate",
        "use_unimol_residue_compat",
    ):
        if bool(getattr(model, flag, cfg.get(flag, False))):
            return True

    # Conservative future-proofing for V12-style paths if reused through this runner.
    if getattr(model, "unimol_to_global", None) is not None:
        return True
    co_blocks = getattr(model, "co_attn_blocks", None)
    if co_blocks is not None and any(getattr(block, "film", None) is not None for block in co_blocks):
        return True

    return False


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
    has_local54_group_adapter = any(k.startswith("local54_group_adapter.") for k in state)
    has_v11m8_probe = (
        any(k.startswith("local_ligand_film") for k in state)
        or any(k.startswith("local54_chem_") for k in state)
        or any(k.startswith("pocket_") for k in state)
        or any(k.startswith("center_aux_") for k in state)
    )
    has_v11m8x_probe = any(
        k.startswith("cross_attention_layers.")
        and (
            ".block.cross_attn." in k
            or ".block.prot_to_lig.cross_attn." in k
            or ".block.prot_self_attn." in k
            or k.endswith(".raw_alpha")
        )
        for k in state
    )
    has_v11m9_probe = any(k.startswith("local_evidence_") for k in state)
    has_v11m10_probe = (
        any(k.startswith("local54_interaction_") for k in state)
        or any(k.startswith("local54_compat_") for k in state)
        or any(k.startswith("ligand_context_prior") for k in state)
    )
    has_v11m12_probe = (
        any(k.startswith("unimol_memory_") for k in state)
        or any(k.startswith("unimol_group_gate") for k in state)
        or any(k.startswith("unimol_atom_") for k in state)
        or any(k.startswith("unimol_residue_") for k in state)
    )
    has_v11m14_probe = any(k.startswith("local54_basecond_") for k in state)
    has_v11m15_probe = any(k.startswith("assignment_") for k in state)
    has_v13_probe = any(
        k.startswith("residue_atom_bias_")
        or k.startswith("residue_atom_triangle_")
        or k.startswith("ligand_slot_")
        or k.startswith("local54_q_reparam")
        or k.startswith("local54_k_reparam")
        or k.startswith("residue_chem_value_enrich_")
        for k in state
    )
    has_v11m13_probe = any(k.startswith("local54_edge_context_") for k in state)
    has_v11m11_probe = (
        any(k.startswith("bimamba_") for k in state)
        or any(k.startswith("mamba_lig_") for k in state)
    )

    if any(tag in exp_name for tag in ("v13m0", "v13m1", "v13m2")) or has_v13_probe:
        return "v13m0"
    if "v11m15" in exp_name or has_v11m15_probe:
        return "v11m15"
    if "v11m14" in exp_name or has_v11m14_probe:
        return "v11m14"
    if "v11m13" in exp_name or has_v11m13_probe:
        return "v11m13"
    if "v11m12" in exp_name or has_v11m12_probe:
        return "v11m12"
    if "v11m11" in exp_name or has_v11m11_probe:
        return "v11m11"
    if "v11m10" in exp_name or has_v11m10_probe:
        return "v11m10"
    if "v11m9" in exp_name or has_v11m9_probe:
        return "v11m9"
    if "v11m8x" in exp_name or has_v11m8x_probe:
        return "v11m8x"
    if "v11m8" in exp_name or has_v11m8_probe:
        return "v11m8"
    if "v11m7" in exp_name or has_local54_group_adapter:
        return "v11m7"
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

    if version == "v13m0":
        from src.models.pocket_gvp_fusion_v13m0 import PocketGVPFusionV13m0 as ModelClass
    elif version == "v11m15":
        from src.models.pocket_gvp_fusion_v11m15 import PocketGVPFusionV11m15 as ModelClass
    elif version == "v11m14":
        from src.models.pocket_gvp_fusion_v11m14 import PocketGVPFusionV11m14 as ModelClass
    elif version == "v11m12":
        from src.models.pocket_gvp_fusion_v11m12 import PocketGVPFusionV11m12 as ModelClass
    elif version == "v11m13":
        from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13 as ModelClass
    elif version == "v11m11":
        from src.models.pocket_gvp_fusion_v11m11 import PocketGVPFusionV11m11 as ModelClass
    elif version == "v11m10":
        from src.models.pocket_gvp_fusion_v11m10 import PocketGVPFusionV11m10 as ModelClass
    elif version == "v11m9":
        from src.models.pocket_gvp_fusion_v11m9 import PocketGVPFusionV11m9 as ModelClass
    elif version == "v11m8x":
        from src.models.pocket_gvp_fusion_v11m8x import PocketGVPFusionV11m8x as ModelClass
    elif version == "v11m8":
        from src.models.pocket_gvp_fusion_v11m8 import PocketGVPFusionV11m8 as ModelClass
    elif version == "v11m7":
        from src.models.pocket_gvp_fusion_v11m7 import PocketGVPFusionV11m7 as ModelClass
    elif version == "v11m6":
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
    if model_needs_unimol(model):
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

    needs_unimol = model_needs_unimol(model)
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
