#!/usr/bin/env python3
"""
No-train input-balance probe for V11-family checkpoints/configs.

This answers two practical questions:
1. Ligand entry: are local54/global55/UniMol64 comparable after the first
   ligand node projection, i.e. RMS(W_group x_group)?
2. Protein entry: are GVP/physchem geometry and ESM comparable after their
   projection branches before the modality gate?

It runs a few normal forward passes on the internal split and writes a JSON
summary.  It does not update model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.infer.infer_v11_internal_ligdep import (  # noqa: E402
    create_v11_model,
    predict_v11_logits,
)
from scripts.infer.internal_ligand_dependency import (  # noqa: E402
    add_common_args,
    batch_unimol,
    build_internal_dataset,
    collate_with_ligand_source,
    load_checkpoint,
    load_config_from_checkpoint_or_yaml,
    load_unimol_cache,
)


def rms_per_channel(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    xf = x.detach().float()
    return float(xf.norm(dim=-1).mean().item() / math.sqrt(max(int(xf.size(-1)), 1)))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def ligand_slices(model) -> Dict[str, Tuple[int, int]]:
    if bool(getattr(model, "use_ligand_groupwise_input_balance", False)):
        a = int(model.ligand_group_local_proj_dim)
        b = a + int(model.ligand_group_global_proj_dim)
        c = b + int(model.ligand_group_unimol_proj_dim)
        return {"local": (0, a), "global": (a, b), "unimol": (b, c)}

    local_dim = 54
    raw_dim = int(getattr(model, "raw_ligand_atom_dim", 109))
    unimol_dim = int(getattr(model, "d_unimol_ligand_proj", 0))
    return {
        "local": (0, local_dim),
        "global": (local_dim, raw_dim),
        "unimol": (raw_dim, raw_dim + unimol_dim),
    }


def add_stat(acc: Dict[str, list], key: str, value: float) -> None:
    if math.isfinite(value):
        acc[key].append(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe V11 ligand/protein input source balance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--max_batches", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = load_config_from_checkpoint_or_yaml(ckpt, args.config)
    model, version = create_v11_model(ckpt, config, device)
    model.eval()
    model.config["cache_intermediates"] = True

    dataset = build_internal_dataset(config, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_ligand_source,
        pin_memory=(device.type == "cuda"),
    )

    needs_unimol = bool(getattr(model, "use_unimol_ligand_input", False))
    d_unimol = int(getattr(model, "d_unimol", config.get("model", {}).get("d_unimol", 512)))
    unimol_cache = load_unimol_cache(config, args, d_unimol=d_unimol) if needs_unimol else {}

    stats: Dict[str, list] = defaultdict(list)
    slices = ligand_slices(model)
    node_proj_w = model.ligand_encoder.node_proj.weight.detach()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            raw_lig = batch["ligand_x"].to(device, non_blocking=True)
            lig_batch = batch["ligand_batch"].to(device, non_blocking=True)
            ue = batch_unimol(
                batch["pdb_ids"],
                cache=unimol_cache,
                d_unimol=d_unimol,
                device=device,
                seed=args.shuffle_seed,
            )

            x_aug = model._augment_ligand_x(
                ligand_x=raw_lig,
                ligand_batch=lig_batch,
                unimol_embed=ue,
                batch_size=len(batch["pdb_ids"]),
            )

            add_stat(stats, "ligand_raw_local54_rms", rms_per_channel(raw_lig[:, :54]))
            add_stat(stats, "ligand_raw_global55_rms", rms_per_channel(raw_lig[:, 54:109]))
            add_stat(stats, "ligand_aug_total_rms", rms_per_channel(x_aug))

            for name, (start, end) in slices.items():
                if end <= start or end > x_aug.size(-1):
                    continue
                group = x_aug[:, start:end]
                contrib = F.linear(group, node_proj_w[:, start:end], None)
                add_stat(stats, f"ligand_{name}_aug_rms", rms_per_channel(group))
                add_stat(stats, f"ligand_{name}_nodeproj_contrib_rms", rms_per_channel(contrib))

            # Normal forward to populate protein projection caches.
            _ = predict_v11_logits(model, batch, device, ue)
            for attr, key in (
                ("_cached_h_gvp_proj", "protein_gvp_proj_rms"),
                ("_cached_h_esm_proj", "protein_esm_proj_rms"),
            ):
                val = getattr(model, attr, None)
                if val is not None:
                    add_stat(stats, key, rms_per_channel(val))
            gate = getattr(model, "_cached_gate", None)
            h_gvp = getattr(model, "_cached_h_gvp_proj", None)
            h_esm = getattr(model, "_cached_h_esm_proj", None)
            if gate is not None and h_gvp is not None and h_esm is not None:
                add_stat(stats, "protein_gate_mean", float(gate.detach().float().mean().item()))
                add_stat(stats, "protein_gated_gvp_rms", rms_per_channel(gate * h_gvp))
                add_stat(stats, "protein_gated_esm_rms", rms_per_channel((1.0 - gate) * h_esm))

            group_stats = getattr(model, "_cached_ligand_group_stats", None)
            if isinstance(group_stats, dict):
                for key, value in group_stats.items():
                    add_stat(stats, key, float(value.detach().float().item()))

    summary = {
        "model_tag": version,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_batches": min(args.max_batches, len(loader)),
        "stats": {key: mean(vals) for key, vals in sorted(stats.items())},
    }

    out = Path(args.output) if args.output else Path("outputs/infer") / f"input_balance_{version}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "input_balance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
