#!/usr/bin/env python
"""Summarize numeric-scale moments from trained V11m6n checkpoints.

This script is intentionally read-only.  It does not evaluate ligand dependency
or tune thresholds; it extracts scale statistics that can guide initialization:

- base/final/delta logit magnitudes
- effective residual delta magnitude relative to base logits
- local_delta QK score distribution and attention entropy
- learned local_delta residual scale
- train/val label priors on the sampled batches

Use train split only when deriving initialization constants.  Val/test summaries
are for sanity checks, not for choosing priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11  # noqa: E402
from src.data.pocket_dataset import load_splits  # noqa: E402
from src.data.pocket_dataset_v8 import PocketDatasetV8  # noqa: E402
from src.models.pocket_gvp_fusion_v11m6 import PocketGVPFusionV11m6  # noqa: E402


def load_unimol_cache(data_root: str, unimol_dir: str) -> Dict[str, torch.Tensor]:
    root = Path(data_root) / unimol_dir
    if not root.exists():
        raise FileNotFoundError(f"UniMol embedding dir not found: {root}")
    cache: Dict[str, torch.Tensor] = {}
    for fpath in root.rglob("*_unimol.pt"):
        pdb_id = fpath.stem.replace("_unimol", "")
        obj = torch.load(fpath, map_location="cpu", weights_only=True)
        embed = obj.get("embedding") if isinstance(obj, dict) else obj
        if isinstance(embed, torch.Tensor) and tuple(embed.shape) == (512,):
            cache[pdb_id] = embed.float()
    if not cache:
        raise RuntimeError(f"No valid UniMol embeddings found in {root}")
    return cache


def get_batch_unimol(
    pdb_ids: list[str],
    cache: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | None:
    if not cache:
        return None
    missing = [pid for pid in pdb_ids if pid not in cache]
    if missing:
        raise KeyError(f"Missing UniMol embeddings for examples: {missing[:5]}")
    return torch.stack([cache[pid] for pid in pdb_ids], dim=0).to(device, non_blocking=True)


class OnlineValues:
    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []

    def add(self, tensor: torch.Tensor | float | int | None) -> None:
        if tensor is None:
            return
        if not torch.is_tensor(tensor):
            tensor = torch.tensor([float(tensor)])
        tensor = tensor.detach().float().reshape(-1).cpu()
        tensor = tensor[torch.isfinite(tensor)]
        if tensor.numel():
            self.values.append(tensor)

    def summary(self, prefix: str) -> dict[str, float]:
        if not self.values:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_abs_mean": float("nan"),
                f"{prefix}_p05": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p95": float("nan"),
            }
        x = torch.cat(self.values)
        return {
            f"{prefix}_n": int(x.numel()),
            f"{prefix}_mean": float(x.mean().item()),
            f"{prefix}_std": float(x.std(unbiased=False).item()),
            f"{prefix}_abs_mean": float(x.abs().mean().item()),
            f"{prefix}_p05": float(torch.quantile(x, 0.05).item()),
            f"{prefix}_p50": float(torch.quantile(x, 0.50).item()),
            f"{prefix}_p95": float(torch.quantile(x, 0.95).item()),
        }


def masked_flat(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor).bool()
    return tensor[mask]


def load_model(checkpoint: Path, device: torch.device) -> tuple[PocketGVPFusionV11m6, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        cfg_path = checkpoint.parent / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config in checkpoint or alongside it: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text())
    model_cfg = dict(cfg["model"])
    model = PocketGVPFusionV11m6(model_cfg)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [k for k in missing if not k.endswith("local_delta_qk_log_scale")]
    if unexpected or relevant_missing:
        print(
            "WARNING checkpoint loaded with non-trivial missing/unexpected keys: "
            f"missing={relevant_missing[:8]} unexpected={list(unexpected)[:8]}",
            file=sys.stderr,
        )
    model.to(device)
    model.eval()
    return model, cfg


def build_dataset(config: dict[str, Any], split: str, limit_ids: int | None, train_use_nma: bool) -> PocketDatasetV8:
    data_cfg = config["data"]
    train_ids, val_ids, test_ids = load_splits(data_cfg["split_file"])
    ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}
    pdb_ids = ids_by_split[split]
    if limit_ids is not None:
        pdb_ids = pdb_ids[:limit_ids]
    return PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=pdb_ids,
        max_seq_len=data_cfg["max_seq_len"],
        max_atoms=data_cfg["max_atoms"],
        distances_file=data_cfg.get("distances_file"),
        distances_file_original=data_cfg.get("distances_file_original"),
        ligand_features_dir=data_cfg.get("ligand_features_dir", "ligand_features_v4"),
        protein_features_dir=data_cfg.get("protein_features_dir", "complete_features_v3"),
        nma_features_dir=data_cfg.get("nma_features_dir", "complete_features_v3_nma"),
        nma_n_confs=data_cfg.get("nma_n_confs", 1),
        nma_max_conf=data_cfg.get("nma_max_conf", 9),
        use_nma=(split == "train" and train_use_nma),
        label_contact_thresh=data_cfg.get("label_contact_thresh", 5.0),
        label_near_thresh=data_cfg.get("label_near_thresh", 8.0),
        use_spatial_clamp=data_cfg.get("use_spatial_clamp", False),
        spatial_clamp_dims=data_cfg.get("spatial_clamp_dims", [62, 76, 78, 79]),
        spatial_clamp_range=data_cfg.get("spatial_clamp_range", [-5.0, 5.0]),
        split=split,
        verbose=False,
    )


def compute_local_delta_score_stats(
    model: PocketGVPFusionV11m6,
    outputs: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not getattr(model, "use_local_delta_only", False):
        return {}
    h_prot = outputs.get("h_fused")
    h_ligand_fused = outputs.get("h_ligand_fused")
    if h_prot is None or h_ligand_fused is None:
        return {}

    ligand_mask = inputs["ligand_mask"]
    protein_mask = inputs["protein_mask"]
    max_atoms = ligand_mask.size(1)
    h_lig_atoms = h_ligand_fused[:, 1 : 1 + max_atoms, :]
    local54 = model._pad_ligand_local_features(  # noqa: SLF001
        inputs["ligand_x"], inputs["ligand_batch"], ligand_mask
    )

    hp = model.local_delta_prot_norm(h_prot)
    hl = model.local_delta_lig_norm(h_lig_atoms)
    lf = model.local_delta_local_norm(local54.to(dtype=hl.dtype))

    q = model.local_delta_q(hp)
    k = torch.zeros((*hl.shape[:2], model.local_delta_dim), device=hl.device, dtype=hl.dtype)
    if model.local_delta_use_ligand_hidden:
        k = k + model.local_delta_k_lig(hl)
    if model.local_delta_use_local_features:
        k = k + model.local_delta_k_local(lf)

    if getattr(model, "local_delta_use_qk_norm", False):
        q_score = F.normalize(q.float(), dim=-1)
        k_score = F.normalize(k.float(), dim=-1)
        qk_scale = model.local_delta_qk_log_scale.exp().clamp(max=model.local_delta_score_clip)
        scores = torch.matmul(q_score, k_score.transpose(-2, -1)) * qk_scale
    else:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.local_delta_dim)
    scores = scores.float().clamp(-model.local_delta_score_clip, model.local_delta_score_clip)

    atom_mask = ligand_mask.bool()
    score_mask = protein_mask.bool().unsqueeze(-1) & atom_mask.unsqueeze(1)
    masked_scores = scores[score_mask]

    softmax_scores = scores.masked_fill(~atom_mask.unsqueeze(1), -1.0e4)
    has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
    softmax_scores = torch.where(has_atom, softmax_scores, torch.zeros_like(softmax_scores))
    attn = torch.softmax(softmax_scores, dim=-1) * atom_mask.unsqueeze(1).float()
    denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    attn = torch.where(has_atom, attn / denom, torch.zeros_like(attn))
    entropy = -(attn.float() * attn.float().clamp(min=1e-8).log()).sum(dim=-1)
    top1 = attn.float().max(dim=-1).values
    residue_mask = protein_mask.bool()
    return {
        "local_score": masked_scores,
        "local_attn_entropy": entropy[residue_mask],
        "local_attn_top1": top1[residue_mask],
    }


def summarize_checkpoint(
    checkpoint: Path,
    split: str,
    max_batches: int,
    batch_size: int,
    num_workers: int,
    limit_ids: int | None,
    train_use_nma: bool,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model, cfg = load_model(checkpoint, device)
    data_cfg = cfg["data"]
    dataset = build_dataset(cfg, split=split, limit_ids=limit_ids, train_use_nma=train_use_nma)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pocket_collate_fn_v10,
        pin_memory=(device.type == "cuda"),
    )
    unimol_cache = {}
    if cfg["model"].get("use_unimol_ligand_input", True):
        unimol_cache = load_unimol_cache(data_cfg["data_root"], data_cfg.get("unimol_embeddings_dir", "unimol_embeddings"))

    stats = defaultdict(OnlineValues)
    label_counts: Counter[int] = Counter()
    n_batches = 0
    n_samples = 0

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if cfg.get("training", {}).get("use_bf16", True) else torch.float16

    with torch.no_grad():
        for batch in loader:
            if n_batches >= max_batches:
                break
            inputs = prepare_model_inputs_v11(batch, device, non_blocking=(device.type == "cuda"))
            labels = batch["labels"].to(device, non_blocking=(device.type == "cuda"))
            mask = batch["protein_mask"].to(device, non_blocking=(device.type == "cuda")).bool()
            unimol = get_batch_unimol(batch["pdb_ids"], unimol_cache, device)
            with torch.amp.autocast(amp_device, enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                outputs = model(**inputs, unimol_embed=unimol)

            base_logits = outputs["base_logits"].float()
            logits = outputs["logits"].float()
            delta_logits = outputs["delta_logits"].float()
            effective_delta = logits - base_logits

            stats["base_logits"].add(masked_flat(base_logits, mask))
            stats["final_logits"].add(masked_flat(logits, mask))
            stats["delta_logits"].add(masked_flat(delta_logits, mask))
            stats["effective_delta"].add(masked_flat(effective_delta, mask))
            stats["delta_to_base_abs_ratio"].add(
                masked_flat(effective_delta.abs(), mask)
                / masked_flat(base_logits.abs().clamp_min(1e-6), mask)
            )
            for key in (
                "pair_delta_scale",
                "local_delta_scale",
                "local_delta_score_abs_mean",
                "local_delta_attn_entropy_mean",
                "local_delta_logits_abs_mean",
                "local_delta_qk_scale",
                "delta_gate_mean",
                "delta_gate_abs_dev",
            ):
                if key in outputs:
                    stats[key].add(outputs[key])

            for key, tensor in compute_local_delta_score_stats(model, outputs, inputs).items():
                stats[key].add(tensor)

            valid_labels = labels[mask].detach().cpu().tolist()
            label_counts.update(int(x) for x in valid_labels)
            n_batches += 1
            n_samples += len(batch["pdb_ids"])

    out: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "run": checkpoint.parent.name,
        "split": split,
        "n_batches": n_batches,
        "n_samples": n_samples,
        "dataset_size": len(dataset),
        "model_grad_scale": cfg["model"].get("local_delta_protein_grad_scale"),
        "model_qk_norm": bool(cfg["model"].get("local_delta_use_qk_norm", False)),
        "model_qk_alpha_init": cfg["model"].get("local_delta_qk_norm_scale_init"),
        "model_delta_scale_init_raw": cfg["model"].get("local_delta_logit_scale_init"),
        "model_final_init_std": cfg["model"].get("local_delta_final_init_std", 0.0),
        "model_zero_init_final": bool(cfg["model"].get("local_delta_zero_init_final", True)),
        "model_no_gate": not bool(cfg["model"].get("use_global_delta_gate", True)),
    }
    total_labels = sum(label_counts.values())
    for klass in range(int(cfg["model"].get("n_classes", 3))):
        count = label_counts.get(klass, 0)
        prior = count / total_labels if total_labels else float("nan")
        out[f"label_count_c{klass}"] = count
        out[f"label_prior_c{klass}"] = prior
        out[f"label_log_prior_c{klass}"] = math.log(max(prior, 1e-12)) if total_labels else float("nan")

    for name, acc in sorted(stats.items()):
        out.update(acc.summary(name))

    raw_scale = getattr(model, "local_delta_logit_scale", None)
    if raw_scale is not None and torch.is_tensor(raw_scale):
        out["learned_delta_scale_raw"] = float(raw_scale.detach().float().cpu().item())
        out["learned_delta_scale_effective"] = float(model._delta_scale().detach().float().cpu().item())  # noqa: SLF001
    qk_log_scale = getattr(model, "local_delta_qk_log_scale", None)
    if qk_log_scale is not None and torch.is_tensor(qk_log_scale):
        out["learned_qk_alpha"] = float(qk_log_scale.detach().float().exp().cpu().item())

    return out


def write_outputs(result: dict[str, Any], output_json: Path, output_csv: Path | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = output_csv.exists()
        with output_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(result.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-ids", type=int, default=None)
    parser.add_argument("--train-use-nma", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = summarize_checkpoint(
        checkpoint=args.checkpoint,
        split=args.split,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_ids=args.limit_ids,
        train_use_nma=args.train_use_nma,
        device=device,
        use_amp=args.amp,
    )
    write_outputs(result, args.output_json, args.output_csv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
