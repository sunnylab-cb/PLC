import copy
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v12
from src.models.pocket_gvp_fusion_v12 import PocketGVPFusionV12
from scripts.train.train_v12 import (
    compute_combined_loss_v4,
    compute_dist_aux_loss,
    get_batch_unimol,
    load_unimol_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def is_finite(x):
    return bool(torch.isfinite(x).all().item())


def stat(name, x):
    if x is None:
        print(f"{name}: None")
        return
    y = x.detach().float()
    print(
        f"{name}: finite={is_finite(y)} nan={torch.isnan(y).sum().item()} "
        f"inf={torch.isinf(y).sum().item()} mean={torch.nan_to_num(y).mean().item():.4g} "
        f"std={torch.nan_to_num(y).std().item():.4g} "
        f"min={torch.nan_to_num(y).min().item():.4g} max={torch.nan_to_num(y).max().item():.4g}"
    )


def make_batch(cfg, device):
    data_cfg = cfg["data"]
    train_ids, _, _ = load_splits(data_cfg["split_file"])
    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=train_ids[: cfg["training"].get("batch_size", 12)],
        max_seq_len=data_cfg["max_seq_len"],
        max_atoms=data_cfg["max_atoms"],
        distances_file=data_cfg.get("distances_file"),
        ligand_features_dir=data_cfg.get("ligand_features_dir", "ligand_features_v5"),
        protein_features_dir=data_cfg.get("protein_features_dir", "complete_features_v4"),
        nma_features_dir=data_cfg.get("nma_features_dir", "complete_features_v4_nma"),
        nma_n_confs=data_cfg.get("nma_n_confs", 1),
        nma_max_conf=data_cfg.get("nma_max_conf", 9),
        label_contact_thresh=data_cfg.get("label_contact_thresh", 5.0),
        label_near_thresh=data_cfg.get("label_near_thresh", 8.0),
        split="train",
        use_nma=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 12),
        shuffle=False,
        num_workers=0,
        collate_fn=pocket_collate_fn_v10,
    )
    batch = next(iter(loader))
    cache = load_unimol_cache(
        data_cfg.get("data_root", "data"),
        data_cfg.get("unimol_embeddings_dir", "unimol_embeddings"),
    )
    unimol = get_batch_unimol(
        batch["pdb_ids"],
        cache,
        cfg["model"].get("d_unimol", 512),
        device,
    )
    inputs = prepare_model_inputs_v12(batch, device, non_blocking=False, unimol_embed=unimol)
    labels = batch["labels"].to(device)
    mask = batch["protein_mask"].to(device)
    return batch, inputs, labels, mask


def install_hooks(model):
    bad = []

    def hook(name):
        def _hook(_module, _inp, out):
            tensors = out if isinstance(out, tuple) else (out,)
            for i, t in enumerate(tensors):
                if not torch.is_tensor(t):
                    continue
                # DistanceBias intentionally contains -inf in masked key columns.
                # Treat NaN as fatal everywhere; treat Inf as fatal outside bias modules.
                has_nan = bool(torch.isnan(t).any().item())
                has_inf = bool(torch.isinf(t).any().item())
                inf_allowed = ("dist_bias" in name)
                if has_nan or (has_inf and not inf_allowed):
                    bad.append((name, i, tuple(t.shape), str(t.dtype)))
                    print(f"FIRST_BAD_HOOK {name}[{i}] shape={tuple(t.shape)} dtype={t.dtype}")
                    stat(f"{name}[{i}]", t)
                    raise RuntimeError(f"non-finite output at {name}")
        return _hook

    for name, module in model.named_modules():
        if any(key in name for key in (
            "co_attn_blocks",
            "prot_dist_bias",
            "lig_dist_bias",
            "prot_self_attn",
            "lig_self_attn",
            "prot_to_lig",
            "lig_to_prot",
            "film",
        )):
            module.register_forward_hook(hook(name))
    return bad


def run_case(name, dtype, model_overrides, cfg, inputs, labels, mask, batch, backward=True):
    print("\n" + "=" * 80)
    print(f"CASE {name} dtype={dtype} overrides={model_overrides}")
    print("=" * 80)
    device = labels.device
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.update(model_overrides)
    model_cfg["use_rsa_input"] = False
    model_cfg["use_sasa_gate"] = False
    model = PocketGVPFusionV12(model_cfg).to(device).train()
    install_hooks(model)
    class_weights = torch.tensor(cfg["training"]["class_weights"], device=device, dtype=torch.float32)
    dist_cfg = cfg["training"].get("dist_aux", {})

    try:
        with torch.amp.autocast("cuda", enabled=(dtype is not None), dtype=dtype or torch.float32):
            out = model(**inputs)
            logits = out["logits"]
            loss, tv, focal = compute_combined_loss_v4(
                logits=logits,
                labels=labels,
                mask=mask,
                class_weights=class_weights,
                focal_gamma=cfg["training"]["focal_gamma"],
                focal_weight=cfg["training"]["focal_weight"],
                tversky_weight=cfg["training"]["tversky_weight"],
                tversky_alpha=cfg["training"]["tversky_alpha"],
                tversky_beta=cfg["training"]["tversky_beta"],
                tversky_smooth=cfg["training"]["tversky_smooth"],
                weighted_dice_weight=cfg["training"].get("weighted_dice_weight", 0.0),
                weighted_dice_smooth=cfg["training"].get("weighted_dice_smooth", 1.0),
                label_smoothing=cfg["training"].get("label_smoothing", 0.05),
            )
            da = torch.tensor(0.0, device=device)
            if dist_cfg.get("enabled", False) and "dist_pred" in out:
                da = compute_dist_aux_loss(
                    out["dist_pred"],
                    batch["raw_distances"].to(device),
                    mask,
                    dist_cfg.get("max_dist", 20.0),
                )
                loss = loss + dist_cfg.get("weight", 0.05) * da
        stat("logits", logits)
        stat("dist_pred", out.get("dist_pred"))
        print(
            f"loss={loss.detach().float().item():.6g} tv={tv.detach().float().item():.6g} "
            f"focal={focal.detach().float().item():.6g} dist_aux={da.detach().float().item():.6g}"
        )
        probs = torch.softmax(logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        print("pred_frac", torch.bincount(pred[mask], minlength=3).float().div(mask.sum()).cpu().tolist())
        print("mean_prob", probs[mask].mean(0).detach().cpu().tolist())
        if backward:
            loss.backward()
            bad_grads = []
            rows = []
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                if not is_finite(g):
                    bad_grads.append((n, int(torch.isnan(g).sum()), int(torch.isinf(g).sum()), tuple(g.shape)))
                rows.append((float(torch.nan_to_num(g).norm().cpu()), n))
            rows.sort(reverse=True)
            print("bad_grad_tensors", len(bad_grads))
            for item in bad_grads[:10]:
                print("bad_grad", item)
            print("top_grad_norms", rows[:8])
    except Exception as exc:
        print(f"CASE_FAILED {name}: {type(exc).__name__}: {exc}")


def main():
    cfg = yaml.safe_load((ROOT / "configs/pocket_gvp_v12.yaml").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(device) if device.type == "cuda" else "")
    batch, inputs, labels, mask = make_batch(cfg, device)
    for key in ("esm_embeddings", "protein_physchem", "protein_spatial_scalar", "protein_coords", "ligand_x", "ligand_pos", "unimol_embed"):
        stat(f"input.{key}", inputs.get(key))
    stat("raw_distances", batch.get("raw_distances"))

    base = {"n_co_attn_blocks": 2}
    cases = [
        ("fp32_full", None, base),
        ("fp16_full", torch.float16, base),
        ("bf16_full", torch.bfloat16, base),
        ("bf16_no_film", torch.bfloat16, {**base, "use_film": False}),
        ("bf16_no_prot_dist_bias", torch.bfloat16, {**base, "use_prot_dist_bias": False}),
        ("bf16_no_lig_dist_bias", torch.bfloat16, {**base, "use_lig_dist_bias": False}),
        ("bf16_no_dist_bias", torch.bfloat16, {**base, "use_prot_dist_bias": False, "use_lig_dist_bias": False}),
        ("bf16_one_block", torch.bfloat16, {"n_co_attn_blocks": 1}),
    ]
    for name, dtype, overrides in cases:
        run_case(name, dtype, overrides, cfg, inputs, labels, mask, batch)


if __name__ == "__main__":
    main()
