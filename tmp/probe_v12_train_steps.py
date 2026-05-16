import copy
import math
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
    compute_metrics,
    compute_combined_loss_v4,
    compute_dist_aux_loss,
    get_batch_unimol,
    load_unimol_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def finite(x):
    return bool(torch.isfinite(x).all().item())


def make_loader(cfg, n_ids=512):
    data_cfg = cfg["data"]
    train_ids, _, _ = load_splits(data_cfg["split_file"])
    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=train_ids[:n_ids],
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
    return DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 12),
        shuffle=False,
        num_workers=0,
        collate_fn=pocket_collate_fn_v10,
    )


def grad_rows(model, topn=6):
    rows = []
    bad = []
    total_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not finite(g):
            bad.append((n, int(torch.isnan(g).sum()), int(torch.isinf(g).sum()), tuple(g.shape)))
        gn = float(torch.nan_to_num(g).norm().cpu())
        total_sq += gn * gn
        rows.append((gn, n))
    rows.sort(reverse=True)
    return math.sqrt(total_sq), bad, rows[:topn]


def run_case(name, cfg, dtype, overrides, steps=30, lr=None, dist_aux=True):
    print("\n" + "=" * 100)
    print(f"TRAIN_CASE {name} dtype={dtype} overrides={overrides} steps={steps} lr={lr} dist_aux={dist_aux}")
    print("=" * 100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.update(overrides)
    model_cfg["use_rsa_input"] = False
    model_cfg["use_sasa_gate"] = False
    if not dist_aux:
        model_cfg["use_dist_aux"] = False
    model = PocketGVPFusionV12(model_cfg).to(device).train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr or cfg["training"].get("learning_rate", 5e-5),
        weight_decay=cfg["training"].get("weight_decay", 0.05),
        betas=tuple(cfg["training"].get("betas", [0.9, 0.999])),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == "cuda"))
    cache = load_unimol_cache(cfg["data"].get("data_root", "data"), cfg["data"].get("unimol_embeddings_dir", "unimol_embeddings"))
    loader = make_loader(cfg, n_ids=max(steps * cfg["training"].get("batch_size", 12), 64))
    class_weights = torch.tensor(cfg["training"]["class_weights"], device=device, dtype=torch.float32)
    dist_cfg = cfg["training"].get("dist_aux", {})

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        opt.zero_grad(set_to_none=True)
        unimol = get_batch_unimol(batch["pdb_ids"], cache, model_cfg.get("d_unimol", 512), device)
        inputs = prepare_model_inputs_v12(batch, device, non_blocking=False, unimol_embed=unimol)
        labels = batch["labels"].to(device)
        mask = batch["protein_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=(dtype is not None and device.type == "cuda"), dtype=dtype or torch.float32):
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
            if dist_aux and dist_cfg.get("enabled", False) and "dist_pred" in out:
                da = compute_dist_aux_loss(out["dist_pred"], batch["raw_distances"].to(device), mask, dist_cfg.get("max_dist", 20.0))
                # Match current train_v12 epoch-0 effective warmup.
                warm = min(1.0, 1.0 / max(dist_cfg.get("warmup_epochs", 5), 1))
                loss = loss + dist_cfg.get("weight", 0.05) * warm * da

        if not finite(loss) or not finite(logits):
            print(f"NONFINITE_FORWARD step={step} loss={loss.detach().float().cpu().item()}")
            return
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        total_norm, bad, top = grad_rows(model)
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clip", 1.0), error_if_nonfinite=False)
        if step % 5 == 0 or bad or (not math.isfinite(total_norm)):
            metrics = compute_metrics(logits.detach(), labels, mask)
            probs = torch.softmax(logits.detach().float(), dim=-1)
            pred = probs.argmax(-1)
            print(
                f"step={step:03d} loss={loss.detach().float().item():.5f} tv={tv.detach().float().item():.5f} "
                f"focal={focal.detach().float().item():.5f} da={da.detach().float().item():.5f} "
                f"grad={total_norm:.4e} clip_ret={float(torch.nan_to_num(clipped).cpu()):.4e} "
                f"f1_c0={metrics.get('f1_c0', 0):.4f} pred_frac={torch.bincount(pred[mask], minlength=3).float().div(mask.sum()).cpu().tolist()}"
            )
            print("  top_grad", top)
        if bad or not math.isfinite(total_norm):
            print("BAD_GRAD", bad[:10])
            return
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        bad_params = [(n, tuple(p.shape)) for n, p in model.named_parameters() if not finite(p.detach())]
        if bad_params:
            print(f"NONFINITE_PARAM_AFTER_STEP step={step}", bad_params[:10])
            return
    print(f"COMPLETED {name}")


def main():
    cfg = yaml.safe_load((ROOT / "configs/pocket_gvp_v12.yaml").read_text())
    cases = [
        ("fp32_full", None, {}, None, True),
        ("bf16_full", torch.bfloat16, {}, None, True),
        ("bf16_no_dist_aux", torch.bfloat16, {}, None, False),
        ("bf16_no_film", torch.bfloat16, {"use_film": False}, None, True),
        ("bf16_no_dist_bias", torch.bfloat16, {"use_prot_dist_bias": False, "use_lig_dist_bias": False}, None, True),
        ("bf16_lr_1e-5", torch.bfloat16, {}, 1e-5, True),
        ("fp16_full", torch.float16, {}, None, True),
    ]
    for name, dtype, overrides, lr, dist_aux in cases:
        run_case(name, cfg, dtype, overrides, steps=30, lr=lr, dist_aux=dist_aux)


if __name__ == "__main__":
    main()
