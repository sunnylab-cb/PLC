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
    get_batch_unimol,
    load_unimol_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def finite(x):
    return bool(torch.isfinite(x).all().item())


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = yaml.safe_load((ROOT / "configs/pocket_gvp_v12.yaml").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_ids, _, _ = load_splits(data_cfg["split_file"])
    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=train_ids[:360],
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
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["use_rsa_input"] = False
    model_cfg["use_sasa_gate"] = False
    model_cfg["use_dist_aux"] = False
    model = PocketGVPFusionV12(model_cfg).to(device).train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 5e-5),
        weight_decay=cfg["training"].get("weight_decay", 0.05),
    )
    cache = load_unimol_cache(data_cfg.get("data_root", "data"), data_cfg.get("unimol_embeddings_dir", "unimol_embeddings"))
    class_weights = torch.tensor(cfg["training"]["class_weights"], device=device, dtype=torch.float32)

    torch.autograd.set_detect_anomaly(True)
    for step, batch in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        unimol = get_batch_unimol(batch["pdb_ids"], cache, model_cfg.get("d_unimol", 512), device)
        inputs = prepare_model_inputs_v12(batch, device, non_blocking=False, unimol_embed=unimol)
        labels = batch["labels"].to(device)
        mask = batch["protein_mask"].to(device)
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
        print(f"step={step} loss={loss.detach().item():.6f} logits_finite={finite(logits)}", flush=True)
        with torch.autograd.detect_anomaly(check_nan=True):
            loss.backward()
        bad = []
        for n, p in model.named_parameters():
            if p.grad is not None and not finite(p.grad.detach()):
                bad.append((n, int(torch.isnan(p.grad).sum()), int(torch.isinf(p.grad).sum()), tuple(p.shape)))
                if len(bad) >= 8:
                    break
        if bad:
            print("BAD_GRADS_AFTER_BACKWARD", bad, flush=True)
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clip", 1.0), error_if_nonfinite=True)
        opt.step()
        if step >= 40:
            break
    print("ANOMALY_PROBE_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
