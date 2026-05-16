#!/usr/bin/env python3
"""Check V12 identity-init invariants on one real batch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v12
from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.models.pocket_gvp_fusion_v12 import PocketGVPFusionV12
from scripts.train.train_v12 import compute_combined_loss_v4, get_batch_unimol, load_unimol_cache


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/pocket_gvp_v12.yaml").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("experiment", {}).get("seed", 42))

    data_cfg = cfg["data"]
    train_ids, _, _ = load_splits(data_cfg["split_file"])
    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=train_ids[: cfg["training"].get("batch_size", 15)],
        max_seq_len=data_cfg["max_seq_len"],
        max_atoms=data_cfg["max_atoms"],
        distances_file=data_cfg.get("distances_file"),
        distances_file_original=data_cfg.get("distances_file_original"),
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
    batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=pocket_collate_fn_v10)))

    model_cfg = dict(cfg["model"])
    model_cfg["n_co_attn_blocks"] = 2
    model_cfg["use_rsa_input"] = False
    model_cfg["use_sasa_gate"] = False
    model = PocketGVPFusionV12(model_cfg).to(device).train()

    cache = load_unimol_cache(
        data_cfg.get("data_root", "data"),
        data_cfg.get("unimol_embeddings_dir", "unimol_embeddings"),
    )
    unimol = get_batch_unimol(batch["pdb_ids"], cache, model_cfg.get("d_unimol", 512), device)
    inputs = prepare_model_inputs_v12(batch, device, non_blocking=False, unimol_embed=unimol)
    labels = batch["labels"].to(device)
    mask = batch["protein_mask"].to(device)
    class_weights = torch.tensor(cfg["training"]["class_weights"], device=device, dtype=torch.float32)

    deltas = []

    def block_hook(idx):
        def hook(_module, in_tup, out_tup):
            hp_in, hl_in = in_tup[0], in_tup[1]
            hp_out, hl_out = out_tup
            deltas.append((
                idx,
                float((hp_out - hp_in).detach().float().abs().max().cpu()),
                float((hl_out - hl_in).detach().float().abs().max().cpu()),
            ))
        return hook

    handles = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(model.co_attn_blocks)]
    out = model(**inputs)
    for h in handles:
        h.remove()

    loss, _, _ = compute_combined_loss_v4(
        logits=out["logits"],
        labels=labels,
        mask=mask,
        class_weights=class_weights,
        focal_gamma=cfg["training"]["focal_gamma"],
        focal_weight=cfg["training"].get("focal_weight", 0.7),
        tversky_weight=cfg["training"].get("tversky_weight", 0.0),
        label_smoothing=cfg["training"].get("label_smoothing", 0.05),
    )
    loss.backward()

    print("block_output_delta_max:")
    for idx, dp, dl in deltas:
        print(f"  block{idx}: prot={dp:.6e} lig={dl:.6e}")

    rows = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "co_attn_blocks" not in name:
            continue
        g = param.grad.detach()
        if any(token in name for token in (".norm", ".w_q", ".w_k", ".w_v", ".gate")):
            rows.append((float(torch.linalg.vector_norm(g.float()).cpu()), name))
    rows.sort(reverse=True)
    print("top_internal_coattn_grads:")
    for norm, name in rows[:30]:
        print(f"  {norm:.6e} {name}")


if __name__ == "__main__":
    main()
