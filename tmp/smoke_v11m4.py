import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
from src.data.pocket_dataset import load_splits
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.models.pocket_gvp_fusion_v11m4 import PocketGVPFusionV11m4


def load_unimol_cache(data_root, unimol_dir):
    cache = {}
    root = Path(data_root) / unimol_dir
    for fpath in root.rglob("*_unimol.pt"):
        pdb_id = fpath.stem.replace("_unimol", "")
        obj = torch.load(fpath, map_location="cpu", weights_only=True)
        embed = obj.get("embedding") if isinstance(obj, dict) else obj
        if isinstance(embed, torch.Tensor) and tuple(embed.shape) == (512,):
            embed = embed.float()
            if torch.isfinite(embed).all() and embed.abs().sum() > 1e-6:
                cache[pdb_id] = embed
    if not cache:
        raise RuntimeError(f"no Uni-Mol embeddings loaded from {root}")
    return cache


def get_batch_unimol(pdb_ids, cache, device):
    missing = [pid for pid in pdb_ids if pid not in cache]
    if missing:
        raise RuntimeError(f"missing Uni-Mol embeddings: {missing[:5]}")
    return torch.stack([cache[pid] for pid in pdb_ids], dim=0).to(device)


def main():
    cfg = yaml.safe_load(open("configs/pocket_gvp_v11m4_hpc.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    _, _, test_ids = load_splits(data_cfg["split_file"])
    pdb_ids = test_ids[:2]

    ds = PocketDatasetV8(
        data_root=data_cfg["data_root"],
        pdb_ids=pdb_ids,
        max_seq_len=data_cfg["max_seq_len"],
        max_atoms=data_cfg["max_atoms"],
        distances_file=data_cfg["distances_file"],
        ligand_features_dir=data_cfg["ligand_features_dir"],
        protein_features_dir=data_cfg["protein_features_dir"],
        nma_features_dir=data_cfg["nma_features_dir"],
        nma_n_confs=1,
        nma_max_conf=data_cfg["nma_max_conf"],
        use_nma=False,
        label_contact_thresh=data_cfg["label_contact_thresh"],
        label_near_thresh=data_cfg["label_near_thresh"],
        use_spatial_clamp=data_cfg.get("use_spatial_clamp", False),
        spatial_clamp_dims=data_cfg.get("spatial_clamp_dims"),
        spatial_clamp_range=data_cfg.get("spatial_clamp_range"),
        split="test",
        verbose=False,
    )
    batch = pocket_collate_fn_v10([ds[0], ds[1]])
    inputs = prepare_model_inputs_v11(batch, device)

    model = PocketGVPFusionV11m4(cfg["model"])
    ckpt = torch.load(cfg["training"]["init_from_checkpoint"], map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:10]}")
    bad_missing = [k for k in missing if not k.startswith("pair_delta_")]
    if bad_missing:
        raise RuntimeError(f"bad missing keys: {bad_missing[:10]}")

    model.to(device)
    model.train()

    cache = load_unimol_cache(data_cfg["data_root"], data_cfg["unimol_embeddings_dir"])
    unimol = get_batch_unimol(batch["pdb_ids"], cache, device=device)
    outputs = model(
        esm_embeddings=inputs["esm_embeddings"],
        protein_physchem=inputs["protein_physchem"],
        protein_spatial_scalar=inputs["protein_spatial_scalar"],
        protein_spatial_vector=inputs["protein_spatial_vector"],
        protein_gvp_vectors=inputs["protein_gvp_vectors"],
        protein_coords=inputs["protein_coords"],
        protein_mask=inputs["protein_mask"],
        chain_ids=inputs.get("chain_ids"),
        ligand_x=inputs["ligand_x"],
        ligand_pos=inputs["ligand_pos"],
        ligand_edge_index=inputs["ligand_edge_index"],
        ligand_edge_attr=inputs["ligand_edge_attr"],
        ligand_batch=inputs["ligand_batch"],
        ligand_mask=inputs["ligand_mask"],
        unimol_embed=unimol,
    )
    diff = (outputs["logits"] - outputs["base_logits"]).detach().abs().max().item()
    if diff > 1e-6:
        raise RuntimeError(f"delta is not zero-init: max_diff={diff}")

    labels = batch["labels"].to(device)
    mask = batch["protein_mask"].bool().to(device)
    loss = F.cross_entropy(outputs["logits"][mask], labels[mask])
    loss.backward()

    delta_grad = 0.0
    delta_params = 0
    for name, p in model.named_parameters():
        if name.startswith("pair_delta_") and p.grad is not None:
            delta_grad += float(p.grad.detach().float().norm().item())
            delta_params += 1
    if delta_grad <= 0:
        raise RuntimeError("no gradient reached pair_delta branch")

    print(
        f"SMOKE_OK device={device} loss={loss.item():.4f} "
        f"missing_delta_keys={len(missing)} max_logit_delta={diff:.3e} "
        f"delta_grad_norm_sum={delta_grad:.3e} delta_grad_params={delta_params}"
    )


if __name__ == "__main__":
    main()
