import copy

import pytest
import torch

from src.models.pocket_local_spatial_scan import PocketLocalSpatialScan, local_neighbors
from scripts.diagnostics.prepare_local_spatial_scan import make_configs, VARIANTS


torch.set_num_threads(2)


def inputs(n=7, m=4):
    torch.manual_seed(10)
    mask = torch.ones(2, n, dtype=torch.bool)
    mask[1, -1] = False
    edges = torch.tensor([[0, 1, 1, 2, m, m + 1], [1, 0, 2, 1, m + 1, m]])
    return dict(esm_embeddings=torch.randn(2, n, 1280), protein_physchem=torch.randn(2, n, 62),
        protein_spatial_scalar=torch.randn(2, n, 89), protein_spatial_vector=torch.randn(2, n, 8, 3),
        protein_gvp_vectors=torch.randn(2, n, 4, 3), protein_coords=torch.randn(2, n, 3) * 3,
        protein_mask=mask, chain_ids=torch.tensor([[0] * n, [1] * n]),
        ligand_x=torch.randn(2 * m, 109), ligand_pos=torch.randn(2 * m, 3),
        ligand_edge_index=edges, ligand_edge_attr=torch.randn(edges.shape[1], 6),
        ligand_batch=torch.arange(2).repeat_interleave(m), ligand_mask=torch.ones(2, m, dtype=torch.bool))


def model(variant):
    return PocketLocalSpatialScan(dict(scan_variant=variant, scan_k=5, scan_chunk=3, dropout=0.0,
        classifier_dropout=0.0, coord_noise_std=0.0, esm_dropout_prob=0.0, physchem_dropout_prob=0.0,
        gvp_dropout_prob=0.0, ligand_source_dropout_p=0.0, ligand_dropout_hidden_p=0.0,
        ligand_dropout_local54_p=0.0, use_gradient_checkpointing=True))


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_forward_backward_checkpoint_roundtrip(variant):
    net = model(variant).train()
    x = inputs()
    out = net(**x)
    assert out["logits"].shape == (2, 7, 3)
    loss = out["logits"].square().mean() + out["pred_dist"].mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert net.esm_proj[0].weight.grad.abs().sum() > 0
    assert net.atom_k.weight.grad.abs().sum() > 0
    if variant != VARIANTS[1]:
        assert net.spatial_blocks[0].conditioner.net[-1].weight.grad.abs().sum() > 0
    net.eval()
    clone = model(variant).eval()
    clone.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net(**x)["logits"], clone(**x)["logits"])
    net.checkpoint = False
    net.train()
    plain = net(**x)["logits"]
    net.checkpoint = True
    torch.testing.assert_close(plain, net(**x)["logits"])


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_rigid_motion_atom_permutation_and_padding(variant):
    net = model(variant).eval()
    x = inputs()
    with torch.no_grad():
        ref = net(**x)["logits"]
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        y = copy.deepcopy(x)
        for key in ("protein_coords", "protein_gvp_vectors", "protein_spatial_vector"):
            y[key] = y[key] @ rotation
        y["protein_coords"] += 17
        y["ligand_pos"] = y["ligand_pos"] @ rotation + 103
        torch.testing.assert_close(ref, net(**y)["logits"], atol=3e-5, rtol=3e-5)
        perm = torch.tensor([2, 0, 3, 1, 7, 6, 4, 5])
        inverse = perm.argsort()
        y = copy.deepcopy(x)
        for key in ("ligand_x", "ligand_pos", "ligand_batch"):
            y[key] = y[key][perm]
        y["ligand_edge_index"] = inverse[y["ligand_edge_index"]]
        torch.testing.assert_close(ref, net(**y)["logits"], atol=2e-5, rtol=2e-5)
        y = copy.deepcopy(x)
        for key in ("esm_embeddings", "protein_physchem", "protein_spatial_scalar", "protein_coords",
                    "protein_gvp_vectors", "protein_spatial_vector"):
            y[key][1, -1] = 1000
        y["chain_ids"] += 25
        torch.testing.assert_close(ref, net(**y)["logits"], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_empty_ligand_missing_coords_and_far_component(variant):
    net = model(variant).eval()
    x = inputs()
    x["protein_coords"][:, 4:] += 1000
    with torch.no_grad():
        ref = net(**x)["logits"]
        y = copy.deepcopy(x)
        y["esm_embeddings"][:, 4:] *= -10
        torch.testing.assert_close(ref[:, :4], net(**y)["logits"][:, :4])
        y["ligand_mask"].zero_()
        y["protein_coords"][0, 0] = float("nan")
        out = net(**y)
        assert torch.isfinite(out["logits"]).all()
        y["ligand_x"] = y["ligand_x"][:0]
        y["ligand_pos"] = y["ligand_pos"][:0]
        y["ligand_batch"] = y["ligand_batch"][:0]
        y["ligand_edge_index"] = y["ligand_edge_index"][:, :0]
        y["ligand_edge_attr"] = y["ligand_edge_attr"][:0]
        y["ligand_mask"] = y["ligand_mask"][:, :0]
        assert torch.isfinite(net(**y)["logits"]).all()


def test_graph_chain_mask_cap_and_singleton():
    xyz = torch.tensor([[0., 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0]])
    ids = torch.zeros(4, dtype=torch.long)
    chains = torch.tensor([0, 1, 0, 0])
    idx, mask, es, _, _, stats = local_neighbors(xyz, ids, chains, torch.arange(4), k=2)
    assert idx[0, 1] == 1 and mask[0, 1]
    assert es[0, 1, 16:21].eq(0).all()
    assert mask[3].sum() == 1 and stats["scan_cap_hit_fraction"] > 0
    single = local_neighbors(xyz[:1], ids[:1], chains[:1], torch.arange(1), k=32)
    assert single[1].sum() == 1


def test_config_controls_preserve_dataset(tmp_path):
    base = dict(experiment=dict(seed=44), data=dict(label_contact_thresh=5, max_seq_len=1800),
        model=dict(protein_hybrid_pattern="GFGF", **{k: 0.0 for k in (
            "d_esm", "dropout", "classifier_dropout", "coord_noise_std", "esm_dropout_prob", "physchem_dropout_prob",
            "gvp_dropout_prob", "ligand_source_dropout_p", "ligand_dropout_hidden_p", "ligand_dropout_local54_p", "ligand_edge_rbf_bins")}),
        training=dict(num_epochs=120, learning_rate=1e-4))
    configs = make_configs(base, tmp_path, tmp_path / "source")
    for cfg in configs.values():
        assert cfg["data"] == base["data"]
        assert cfg["training"]["init_from_checkpoint"] is None
        assert cfg["training"]["early_stopping_metric"] == "f1_c0"
        assert cfg["training"]["save_every"] == 0
    assert configs[VARIANTS[0]]["model"]["d_model"] == 256
    assert "protein_hybrid_pattern" not in configs[VARIANTS[1]]["model"]
