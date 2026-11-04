"""Chain-boundary topology, position propagation and checkpoint replay controls."""
import copy

import pytest
import torch

from src.models.gvp_layers import build_protein_graph
from src.models.chain_encoding import get_cumulative_gap_position_ids
from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
from scripts.diagnostics.prepare_chain_boundary_controls import configs, FLAGS, VARIANTS

torch.set_num_threads(2)


def graph(xs, chains, fixed):
    ca = torch.tensor([[x, 0., 0.] for x in xs])
    return build_protein_graph(ca, ca + torch.tensor([0., 1., 0.]), ca + torch.tensor([1., 0., 0.]),
        k=1, radius=15, num_rbf=16, chain_ids=chains, respect_chain_boundaries=fixed)


def test_far_cross_chain_bridge_removed_but_same_chain_backbone_preserved():
    chains = torch.tensor([0, 0, 1, 1])
    old, _, _ = graph([0., 3.8, 100., 103.8], chains, False)
    new, features, _ = graph([0., 3.8, 100., 103.8], chains, True)
    assert (1, 2) in set(map(tuple, old.T.tolist()))
    assert set(map(tuple, new.T.tolist())) == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert features[:, 16].eq(1).all()
    far, _, _ = graph([0., 100.], torch.zeros(2, dtype=torch.long), True)
    assert far.shape[1] == 2


def test_real_spatial_cross_chain_edge_retained_without_sequence_flags():
    chains = torch.tensor([0, 0, 1, 1])
    edges, features, _ = graph([0., 3.8, 4.2, 8.], chains, True)
    cross = chains[edges[0]] != chains[edges[1]]
    assert cross.any()
    assert features[cross, 16:20].eq(0).all()
    assert features[cross, -1].eq(0).all()


def test_single_chain_graph_exact_parity_and_missing_ids_fail():
    chains = torch.ones(4, dtype=torch.long)
    for a, b in zip(graph([0., 3.8, 15., 30.], chains, False), graph([0., 3.8, 15., 30.], chains, True)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match="chain_ids"):
        graph([0., 3.8], None, True)


def tiny_config(**extra):
    cfg = dict(d_model=32, d_esm=16, d_gvp_internal=16, n_heads=4, n_gvp_layers=1,
        n_transformer_layers=4, protein_hybrid_pattern="GFGF", protein_gdn_backend="reference",
        n_cross_attention_layers=2, n_ligand_layers=1, n_ligand_self_attn_layers=1,
        local_delta_dim=8, local_delta_hidden=16, use_local_delta_only=True,
        use_global_delta_gate=False, use_center_aux_head=True, use_unimol_ligand_input=False,
        use_v12_coattention_block=True, v12_coattn_zero_init_residuals=False,
        use_gradient_checkpointing=False, use_chain_features=True, gvp_k_neighbors=2,
        use_local54_interaction_kv=True, use_local54_residue_chem_compat=True,
        dropout=0., attn_dropout=0., classifier_dropout=0., stochastic_depth_prob=0.,
        coord_noise_std=0., esm_dropout_prob=0., physchem_dropout_prob=0., gvp_dropout_prob=0.,
        ligand_source_dropout_p=0., ligand_dropout_hidden_p=0., ligand_dropout_local54_p=0.,
        local_delta_dropout=0.)
    cfg.update(extra)
    return cfg


def inputs(single=False):
    return dict(esm_embeddings=torch.randn(1, 5, 16), protein_physchem=torch.randn(1, 5, 62),
        protein_spatial_scalar=torch.randn(1, 5, 89), protein_spatial_vector=torch.randn(1, 5, 8, 3),
        protein_gvp_vectors=torch.randn(1, 5, 4, 3), protein_coords=torch.randn(1, 5, 3),
        protein_mask=torch.ones(1, 5, dtype=torch.bool),
        chain_ids=torch.tensor([[0, 0, 0, 0, 0] if single else [0, 0, 0, 1, 1]]),
        ligand_x=torch.randn(3, 109), ligand_pos=torch.randn(3, 3),
        ligand_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ligand_edge_attr=torch.zeros(4, 6), ligand_batch=torch.zeros(3, dtype=torch.long),
        ligand_mask=torch.ones(1, 3, dtype=torch.bool))


def test_no_new_parameters_or_rng_and_single_chain_logits_parity():
    torch.manual_seed(44)
    old = PocketGVPFusionV11m13(tiny_config()).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(44)
    new = PocketGVPFusionV11m13(tiny_config(**dict(zip(FLAGS, (True, True))))).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
    x = inputs(single=True)
    with torch.no_grad():
        torch.testing.assert_close(old(**x)["logits"], new(**x)["logits"], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("flags", VARIANTS.values())
@pytest.mark.parametrize("amp", [False, True])
def test_positions_and_full_model_checkpoint_gradients(flags, amp):
    torch.manual_seed(44)
    plain = PocketGVPFusionV11m13(tiny_config(**dict(zip(FLAGS, flags)))).train()
    replay = copy.deepcopy(plain)
    replay.config["use_gradient_checkpointing"] = True
    x = inputs()
    positions = []
    hooks = [b.block.prot_self_attn.register_forward_pre_hook(
        lambda m, a, k: positions.append(k.get("position_ids")), with_kwargs=True)
        for b in replay.cross_attention_layers]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        a = plain(**x)["logits"]
        la = a.float().square().sum()
    la.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        b = replay(**x)["logits"]
        lb = b.float().square().sum()
    lb.backward()
    for hook in hooks:
        hook.remove()
    assert positions
    expected = get_cumulative_gap_position_ids(x["chain_ids"])
    for position in positions:
        if flags[1]:
            torch.testing.assert_close(position, expected, atol=0, rtol=0)
        else:
            assert position is None
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    for p, q in zip(plain.parameters(), replay.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            assert torch.isfinite(q.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5)


def test_config_changes_only_flags_and_output_names(tmp_path):
    base = dict(experiment=dict(seed=44), training=dict(early_stopping_metric="f1_c0"),
        data=dict(label_contact_thresh=5), model=dict(scan_variant="G0_small_global", d_model=256,
        use_chain_features=True, use_v12_coattention_block=True, use_compile=False))
    for name, cfg in configs(base, tmp_path).items():
        assert tuple(cfg["model"].pop(k) for k in FLAGS) == VARIANTS[name]
        cfg["experiment"] = base["experiment"]
        assert cfg == base


def test_chain_reversal_restores_every_residue_input_without_touching_ligand():
    from scripts.diagnostics.eval_chain_boundary_controls import reverse_chains, RESIDUE_INPUTS
    x = inputs()
    other, inverse = reverse_chains(x)
    assert other["chain_ids"].tolist() == [[1, 1, 0, 0, 0]]
    for key in RESIDUE_INPUTS:
        torch.testing.assert_close(other[key][:, inverse], x[key], atol=0, rtol=0)
    for key in x.keys() - set(RESIDUE_INPUTS):
        assert other[key] is x[key]
