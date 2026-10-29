"""Front-encoder isolation, chain boundaries, padding, AMP and checkpoint parity."""
import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from src.models.protein_sequence_context import ProteinSequenceContext, segment_indices
from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
from scripts.diagnostics.prepare_sequence_context_controls import make_configs, VARIANTS
from scripts.diagnostics.prepare_local_spatial_scan import make_configs as local_configs


torch.set_num_threads(2)
OPERATORS = ("pointwise", "tcn", "bigru")


def context_inputs():
    x = torch.randn(2, 7, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    chains = torch.tensor([[2, 2, 2, 4, 4, 4, 4], [2, 2, 2, 0, 0, 0, 0]])
    return x, dict(mask=mask, chain_ids=chains)


def test_segment_resets_and_geometry_gaps():
    _, kwargs = context_inputs()
    index, ids, local, lengths = segment_indices(**kwargs)
    assert index.tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert ids.tolist() == [0, 0, 0, 1, 1, 2, 3, 3, 3]
    assert local.tolist() == [0, 1, 2, 0, 1, 0, 0, 1, 2]
    assert lengths.tolist() == [3, 2, 1, 3]
    mask = torch.ones(1, 6, dtype=torch.bool)
    chains = torch.zeros_like(mask, dtype=torch.long)
    coords = torch.zeros(1, 6, 3)
    coords[0, :, 0] = torch.tensor([0, 3.8, 7.6, 30, 33.8, 37.6])
    assert segment_indices(mask, chains, coords)[-1].tolist() == [3, 3]
    positions = torch.tensor([[0, 1, 3, 4, 5, 6]])
    assert segment_indices(mask, chains, coords, positions)[-1].tolist() == [2, 1, 3]
    coords[0, 1] = float("nan")
    assert segment_indices(mask, chains, coords)[-1].tolist() == [1, 1, 1, 3]


@pytest.mark.parametrize("operator", OPERATORS)
def test_padding_isolation_and_roundtrip(operator):
    torch.manual_seed(44)
    model = ProteinSequenceContext(16, operator, dropout=0).eval()
    x, kwargs = context_inputs()
    x.requires_grad_()
    ref = model(x, **kwargs)
    changed = x.detach().clone()
    changed[0, 3:] += 100
    changed[~kwargs["mask"]] = float("nan")
    observed = model(changed, **kwargs)
    torch.testing.assert_close(ref[0, :3], observed[0, :3])
    torch.testing.assert_close(ref[1], observed[1])
    assert torch.isfinite(observed).all()
    assert observed[~kwargs["mask"]].eq(0).all()
    alone = model(x[:1, :3], chain_ids=kwargs["chain_ids"][:1, :3])
    torch.testing.assert_close(ref[:1, :3], alone, atol=2e-6, rtol=2e-6)
    weights = torch.randn_like(ref)
    (ref * weights).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[~kwargs["mask"]].eq(0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = ProteinSequenceContext(16, operator, dropout=0).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(ref, clone(x, **kwargs), atol=0, rtol=0)


@pytest.mark.parametrize("operator", OPERATORS)
def test_empty_mask_and_shape_validation(operator):
    net = ProteinSequenceContext(16, operator)
    x = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    out = net(x, mask=mask, chain_ids=torch.zeros_like(mask))
    assert out.eq(0).all()
    out.sum().backward()
    assert x.grad.eq(0).all()
    with pytest.raises(ValueError, match="chain_ids"):
        net(x)
    with pytest.raises(ValueError, match="coords"):
        net(x, chain_ids=torch.zeros_like(mask), coords=torch.zeros(2, 3))


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_checkpoint_gradient_equivalence(operator, dtype):
    torch.manual_seed(44)
    plain = ProteinSequenceContext(16, operator, dropout=.15).train()
    replay = copy.deepcopy(plain)
    x, kwargs = context_inputs()
    x = x.to(dtype).requires_grad_()
    y = x.detach().clone().requires_grad_()
    weights = torch.randn_like(x)
    torch.manual_seed(133)
    with torch.autocast("cpu", dtype=dtype):
        reference = plain(x, **kwargs)
        loss = (reference * weights).sum()
    loss.backward()
    torch.manual_seed(133)
    with torch.autocast("cpu", dtype=dtype):
        tested = checkpoint(lambda z: replay(z, **kwargs), y, use_reentrant=False)
        other_loss = (tested * weights).sum()
    other_loss.backward()
    torch.testing.assert_close(reference, tested, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
    for a, b in zip(plain.parameters(), replay.parameters()):
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)


def test_tcn_bidirectional_receptive_field_61():
    torch.manual_seed(7)
    net = ProteinSequenceContext(16, "tcn", dropout=0).eval()
    x = torch.randn(1, 81, 16, requires_grad=True)
    out = net(x, chain_ids=torch.zeros(1, 81, dtype=torch.long))
    out[0, 40, 0].backward()
    assert x.grad[:, :10].eq(0).all() and x.grad[:, 71:].eq(0).all()
    assert x.grad[0, 39].abs().sum() > 0 and x.grad[0, 41].abs().sum() > 0


@pytest.mark.parametrize("operator", OPERATORS)
def test_geometry_breaks_and_variable_length_buckets(operator):
    net = ProteinSequenceContext(16, operator, dropout=0).eval()
    x = torch.randn(1, 45, 16)
    chains = torch.zeros(1, 45, dtype=torch.long)
    coords = torch.zeros(1, 45, 3)
    coords[0, :, 0] = torch.arange(45) * 3.8
    coords[:, 4:, 1] += 100
    ref = net(x, chain_ids=chains, coords=coords)
    for start, end in ((0, 4), (4, 45)):
        standalone = net(x[:, start:end], chain_ids=chains[:, start:end], coords=coords[:, start:end])
        torch.testing.assert_close(ref[:, start:end], standalone, atol=2e-6, rtol=2e-6)


def tiny_config(**extra):
    cfg = dict(d_model=32, d_esm=16, d_gvp_internal=16, n_heads=4, n_gvp_layers=1,
        n_transformer_layers=4, protein_hybrid_pattern="GFGF", protein_gdn_backend="reference",
        n_cross_attention_layers=2, n_ligand_layers=1, n_ligand_self_attn_layers=1,
        local_delta_dim=8, local_delta_hidden=16, use_local_delta_only=True,
        use_global_delta_gate=False, use_center_aux_head=True, use_unimol_ligand_input=False,
        use_v12_coattention_block=True, use_gradient_checkpointing=True, gvp_k_neighbors=2,
        use_local54_interaction_kv=True, use_local54_residue_chem_compat=True)
    cfg.update(extra)
    return cfg


@pytest.mark.parametrize("operator", OPERATORS)
def test_only_context_changes_and_rng_is_preserved(operator):
    torch.manual_seed(44)
    ref = PocketGVPFusionV11m13(tiny_config())
    rng = torch.get_rng_state().clone()
    torch.manual_seed(44)
    new = PocketGVPFusionV11m13(tiny_config(protein_context_operator=operator))
    assert torch.equal(rng, torch.get_rng_state())
    ref_state = {k: v for k, v in ref.state_dict().items() if not k.startswith("protein_context_encoder.")}
    new_state = {k: v for k, v in new.state_dict().items() if not k.startswith("protein_context_encoder.")}
    assert ref_state.keys() == new_state.keys()
    for key, value in ref_state.items():
        torch.testing.assert_close(value, new_state[key], atol=0, rtol=0)
    assert all(p.requires_grad for p in new.parameters())


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_full_model_checkpoint_backward_and_restore(operator, amp_dtype):
    cfg = tiny_config(protein_context_operator=operator)
    model = PocketGVPFusionV11m13(cfg).train()
    assert model.use_local54_interaction_kv and model.use_local54_residue_chem_compat
    x = dict(esm_embeddings=torch.randn(1, 5, 16), protein_physchem=torch.randn(1, 5, 62),
        protein_spatial_scalar=torch.randn(1, 5, 89), protein_spatial_vector=torch.randn(1, 5, 8, 3),
        protein_gvp_vectors=torch.randn(1, 5, 4, 3), protein_coords=torch.randn(1, 5, 3),
        protein_mask=torch.ones(1, 5, dtype=torch.bool), chain_ids=torch.tensor([[0, 0, 0, 1, 1]]),
        ligand_x=torch.randn(3, 109), ligand_pos=torch.randn(3, 3),
        ligand_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ligand_edge_attr=torch.zeros(4, 6), ligand_batch=torch.zeros(3, dtype=torch.long),
        ligand_mask=torch.ones(1, 3, dtype=torch.bool))
    with torch.autocast("cpu", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        out = model(**x)
        loss = torch.nn.functional.cross_entropy(out["logits"].float().flatten(0, 1), torch.tensor([0, 1, 2, 0, 2]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    context_grads = [p.grad for p in model.protein_context_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in context_grads)
    assert sum(g.abs().sum() for g in context_grads) > 0
    model.eval()
    restored = PocketGVPFusionV11m13(cfg).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(**x)["logits"], restored(**x)["logits"], atol=0, rtol=0)


def test_control_configs_keep_g0_recipe(tmp_path):
    base = dict(experiment=dict(seed=44), data=dict(label_contact_thresh=5, max_seq_len=1800),
        model=dict(protein_hybrid_pattern="GFGF", **{k: .1 for k in (
            "d_esm", "dropout", "classifier_dropout", "coord_noise_std", "esm_dropout_prob",
            "physchem_dropout_prob", "gvp_dropout_prob", "ligand_source_dropout_p",
            "ligand_dropout_hidden_p", "ligand_dropout_local54_p", "ligand_edge_rbf_bins")}),
        training=dict(num_epochs=120, learning_rate=1e-4))
    reference = local_configs(base, tmp_path, tmp_path / "source")["G0_small_global"]
    configs = make_configs(base, tmp_path, tmp_path / "source")
    for name, cfg in configs.items():
        cfg = copy.deepcopy(cfg)
        assert cfg["model"].pop("protein_context_operator") == VARIANTS[name]
        assert cfg["model"].pop("context_variant") == name
        assert cfg["model"].pop("protein_context_max_ca_step") == 4.5
        cfg["experiment"] = reference["experiment"]
        assert cfg == reference


def test_context_budgets_are_close():
    sizes = [sum(p.numel() for p in ProteinSequenceContext(256, op).parameters()) for op in OPERATORS]
    assert max(sizes) / min(sizes) < 1.1
