"""Original-model noise controls: no teacher, label input or inference change."""

import ast
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import yaml
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.protein_context_noise import ContextNoiseOptions, ProteinContextNoise, geometry_protection
from scripts.diagnostics.prepare_context_noise import make_configs, VARIANTS


def inputs():
    rng = torch.Generator().manual_seed(7)
    h = torch.randn(2, 40, 32, generator=rng)
    spatial = torch.randn(2, 40, 89, generator=rng)
    xyz = torch.randn(2, 40, 3, generator=rng) * 18
    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[1, 27:] = False
    return h, spatial, xyz, mask


@pytest.mark.parametrize("change", [{"sigma": -.1}, {"sigma": float("nan")},
    {"foreground_floor": 1.1}, {"batch_probability": -1}, {"mode": "gt"},
    {"num_candidates": 0}, {"membership_sigma": 0}])
def test_invalid_options(change):
    with pytest.raises(ValueError):
        ContextNoiseOptions(**change)


@pytest.mark.parametrize("change", [{"sigma": 0}, {"batch_probability": 0}])
def test_zero_is_exact_identity_and_rng_noop(change):
    data = inputs()
    state = torch.get_rng_state().clone()
    noise = ProteinContextNoise(ContextNoiseOptions(**change))
    assert noise(*data) is data[0]
    assert torch.equal(torch.get_rng_state(), state)
    assert noise.generators == {} and noise.calls == 0


def test_mask_geometry_only_rigid_invariance_padding_and_empty_rows():
    h, spatial, xyz, mask = inputs()
    o = ContextNoiseOptions()
    expected = geometry_protection(spatial, xyz, mask, o)
    rotated = xyz[..., [1, 2, 0]] + torch.tensor([15., 3., -9.])
    torch.testing.assert_close(expected, geometry_protection(spatial, rotated, mask, o))
    xyz[~mask] = float("nan"); spatial[~mask] = float("nan")
    torch.testing.assert_close(expected, geometry_protection(spatial, xyz, mask, o))
    empty = torch.zeros_like(mask)
    assert not geometry_protection(spatial, xyz, empty, o).count_nonzero()
    assert (expected[mask] >= 0).all() and (expected[mask] <= 1).all()
    assert not expected[~mask].count_nonzero()


def test_mask_protein_features_are_required_and_fail_closed():
    h, s, xyz, mask = inputs()
    with pytest.raises(ValueError):
        geometry_protection(s[..., :40], xyz, mask, ContextNoiseOptions())
    xyz[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        geometry_protection(s, xyz, mask, ContextNoiseOptions())


def test_noise_gradient_padding_rng_and_matched_control():
    h, s, xyz, mask = inputs()
    h.requires_grad_()
    state = torch.get_rng_state().clone()
    o = ContextNoiseOptions(batch_probability=1)
    bg = ProteinContextNoise(o)
    rand = ProteinContextNoise(replace(o, mode="random"))
    y, z = bg(h, s, xyz, mask), rand(h, s, xyz, mask)
    assert torch.equal(torch.get_rng_state(), state)
    assert not torch.equal(y[mask], h[mask]) and not torch.equal(y[mask], z[mask])
    torch.testing.assert_close(y[~mask], h[~mask], rtol=0, atol=0)
    for b in range(len(mask)):
        torch.testing.assert_close(bg.last_amplitudes[b, mask[b]].sort().values,
                                   rand.last_amplitudes[b, mask[b]].sort().values, rtol=0, atol=0)
    assert bg.last["expected_relative_noise_rms"] == pytest.approx(rand.last["expected_relative_noise_rms"])
    assert bg.last_amplitudes[mask].min() >= .5
    y[mask].sum().backward()
    torch.testing.assert_close(h.grad, mask[..., None].expand_as(h).float(), rtol=0, atol=0)
    assert torch.equal(bg._generator(h.device).get_state(), rand._generator(h.device).get_state())


def test_strength_linear_and_repeatable():
    data = inputs()
    low = ProteinContextNoise(ContextNoiseOptions(sigma=.05, batch_probability=1))
    high = ProteinContextNoise(ContextNoiseOptions(sigma=.20, batch_probability=1))
    a, b = low(*data), high(*data)
    torch.testing.assert_close(b - data[0], 4 * (a - data[0]), atol=5e-7, rtol=1e-4)
    clone = ProteinContextNoise(low.options)
    torch.testing.assert_close(a, clone(*data), rtol=0, atol=0)


def test_checkpoint_does_not_resample_and_gradients_match():
    h, s, xyz, mask = inputs()
    noise = ProteinContextNoise(ContextNoiseOptions(batch_probability=1))
    h.requires_grad_()
    perturbed = noise(h, s, xyz, mask)
    noise_state = noise._generator(h.device).get_state().clone()
    layer = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.GELU(), torch.nn.Linear(16, 3))
    direct = layer(perturbed)
    expected, = torch.autograd.grad(direct.square().sum(), h, retain_graph=True)
    actual = checkpoint(layer, perturbed, use_reentrant=False)
    actual.square().sum().backward()
    torch.testing.assert_close(h.grad, expected)
    assert noise.calls == noise.applied == 1
    assert torch.equal(noise_state, noise._generator(h.device).get_state())


def test_configs_change_only_noise_and_output_metadata():
    root = Path(__file__).parents[1]
    base = yaml.safe_load((root / "tmp/libre_architecture_review/plc_gfgf_config.yaml").read_text())
    configs = make_configs(base, "/tmp/noise")
    assert len(configs) == len(VARIANTS) == 6
    for cfg in configs.values():
        assert "self_distillation" not in cfg
        cfg.pop("context_noise")
        cfg["experiment"] = base["experiment"]
        assert cfg == base
    base["self_distillation"] = {"weight": 0}
    with pytest.raises(ValueError):
        make_configs(base, "/tmp/noise")


def test_train_loop_only_adds_logging_not_loss():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/train"))
    import train_context_noise as module
    source = (module.ROOT / "scripts/train/train_v11m3.py").read_text()
    source = module.trainer.shared._patch_v11m3_source_for_v11m4(source)
    original, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "train_epoch"]
    patched = module.logging_epoch_ast(source)

    class RemoveLog(ast.NodeTransformer):
        def visit_If(self, node):
            if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                    and ast.unparse(node.body[0]).startswith("log_context_noise(")):
                return None
            return self.generic_visit(node)

    restored = RemoveLog().visit(patched).body[0]
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)


@pytest.mark.parametrize("gradient_checkpointing,coattention", [(False, True), (True, False), (True, True)])
def test_original_model_eval_state_dict_and_train_hook(gradient_checkpointing, coattention):
    from test_libre_transfer_training import tiny_config
    from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
    cfg = tiny_config(use_gradient_checkpointing=gradient_checkpointing, gvp_k_neighbors=2,
                      use_unimol_ligand_input=False, use_v12_coattention_block=coattention)
    torch.manual_seed(44)
    model = PocketGVPFusionV11m13(cfg)
    kwargs = dict(esm_embeddings=torch.randn(1, 4, 16), protein_physchem=torch.zeros(1, 4, 62),
        protein_spatial_scalar=torch.randn(1, 4, 89), protein_spatial_vector=torch.zeros(1, 4, 8, 3),
        protein_gvp_vectors=torch.randn(1, 4, 4, 3), protein_coords=torch.randn(1, 4, 3),
        protein_mask=torch.ones(1, 4, dtype=torch.bool), chain_ids=torch.zeros(1, 4, dtype=torch.long),
        ligand_x=torch.randn(3, 109), ligand_pos=torch.randn(3, 3),
        ligand_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ligand_edge_attr=torch.zeros(4, 6), ligand_batch=torch.zeros(3, dtype=torch.long),
        ligand_mask=torch.ones(1, 3, dtype=torch.bool))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        expected = model(**kwargs)["logits"]
        model._protein_context_noise = ProteinContextNoise(ContextNoiseOptions(batch_probability=1))
        actual = model(**kwargs)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model._protein_context_noise.calls == 0
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    clone = PocketGVPFusionV11m13(cfg)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.train()
    out = model(**kwargs)
    out["logits"].square().mean().backward()
    assert model._protein_context_noise.calls == model._protein_context_noise.applied == 1
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
