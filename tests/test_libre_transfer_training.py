"""Default compatibility, matched readouts, mask safety and class-mass semantics."""

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.conditional_local_readout import ConditionalLocalReadout, widen_local_head
from src.models.pocket_gvp_fusion_v11m10 import PocketGVPFusionV11m10
from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
from src.losses.residue_classmass import residue_classmass_loss
from scripts.diagnostics.prepare_libre_transfer import make_configs, VARIANTS


def tiny_config(**extra):
    return dict(d_model=32, d_esm=16, d_gvp_internal=16, n_heads=4, n_transformer_layers=1,
                n_cross_attention_layers=1, n_ligand_layers=1, n_ligand_self_attn_layers=1,
                local_delta_dim=8, local_delta_hidden=16, use_local_delta_only=True,
                use_global_delta_gate=False, use_center_aux_head=True, **extra)


@pytest.mark.parametrize("mode", ["product", "additive"])
def test_readout_arithmetic_and_nan_mask(mode):
    block = ConditionalLocalReadout(8, 4, mode)
    p, l = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    valid = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    u, v = block.protein(p), block.ligand(l)
    expected = u * v if mode == "product" else (u + v) / math.sqrt(2)
    p[~valid], l[~valid] = float("nan"), float("nan")
    p.requires_grad_(); l.requires_grad_()
    z = block(p, l, valid)
    torch.testing.assert_close(z[valid], expected[valid])
    assert not z[~valid].count_nonzero()
    z.square().sum().backward()
    assert torch.isfinite(p.grad).all() and not p.grad[~valid].count_nonzero()
    assert all(torch.isfinite(w.grad).all() for w in block.parameters())


def test_exact_parameter_budget():
    head = torch.nn.Sequential(torch.nn.LayerNorm(1028), torch.nn.Linear(1028, 256),
                               torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(256, 3))
    before = sum(p.numel() for p in head.parameters())
    widen_local_head(head, 64)
    extra = sum(p.numel() for p in head.parameters()) - before
    a = ConditionalLocalReadout(512, 64, "product")
    b = ConditionalLocalReadout(512, 64, "additive")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert extra + sum(p.numel() for p in a.parameters()) == 82304


def test_defaults_and_shared_initialization():
    torch.manual_seed(44)
    old = PocketGVPFusionV11m10(tiny_config())
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(44)
    base = PocketGVPFusionV11m13(tiny_config())
    assert torch.equal(old_rng, torch.get_rng_state())
    base.load_state_dict(old.state_dict(), strict=True)
    models = []
    for mode in ("product", "additive"):
        torch.manual_seed(44)
        m = PocketGVPFusionV11m13(tiny_config(conditional_readout_mode=mode, conditional_readout_dim=4))
        assert torch.equal(old_rng, torch.get_rng_state())
        for key, val in base.state_dict().items():
            current = m.state_dict()[key]
            if key in ("local_delta_head.0.weight", "local_delta_head.0.bias"):
                current = current[:len(val)]
            if key == "local_delta_head.1.weight":
                current = current[:, :val.shape[1]]
            torch.testing.assert_close(val, current, rtol=0, atol=0)
        assert m.center_aux_head[0].normalized_shape == base.center_aux_head[0].normalized_shape
        models.append(m)
    for k, v in models[0].state_dict().items():
        torch.testing.assert_close(v, models[1].state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 5, 68)
    mask = torch.ones(2, 5, dtype=torch.bool)
    lig = torch.ones(2, 4, dtype=torch.bool)
    old.eval(); base.eval()
    torch.testing.assert_close(old._classify_local_delta(x, mask, lig), base._classify_local_delta(x, mask, lig))


def test_classmass_closed_form_and_padding_gradient():
    h = torch.ones(1, 6, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 2, 2, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    loss, stats = residue_classmass_loss(h, labels, mask)
    torch.testing.assert_close(loss, torch.tensor(math.log(3.0)))
    assert stats == dict(sampled_contact=2, sampled_far=2, eligible_anchors=4, eligible_rows=1)
    loss.backward()
    assert not h.grad[:, 4:].count_nonzero()
    assert torch.isfinite(h.grad).all()


@pytest.mark.parametrize("labels", [[0], [0, 2], [1, 1], [0, 0, 0]])
def test_classmass_degenerate(labels):
    h = torch.randn(1, len(labels), 8, requires_grad=True)
    loss, _ = residue_classmass_loss(h, torch.tensor([labels]), torch.ones(1, len(labels), dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(h.grad).all()


def test_classmass_sampling_cap_and_private_rng():
    h = torch.randn(2, 100, 8, requires_grad=True)
    labels = torch.tensor([[0] * 50 + [2] * 50] * 2)
    mask = torch.ones(2, 100, dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    a, stats = residue_classmass_loss(h, labels, mask, max_per_class=3)
    b, _ = residue_classmass_loss(h, labels, mask, max_per_class=3)
    assert torch.equal(rng, torch.get_rng_state())
    assert stats["sampled_contact"] == stats["sampled_far"] == 6
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    assert torch.isfinite(h.grad).all() and h.grad.norm() > 0


def test_classmass_row_class_balancing():
    h = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 2], [0, 2, 2, 1]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    actual, _ = residue_classmass_loss(h, labels, mask)
    flat = torch.nn.functional.normalize(h[labels != 1], dim=-1)
    c = labels[labels != 1]
    sims = flat @ flat.T / .2
    individual = []
    for i in range(7):
        nonself = torch.arange(7) != i
        pos = nonself & (c == c[i])
        individual.append(sims[i, nonself].logsumexp(0) - sims[i, pos].logsumexp(0))
    expected = ((torch.stack(individual[:3]).mean() + individual[3]) / 2
                + (individual[4] + torch.stack(individual[5:]).mean()) / 2) / 2
    torch.testing.assert_close(actual, expected)


def test_auxiliary_gradient_without_changing_dropout_or_eval():
    torch.manual_seed(44)
    base = PocketGVPFusionV11m13(tiny_config())
    torch.manual_seed(44)
    aux = PocketGVPFusionV11m13(tiny_config(residue_classmass_weight=.02))
    aux.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(2, 6, 68, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    lig = torch.ones(2, 3, dtype=torch.bool)
    torch.manual_seed(10)
    expected = base._classify_local_delta(x, mask, lig)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(10)
    actual = aux._classify_local_delta(x, mask, lig)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    loss, stats = aux.compute_residue_classmass_loss(
        {"residue_classmass_hidden": aux._cached_classmass_hidden},
        torch.tensor([[0, 0, 2, 2, 1, 1]] * 2), mask)
    loss.backward()
    assert stats["eligible_anchors"] == 8 and x.grad.norm() > 0
    aux.eval()
    assert aux.compute_residue_classmass_loss({}, None, None) == (None, {})


def test_configs_only_change_planned_keys():
    path = Path(__file__).parents[1] / "tmp/libre_architecture_review/plc_gfgf_config.yaml"
    if not path.exists():
        pytest.skip("Frozen config available on deployment host")
    base = yaml.safe_load(path.read_text())
    configs = make_configs(base, "/tmp/probes")
    for name, cfg in configs.items():
        cfg["experiment"] = base["experiment"]
        for dotted in VARIANTS[name]:
            parts, target, original = dotted.split("."), cfg, base
            for p in parts[:-1]:
                target, original = target[p], original[p]
            key = parts[-1]
            if key in original:
                target[key] = original[key]
            else:
                del target[key]
        assert cfg == base
    base["training"]["init_from_checkpoint"] = "not-allowed.pt"
    with pytest.raises(ValueError, match="scratch"):
        make_configs(base, "/tmp/probes")


@pytest.mark.parametrize("mode,weight", [("none", .02), ("product", 0.), ("additive", 0.)])
def test_full_model_train_eval_and_state_roundtrip(mode, weight):
    cfg = tiny_config(conditional_readout_mode=mode, conditional_readout_dim=4,
                      residue_classmass_weight=weight, gvp_k_neighbors=2,
                      use_unimol_ligand_input=False, use_v12_coattention_block=True)
    model = PocketGVPFusionV11m13(cfg)
    inputs = dict(esm_embeddings=torch.randn(1, 4, 16),
                  protein_physchem=torch.zeros(1, 4, 62),
                  protein_spatial_scalar=torch.zeros(1, 4, 89),
                  protein_spatial_vector=torch.zeros(1, 4, 8, 3),
                  protein_gvp_vectors=torch.randn(1, 4, 4, 3),
                  protein_coords=torch.randn(1, 4, 3),
                  protein_mask=torch.ones(1, 4, dtype=torch.bool),
                  chain_ids=torch.zeros(1, 4, dtype=torch.long),
                  ligand_x=torch.randn(3, 109), ligand_pos=torch.randn(3, 3),
                  ligand_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                  ligand_edge_attr=torch.zeros(4, 6), ligand_batch=torch.zeros(3, dtype=torch.long),
                  ligand_mask=torch.ones(1, 3, dtype=torch.bool))
    outputs = model(**inputs)
    assert outputs["logits"].shape == (1, 4, 3)
    assert ("residue_classmass_hidden" in outputs) == (weight > 0)
    assert model._cached_classmass_hidden is None
    labels = torch.tensor([[0, 0, 2, 2]])
    loss = torch.nn.functional.cross_entropy(outputs["logits"].flatten(0, 1), labels.flatten())
    aux, _ = model.compute_residue_classmass_loss(outputs, labels, inputs["protein_mask"])
    (loss + aux if aux is not None else loss).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        original = model(**inputs)
        clone = PocketGVPFusionV11m13(cfg).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        restored = clone(**inputs)
    assert "residue_classmass_hidden" not in original
    torch.testing.assert_close(original["logits"], restored["logits"], rtol=0, atol=0)


def test_trainer_has_separate_training_only_classmass_hook():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
    import train_v11m4
    source = (Path(__file__).parents[1] / "scripts/train/train_v11m3.py").read_text()
    patched = train_v11m4._patch_v11m3_source_for_v11m4(source)
    compile(patched, "patched_trainer", "exec")
    train_source = patched.split("def train_epoch(", 1)[1].split("def evaluate(", 1)[0]
    assert "compute_residue_classmass_loss" in train_source
    assert "ResidueContrast/" in train_source
    assert "conditional_readout_norm" not in train_source  # Module-name-based gradient logging.
    assert "compute_residue_classmass_loss" not in patched.split("def evaluate(", 1)[1]


def test_submission_guard_rejects_failed_stale_or_partial_smoke():
    from copy import deepcopy
    from scripts.diagnostics.submit_libre_transfer import validate_smoke
    valid = dict(passed=True, provenance_sha256="expected",
                 runs=[dict(name=n, extra_parameters=82304 if "readout" in n else 0) for n in VARIANTS])
    validate_smoke(valid, "expected", VARIANTS)
    with pytest.raises(RuntimeError, match="different"):
        validate_smoke(valid, "changed", VARIANTS)
    broken = deepcopy(valid)
    broken["passed"] = False
    with pytest.raises(RuntimeError, match="failed"):
        validate_smoke(broken, "expected", VARIANTS)
    broken = deepcopy(valid)
    broken["runs"].pop()
    with pytest.raises(RuntimeError, match="four"):
        validate_smoke(broken, "expected", VARIANTS)
    broken = deepcopy(valid)
    broken["runs"][0]["extra_parameters"] += 1
    with pytest.raises(RuntimeError, match="budget"):
        validate_smoke(broken, "expected", VARIANTS)
