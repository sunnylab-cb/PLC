"""Co-attention must replay both outputs without mutable ligand caches."""
import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from src.models.pocket_gvp_fusion_v11m8x import _V12CoAttentionAsV11CrossBlock


def block(dropout):
    return _V12CoAttentionAsV11CrossBlock(
        d_model=32, n_heads=4, dropout=dropout, drop_path_prob=dropout,
        attn_dropout=dropout, cross_attn_logit_clip=30., self_attn_logit_clip=30.,
        d_unimol=16, use_film=False, use_prot_dist_bias=False, use_lig_dist_bias=False,
        inner_residual_scale=.05, cross_residual_scale=.25, use_rezero_branch_gates=True,
        rezero_gate_init=.05, p2l_length_scale=False, p2l_length_ref=256.,
        p2l_rezero_gate_max=.2, use_flash_attn=False, flash_attn_fallback=True,
        zero_init_residuals=False)


@pytest.mark.parametrize("dropout", [0., .15])
def test_joint_checkpoint_matches_legacy_outputs_and_gradients(dropout):
    torch.manual_seed(44)
    legacy = torch.nn.ModuleList([block(dropout), block(dropout)]).train()
    replay = copy.deepcopy(legacy)
    p = torch.randn(2, 5, 32, requires_grad=True)
    l = torch.randn(2, 3, 32, requires_grad=True)
    p2, l2 = p.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
    pm = torch.tensor([[True]*5, [True]*4 + [False]])
    lm = torch.tensor([[True]*3, [True]*2 + [False]])
    torch.manual_seed(100)
    hp, hl = p, l
    for layer in legacy:
        hp = layer.prot_to_lig(hp, hl, pm, lm)
        hl = layer.lig_to_prot(hl, hp, lm, pm)
    loss = hp.square().sum() + hl.square().sum()
    loss.backward()
    torch.manual_seed(100)
    rp, rl = p2, l2
    for layer in replay:
        rp, rl = checkpoint(layer.forward_pair, rp, rl, pm, lm, use_reentrant=False)
    (rp.square().sum() + rl.square().sum()).backward()
    torch.testing.assert_close(rp, hp, rtol=0, atol=0)
    torch.testing.assert_close(rl, hl, rtol=0, atol=0)
    torch.testing.assert_close(p2.grad, p.grad)
    torch.testing.assert_close(l2.grad, l.grad)
    for (name, a), (other, b) in zip(legacy.named_parameters(), replay.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(b.grad, a.grad)
    assert all(layer._pending_lig is None for layer in replay)
    replay.load_state_dict(legacy.state_dict(), strict=True)
