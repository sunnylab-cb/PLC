import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.protein_bidirectional_gdn import BidirectionalProteinGDN, chain_pack_indices
from src.models.pocket_gvp_fusion_v11 import RoPEProteinContextEncoder


def test_pack_resets_at_chains_padding_holes_and_batch_boundary():
    mask = torch.tensor([[1, 1, 1, 0, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    chains = torch.tensor([[7, 7, 8, 8, 8], [8, 8, 0, 0, 0]])
    index, cu, reverse = chain_pack_indices(mask, chains)
    assert index.tolist() == [0, 1, 2, 4, 5, 6]
    assert cu.tolist() == [0, 2, 3, 4, 6]
    assert reverse.tolist() == [1, 0, 2, 3, 5, 4]


def test_chain_isolation_padding_reverse_and_backward():
    torch.manual_seed(44)
    module = BidirectionalProteinGDN(16, 2, backend='reference').eval()
    x = torch.randn(1, 9, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    chains = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0]])
    out = module(x, mask=mask, chain_ids=chains)
    changed = x.detach().clone()
    changed[:, 3:6] += 100
    changed[:, 6:] = float('nan')
    torch.testing.assert_close(out[:, :3], module(changed, mask=mask, chain_ids=chains)[:, :3])
    assert torch.count_nonzero(out[:, 6:]) == 0
    alone = module(x[:, :3], chain_ids=chains[:, :3])
    torch.testing.assert_close(out[:, :3], alone)
    reverse = module(x[:, :3].flip(1), chain_ids=chains[:, :3]).flip(1)
    torch.testing.assert_close(alone, reverse)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 6:]) == 0
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_requires_explicit_chains_and_no_dense_bias():
    module = BidirectionalProteinGDN(16, 2, backend='reference')
    x = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match='chain_ids'):
        module(x)
    with pytest.raises(ValueError, match='distance bias'):
        module(x, attn_bias=torch.zeros(1), chain_ids=torch.zeros(1, 3))


def test_empty_mask():
    module = BidirectionalProteinGDN(16, 2, backend='reference')
    x = torch.randn(1, 3, 16, requires_grad=True)
    out = module(x, mask=torch.zeros(1, 3, dtype=torch.bool), chain_ids=torch.zeros(1, 3))
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert torch.count_nonzero(x.grad) == 0


def test_default_state_dict_and_unchanged_initialization():
    kwargs = dict(d_model=32, n_layers=4, n_heads=4, dropout=0., use_dist_bias=True)
    torch.manual_seed(44)
    original = RoPEProteinContextEncoder(**kwargs)
    after_original = torch.randn(10)
    torch.manual_seed(44)
    explicit = RoPEProteinContextEncoder(**kwargs, hybrid_pattern='FFFF')
    assert original.state_dict().keys() == explicit.state_dict().keys()
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, explicit.state_dict()[k], rtol=0, atol=0)
    torch.manual_seed(44)
    hybrid = RoPEProteinContextEncoder(**kwargs, hybrid_pattern='GFGF', gdn_backend='reference')
    torch.testing.assert_close(after_original, torch.randn(10), rtol=0, atol=0)
    for k, v in original.state_dict().items():
        if k in hybrid.state_dict():
            torch.testing.assert_close(v, hybrid.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 6, 32, requires_grad=True)
    result = hybrid(x, mask=torch.ones(2, 6, dtype=torch.bool),
                    coords=torch.randn(2, 6, 3), chain_ids=torch.zeros(2, 6, dtype=torch.long))
    assert result.shape == x.shape
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_pattern():
    with pytest.raises(ValueError, match='one F/G'):
        RoPEProteinContextEncoder(32, n_heads=4, hybrid_pattern='GFF')
