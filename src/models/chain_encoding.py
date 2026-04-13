"""
Chain-aware positional encoding utilities for multi-chain protein complexes.

Provides get_cumulative_gap_position_ids for RoPE chain-gap encoding:
chains A/B/C get position offsets 0, gap_size, 2*gap_size, ...
so RoPE naturally attenuates cross-chain attention.
"""

import torch


def get_cumulative_gap_position_ids(
    chain_ids: torch.Tensor,
    gap_size: int = 1000,
) -> torch.Tensor:
    """
    Generate positional indices with guaranteed gaps between chains.

    Args:
        chain_ids: [B, L] integer chain indices (0, 1, 2, ...)
        gap_size:  position jump inserted between consecutive chains

    Returns:
        position_ids: [B, L]  e.g. chain A->[0,1,2], chain B->[1003,1004,...]
    """
    device = chain_ids.device
    B, L = chain_ids.shape

    chain_change = torch.zeros_like(chain_ids)
    chain_change[:, 1:] = (chain_ids[:, 1:] != chain_ids[:, :-1]).long()

    num_jumps = torch.cumsum(chain_change, dim=1)
    offsets = num_jumps * gap_size

    standard_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    return standard_idx + offsets
