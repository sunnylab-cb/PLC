"""Matched product/additive channels for the existing late local-delta head."""

import math

import torch
from torch import nn


class ConditionalLocalReadout(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mode: str):
        super().__init__()
        if mode not in ("product", "additive") or output_dim <= 0:
            raise ValueError("Expected product/additive readout with positive width")
        self.mode = mode
        self.protein = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False), nn.LayerNorm(output_dim))
        self.ligand = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False), nn.LayerNorm(output_dim))

    def forward(self, protein, ligand_read, valid):
        # Mask before projections: NaN padding must not contaminate weight gradients.
        valid = valid.bool().unsqueeze(-1)
        u = self.protein(protein.masked_fill(~valid, 0))
        v = self.ligand(ligand_read.masked_fill(~valid, 0))
        z = u * v if self.mode == "product" else (u + v) / math.sqrt(2.0)
        return z.masked_fill(~valid, 0)


def widen_local_head(head: nn.Sequential, extra_dim: int) -> None:
    """Preserve all existing weights; initialize only the extra input columns."""
    norm, linear = head[0], head[1]
    old_dim = linear.in_features
    new_norm = nn.LayerNorm(old_dim + extra_dim, eps=norm.eps)
    new_linear = nn.Linear(old_dim + extra_dim, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        new_norm.weight[:old_dim].copy_(norm.weight)
        new_norm.bias[:old_dim].copy_(norm.bias)
        new_linear.weight[:, :old_dim].copy_(linear.weight)
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    head[0], head[1] = new_norm, new_linear
