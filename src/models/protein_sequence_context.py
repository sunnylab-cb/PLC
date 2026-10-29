"""Small chain-segmented alternatives to the front protein context stack.

Only this encoder is replaced; downstream co-attention and ligand delta remain.
CA jumps are a conservative break heuristic, not complete sequence alignment.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def segment_indices(mask, chain_ids, coords=None, position_ids=None, max_ca_step=4.5):
    """Pack contiguous valid runs, resetting at chains, padding and known gaps."""
    if mask.ndim != 2 or chain_ids is None or chain_ids.shape != mask.shape:
        raise ValueError("Explicit chain_ids and mask must have shape [B,L]")
    if coords is not None and coords.shape != (*mask.shape, 3):
        raise ValueError("coords must have shape [B,L,3]")
    if position_ids is not None and position_ids.shape != mask.shape:
        raise ValueError("position_ids must have shape [B,L]")
    if max_ca_step <= 0:
        raise ValueError("max_ca_step must be positive")
    index = mask.bool().flatten().nonzero().flatten()
    if not index.numel():
        return index, index, index, index
    length = mask.shape[1]
    chains = chain_ids.flatten()[index]
    starts = torch.ones_like(index, dtype=torch.bool)
    starts[1:] = ((index[1:] != index[:-1] + 1)
                  | (index[1:] // length != index[:-1] // length)
                  | (chains[1:] != chains[:-1]))
    if position_ids is not None:
        positions = position_ids.flatten()[index]
        starts[1:] |= positions[1:] - positions[:-1] != 1
    if coords is not None:
        xyz = coords.reshape(-1, 3)[index].float()
        finite = torch.isfinite(xyz).all(-1)
        distances = (xyz[1:] - xyz[:-1]).norm(dim=-1)
        starts[1:] |= (~finite[1:] | ~finite[:-1] | (distances > max_ca_step))
    offsets = starts.nonzero().flatten()
    lengths = torch.diff(torch.cat((offsets, offsets.new_tensor([len(index)]))))
    segments = starts.long().cumsum(0) - 1
    local = torch.arange(len(index), device=index.device) - offsets[segments]
    return index, segments, local, lengths


class FeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, x):
        return self.net(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, width, dropout, dilation=None, kernel_size=5):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = (nn.Conv1d(width, width, kernel_size, groups=width,
                              dilation=dilation, padding=dilation * (kernel_size // 2))
                     if dilation is not None else None)
        self.ffn = FeedForward(width, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.norm(x).masked_fill(~mask[..., None], 0)
        if self.conv is not None:
            y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        return (x + self.drop(self.ffn(y))).masked_fill(~mask[..., None], 0)


class RecurrentBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gru = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, cpu_lengths):
        # Keep recurrent state math FP32 in both BF16 TRAIN and FP16 VAL.
        with torch.autocast(device_type=x.device.type, enabled=False):
            packed = pack_padded_sequence(self.norm(x.float()), cpu_lengths,
                                          batch_first=True, enforce_sorted=False)
            packed, _ = self.gru(packed)
            y, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        x = (x + self.drop(y.to(x.dtype))).masked_fill(~mask[..., None], 0)
        return (x + self.drop(self.ffn(self.ffn_norm(x)))).masked_fill(~mask[..., None], 0)


class ProteinSequenceContext(nn.Module):
    def __init__(self, d_model, operator, dropout=0.15, max_ca_step=4.5):
        super().__init__()
        if operator not in ("pointwise", "tcn", "bigru"):
            raise ValueError(f"Unknown protein context operator: {operator}")
        if d_model < 2 or d_model % 2 or max_ca_step <= 0:
            raise ValueError("Sequence context needs positive even width and CA cutoff")
        self.operator = operator
        self.max_ca_step = float(max_ca_step)
        if operator == "bigru":
            self.blocks = nn.ModuleList(RecurrentBlock(d_model, dropout) for _ in range(2))
        else:
            self.blocks = nn.ModuleList(ConvolutionBlock(
                d_model, dropout, dilation=d if operator == "tcn" else None)
                for d in (1, 2, 4, 8))
        self.norm = nn.LayerNorm(d_model)
        self._last_context_stats = {}

    def forward(self, x, mask=None, coords=None, position_ids=None, chain_ids=None):
        if x.ndim != 3:
            raise ValueError("Sequence context expects [B,L,D]")
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        if mask.shape != x.shape[:2]:
            raise ValueError("Protein mask shape mismatch")
        mask = mask.bool()
        index, segments, local, lengths = segment_indices(
            mask, chain_ids, coords, position_ids, self.max_ca_step)
        if not index.numel():
            return x.masked_fill(torch.ones_like(x, dtype=torch.bool), 0)
        self._last_context_stats = {
            "context_segments": lengths.new_tensor(len(lengths)).detach(),
            "context_max_segment_length": lengths.max().detach(),
        }
        if self.operator == "pointwise":
            result = x.masked_fill(~mask[..., None], 0)
            for block in self.blocks:
                result = block(result, mask)
            return self.norm(result).masked_fill(~mask[..., None], 0)
        flat = x.new_zeros(x.shape[0] * x.shape[1], x.shape[2])
        # Length buckets avoid padding every tiny fragment to the longest chain.
        buckets = (lengths.float().log2().ceil().long()).clamp_min(4)
        for bucket in buckets.unique().tolist():
            selected = buckets == bucket
            ids = selected.nonzero().flatten()
            these_lengths = lengths[ids]
            max_length = int(these_lengths.max())
            packed_mask = torch.arange(max_length, device=x.device)[None] < these_lengths[:, None]
            rows = selected[segments].nonzero().flatten()
            packed_ids = selected.long().cumsum(0)[segments[rows]] - 1
            positions = local[rows]
            packed = x.new_zeros(len(ids), max_length, x.shape[-1])
            packed[packed_ids, positions] = x.reshape(-1, x.shape[-1])[index[rows]]
            cpu_lengths = these_lengths.cpu() if self.operator == "bigru" else None
            for block in self.blocks:
                packed = (block(packed, packed_mask, cpu_lengths) if self.operator == "bigru"
                          else block(packed, packed_mask))
            valid_out = self.norm(packed)[packed_ids, positions]
            flat = flat.index_copy(0, index[rows], valid_out.to(x.dtype))
        return flat.reshape_as(x)
