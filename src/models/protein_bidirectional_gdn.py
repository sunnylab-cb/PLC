"""Chain-segmented bidirectional Gated DeltaNet for the protein context trunk.

Production uses fla-core 0.4.1 chunk kernels, not a token-by-token Python loop.
The small reference backend is exclusively for numerical/unit tests. Both
directions share parameters and are averaged after restoring residue order.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def chain_pack_indices(mask, chain_ids):
    """Return valid flat indices, segment boundaries, and within-chain reversal.

    Padding holes and batch boundaries also reset state. Chain IDs need not be
    consecutive; only contiguous runs are joined, never separate chain copies.
    """
    if mask.ndim != 2 or chain_ids.shape != mask.shape:
        raise ValueError("mask and chain_ids must both have shape [B,L]")
    length = mask.shape[1]
    index = mask.bool().flatten().nonzero().flatten()
    if index.numel() == 0:
        return index, index.new_zeros(1), index
    chains = chain_ids.flatten()[index]
    starts = torch.ones_like(index, dtype=torch.bool)
    starts[1:] = ((index[1:] != index[:-1] + 1)
                  | (index[1:] // length != index[:-1] // length)
                  | (chains[1:] != chains[:-1]))
    offsets = starts.nonzero().flatten()
    cu = torch.cat((offsets, offsets.new_tensor([index.numel()])))
    segment = starts.long().cumsum(0) - 1
    reverse = cu[segment] + cu[segment + 1] - 1 - torch.arange(index.numel(), device=index.device)
    return index, cu, reverse


def reference_delta_rule(q, k, v, g, beta, cu_seqlens):
    """FP32 oracle with the same key-by-value state and scaling as FLA."""
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    values = []
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        state = q.new_zeros(q.shape[2], q.shape[3], v.shape[3])
        for t in range(start, end):
            state = state * g[0, t].float().exp()[:, None, None]
            error = v[0, t].float() - torch.einsum('hk,hkv->hv', k[0, t], state)
            state = state + beta[0, t].float()[:, None, None] * k[0, t, :, :, None] * error[:, None, :]
            values.append(torch.einsum('hk,hkv->hv', q[0, t], state) * q.shape[-1] ** -.5)
    return torch.stack(values).unsqueeze(0).to(v.dtype)


class BidirectionalProteinGDN(nn.Module):
    requires_chain_ids = True

    def __init__(self, d_model, n_heads=8, conv_size=4, dropout=0., backend="fla"):
        super().__init__()
        if d_model % n_heads or conv_size < 1 or backend not in ("fla", "reference"):
            raise ValueError("Invalid GDN dimensions/backend")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim, self.conv_size = d_model // n_heads, conv_size
        self.backend = backend
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(3 * d_model, conv_size))
        nn.init.uniform_(self.conv_weight, -conv_size ** -.5, conv_size ** -.5)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        # Naming excludes decay constants from this repo's keyword-based AdamW decay.
        self.decay_log_bias = nn.Parameter(torch.empty(n_heads).uniform_(.001, 16).log())
        dt = torch.empty(n_heads).uniform_(math.log(.001), math.log(.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.output_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.output_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, x, cu):
        # Four vectorized taps, NOT a loop over sequence positions. Every tap is
        # bounded by the segment start, including the reversed-direction chains.
        projected = self.qkv(x)
        pos = torch.arange(x.shape[0], device=x.device)
        segment = torch.bucketize(pos, cu[1:], right=True)
        begin = cu[segment]
        convolved = torch.zeros_like(projected)
        for lag in range(self.conv_size):
            source = pos - lag
            tap = projected[source.clamp_min(0)].masked_fill((source < begin)[:, None], 0)
            convolved = convolved + tap * self.conv_weight[:, self.conv_size - 1 - lag].to(tap.dtype)
        q, k, v = (t.reshape(1, -1, self.n_heads, self.head_dim)
                   for t in F.silu(convolved).chunk(3, dim=-1))
        g = -self.decay_log_bias.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        beta = self.beta_proj(x).sigmoid().unsqueeze(0)
        if self.backend == "reference":
            if x.shape[0] > 512:
                raise RuntimeError("Reference GDN is for tiny tests only, not training")
            out = reference_delta_rule(q, k, v, g.unsqueeze(0), beta, cu)
        else:
            if x.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError("FLA GDN requires CUDA with BF16/FP16 autocast")
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            out, _ = chunk_gated_delta_rule(q, k, v, g.unsqueeze(0), beta,
                                           cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
        out = self.output_norm(out.float()).to(v.dtype).reshape(-1, self.d_model)
        return self.out_proj(out * F.silu(self.output_gate(x)))

    def forward(self, x, mask=None, attn_bias=None, position_ids=None, chain_ids=None):
        if attn_bias is not None:
            raise ValueError("GDN cannot consume dense distance bias; retain full-attention layers")
        if chain_ids is None:
            raise ValueError("Protein GDN requires explicit chain_ids")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        index, cu, reverse = chain_pack_indices(mask, chain_ids)
        if not index.numel():
            return x.masked_fill(torch.ones_like(x, dtype=torch.bool), 0)
        packed = x.reshape(-1, x.shape[-1])[index]
        both = torch.cat((packed, packed[reverse]), dim=0)
        both_cu = torch.cat((cu, cu[1:] + index.numel()))
        out = self._scan(both, both_cu)
        forward, backward = out.chunk(2, dim=0)
        merged = .5 * (forward + backward[reverse])
        flat = x.new_zeros(x.shape[0] * x.shape[1], x.shape[2])
        return self.dropout(flat.index_copy(0, index, merged.to(x.dtype)).reshape_as(x))
