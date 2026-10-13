"""Sampled positive-class mass, not canonical SupCon or ligand assignment loss."""

import math

import torch
from torch.nn import functional as F


def residue_classmass_loss(hidden, labels, mask, *, temperature=0.2, max_per_class=32, seed=44):
    """Compare contact/far residues; balance classes within input rows, then rows.

    Other input rows contribute candidates. Near/padding and self-pairs are
    excluded. No exact-protein grouping is inferred from a batch row index.
    Sampling uses a private CPU generator, leaving training RNG streams intact.
    """
    if not math.isfinite(temperature) or temperature <= 0 or max_per_class <= 0:
        raise ValueError("temperature and max_per_class must be positive")
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or mask.shape != labels.shape:
        raise ValueError("Expected hidden[B,L,D] and labels/mask[B,L]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    cpu_labels = labels.detach().cpu()
    cpu_mask = mask.detach().bool().cpu()
    selected, classes, rows = [], [], []
    for row in range(hidden.shape[0]):
        for cls in (0, 2):
            ids = torch.where(cpu_mask[row] & (cpu_labels[row] == cls))[0]
            if ids.numel() > max_per_class:
                ids = ids[torch.randperm(ids.numel(), generator=generator)[:max_per_class]]
            selected.extend((ids + row * hidden.shape[1]).tolist())
            classes.extend([cls] * len(ids))
            rows.extend([row] * len(ids))
    indices = torch.tensor(selected, device=hidden.device, dtype=torch.long)
    cls = torch.tensor(classes, device=hidden.device, dtype=torch.long)
    row_ids = torch.tensor(rows, device=hidden.device, dtype=torch.long)
    stats = {"sampled_contact": classes.count(0), "sampled_far": classes.count(2),
             "eligible_anchors": 0, "eligible_rows": 0}
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        sampled = hidden.reshape(-1, hidden.shape[-1])[indices].float()
        # Empty selection still returns a graph-connected scalar, without reading padding.
        zero = sampled.sum() * 0.0
        if len(selected) < 2:
            return zero, stats
        z = F.normalize(sampled, dim=-1, eps=1e-8)
        logits = (z @ z.T) / temperature
        nonself = ~torch.eye(len(selected), device=hidden.device, dtype=torch.bool)
        positive = (cls[:, None] == cls[None, :]) & nonself
        eligible = positive.any(dim=1)
        if not eligible.any():
            return zero, stats
        logits = logits[eligible]
        losses = torch.logsumexp(logits.masked_fill(~nonself[eligible], -torch.inf), dim=-1)
        losses = losses - torch.logsumexp(logits.masked_fill(~positive[eligible], -torch.inf), dim=-1)
        cls, row_ids = cls[eligible], row_ids[eligible]
        row_losses = []
        for row in row_ids.unique():
            per_class = [losses[(row_ids == row) & (cls == c)].mean()
                         for c in (0, 2) if ((row_ids == row) & (cls == c)).any()]
            row_losses.append(torch.stack(per_class).mean())
        stats.update(eligible_anchors=int(eligible.sum()), eligible_rows=len(row_losses))
        return torch.stack(row_losses).mean(), stats
