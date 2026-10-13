"""Standalone candidate-quality probe; never changes the frozen V11 backbone."""

import torch
from torch import nn
from torch.nn import functional as F


class SiteQualityHead(nn.Module):
    def __init__(self, mode, input_dim, stats_mean, stats_std, width=128, projection=64):
        super().__init__()
        if mode not in ("stats_only", "protein_site", "ligand_site"):
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer("stats_mean", torch.as_tensor(stats_mean, dtype=torch.float32))
        self.register_buffer("stats_std", torch.as_tensor(stats_std, dtype=torch.float32).clamp_min(.01))
        dim = len(stats_mean)
        self.pocket = self.ligand = None
        if mode != "stats_only":
            self.pocket = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, projection), nn.GELU())
            dim += projection
        if mode == "ligand_site":
            self.ligand = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, projection), nn.GELU())
            dim += 3 * projection
        self.readout = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def forward(self, pocket, ligand, stats):
        parts = [((stats.float() - self.stats_mean) / self.stats_std).clamp(-10, 10)]
        if self.pocket is not None:
            p = self.pocket(pocket.float())
            parts.append(p)
        if self.ligand is not None:
            q = self.ligand(ligand.float()).unsqueeze(1).expand_as(p)
            parts.extend([q, p * q, (p - q).abs()])
        return self.readout(torch.cat(parts, dim=-1)).squeeze(-1).sigmoid()


def quality_loss(scores, targets, valid, group_weights):
    """Equal sample candidate means, inverse exact-axis group weights, soft ties."""
    count = valid.sum(-1).clamp_min(1)
    regression = ((scores - targets).square() * valid).sum(-1) / count
    target_distribution = torch.softmax((targets / .1).masked_fill(~valid, -1e4), dim=-1)
    log_scores = F.log_softmax((scores / .1).masked_fill(~valid, -1e4), dim=-1)
    ranking = -(target_distribution * log_scores).sum(-1)
    return ((regression + .1 * ranking) * group_weights).mean()


def select_with_fallback(scores, valid, margin=.02):
    """Scores only: label-free selection, first mask is original, ties keep it."""
    if margin < 0 or not valid[:, 0].all() or not torch.isfinite(scores[valid]).all():
        raise ValueError("Invalid scores, fallback or margin")
    best = scores.masked_fill(~valid, -torch.inf).argmax(-1)
    best_score = scores.gather(1, best[:, None]).squeeze(1)
    return torch.where(best_score > scores[:, 0] + margin, best, torch.zeros_like(best))
