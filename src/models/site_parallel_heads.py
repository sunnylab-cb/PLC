"""Small frozen-feature assignment probes, with no GT input to any forward."""
import torch
from torch import nn
from torch.nn import functional as F

from .site_quality_head import SiteQualityHead


class PairedQualityHead(SiteQualityHead):
    """Matched paired-feature controls differing only in quality vs gain target."""
    def __init__(self, input_dim, mean, std, gain=False):
        super().__init__('ligand_site', input_dim, mean, std)
        self.gain = gain
        dim = len(mean) + 4 * 64
        self.readout = nn.Sequential(nn.Linear(dim * 3, 128), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, pocket, ligand, stats):
        p = self.pocket(pocket.float())
        q = self.ligand(ligand.float()).unsqueeze(1).expand_as(p)
        x = torch.cat([((stats.float()-self.stats_mean)/self.stats_std).clamp(-10, 10), p, q, p*q, (p-q).abs()], -1)
        original = x[:, :1].expand_as(x)
        score = self.readout(torch.cat([x, original, x-original], -1)).squeeze(-1)
        if not self.gain:
            return score.sigmoid()
        score = score.tanh()
        # The no-change decision has exactly zero gain, including during training.
        return torch.cat([torch.zeros_like(score[:, :1]), score[:, 1:]], 1)


class TokenQualityHead(SiteQualityHead):
    """Same parameters for fine token matching, mean matching and blind control."""
    def __init__(self, input_dim, mean, std, mode='tokens'):
        super().__init__('ligand_site', input_dim, mean, std)
        if mode not in ('tokens', 'mean', 'blind'):
            raise ValueError(mode)
        self.token_mode = mode
        dim = len(mean) + 4 * 64 + 2
        self.readout = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, pocket, ligand, stats, residues, atoms, membership, chunk=8):
        if self.token_mode == 'blind':
            atoms, ligand = torch.zeros_like(atoms), torch.zeros_like(ligand)
        p = self.pocket(pocket.float())
        q = self.ligand(ligand.float()).expand_as(p)
        if self.token_mode == 'mean':
            similarity = (F.normalize(p, dim=-1) * F.normalize(q, dim=-1)).sum(-1)
            matching = torch.stack([similarity, similarity], -1)
        else:
            r = F.normalize(self.pocket(residues.float()), dim=-1)
            a = F.normalize(self.ligand(atoms.float()), dim=-1)
            sim = r @ a.T / .1
            per_residue = .1 * (torch.logsumexp(sim, -1) - sim.new_tensor(len(atoms)).log())
            m = membership.float()
            count = m.sum(-1).clamp_min(1)
            # Normalized cosine / 0.1 is bounded by [-10,10]; exponentiation
            # stays finite and avoids any [candidates,residues,atoms] tensor.
            ligand_read = .1 * ((m @ sim.exp()) / count[:, None]).clamp_min(1e-12).log()
            matching = torch.stack([ligand_read.mean(-1), (m * per_residue).sum(-1)/count], -1)
            matching = matching.masked_fill(~membership.any(-1)[:, None], 0)
        x = torch.cat([((stats.float()-self.stats_mean)/self.stats_std).clamp(-10, 10),
                       p, q, p*q, (p-q).abs(), matching], -1)
        return self.readout(x).squeeze(-1).sigmoid()


class BoundaryResidualHead(nn.Module):
    """Only refine visible residues in protein-defined proposal support."""
    def __init__(self, dim, ligand_aware=True, amplitude=.5):
        super().__init__()
        self.ligand_aware, self.amplitude = ligand_aware, amplitude
        self.protein = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU())
        self.ligand = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU())
        self.readout = nn.Sequential(nn.Linear(64*3+2, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, residues, ligand, pocket, log_odds, membership, support):
        r = self.protein(residues.float())
        c = self.protein(pocket.float())
        q = self.ligand((ligand if self.ligand_aware else torch.zeros_like(ligand)).float())
        k, n = membership.shape
        x = torch.cat([r[None].expand(k, -1, -1), c[:, None].expand(-1, n, -1),
                       q[None, None].expand(k, n, -1), log_odds[None, :, None].expand(k, -1, -1),
                       membership.float()[:, :, None]], -1)
        delta = self.amplitude * self.readout(x).squeeze(-1).tanh()
        return log_odds[None] + delta * support


def gain_loss(scores, targets, valid, weights):
    gain = targets - targets[:, :1]
    regression = F.smooth_l1_loss(scores, gain, reduction='none', beta=.1)
    comparable = valid & (gain.abs() > .02)
    ranking = F.softplus(-gain.sign() * scores / .1) * gain.abs()
    each = (regression * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    each += .1 * (ranking * comparable).sum(-1) / comparable.sum(-1).clamp_min(1)
    return (each * weights).mean()
