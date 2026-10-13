"""Training-only geometry-guided noise. No labels, ligand, teacher or new weights.

The protected region is a soft union of protein-only geometric candidates, NOT
a ground-truth pocket. The random control permutes the same amplitude multiset.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ContextNoiseOptions:
    mode: str = "background"
    sigma: float = .10
    foreground_floor: float = .5
    batch_probability: float = .25
    seed: int = 190765
    num_candidates: int = 8
    min_center_distance: float = 10.
    membership_sigma: float = 8.

    def __post_init__(self):
        if self.mode not in ("background", "random"):
            raise ValueError("Context noise mode must be background or random")
        for name in ("sigma", "foreground_floor", "batch_probability",
                     "min_center_distance", "membership_sigma"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Non-finite context noise {name}")
        if self.sigma < 0 or not 0 <= self.foreground_floor <= 1:
            raise ValueError("Invalid sigma or foreground noise floor")
        if not 0 <= self.batch_probability <= 1:
            raise ValueError("Invalid augmentation probability")
        if self.num_candidates < 1 or self.min_center_distance <= 0 or self.membership_sigma <= 0:
            raise ValueError("Invalid geometric candidate settings")


@torch.no_grad()
def geometry_protection(spatial, coords, mask, options):
    """Soft max-union around high-pocketness, spatially separated CA centers.

    V4 channels 46:49 are CA-only CX/packing/shell density (not ligand distance).
    This reproduces V11m16's .50/.25/.25 candidate score without its rank head.
    """
    if spatial.ndim != 3 or spatial.size(-1) < 49 or coords.shape != (*mask.shape, 3):
        raise ValueError("Expected V4 spatial features and aligned CA coordinates")
    if spatial.shape[:2] != mask.shape or mask.ndim != 2 or mask.size(1) == 0:
        raise ValueError("Invalid protein mask/feature shape")
    valid = mask.bool()
    if not torch.isfinite(coords[valid]).all() or not torch.isfinite(spatial[..., 46:49][valid]).all():
        raise ValueError("Non-finite valid protein geometry")
    xyz = coords.detach().float().masked_fill(~valid[..., None], 0.)
    features = spatial[..., 46:49].detach().float().masked_fill(~valid[..., None], 0.)
    scores = (features * features.new_tensor([.5, .25, .25])).sum(-1)
    available = valid.clone()
    protection = torch.zeros_like(scores)
    rows = torch.arange(len(mask), device=mask.device)
    for _ in range(options.num_candidates):
        has = available.any(-1)
        index = scores.masked_fill(~available, -torch.inf).argmax(-1)
        center = xyz[rows, index]
        d2 = (xyz - center[:, None]).square().sum(-1)
        member = torch.exp(-.5 * d2 / options.membership_sigma ** 2)
        member = member * (valid & has[:, None])
        protection = torch.maximum(protection, member)
        available &= (d2 >= options.min_center_distance ** 2) & has[:, None]
        available[rows, index] = False
    return protection.masked_fill(~valid, 0.)


class ProteinContextNoise:
    """A parameter-free controller, deliberately absent from the state_dict.

    Call once BEFORE checkpointed protein encoding. Private RNG leaves training
    dropout/data RNG intact; both modes consume identical numbers of draws.
    Scratch-only runs do not require serializing the private RNG for resume.
    """

    def __init__(self, options):
        self.options = options if isinstance(options, ContextNoiseOptions) else ContextNoiseOptions(**options)
        self.generators = {}
        self.calls = 0
        self.applied = 0
        self.last = {}
        self.last_protection = None
        self.last_amplitudes = None

    def _generator(self, device):
        key = str(device)
        if key not in self.generators:
            self.generators[key] = torch.Generator(device=device).manual_seed(self.options.seed)
        return self.generators[key]

    def __call__(self, hidden, spatial, coords, mask):
        o = self.options
        self.last_protection = self.last_amplitudes = None
        self.last = {"applied": 0.}
        if o.sigma == 0 or o.batch_probability == 0:
            return hidden
        self.calls += 1
        rng = self._generator(hidden.device)
        take = torch.rand((), device=hidden.device, generator=rng).item() < o.batch_probability
        self.last["applied_fraction_running"] = self.applied / self.calls
        if not take:
            return hidden
        if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
            raise ValueError("Hidden and mask must be residue-aligned")
        valid = mask.bool()
        if not valid.any():
            return hidden
        with torch.no_grad():
            protection = geometry_protection(spatial, coords, mask, o)
            amplitude = o.foreground_floor + (1 - o.foreground_floor) * (1 - protection)
            # Generate permutations in BOTH modes so subsequent RNG draws match.
            randomized = amplitude.clone()
            for b in range(len(mask)):
                ids = valid[b].nonzero(as_tuple=True)[0]
                order = torch.randperm(len(ids), device=hidden.device, generator=rng)
                randomized[b, ids] = amplitude[b, ids[order]]
            chosen = randomized if o.mode == "random" else amplitude
            scale = o.sigma * (.5 + torch.rand((len(mask), 1), device=hidden.device, generator=rng))
            epsilon = torch.randn(hidden.shape, device=hidden.device, dtype=torch.float32, generator=rng)
            clean = hidden.detach().float().masked_fill(~valid[..., None], 0.)
            if not torch.isfinite(clean).all():
                raise ValueError("Non-finite valid context hidden state")
            rms = clean.square().mean(-1).sqrt()
            delta = epsilon * (rms * scale * chosen)[..., None]
            delta = delta.masked_fill(~valid[..., None], 0.)
            relative = delta.square().mean(-1) / rms.square().clamp_min(1e-12)
            self.applied += 1
            self.last = {
                "applied": 1., "applied_fraction_running": self.applied / self.calls,
                "relative_noise_rms": relative[valid].mean().sqrt().item(),
                "expected_relative_noise_rms": (scale * chosen)[valid].square().mean().sqrt().item(),
                "hidden_rms": clean[valid].square().mean().sqrt().item(),
                "absolute_noise_rms": delta[valid].square().mean().sqrt().item(),
                "protection_mean": protection[valid].mean().item(),
                "protected_fraction": (protection[valid] >= .5).float().mean().item(),
                "amplitude_mean": chosen[valid].mean().item(),
            }
            self.last_protection = protection
            self.last_amplitudes = chosen.masked_fill(~valid, 0.)
        return hidden + delta.to(dtype=hidden.dtype)


def log_context_noise(model, labels, mask, writer, step):
    """Labels are used ONLY for post-forward diagnostics, never augmentation."""
    controller = getattr(getattr(model, "module", model), "_protein_context_noise", None)
    if controller is None or writer is None:
        return
    for key, value in controller.last.items():
        writer.add_scalar("ContextNoise/" + key, value, step)
    if controller.last_protection is None:
        return
    with torch.no_grad():
        for c in range(3):
            selected = mask.bool() & (labels == c)
            count = selected.sum().item()
            writer.add_scalar(f"ContextNoise/true_class{c}_count", count, step)
            if count:
                writer.add_scalar(f"ContextNoise/true_class{c}_protection_mean",
                                  controller.last_protection[selected].mean().item(), step)
                writer.add_scalar(f"ContextNoise/true_class{c}_amplitude_mean",
                                  controller.last_amplitudes[selected].mean().item(), step)
