"""V11m16: ligand-ranked protein-only pocket candidates.

The established V11m13/V11m15 residue path remains unchanged.  This module
adds a deliberately small assignment path:

1. protein-only V4 pocketness channels propose spatially separated sites;
2. pre-cross protein and ligand states form independent pocket/query vectors;
3. ligand-to-candidate scores produce a bounded class-0 logit residual.

No native protein-ligand cross-distance is used.  Candidate generation is
non-parametric in the first probe so the experiment isolates whether explicit
pocket assignment helps, rather than confounding it with proposal learning.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import _atanh_ratio
from .pocket_gvp_fusion_v11m15 import PocketGVPFusionV11m15


@torch.no_grad()
def select_spatial_candidate_centers(
    scores: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    num_candidates: int,
    min_distance: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily select high-scoring residue centers with spatial NMS."""
    if scores.ndim != 2 or coords.ndim != 3 or mask.ndim != 2:
        raise ValueError("scores/mask must be [B,L] and coords must be [B,L,3]")
    if scores.shape != mask.shape or coords.shape[:2] != scores.shape:
        raise ValueError("candidate center input shapes do not match")

    batch_size, _ = scores.shape
    n_candidates = max(1, int(num_candidates))
    valid_mask = mask.bool() & torch.isfinite(coords).all(dim=-1)
    work_scores = torch.nan_to_num(scores.float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
    available = valid_mask.clone()
    batch_idx = torch.arange(batch_size, device=scores.device)

    indices = []
    centers = []
    center_valid = []
    for _ in range(n_candidates):
        has_candidate = available.any(dim=-1)
        masked_scores = work_scores.masked_fill(~available, -torch.inf)
        index = masked_scores.argmax(dim=-1)
        safe_index = torch.where(has_candidate, index, torch.zeros_like(index))
        center = coords[batch_idx, safe_index]
        center = torch.where(has_candidate.unsqueeze(-1), center, torch.zeros_like(center))

        indices.append(safe_index)
        centers.append(center)
        center_valid.append(has_candidate)

        distance = torch.linalg.vector_norm(coords.float() - center.float().unsqueeze(1), dim=-1)
        separated = distance >= float(min_distance)
        available = available & separated & has_candidate.unsqueeze(-1)
        if has_candidate.any():
            available[batch_idx[has_candidate], safe_index[has_candidate]] = False

    return (
        torch.stack(indices, dim=1),
        torch.stack(centers, dim=1),
        torch.stack(center_valid, dim=1),
    )


def gaussian_candidate_membership(
    coords: torch.Tensor,
    centers: torch.Tensor,
    protein_mask: torch.Tensor,
    candidate_valid: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return bounded spatial membership [B,K,L] around candidate centers."""
    sigma = max(float(sigma), 1.0e-3)
    distance_sq = (
        coords.float().unsqueeze(1) - centers.float().unsqueeze(2)
    ).square().sum(dim=-1)
    membership = torch.exp(-0.5 * distance_sq / (sigma * sigma))
    membership = membership * protein_mask.bool().unsqueeze(1).to(membership.dtype)
    membership = membership * candidate_valid.bool().unsqueeze(-1).to(membership.dtype)
    return membership


class PocketGVPFusionV11m16(PocketGVPFusionV11m15):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m15.DEFAULT_CONFIG,
        "use_candidate_rank_gate": False,
        "candidate_num_sites": 4,
        "candidate_min_center_distance": 10.0,
        "candidate_membership_sigma": 8.0,
        "candidate_pocketness_indices": [46, 47, 48],
        "candidate_pocketness_weights": [0.50, 0.25, 0.25],
        "candidate_embed_dim": 128,
        "candidate_temperature": 0.10,
        "candidate_gate_alpha_init": 0.02,
        "candidate_gate_alpha_max": 0.30,
        "candidate_aux_weight": 0.02,
        "candidate_aux_min_coverage": 0.05,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_candidate_rank_gate = bool(cfg.get("use_candidate_rank_gate", False))
        self.candidate_num_sites = max(1, int(cfg.get("candidate_num_sites", 4)))
        self.candidate_min_center_distance = float(
            cfg.get("candidate_min_center_distance", 10.0)
        )
        self.candidate_membership_sigma = float(cfg.get("candidate_membership_sigma", 8.0))
        self.candidate_pocketness_indices = tuple(
            int(index) for index in cfg.get("candidate_pocketness_indices", [46, 47, 48])
        )
        pocketness_weights = tuple(
            float(value) for value in cfg.get("candidate_pocketness_weights", [0.50, 0.25, 0.25])
        )
        if len(pocketness_weights) != len(self.candidate_pocketness_indices):
            raise ValueError("candidate pocketness indices and weights must have equal length")
        weight_tensor = torch.tensor(pocketness_weights, dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.abs().sum().clamp(min=1.0e-6)
        self.register_buffer(
            "assignment_candidate_pocketness_weights",
            weight_tensor,
            persistent=True,
        )

        self.candidate_temperature = float(cfg.get("candidate_temperature", 0.10))
        self.candidate_gate_alpha_max = float(cfg.get("candidate_gate_alpha_max", 0.30))
        self.candidate_aux_weight = float(cfg.get("candidate_aux_weight", 0.02))
        self.candidate_aux_min_coverage = float(cfg.get("candidate_aux_min_coverage", 0.05))

        d_model = int(self.config["d_model"])
        embed_dim = int(cfg.get("candidate_embed_dim", 128))
        if self.use_candidate_rank_gate:
            self.assignment_candidate_lig_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            self.assignment_candidate_pocket_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            init = float(cfg.get("candidate_gate_alpha_init", 0.02))
            self.assignment_candidate_gate_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.candidate_gate_alpha_max))
            )
        else:
            self.assignment_candidate_lig_proj = None
            self.assignment_candidate_pocket_proj = None
            self.register_buffer(
                "assignment_candidate_gate_raw_alpha",
                torch.tensor(0.0),
                persistent=False,
            )

        self.config["use_candidate_rank_gate"] = self.use_candidate_rank_gate

    @staticmethod
    def _forward_arg(args: Tuple, kwargs: Dict, name: str, index: int):
        if name in kwargs:
            return kwargs[name]
        if index < len(args):
            return args[index]
        raise KeyError(f"V11m16 forward requires {name}")

    def _pocketness_score(self, protein_spatial_scalar: torch.Tensor) -> torch.Tensor:
        max_index = max(self.candidate_pocketness_indices)
        if protein_spatial_scalar.size(-1) <= max_index:
            raise ValueError(
                "protein_spatial_scalar does not contain configured pocketness channels: "
                f"shape={tuple(protein_spatial_scalar.shape)}, max_index={max_index}"
            )
        features = protein_spatial_scalar[..., list(self.candidate_pocketness_indices)].float()
        weights = self.assignment_candidate_pocketness_weights.to(
            device=features.device,
            dtype=features.dtype,
        )
        return torch.sum(features * weights, dim=-1)

    def _candidate_rank_forward(
        self,
        outputs: Dict[str, torch.Tensor],
        protein_spatial_scalar: torch.Tensor,
        protein_coords: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.assignment_candidate_lig_proj is None or self.assignment_candidate_pocket_proj is None:
            return outputs
        if "h_ligand_context" not in outputs:
            raise RuntimeError("V11m16 requires pre-cross h_ligand_context from V11")

        h_protein = outputs["h_prot_global"]
        h_ligand = outputs["h_ligand_context"][:, 1 : 1 + ligand_mask.size(1)]
        pocketness = self._pocketness_score(protein_spatial_scalar)
        candidate_indices, candidate_centers, candidate_valid = select_spatial_candidate_centers(
            scores=pocketness,
            coords=protein_coords,
            mask=protein_mask,
            num_candidates=self.candidate_num_sites,
            min_distance=self.candidate_min_center_distance,
        )
        membership = gaussian_candidate_membership(
            coords=protein_coords,
            centers=candidate_centers,
            protein_mask=protein_mask,
            candidate_valid=candidate_valid,
            sigma=self.candidate_membership_sigma,
        ).to(dtype=h_protein.dtype)

        candidate_denom = membership.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        pocket_pool = torch.einsum("bkl,bld->bkd", membership, h_protein) / candidate_denom
        lig_pool, ligand_valid = self._masked_mean(h_ligand, ligand_mask)

        query = F.normalize(self.assignment_candidate_lig_proj(lig_pool.float()), dim=-1)
        key = F.normalize(self.assignment_candidate_pocket_proj(pocket_pool.float()), dim=-1)
        rank_logits = torch.einsum("bd,bkd->bk", query, key)
        rank_valid = candidate_valid & ligand_valid.unsqueeze(-1)
        rank_logits = rank_logits.masked_fill(~rank_valid, -1.0e4)
        rank_weights = torch.softmax(
            rank_logits / max(self.candidate_temperature, 1.0e-6), dim=-1
        )
        rank_weights = rank_weights * rank_valid.to(dtype=rank_weights.dtype)
        rank_weights = rank_weights / rank_weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)

        residue_gate = torch.einsum("bk,bkl->bl", rank_weights.to(membership.dtype), membership)
        valid = protein_mask.bool()
        valid_count = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(residue_gate.dtype)
        gate_mean = (
            residue_gate * valid.to(dtype=residue_gate.dtype)
        ).sum(dim=-1, keepdim=True) / valid_count
        centered_gate = (residue_gate - gate_mean) * valid.to(dtype=residue_gate.dtype)

        alpha = self._bounded_scale(
            self.assignment_candidate_gate_raw_alpha.to(dtype=outputs["logits"].dtype),
            self.candidate_gate_alpha_max,
        )
        logits = outputs["logits"].clone()
        logits[..., 0] = logits[..., 0] + alpha * centered_gate.to(dtype=logits.dtype)
        outputs["logits"] = logits

        outputs["candidate_rank_logits"] = rank_logits
        outputs["candidate_rank_weights"] = rank_weights.detach()
        outputs["candidate_membership"] = membership.detach()
        outputs["candidate_indices"] = candidate_indices.detach()
        outputs["candidate_centers"] = candidate_centers.detach()
        outputs["candidate_valid"] = candidate_valid.detach()
        outputs["candidate_gate_alpha"] = alpha.detach()
        outputs["candidate_gate_abs_mean"] = (
            centered_gate.detach()[valid].abs().mean()
            if valid.any()
            else logits.new_tensor(0.0, dtype=torch.float32)
        )
        rank_entropy = -(
            rank_weights.float() * rank_weights.float().clamp(min=1.0e-8).log()
        ).sum(dim=-1)
        valid_rows = rank_valid.any(dim=-1)
        outputs["candidate_rank_entropy"] = (
            rank_entropy[valid_rows].mean().detach()
            if valid_rows.any()
            else rank_entropy.new_tensor(0.0)
        )
        return outputs

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = super().forward(*args, **kwargs)
        if not self.use_candidate_rank_gate:
            return outputs
        return self._candidate_rank_forward(
            outputs=outputs,
            protein_spatial_scalar=self._forward_arg(args, kwargs, "protein_spatial_scalar", 2),
            protein_coords=self._forward_arg(args, kwargs, "protein_coords", 5),
            protein_mask=self._forward_arg(args, kwargs, "protein_mask", 6),
            ligand_mask=self._forward_arg(args, kwargs, "ligand_mask", 12),
        )

    def compute_assignment_auxiliary_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_loss, stats = super().compute_assignment_auxiliary_loss(
            outputs=outputs,
            labels=labels,
            mask=mask,
            batch=batch,
        )
        if not self.use_candidate_rank_gate or "candidate_rank_logits" not in outputs:
            return base_loss, stats

        membership = outputs["candidate_membership"].to(device=labels.device)
        candidate_valid = outputs["candidate_valid"].to(device=labels.device).bool()
        positive = ((labels == 0) & mask.bool()).to(dtype=membership.dtype)
        positive_count = positive.sum(dim=-1, keepdim=True).clamp(min=1.0)
        coverage = (membership * positive.unsqueeze(1)).sum(dim=-1) / positive_count
        coverage = coverage.masked_fill(~candidate_valid, -1.0)
        target = coverage.argmax(dim=-1)
        best_coverage = coverage.max(dim=-1).values
        valid = positive.bool().any(dim=-1) & candidate_valid.any(dim=-1)
        valid = valid & (best_coverage >= self.candidate_aux_min_coverage)

        if valid.any():
            raw_loss = F.cross_entropy(
                outputs["candidate_rank_logits"][valid]
                / max(self.candidate_temperature, 1.0e-6),
                target[valid],
            )
            candidate_loss = raw_loss * self.candidate_aux_weight
            with torch.no_grad():
                accuracy = (
                    outputs["candidate_rank_logits"][valid].argmax(dim=-1) == target[valid]
                ).float().mean()
                coverage_mean = best_coverage[valid].mean()
        else:
            raw_loss = outputs["logits"].new_tensor(0.0)
            candidate_loss = raw_loss
            accuracy = raw_loss
            coverage_mean = raw_loss

        total = base_loss + candidate_loss
        stats.update(
            {
                "candidate_loss": raw_loss.detach(),
                "candidate_weighted_loss": candidate_loss.detach(),
                "candidate_valid_frac": valid.float().mean().detach(),
                "candidate_rank_acc": accuracy.detach(),
                "candidate_best_coverage": coverage_mean.detach(),
                "candidate_gate_alpha": outputs["candidate_gate_alpha"].detach(),
                "candidate_gate_abs_mean": outputs["candidate_gate_abs_mean"].detach(),
                "candidate_rank_entropy": outputs["candidate_rank_entropy"].detach(),
            }
        )
        stats["total_loss"] = total.detach()
        return total, stats
