"""Pocket-GVP-Fusion V11m15: auxiliary ligand-pocket assignment objective.

The model architecture stays on the current V11m13 local-evidence line.  V11m15
only adds a lightweight embedding head that learns a ligand-query / pocket-key
space from existing hidden states:

    q_L = pooled ligand atom hidden -> projection -> L2 normalize
    k_C = pooled GT contact residue hidden -> projection -> L2 normalize

The auxiliary loss is intentionally separate from residue logits.  It can later
be wired into inference if retrieval improves but residue-level LAA does not.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13


class PocketGVPFusionV11m15(PocketGVPFusionV11m13):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m13.DEFAULT_CONFIG,
        "use_assignment_embed_loss": False,
        "assignment_embed_dim": 128,
        "assignment_temperature": 0.10,
        "assignment_embed_weight": 0.05,
        "assignment_mask_same_pdb_inbatch": True,
        "assignment_samevalid_mode": "none",  # none | softmax | pairwise
        "assignment_samevalid_weight": 0.02,
        "assignment_samevalid_temperature": 0.10,
        "assignment_samevalid_neg_weight": 3.0,
        "assignment_pair_beta": 10.0,
        "assignment_pair_topk": 1,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)
        self.use_assignment_embed_loss = bool(cfg.get("use_assignment_embed_loss", False))
        self.assignment_temperature = float(cfg.get("assignment_temperature", 0.10))
        self.assignment_embed_weight = float(cfg.get("assignment_embed_weight", 0.05))
        self.assignment_mask_same_pdb_inbatch = bool(
            cfg.get("assignment_mask_same_pdb_inbatch", True)
        )
        self.assignment_samevalid_mode = str(
            cfg.get("assignment_samevalid_mode", "none")
        ).lower()
        self.assignment_samevalid_weight = float(cfg.get("assignment_samevalid_weight", 0.02))
        self.assignment_samevalid_temperature = float(
            cfg.get("assignment_samevalid_temperature", self.assignment_temperature)
        )
        self.assignment_samevalid_neg_weight = float(
            cfg.get("assignment_samevalid_neg_weight", 3.0)
        )
        self.assignment_pair_beta = float(cfg.get("assignment_pair_beta", 10.0))
        self.assignment_pair_topk = max(1, int(cfg.get("assignment_pair_topk", 1)))

        d_model = int(self.config["d_model"])
        embed_dim = int(cfg.get("assignment_embed_dim", 128))
        if self.use_assignment_embed_loss:
            self.assignment_lig_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            self.assignment_pocket_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.assignment_lig_proj = None
            self.assignment_pocket_proj = None

        self.config["use_assignment_embed_loss"] = self.use_assignment_embed_loss
        self.config["assignment_samevalid_mode"] = self.assignment_samevalid_mode

    def _masked_mean(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)).sum(dim=-2) / denom
        valid = mask.any(dim=-1)
        return pooled, valid

    def _assignment_embeddings(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.assignment_lig_proj is None or self.assignment_pocket_proj is None:
            raise RuntimeError("assignment projections are disabled")
        if "h_fused" not in outputs or "h_ligand_fused" not in outputs:
            raise RuntimeError("V11m15 assignment loss requires h_fused and h_ligand_fused outputs")

        h_prot = outputs["h_fused"]
        h_lig = outputs["h_ligand_fused"][:, 1 : 1 + ligand_mask.size(1), :]
        pos_mask = (labels == 0) & protein_mask.bool()

        lig_pool, lig_valid = self._masked_mean(h_lig, ligand_mask)
        pocket_pool, pocket_valid = self._masked_mean(h_prot, pos_mask)
        valid = lig_valid & pocket_valid

        q = F.normalize(self.assignment_lig_proj(lig_pool.float()), dim=-1)
        k = F.normalize(self.assignment_pocket_proj(pocket_pool.float()), dim=-1)
        return q, k, valid

    def _inbatch_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        valid: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = q.device
        batch_size = q.size(0)
        if valid.sum() <= 1:
            zero = q.new_tensor(0.0)
            return zero, {
                "inbatch_loss": zero,
                "inbatch_acc": zero,
                "inbatch_valid_frac": valid.float().mean(),
                "inbatch_rank_mean": zero,
            }

        sim = torch.matmul(q, k.t()) / max(self.assignment_temperature, 1e-6)
        col_mask = valid.view(1, batch_size).expand(batch_size, batch_size).clone()
        row_mask = valid.clone()

        if self.assignment_mask_same_pdb_inbatch:
            pdb_ids = list(batch.get("pdb_ids", []))
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and i < len(pdb_ids) and j < len(pdb_ids) and pdb_ids[i] == pdb_ids[j]:
                        col_mask[i, j] = False

        sim = sim.masked_fill(~col_mask, -1.0e4)
        targets = torch.arange(batch_size, device=device)
        row_idx = torch.where(row_mask)[0]
        loss = F.cross_entropy(sim[row_idx], targets[row_idx])

        with torch.no_grad():
            pred = sim[row_idx].argmax(dim=-1)
            acc = (pred == targets[row_idx]).float().mean()
            pos_scores = sim[row_idx, targets[row_idx]].unsqueeze(-1)
            ranks = ((sim[row_idx] > pos_scores) & col_mask[row_idx]).sum(dim=-1).float() + 1.0
        return loss * self.assignment_embed_weight, {
            "inbatch_loss": loss.detach(),
            "inbatch_acc": acc.detach(),
            "inbatch_valid_frac": valid.float().mean().detach(),
            "inbatch_rank_mean": ranks.mean().detach(),
        }

    def _samevalid_loss(
        self,
        q: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = self.assignment_samevalid_mode
        if mode not in {"softmax", "pairwise"}:
            zero = q.new_tensor(0.0)
            return zero, {"samevalid_loss": zero, "samevalid_valid_frac": zero}
        if "assignment_neg_masks" not in batch or "assignment_neg_valid" not in batch:
            zero = q.new_tensor(0.0)
            return zero, {"samevalid_loss": zero, "samevalid_valid_frac": zero}

        neg_masks = batch["assignment_neg_masks"].to(device=q.device).bool()
        neg_valid = batch["assignment_neg_valid"].to(device=q.device).bool()
        h_prot = outputs["h_fused"].float()
        batch_size, max_negs, _ = neg_masks.shape
        if max_negs == 0 or neg_valid.sum() == 0:
            zero = q.new_tensor(0.0)
            return zero, {"samevalid_loss": zero, "samevalid_valid_frac": zero}

        flat_masks = neg_masks.view(batch_size * max_negs, -1)
        neg_pool, pooled_valid = self._masked_mean(
            h_prot.repeat_interleave(max_negs, dim=0),
            flat_masks,
        )
        neg_k = F.normalize(self.assignment_pocket_proj(neg_pool), dim=-1)
        neg_k = neg_k.view(batch_size, max_negs, -1)
        valid = neg_valid & pooled_valid.view(batch_size, max_negs)
        row_valid = valid.any(dim=-1)
        if row_valid.sum() == 0:
            zero = q.new_tensor(0.0)
            return zero, {"samevalid_loss": zero, "samevalid_valid_frac": row_valid.float().mean()}

        # Current-pocket positive key is the diagonal in-batch pocket embedding.
        labels = batch["labels"].to(device=q.device)
        protein_mask = batch["protein_mask"].to(device=q.device)
        pos_pool, _ = self._masked_mean(h_prot, (labels == 0) & protein_mask.bool())
        pos_k = F.normalize(self.assignment_pocket_proj(pos_pool), dim=-1)
        pos_score = (q * pos_k).sum(dim=-1)
        neg_score = torch.einsum("bd,bkd->bk", q, neg_k)
        neg_score = neg_score.masked_fill(~valid, -1.0e4)

        if mode == "softmax":
            tau = max(self.assignment_samevalid_temperature, 1e-6)
            pos_logit = pos_score[row_valid] / tau
            neg_logits = neg_score[row_valid] / tau
            log_weight = torch.log(
                neg_logits.new_tensor(max(self.assignment_samevalid_neg_weight, 1e-6))
            )
            denom = torch.logsumexp(
                torch.cat([pos_logit.unsqueeze(-1), neg_logits + log_weight], dim=-1),
                dim=-1,
            )
            loss = (denom - pos_logit).mean()
        else:
            topk = min(self.assignment_pair_topk, max_negs)
            hardest = neg_score[row_valid].topk(topk, dim=-1).values
            margins = pos_score[row_valid].unsqueeze(-1) - hardest
            loss = F.softplus(-self.assignment_pair_beta * margins).mean()

        return loss * self.assignment_samevalid_weight, {
            "samevalid_loss": loss.detach(),
            "samevalid_valid_frac": row_valid.float().mean().detach(),
            "samevalid_neg_per_valid": valid[row_valid].sum(dim=-1).float().mean().detach(),
        }

    def compute_assignment_auxiliary_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_assignment_embed_loss:
            zero = outputs["logits"].new_tensor(0.0)
            return zero, {}
        q, k, valid = self._assignment_embeddings(
            outputs=outputs,
            labels=labels,
            protein_mask=mask,
            ligand_mask=batch["ligand_mask"].to(device=labels.device),
        )
        inbatch_loss, stats = self._inbatch_loss(q, k, valid, batch)
        same_loss, same_stats = self._samevalid_loss(q, outputs, batch)
        stats.update(same_stats)
        total = inbatch_loss + same_loss
        stats["total_loss"] = total.detach()
        stats["embed_weight"] = total.new_tensor(self.assignment_embed_weight)
        stats["samevalid_weight"] = total.new_tensor(self.assignment_samevalid_weight)
        return total, stats
