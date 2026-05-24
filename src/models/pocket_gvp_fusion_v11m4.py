"""
Pocket-GVP-Fusion V11m4: V11m3 + local ligand-evidence delta head.

V11m4 is intentionally built as a small graft on top of V11m3:
- The full V11m3 trunk and classifier produce base_logits unchanged.
- A coordinate-free pair-evidence branch lets each residue read ligand atom
  memory from local54 atom chemistry + ligand hidden states.
- The branch predicts delta_logits and returns:
      logits = base_logits + scale * delta_logits

No protein-ligand cross-coordinate distance, angle, or holo geometry is used.
The delta head final projection is zero-initialized, so the initial forward is
identical to V11m3 up to floating-point roundoff.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m3 import PocketGVPFusionV11m3


class PocketGVPFusionV11m4(PocketGVPFusionV11m3):
    """
    V11m4: V11m3 stable trunk plus local54/edge-informed ligand delta.

    The branch is late-fusion by design.  It first tests whether a small
    residue-level ligand correction can improve V11m3 without lowering the
    protein-side decision floor.  If this works, pair evidence can later be
    moved earlier into cross-attention logits.
    """

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m3.DEFAULT_CONFIG,
        "use_local_pair_delta": True,
        "pair_delta_local_dim": 54,
        "pair_delta_dim": 128,
        "pair_delta_hidden": 256,
        "pair_delta_dropout": 0.1,
        "pair_delta_score_clip": 20.0,
        "pair_delta_logit_scale_init": 0.5,
        "pair_delta_logit_scale_max": 1.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_local_pair_delta = bool(cfg.get("use_local_pair_delta", True))
        self.pair_delta_local_dim = int(cfg.get("pair_delta_local_dim", 54))
        self.pair_delta_dim = int(cfg.get("pair_delta_dim", 128))
        self.pair_delta_score_clip = float(cfg.get("pair_delta_score_clip", 20.0))
        self.pair_delta_logit_scale_max = float(cfg.get("pair_delta_logit_scale_max", 1.0))

        d_model = int(self.config["d_model"])
        n_classes = int(self.config["n_classes"])
        hidden = int(cfg.get("pair_delta_hidden", 256))
        dropout = float(cfg.get("pair_delta_dropout", 0.1))

        if self.use_local_pair_delta:
            self.pair_delta_prot_norm = nn.LayerNorm(d_model)
            self.pair_delta_lig_norm = nn.LayerNorm(d_model)
            self.pair_delta_local_norm = nn.LayerNorm(self.pair_delta_local_dim)

            self.pair_delta_q = nn.Linear(d_model, self.pair_delta_dim, bias=False)
            self.pair_delta_k_lig = nn.Linear(d_model, self.pair_delta_dim, bias=False)
            self.pair_delta_k_local = nn.Linear(
                self.pair_delta_local_dim, self.pair_delta_dim, bias=False
            )
            self.pair_delta_v = nn.Linear(d_model, d_model, bias=False)

            delta_in_dim = d_model + d_model + 3
            self.pair_delta_head = nn.Sequential(
                nn.LayerNorm(delta_in_dim),
                nn.Linear(delta_in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            # Exact V11m3 behavior at initialization. The final layer learns the
            # delta first; earlier pair modules start receiving gradients after
            # the first update, avoiding a large initial branch perturbation.
            final = self.pair_delta_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

            init_scale = float(cfg.get("pair_delta_logit_scale_init", 0.5))
            self.pair_delta_logit_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.pair_delta_head = None
            self.register_buffer("pair_delta_logit_scale", torch.tensor(0.0), persistent=False)

        self.config["use_local_pair_delta"] = self.use_local_pair_delta
        self.config["pair_delta_local_dim"] = self.pair_delta_local_dim
        self.config["pair_delta_dim"] = self.pair_delta_dim

    def _pad_ligand_local54(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter flat ligand local features to [B, M, local_dim]."""
        batch_size, max_atoms = ligand_mask.shape
        local = ligand_x[:, : self.pair_delta_local_dim]
        out = ligand_x.new_zeros((batch_size, max_atoms, self.pair_delta_local_dim))

        n_atoms = int(local.size(0))
        if n_atoms == 0:
            return out

        if not bool((ligand_batch[1:] >= ligand_batch[:-1]).all()):
            raise ValueError("ligand_batch must be non-decreasing for V11m4 local scatter")

        counts = torch.bincount(ligand_batch, minlength=batch_size)
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=ligand_x.device)
        cumsum[1:] = counts.cumsum(0)
        local_idx = torch.arange(n_atoms, device=ligand_x.device) - cumsum[ligand_batch]
        valid = local_idx < max_atoms
        out[ligand_batch[valid], local_idx[valid]] = local[valid]
        return out

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Coordinate-free residue-atom compatibility delta logits."""
        if not self.use_local_pair_delta:
            return h_prot.new_zeros((*h_prot.shape[:2], int(self.config["n_classes"])))

        # h_ligand_fused includes the dynamic global token at position 0.
        max_atoms = ligand_mask.size(1)
        h_lig_atoms = h_ligand_fused[:, 1 : 1 + max_atoms, :]
        local54 = self._pad_ligand_local54(ligand_x, ligand_batch, ligand_mask)

        hp = self.pair_delta_prot_norm(h_prot)
        hl = self.pair_delta_lig_norm(h_lig_atoms)
        lf = self.pair_delta_local_norm(local54.to(dtype=hl.dtype))

        q = self.pair_delta_q(hp)
        k = self.pair_delta_k_lig(hl) + self.pair_delta_k_local(lf)
        v = self.pair_delta_v(hl)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.pair_delta_dim)
        scores = scores.float().clamp(-self.pair_delta_score_clip, self.pair_delta_score_clip)

        atom_mask = ligand_mask.bool()
        score_mask = atom_mask.unsqueeze(1)  # [B, 1, M]
        scores = scores.masked_fill(~score_mask, -1.0e4)

        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))

        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        attn = attn * atom_mask.unsqueeze(1).to(dtype=attn.dtype)

        # Keep no-atom rows exactly zero instead of uniform.
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = torch.where(has_atom, attn / denom, torch.zeros_like(attn))

        lig_read = torch.matmul(attn, v)

        masked_scores = scores.masked_fill(~score_mask, -1.0e4)
        pair_max = masked_scores.max(dim=-1).values
        pair_lse = torch.logsumexp(masked_scores, dim=-1)
        n_valid = atom_mask.sum(dim=1).clamp(min=1).to(dtype=pair_lse.dtype).view(-1, 1)
        pair_lse = pair_lse - n_valid.log()
        entropy = -(attn.float() * (attn.float().clamp(min=1e-8).log())).sum(dim=-1)

        stats = torch.stack([pair_max, pair_lse, entropy], dim=-1).to(dtype=h_prot.dtype)
        delta_in = torch.cat([h_prot, lig_read.to(dtype=h_prot.dtype), stats], dim=-1)
        delta_logits = self.pair_delta_head(delta_in)
        delta_logits = delta_logits * protein_mask.unsqueeze(-1).to(dtype=delta_logits.dtype)
        return delta_logits

    def _delta_scale(self) -> torch.Tensor:
        if self.pair_delta_logit_scale_max <= 0:
            return self.pair_delta_logit_scale
        return self.pair_delta_logit_scale.tanh() * self.pair_delta_logit_scale_max

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        protein_physchem: torch.Tensor,
        protein_spatial_scalar: torch.Tensor,
        protein_spatial_vector: torch.Tensor,
        protein_gvp_vectors: torch.Tensor,
        protein_coords: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_edge_index: torch.Tensor,
        ligand_edge_attr: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
        chain_ids: Optional[torch.Tensor] = None,
        ligand_to_protein: Optional[torch.Tensor] = None,
        unimol_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = super().forward(
            esm_embeddings=esm_embeddings,
            protein_physchem=protein_physchem,
            protein_spatial_scalar=protein_spatial_scalar,
            protein_spatial_vector=protein_spatial_vector,
            protein_gvp_vectors=protein_gvp_vectors,
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_x=ligand_x,
            ligand_pos=ligand_pos,
            ligand_edge_index=ligand_edge_index,
            ligand_edge_attr=ligand_edge_attr,
            ligand_batch=ligand_batch,
            ligand_mask=ligand_mask,
            chain_ids=chain_ids,
            ligand_to_protein=ligand_to_protein,
            unimol_embed=unimol_embed,
        )

        base_logits = outputs["logits"]
        delta_logits = self._compute_pair_delta(
            h_prot=outputs["h_fused"],
            h_ligand_fused=outputs["h_ligand_fused"],
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
        )
        scale = self._delta_scale().to(dtype=base_logits.dtype)
        logits = base_logits + scale * delta_logits

        outputs["base_logits"] = base_logits
        outputs["delta_logits"] = delta_logits
        outputs["pair_delta_scale"] = scale.detach()
        outputs["pair_delta_abs_mean"] = delta_logits.detach().abs().mean()
        outputs["logits"] = logits
        return outputs
