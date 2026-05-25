"""
Pocket-GVP-Fusion V11m4t: V11m4 + global55/UniMol delta calibration gate.

This is the clean ablation for V11m5t:
- keep the proven V11m4 late local pair-delta branch
- do NOT add local pair evidence into cross-attention logits
- use global55 + UniMol CLS only as a delta-amplitude calibration gate

    logits = base_logits + scale * global_gate(global55, UniMol_CLS) * delta_logits

The gate is initialized to 1.0 exactly, so epoch-0 behavior matches V11m4.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m4 import PocketGVPFusionV11m4


class PocketGVPFusionV11m4t(PocketGVPFusionV11m4):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m4.DEFAULT_CONFIG,
        "use_global_delta_gate": True,
        "delta_gate_global_start": 54,
        "delta_gate_global_dim": 55,
        "delta_gate_hidden": 128,
        "delta_gate_per_class": True,
        "delta_gate_max": 2.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_global_delta_gate = bool(cfg.get("use_global_delta_gate", True))
        self.delta_gate_global_start = int(cfg.get("delta_gate_global_start", 54))
        self.delta_gate_global_dim = int(cfg.get("delta_gate_global_dim", 55))
        self.delta_gate_per_class = bool(cfg.get("delta_gate_per_class", True))
        self.delta_gate_max = float(cfg.get("delta_gate_max", 2.0))

        if self.use_global_delta_gate:
            d_unimol = int(cfg.get("d_unimol", 512))
            hidden = int(cfg.get("delta_gate_hidden", 128))
            gate_out = int(self.config["n_classes"]) if self.delta_gate_per_class else 1
            gate_in = self.delta_gate_global_dim + d_unimol
            self.global_delta_gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, gate_out),
            )
            # gate = max * sigmoid(raw); raw=0 and max=2 -> exact gate=1.
            final = self.global_delta_gate[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            self.global_delta_gate = None

        self.config["use_global_delta_gate"] = self.use_global_delta_gate

    def _mean_ligand_global55(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_atoms = ligand_mask.shape
        start = self.delta_gate_global_start
        end = start + self.delta_gate_global_dim
        global55 = ligand_x[:, start:end]
        out = ligand_x.new_zeros((batch_size, self.delta_gate_global_dim))

        n_atoms = int(global55.size(0))
        if n_atoms == 0:
            return out
        if not bool((ligand_batch[1:] >= ligand_batch[:-1]).all()):
            raise ValueError("ligand_batch must be non-decreasing for V11m4t global scatter")

        counts = torch.bincount(ligand_batch, minlength=batch_size)
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=ligand_x.device)
        cumsum[1:] = counts.cumsum(0)
        local_idx = torch.arange(n_atoms, device=ligand_x.device) - cumsum[ligand_batch]
        valid = local_idx < max_atoms

        out.index_add_(0, ligand_batch[valid], global55[valid])
        denom = ligand_mask.to(dtype=out.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        return out / denom

    def _compute_delta_gate(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
        unimol_embed: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if not self.use_global_delta_gate:
            return ligand_x.new_ones((batch_size, 1))

        global55 = self._mean_ligand_global55(ligand_x, ligand_batch, ligand_mask)
        if unimol_embed is None:
            unimol = ligand_x.new_zeros((batch_size, int(self.config.get("d_unimol", 512))))
        else:
            unimol = unimol_embed.to(device=ligand_x.device, dtype=ligand_x.dtype)

        gate_in = torch.cat([global55, unimol], dim=-1)
        raw = self.global_delta_gate(gate_in)
        return self.delta_gate_max * torch.sigmoid(raw)

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

        gate = self._compute_delta_gate(
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            ligand_mask=ligand_mask,
            unimol_embed=unimol_embed,
            batch_size=esm_embeddings.size(0),
        ).to(dtype=outputs["delta_logits"].dtype)

        gated_delta = outputs["delta_logits"] * gate.unsqueeze(1)
        scale = outputs["pair_delta_scale"].to(dtype=outputs["base_logits"].dtype)
        outputs["logits"] = outputs["base_logits"] + scale * gated_delta
        outputs["delta_gate"] = gate.detach()
        outputs["delta_gate_mean"] = gate.detach().mean()
        outputs["delta_gate_abs_dev"] = (gate.detach() - 1.0).abs().mean()
        return outputs
