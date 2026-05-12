"""
Pocket-GVP-Fusion V11m1: Ligand Global Gate.

V11m1 adds over V11:
- Ligand Global Gate (lig_gate_proj):
  After cross-attention, the cross-attended ligand global token h_lig_fused[:,0,:]
  is projected through a sigmoid gate and applied multiplicatively to h_prot_fused
  before the classifier.  This lets the ligand modulate which protein feature
  dimensions are relevant for this specific ligand, rather than adding a uniform
  global bias (as concat would).

      gate = σ(W · lig_global)                       [B, d_model]
      h_prot_gated = h_prot_fused * gate[:, None, :] [B, L, d_model]
      x_final = [h_prot_gated | physchem_encoded]

  Classifier input dimension is unchanged (d_model + d_phys_encoded = 640).

  Gate init: W=0, b=0  →  σ(0)=0.5 (half-open at start), converges gradually.

繼承 V11:
  Per-layer distance bias, Distance regression aux head,
  Dynamic Ligand Global Token, Ligand Internal Self-Attention,
  GVP Output Dropout, Coordinate Noise, RoPE, AttnRes
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional

from .pocket_gvp_fusion_v11 import PocketGVPFusionV11


class PocketGVPFusionV11m1(PocketGVPFusionV11):
    """
    V11m1: Ligand global gate on protein representation before classifier.
    All other modules identical to V11.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        d_model = self.config['d_model']

        # Gate projects cross-attended ligand global token → per-dim sigmoid gate
        # applied to h_prot_fused.  Zero-init → σ(0)=0.5 at start for stability.
        self.lig_gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.lig_gate_proj.weight)
        nn.init.zeros_(self.lig_gate_proj.bias)

    def forward(
        self,
        esm_embeddings,
        protein_physchem,
        protein_spatial_scalar,
        protein_spatial_vector,
        protein_gvp_vectors,
        protein_coords,
        protein_mask,
        ligand_x,
        ligand_pos,
        ligand_edge_index,
        ligand_edge_attr,
        ligand_batch,
        ligand_to_protein,
        ligand_mask,
        chain_ids=None,
    ) -> Dict[str, torch.Tensor]:
        # Run V11 forward.  V11 exposes 'protein_physchem_encoded' and
        # 'h_ligand_fused' in its result dict so we can avoid recomputing.
        result = super().forward(
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
            ligand_to_protein=ligand_to_protein,
            ligand_mask=ligand_mask,
            chain_ids=chain_ids,
        )

        # ── Ligand global gate ────────────────────────────────────────────────
        h_prot_fused           = result['h_fused']                   # [B, L, d_model]
        h_lig_fused            = result['h_ligand_fused']            # [B, M+1, d_model]
        protein_physchem_enc   = result['protein_physchem_encoded']  # [B, L, d_phys_enc]

        lig_global   = h_lig_fused[:, 0, :]                          # [B, d_model]
        gate         = torch.sigmoid(self.lig_gate_proj(lig_global))  # [B, d_model]
        h_prot_gated = h_prot_fused * gate.unsqueeze(1)              # [B, L, d_model]

        # ── Re-classify with gated features (same head weights as V11) ───────
        x_final = torch.cat([h_prot_gated, protein_physchem_enc], dim=-1)
        x_final = self.classifier_input_norm(x_final)
        logits  = self.classifier(x_final)

        result['logits'] = logits  # overwrite with gated logits
        return result
