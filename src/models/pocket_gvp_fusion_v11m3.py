"""
Pocket-GVP-Fusion V11m3: V11m2 + Uni-Mol ligand input.

V11m3 adds over V11m2:
0. Uni-Mol global embedding is projected and concatenated to every ligand
   atom feature before Ligand3DEncoder.  This is an isolated lab version:
   no V11/V11m1/V11m2 behavior is changed.

V11m2 features retained:
1. Ligand self-attention 3D distance bias (use_lig_dist_bias=True):
   Per-layer RBF(inter-atomic distance) → per-head additive bias in
   LigandSelfAttentionEncoder, symmetric to protein's DistanceBias.
   Global token row/col = 0 (no position); atom-atom pairs use LigandDistanceBias.
   Zero-init → safe, no change at epoch 0.

2. b_lig1 AttnRes anchor (use_lig_attnres_b1=True):
   After ligand self-attention, h_lig_ctx (post-self-attn) is added as a
   2nd ligand anchor in lig_attn_res_blocks = [b_lig0, h_lig_ctx].
   InterBlockAttnRes.proj is Linear(d,1), source-count agnostic — no new params.
   Lets cross-attention layers adaptively interpolate between raw GNN output
   and self-attention-enriched representation.

3. Ordinal label smoothing (training only, in train_v11m3.py):
   Contact/Near/Far is an ordinal 3-class label. Uniform smoothing sends
   mass to non-adjacent classes (Contact↔Far). Ordinal smoothing restricts
   mass to adjacent classes only (Contact↔Near, Near↔Far).

繼承 V11:
  Per-layer distance bias (protein), Distance regression aux head,
  Dynamic Ligand Global Token, Ligand Internal Self-Attention,
  GVP Output Dropout, Coordinate Noise, RoPE, AttnRes
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m2 import PocketGVPFusionV11m2


class PocketGVPFusionV11m3(PocketGVPFusionV11m2):
    """
    V11m3: V11m2 plus optional Uni-Mol ligand atom feature augmentation.

    Compatibility:
    - Existing callers may omit unimol_embed; a zero embedding is used.
    - Existing collate/dataset stay unchanged; augmentation happens inside
      this model before Ligand3DEncoder.
    - The underlying Ligand3DEncoder input dim is expanded only in V11m3.
    """

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m2.DEFAULT_CONFIG,
        'use_unimol_ligand_input': True,
        'd_unimol': 512,
        'd_unimol_ligand_proj': 64,
        'unimol_ligand_alpha_init': 0.02,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.raw_ligand_atom_dim = int(cfg['d_ligand_atom'])
        self.use_unimol_ligand_input = bool(cfg.get('use_unimol_ligand_input', True))
        self.d_unimol = int(cfg.get('d_unimol', 512))
        self.d_unimol_ligand_proj = int(cfg.get('d_unimol_ligand_proj', 64))

        super_cfg = dict(cfg)
        if self.use_unimol_ligand_input:
            super_cfg['d_ligand_atom'] = self.raw_ligand_atom_dim + self.d_unimol_ligand_proj
        super().__init__(super_cfg)

        # Restore explicit metadata for diagnostics while keeping the actual
        # Ligand3DEncoder input dimension in self.config['d_ligand_atom'].
        self.config['raw_ligand_atom_dim'] = self.raw_ligand_atom_dim
        self.config['d_unimol'] = self.d_unimol
        self.config['d_unimol_ligand_proj'] = self.d_unimol_ligand_proj
        self.config['use_unimol_ligand_input'] = self.use_unimol_ligand_input

        if self.use_unimol_ligand_input:
            self.unimol_ligand_proj = nn.Sequential(
                nn.LayerNorm(self.d_unimol),
                nn.Linear(self.d_unimol, self.d_unimol_ligand_proj),
                nn.GELU(),
                nn.LayerNorm(self.d_unimol_ligand_proj),
            )
            self.unimol_ligand_alpha = nn.Parameter(
                torch.tensor(float(cfg.get('unimol_ligand_alpha_init', 0.02)))
            )
        else:
            self.unimol_ligand_proj = None
            self.register_buffer('unimol_ligand_alpha', torch.tensor(0.0), persistent=False)

    def _augment_ligand_x(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        unimol_embed: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if not self.use_unimol_ligand_input:
            return ligand_x

        expected_aug_dim = self.raw_ligand_atom_dim + self.d_unimol_ligand_proj
        if ligand_x.size(-1) == expected_aug_dim:
            return ligand_x
        if ligand_x.size(-1) != self.raw_ligand_atom_dim:
            raise ValueError(
                f"Unexpected ligand_x dim={ligand_x.size(-1)}; expected "
                f"{self.raw_ligand_atom_dim} or {expected_aug_dim}"
            )

        if unimol_embed is None:
            unimol_embed = torch.zeros(
                batch_size, self.d_unimol, device=ligand_x.device, dtype=ligand_x.dtype
            )
        else:
            unimol_embed = unimol_embed.to(device=ligand_x.device, dtype=ligand_x.dtype)
            if unimol_embed.shape != (batch_size, self.d_unimol):
                raise ValueError(
                    f"unimol_embed shape={tuple(unimol_embed.shape)}, "
                    f"expected ({batch_size}, {self.d_unimol})"
                )

        if ligand_x.size(0) == 0:
            unimol_atom = ligand_x.new_zeros((0, self.d_unimol_ligand_proj))
        else:
            if ligand_batch.min().item() < 0 or ligand_batch.max().item() >= batch_size:
                raise ValueError(
                    f"ligand_batch out of range for batch_size={batch_size}: "
                    f"min={ligand_batch.min().item()} max={ligand_batch.max().item()}"
                )
            unimol_atom = self.unimol_ligand_proj(unimol_embed)[ligand_batch]
            unimol_atom = self.unimol_ligand_alpha.to(dtype=ligand_x.dtype) * unimol_atom

        self._cached_unimol_ligand_delta = unimol_atom.detach()
        return torch.cat([ligand_x, unimol_atom], dim=-1)

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
        ligand_x = self._augment_ligand_x(
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            unimol_embed=unimol_embed,
            batch_size=esm_embeddings.size(0),
        )
        return super().forward(
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
        )
