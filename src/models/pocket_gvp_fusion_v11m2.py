"""
Pocket-GVP-Fusion V11m2: Ligand Distance Bias + b_lig1 AttnRes Anchor.

V11m2 adds over V11:
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

3. Ordinal label smoothing (training only, in train_v11m2.py):
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

from .pocket_gvp_fusion_v11 import PocketGVPFusionV11


class PocketGVPFusionV11m2(PocketGVPFusionV11):
    """
    V11m2: Ligand distance bias + b_lig1 AttnRes anchor.
    All other modules identical to V11; changes activated via config flags.
    """

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11.DEFAULT_CONFIG,
        'n_ligand_self_attn_layers': 2,   # V11m2 requires self-attn for dist_bias / b_lig1
        # V11m2 features ON by default
        'use_lig_dist_bias': True,
        'lig_dist_bias_n_rbf': 8,
        'lig_dist_bias_max_dist': 8.0,
        'use_lig_attnres_b1': True,
    }
