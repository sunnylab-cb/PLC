"""
Pocket-GVP-Fusion V11m4tp: V11m4t with probe-only ligand input diagnostics.

This model is intentionally behavior-identical to V11m4t.  It adds no trainable
parameters and does not change logits.  The only extra work is caching the old
raw-concat ligand entry contributions:

    local54  -> W[:, 0:54]   x_local
    global55 -> W[:, 54:109] x_global
    UniMol64 -> W[:,109:173] x_unimol

These metrics are logged by the shared V11m3 training loop through
``_cached_ligand_group_stats`` and provide a direct control for the balanced
V11m5t input adapters.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m4t import PocketGVPFusionV11m4t


def _rms_per_channel(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.detach().float().norm(dim=-1).mean() / math.sqrt(max(int(x.size(-1)), 1))


class PocketGVPFusionV11m4tp(PocketGVPFusionV11m4t):
    """Probe-only V11m4t control; no parameter/logit changes."""

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m4t.DEFAULT_CONFIG,
        "use_ligand_input_probe": True,
        "ligand_probe_local_dim": 54,
        "ligand_probe_global_dim": 55,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)
        self.use_ligand_input_probe = bool(cfg.get("use_ligand_input_probe", True))
        self.ligand_probe_local_dim = int(cfg.get("ligand_probe_local_dim", 54))
        self.ligand_probe_global_dim = int(cfg.get("ligand_probe_global_dim", 55))
        self.config["use_ligand_input_probe"] = self.use_ligand_input_probe
        self.config["ligand_probe_local_dim"] = self.ligand_probe_local_dim
        self.config["ligand_probe_global_dim"] = self.ligand_probe_global_dim

    def _augment_ligand_x(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        unimol_embed: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        x_aug = super()._augment_ligand_x(
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            unimol_embed=unimol_embed,
            batch_size=batch_size,
        )
        if not self.use_ligand_input_probe:
            return x_aug

        local_end = self.ligand_probe_local_dim
        global_end = local_end + self.ligand_probe_global_dim
        unimol_end = x_aug.size(-1)

        local = x_aug[:, :local_end]
        global55 = x_aug[:, local_end:global_end]
        unimol = x_aug[:, global_end:unimol_end]

        w = self.ligand_encoder.node_proj.weight.detach()
        local_contrib = F.linear(local, w[:, :local_end], None)
        global_contrib = F.linear(global55, w[:, local_end:global_end], None)
        unimol_contrib = F.linear(unimol, w[:, global_end:unimol_end], None)

        with torch.no_grad():
            self._cached_ligand_group_stats = {
                "ligand_raw_local54_rms": _rms_per_channel(ligand_x[:, :local_end]),
                "ligand_raw_global55_rms": _rms_per_channel(ligand_x[:, local_end:global_end]),
                "ligand_group_local_proj_rms": _rms_per_channel(local),
                "ligand_group_global_proj_rms": _rms_per_channel(global55),
                "ligand_group_unimol_proj_rms": _rms_per_channel(unimol),
                "ligand_local_nodeproj_contrib_rms": _rms_per_channel(local_contrib),
                "ligand_global_nodeproj_contrib_rms": _rms_per_channel(global_contrib),
                "ligand_unimol_nodeproj_contrib_rms": _rms_per_channel(unimol_contrib),
            }
        return x_aug
