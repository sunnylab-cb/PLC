"""
Pocket-GVP-Fusion V11m5t: V11m4t + balanced ligand input adapters.

The V11m5 local pair-bias path did not improve the trend, so V11m5t is now
repurposed as the cleaner follow-up to V11m4t:

1. Keep V11m4t's stable late local-pair delta and global55/UniMol calibration
   gate.
2. Replace the raw ligand input concatenation
      [local54 | global55 | UniMol64] -> Ligand3DEncoder.node_proj
   with group-wise adapters:
      local54  -> LN/MLP/LN -> local_proj
      global55 -> LN/MLP/LN -> global_proj
      UniMol   -> LN/MLP/LN -> unimol_proj
   followed by learnable positive group scales and a final LayerNorm.

This is not a permanent equal-weight constraint.  It only makes the training
start numerically comparable across ligand sources so the model can learn the
relative source weights instead of inheriting them from raw feature scale.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m4t import PocketGVPFusionV11m4t


def _rms_per_channel(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.detach().float().norm(dim=-1).mean() / math.sqrt(max(int(x.size(-1)), 1))


class PocketGVPFusionV11m5t(PocketGVPFusionV11m4t):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m4t.DEFAULT_CONFIG,
        "use_ligand_groupwise_input_balance": True,
        "ligand_source_atom_dim": 109,
        "ligand_local_dim": 54,
        "ligand_global_dim": 55,
        "ligand_group_local_proj_dim": 96,
        "ligand_group_global_proj_dim": 64,
        "ligand_group_unimol_proj_dim": 64,
        "ligand_group_output_norm": True,
        "ligand_group_scale_init": 1.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.use_ligand_groupwise_input_balance = bool(
            cfg.get("use_ligand_groupwise_input_balance", True)
        )

        if self.use_ligand_groupwise_input_balance:
            source_dim = int(cfg.get("ligand_source_atom_dim", cfg.get("d_ligand_atom", 109)))
            local_dim = int(cfg.get("ligand_local_dim", 54))
            global_dim = int(cfg.get("ligand_global_dim", source_dim - local_dim))
            local_proj = int(cfg.get("ligand_group_local_proj_dim", 96))
            global_proj = int(cfg.get("ligand_group_global_proj_dim", 64))
            unimol_proj = int(cfg.get("ligand_group_unimol_proj_dim", 64))

            # Parent V11m3 builds Ligand3DEncoder with
            # d_ligand_atom + d_unimol_ligand_proj.  Here d_ligand_atom means
            # the balanced local+global adapter output, not the raw V5 input.
            parent_cfg = dict(cfg)
            parent_cfg["d_ligand_atom"] = local_proj + global_proj
            parent_cfg["d_unimol_ligand_proj"] = unimol_proj
            super().__init__(parent_cfg)

            self.ligand_source_atom_dim = source_dim
            self.ligand_local_dim = local_dim
            self.ligand_global_dim = global_dim
            self.ligand_group_local_proj_dim = local_proj
            self.ligand_group_global_proj_dim = global_proj
            self.ligand_group_unimol_proj_dim = unimol_proj
            self.ligand_group_total_dim = local_proj + global_proj + unimol_proj

            self.ligand_local_adapter = nn.Sequential(
                nn.LayerNorm(local_dim),
                nn.Linear(local_dim, local_proj),
                nn.GELU(),
                nn.LayerNorm(local_proj),
            )
            self.ligand_global_adapter = nn.Sequential(
                nn.LayerNorm(global_dim),
                nn.Linear(global_dim, global_proj),
                nn.GELU(),
                nn.LayerNorm(global_proj),
            )

            init_scale = float(cfg.get("ligand_group_scale_init", 1.0))
            self.ligand_group_log_scales = nn.Parameter(
                torch.full((3,), math.log(max(init_scale, 1e-6)))
            )
            self.ligand_group_output_norm = (
                nn.LayerNorm(self.ligand_group_total_dim)
                if bool(cfg.get("ligand_group_output_norm", True))
                else nn.Identity()
            )

            # Metadata for diagnostics/checkpoint config.
            self.raw_ligand_atom_dim = source_dim
            self.config["use_ligand_groupwise_input_balance"] = True
            self.config["ligand_source_atom_dim"] = source_dim
            self.config["ligand_local_dim"] = local_dim
            self.config["ligand_global_dim"] = global_dim
            self.config["ligand_group_local_proj_dim"] = local_proj
            self.config["ligand_group_global_proj_dim"] = global_proj
            self.config["ligand_group_unimol_proj_dim"] = unimol_proj
            self.config["ligand_group_total_dim"] = self.ligand_group_total_dim
        else:
            super().__init__(cfg)
            self.ligand_group_output_norm = nn.Identity()

    def _group_scales(self) -> torch.Tensor:
        return self.ligand_group_log_scales.clamp(-3.0, 3.0).exp()

    def _augment_ligand_x(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        unimol_embed: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if not self.use_ligand_groupwise_input_balance:
            return super()._augment_ligand_x(ligand_x, ligand_batch, unimol_embed, batch_size)

        # Already adapted, useful for smoke tests that call the base forward
        # with cached/generated augmented features.
        if ligand_x.size(-1) == self.ligand_group_total_dim:
            return ligand_x

        if ligand_x.size(-1) != self.ligand_source_atom_dim:
            raise ValueError(
                f"Unexpected ligand_x dim={ligand_x.size(-1)}; expected raw "
                f"{self.ligand_source_atom_dim} or adapted {self.ligand_group_total_dim}"
            )

        local = ligand_x[:, : self.ligand_local_dim]
        global55 = ligand_x[
            :, self.ligand_local_dim : self.ligand_local_dim + self.ligand_global_dim
        ]

        local_z = self.ligand_local_adapter(local)
        global_z = self.ligand_global_adapter(global55)

        if unimol_embed is None:
            unimol_embed = torch.zeros(
                batch_size,
                self.d_unimol,
                device=ligand_x.device,
                dtype=ligand_x.dtype,
            )
        else:
            unimol_embed = unimol_embed.to(device=ligand_x.device, dtype=ligand_x.dtype)
            if unimol_embed.shape != (batch_size, self.d_unimol):
                raise ValueError(
                    f"unimol_embed shape={tuple(unimol_embed.shape)}, "
                    f"expected ({batch_size}, {self.d_unimol})"
                )

        if ligand_x.size(0) == 0:
            unimol_z = ligand_x.new_zeros((0, self.ligand_group_unimol_proj_dim))
        else:
            if ligand_batch.min().item() < 0 or ligand_batch.max().item() >= batch_size:
                raise ValueError(
                    f"ligand_batch out of range for batch_size={batch_size}: "
                    f"min={ligand_batch.min().item()} max={ligand_batch.max().item()}"
                )
            unimol_z = self.unimol_ligand_proj(unimol_embed)[ligand_batch]

        scales = self._group_scales().to(device=ligand_x.device, dtype=ligand_x.dtype)
        local_z = scales[0] * local_z
        global_z = scales[1] * global_z
        unimol_z = scales[2] * unimol_z

        with torch.no_grad():
            self._cached_ligand_group_stats = {
                "ligand_raw_local54_rms": _rms_per_channel(local),
                "ligand_raw_global55_rms": _rms_per_channel(global55),
                "ligand_group_local_proj_rms": _rms_per_channel(local_z),
                "ligand_group_global_proj_rms": _rms_per_channel(global_z),
                "ligand_group_unimol_proj_rms": _rms_per_channel(unimol_z),
                "ligand_group_local_scale": scales[0].detach().float(),
                "ligand_group_global_scale": scales[1].detach().float(),
                "ligand_group_unimol_scale": scales[2].detach().float(),
            }
            self._cached_unimol_ligand_delta = unimol_z.detach()

        out = torch.cat([local_z, global_z, unimol_z], dim=-1)
        return self.ligand_group_output_norm(out)
