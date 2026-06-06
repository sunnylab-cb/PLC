"""
Pocket-GVP-Fusion V11m9: structured local ligand evidence probes.

V11m9 keeps the V11m8 all_light trunk unchanged and tests two low-risk ways to
make local54 chemistry more learnable without changing UniMol/global gating:

- group_kv_delta: add bounded group-specific local54 key/value residuals to the
  existing local_delta atom attention.
- residue_chem_compat: add a bounded residue-conditioned chemistry readout from
  base/electronic/polar/flex local54 groups.

Both probes are default-off and zero-leak at initialization.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import (
    PocketGVPFusionV11m8,
    _LOCAL54_GROUP_RANGES,
    _atanh_ratio,
)


_DEFAULT_EVIDENCE_GROUPS = ("base33", "electronic3", "sasa_hbond4", "flex_rot5")


def _group_dim(group: str) -> int:
    return sum(end - start for start, end in _LOCAL54_GROUP_RANGES[group])


def _parse_groups(value) -> Tuple[str, ...]:
    if value is None:
        return _DEFAULT_EVIDENCE_GROUPS
    if isinstance(value, str):
        groups = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        groups = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [group for group in groups if group not in _LOCAL54_GROUP_RANGES]
    if unknown:
        raise ValueError(f"unknown local54 evidence groups: {unknown}")
    return groups or _DEFAULT_EVIDENCE_GROUPS


class PocketGVPFusionV11m9(PocketGVPFusionV11m8):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m8.DEFAULT_CONFIG,
        "use_local_evidence_group_kv_delta": False,
        "local_evidence_groups": list(_DEFAULT_EVIDENCE_GROUPS),
        "local_evidence_group_kv_alpha_init": 0.05,
        "local_evidence_group_kv_alpha_max": 0.30,
        "use_local_evidence_residue_chem_compat": False,
        "local_evidence_compat_hidden": 128,
        "local_evidence_compat_alpha_init": 0.05,
        "local_evidence_compat_alpha_max": 0.30,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        d_model = int(self.config["d_model"])
        self.local_evidence_groups = _parse_groups(cfg.get("local_evidence_groups"))

        self.use_local_evidence_group_kv_delta = bool(
            cfg.get("use_local_evidence_group_kv_delta", False)
        )
        self.local_evidence_group_kv_alpha_max = float(
            cfg.get("local_evidence_group_kv_alpha_max", 0.30)
        )
        if self.use_local_evidence_group_kv_delta:
            self.local_evidence_group_k = nn.ModuleDict()
            self.local_evidence_group_v = nn.ModuleDict()
            for group in self.local_evidence_groups:
                dim = _group_dim(group)
                k_proj = nn.Sequential(
                    nn.LayerNorm(dim, elementwise_affine=False),
                    nn.Linear(dim, self.local_delta_dim, bias=False),
                )
                v_proj = nn.Sequential(
                    nn.LayerNorm(dim, elementwise_affine=False),
                    nn.Linear(dim, d_model, bias=False),
                )
                nn.init.zeros_(k_proj[-1].weight)
                nn.init.zeros_(v_proj[-1].weight)
                self.local_evidence_group_k[group] = k_proj
                self.local_evidence_group_v[group] = v_proj
            init = float(cfg.get("local_evidence_group_kv_alpha_init", 0.05))
            self.local_evidence_group_kv_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local_evidence_group_kv_alpha_max))
            )
        else:
            self.local_evidence_group_k = None
            self.local_evidence_group_v = None
            self.register_buffer(
                "local_evidence_group_kv_raw_alpha", torch.tensor(0.0), persistent=False
            )

        self.use_local_evidence_residue_chem_compat = bool(
            cfg.get("use_local_evidence_residue_chem_compat", False)
        )
        self.local_evidence_compat_alpha_max = float(
            cfg.get("local_evidence_compat_alpha_max", 0.30)
        )
        if self.use_local_evidence_residue_chem_compat:
            self.local_evidence_group_value = nn.ModuleDict()
            for group in self.local_evidence_groups:
                dim = _group_dim(group)
                self.local_evidence_group_value[group] = nn.Sequential(
                    nn.LayerNorm(dim, elementwise_affine=False),
                    nn.Linear(dim, d_model, bias=False),
                )
            hidden = int(cfg.get("local_evidence_compat_hidden", 128))
            self.local_evidence_compat_gate = nn.Sequential(
                nn.LayerNorm(d_model + d_model + self.local_delta_local_dim),
                nn.Linear(d_model + d_model + self.local_delta_local_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, len(self.local_evidence_groups)),
            )
            nn.init.zeros_(self.local_evidence_compat_gate[-1].weight)
            nn.init.zeros_(self.local_evidence_compat_gate[-1].bias)
            init = float(cfg.get("local_evidence_compat_alpha_init", 0.05))
            self.local_evidence_compat_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local_evidence_compat_alpha_max))
            )
        else:
            self.local_evidence_group_value = None
            self.local_evidence_compat_gate = None
            self.register_buffer(
                "local_evidence_compat_raw_alpha", torch.tensor(0.0), persistent=False
            )

        self.config["use_local_evidence_group_kv_delta"] = self.use_local_evidence_group_kv_delta
        self.config["use_local_evidence_residue_chem_compat"] = (
            self.use_local_evidence_residue_chem_compat
        )
        self.config["local_evidence_groups"] = list(self.local_evidence_groups)

    def _local54_group(self, local54: torch.Tensor, group: str, dtype: torch.dtype) -> torch.Tensor:
        return self._extract_local54_group(local54, group).to(dtype=dtype)

    def _apply_group_kv_delta(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local_evidence_group_k is None or self.local_evidence_group_v is None:
            return k, v
        k_res = torch.zeros_like(k)
        v_res = torch.zeros_like(v)
        for group in self.local_evidence_groups:
            group_x = self._local54_group(local54, group, k.dtype)
            k_res = k_res + self.local_evidence_group_k[group](group_x)
            v_res = v_res + self.local_evidence_group_v[group](group_x).to(dtype=v.dtype)
        alpha = self._bounded_scale(
            self.local_evidence_group_kv_raw_alpha.to(dtype=k.dtype),
            self.local_evidence_group_kv_alpha_max,
        )
        return k + alpha * k_res, v + alpha.to(dtype=v.dtype) * v_res

    def _apply_residue_chem_compat(
        self,
        hp: torch.Tensor,
        lig_read: torch.Tensor,
        local54: torch.Tensor,
        ligand_context: torch.Tensor,
        attn: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_evidence_compat_gate is None or self.local_evidence_group_value is None:
            return lig_read

        group_reads = []
        for group in self.local_evidence_groups:
            group_x = self._local54_group(local54, group, lig_read.dtype)
            group_v = self.local_evidence_group_value[group](group_x).to(dtype=lig_read.dtype)
            group_reads.append(torch.matmul(attn, group_v))
        reads = torch.stack(group_reads, dim=-2)

        context = ligand_context.to(dtype=hp.dtype).unsqueeze(1).expand(-1, hp.size(1), -1)
        gate_in = torch.cat([hp, context], dim=-1)
        gates = torch.tanh(self.local_evidence_compat_gate(gate_in)).to(dtype=reads.dtype)
        compat_read = (reads * gates.unsqueeze(-1)).sum(dim=-2)
        alpha = self._bounded_scale(
            self.local_evidence_compat_raw_alpha.to(dtype=lig_read.dtype),
            self.local_evidence_compat_alpha_max,
        )
        mask = protein_mask.unsqueeze(-1).to(dtype=lig_read.dtype)

        valid = protein_mask.bool()
        zero = hp.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m9_stats["local_evidence_compat_gate_abs_mean"] = (
            gates.detach()[valid].abs().mean() if valid.any() else zero
        )
        self._cached_v11m9_stats["local_evidence_compat_read_abs_mean"] = (
            compat_read.detach()[valid].abs().mean() if valid.any() else zero
        )
        return lig_read + alpha * compat_read * mask

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """V11m8 all_light local delta plus optional structured evidence residuals."""
        self._cached_v11m8_stats = {}
        self._cached_v11m9_stats = {}
        self._cached_center_aux_logits = None

        if not self.use_local_delta_only:
            return h_prot.new_zeros((*h_prot.shape[:2], int(self.config["n_classes"])))

        max_atoms = ligand_mask.size(1)
        h_lig_atoms = h_ligand_fused[:, 1 : 1 + max_atoms, :]
        local54 = self._pad_ligand_local_features(ligand_x, ligand_batch, ligand_mask)

        if self.local_delta_detach_protein:
            hp_src = h_prot.detach()
        elif self.local_delta_protein_grad_scale < 1.0:
            scale = max(0.0, self.local_delta_protein_grad_scale)
            hp_src = h_prot.detach() + (h_prot - h_prot.detach()) * scale
        else:
            hp_src = h_prot
        hl_src = h_lig_atoms.detach() if self.local_delta_detach_ligand else h_lig_atoms
        hl_src, local54 = self._apply_source_dropout(hl_src, local54)
        local54 = self._apply_local54_chemistry(local54)

        ligand_context = self._masked_ligand_context(hl_src, local54, ligand_mask)
        hp_src = self._apply_local_ligand_film(hp_src, ligand_context, protein_mask)

        hp = self.local_delta_prot_norm(hp_src)
        hl = self.local_delta_lig_norm(hl_src)
        lf = self._encode_local_delta_features(local54, hl.dtype)

        q = self.local_delta_q(hp)
        k = torch.zeros((*hl.shape[:2], self.local_delta_dim), device=hl.device, dtype=hl.dtype)
        v = torch.zeros_like(hl)
        if self.local_delta_use_ligand_hidden:
            k = k + self.local_delta_k_lig(hl)
            v = v + self.local_delta_v_lig(hl)
        if self.local_delta_use_local_features:
            k = k + self.local_delta_k_local(lf)
            v = v + self.local_delta_v_local(lf).to(dtype=hl.dtype)
        k, v = self._apply_group_kv_delta(k, v, local54)

        if self.local_delta_use_qk_norm:
            q_score = F.normalize(q.float(), dim=-1)
            k_score = F.normalize(k.float(), dim=-1)
            qk_scale = self.local_delta_qk_log_scale.exp().clamp(max=self.local_delta_score_clip)
            scores = torch.matmul(q_score, k_score.transpose(-2, -1)) * qk_scale
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.local_delta_dim)
        scores = scores.float().clamp(-self.local_delta_score_clip, self.local_delta_score_clip)

        atom_mask = ligand_mask.bool()
        valid_pair_mask = protein_mask.bool().unsqueeze(-1) & atom_mask.unsqueeze(1)
        valid_scores = scores[valid_pair_mask]
        score_mask = atom_mask.unsqueeze(1)
        scores = scores.masked_fill(~score_mask, -1.0e4)
        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))

        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        attn = attn * atom_mask.unsqueeze(1).to(dtype=attn.dtype)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = torch.where(has_atom, attn / denom, torch.zeros_like(attn))

        lig_read = torch.matmul(attn, v)
        lig_read = self._apply_residue_chem_compat(
            hp=hp,
            lig_read=lig_read,
            local54=local54,
            ligand_context=ligand_context,
            attn=attn,
            protein_mask=protein_mask,
        )
        lig_read = self._apply_pocket_virtual_tokens(hp, lig_read, ligand_context, protein_mask)

        masked_scores = scores.masked_fill(~score_mask, -1.0e4)
        pair_max = masked_scores.max(dim=-1).values
        pair_lse = torch.logsumexp(masked_scores, dim=-1)
        n_valid = atom_mask.sum(dim=1).clamp(min=1).to(dtype=pair_lse.dtype).view(-1, 1)
        pair_lse = pair_lse - n_valid.log()
        entropy = -(attn.float() * attn.float().clamp(min=1e-8).log()).sum(dim=-1)
        pair_exp = (attn.float() * scores).sum(dim=-1)

        no_atom_rows = ~has_atom.squeeze(-1)
        pair_max = torch.where(no_atom_rows, torch.zeros_like(pair_max), pair_max)
        pair_lse = torch.where(no_atom_rows, torch.zeros_like(pair_lse), pair_lse)
        pair_exp = torch.where(no_atom_rows, torch.zeros_like(pair_exp), pair_exp)

        stats = torch.stack([pair_max, pair_lse, entropy, pair_exp], dim=-1).to(dtype=h_prot.dtype)
        delta_in = torch.cat([hp_src.to(dtype=h_prot.dtype), lig_read.to(dtype=h_prot.dtype), stats], dim=-1)
        delta_logits = self.local_delta_head(delta_in)

        if self.center_aux_head is not None:
            center_aux_logits = self.center_aux_head(delta_in).squeeze(-1)
            center_aux_logits = center_aux_logits * protein_mask.to(dtype=center_aux_logits.dtype)
            center_scale = self._bounded_scale(
                self.center_aux_raw_scale.to(dtype=delta_logits.dtype),
                self.center_aux_logit_scale_max,
            )
            delta_logits = delta_logits.clone()
            delta_logits[..., 0] = delta_logits[..., 0] + center_scale * center_aux_logits
            self._cached_center_aux_logits = center_aux_logits.detach()
            self._cached_v11m8_stats["center_aux_abs_mean"] = (
                center_aux_logits.detach()[protein_mask.bool()].abs().mean()
                if protein_mask.bool().any()
                else h_prot.new_tensor(0.0, dtype=torch.float32)
            )

        delta_logits = delta_logits * protein_mask.unsqueeze(-1).to(dtype=delta_logits.dtype)

        valid_residue_mask = protein_mask.bool() & atom_mask.any(dim=1).view(-1, 1)
        valid_entropy = entropy[valid_residue_mask]
        valid_delta_logits = delta_logits[protein_mask.bool()]
        zero = h_prot.new_tensor(0.0, dtype=torch.float32)
        self._cached_local_delta_stats = {
            "local_delta_score_abs_mean": (
                valid_scores.detach().abs().mean() if valid_scores.numel() else zero
            ),
            "local_delta_attn_entropy_mean": (
                valid_entropy.detach().mean() if valid_entropy.numel() else zero
            ),
            "local_delta_logits_abs_mean": (
                valid_delta_logits.detach().abs().mean() if valid_delta_logits.numel() else zero
            ),
            "local_delta_qk_scale": self.local_delta_qk_log_scale.detach().exp(),
        }
        return delta_logits

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = super().forward(*args, **kwargs)
        outputs["local_evidence_group_kv_alpha"] = self._bounded_scale(
            self.local_evidence_group_kv_raw_alpha.detach(),
            self.local_evidence_group_kv_alpha_max,
        )
        outputs["local_evidence_compat_alpha"] = self._bounded_scale(
            self.local_evidence_compat_raw_alpha.detach(),
            self.local_evidence_compat_alpha_max,
        )
        for key, value in getattr(self, "_cached_v11m9_stats", {}).items():
            outputs[key] = value
        return outputs
