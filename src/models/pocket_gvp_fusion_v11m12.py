"""
Pocket-GVP-Fusion V11m12: UniMol utilization probes on top of V11m10.

V11m12 keeps the V11m10 local54/graph evidence path intact and adds UniMol as a
separate late branch. The goal is diagnostic: test whether pretrained molecule
representations can provide complementary signal without being washed out by
dense atom/local54 features.

All UniMol branches are bounded and zero-safe: a zero UniMol embedding produces
no memory tokens/gates/logit residuals, so zero_unimol_only dependency remains
interpretable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import _LOCAL54_GROUP_RANGES, _atanh_ratio
from .pocket_gvp_fusion_v11m10 import PocketGVPFusionV11m10


_DEFAULT_UNIMOL_GROUPS = ("electronic3", "sasa_hbond4", "flex_rot5")


def _parse_unimol_groups(value) -> Tuple[str, ...]:
    if value is None:
        groups = _DEFAULT_UNIMOL_GROUPS
    elif isinstance(value, str):
        groups = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        groups = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [group for group in groups if group not in _LOCAL54_GROUP_RANGES]
    if unknown:
        raise ValueError(f"unknown UniMol gate local54 groups: {unknown}")
    return groups or _DEFAULT_UNIMOL_GROUPS


class PocketGVPFusionV11m12(PocketGVPFusionV11m10):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m10.DEFAULT_CONFIG,
        "use_unimol_memory_kv": False,
        "unimol_memory_tokens": 1,
        "unimol_memory_hidden": 192,
        "unimol_memory_alpha_init": 0.03,
        "unimol_memory_alpha_max": 0.20,
        "unimol_memory_score_bias_init": -4.0,
        "use_unimol_group_gate": False,
        "unimol_group_gate_groups": list(_DEFAULT_UNIMOL_GROUPS),
        "unimol_group_gate_hidden": 128,
        "unimol_group_gate_alpha_init": 0.03,
        "unimol_group_gate_alpha_max": 0.25,
        "use_unimol_atom_gate": False,
        "unimol_atom_gate_hidden": 96,
        "unimol_atom_gate_alpha_init": 0.03,
        "unimol_atom_gate_alpha_max": 0.20,
        "use_unimol_residue_compat": False,
        "unimol_residue_compat_hidden": 128,
        "unimol_residue_compat_alpha_init": 0.03,
        "unimol_residue_compat_alpha_max": 0.20,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        d_model = int(self.config["d_model"])
        d_unimol = int(self.config.get("d_unimol", 512))
        local_dim = int(self.local_delta_local_dim)
        n_classes = int(self.config["n_classes"])

        self.use_unimol_memory_kv = bool(cfg.get("use_unimol_memory_kv", False))
        self.unimol_memory_tokens = int(cfg.get("unimol_memory_tokens", 1))
        self.unimol_memory_alpha_max = float(cfg.get("unimol_memory_alpha_max", 0.20))
        if self.use_unimol_memory_kv:
            hidden = int(cfg.get("unimol_memory_hidden", 192))
            out_dim = self.unimol_memory_tokens * (self.local_delta_dim + d_model)
            self.unimol_memory_proj = nn.Sequential(
                nn.LayerNorm(d_unimol, elementwise_affine=False),
                nn.Linear(d_unimol, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden, elementwise_affine=False),
                nn.Linear(hidden, out_dim, bias=False),
            )
            nn.init.zeros_(self.unimol_memory_proj[-1].weight)
            init = float(cfg.get("unimol_memory_alpha_init", 0.03))
            self.unimol_memory_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.unimol_memory_alpha_max))
            )
            self.unimol_memory_score_bias = nn.Parameter(
                torch.tensor(float(cfg.get("unimol_memory_score_bias_init", -4.0)))
            )
        else:
            self.unimol_memory_proj = None
            self.register_buffer("unimol_memory_raw_alpha", torch.tensor(0.0), persistent=False)
            self.register_buffer("unimol_memory_score_bias", torch.tensor(-4.0), persistent=False)

        self.use_unimol_group_gate = bool(cfg.get("use_unimol_group_gate", False))
        self.unimol_group_gate_groups = _parse_unimol_groups(
            cfg.get("unimol_group_gate_groups")
        )
        self.unimol_group_gate_alpha_max = float(cfg.get("unimol_group_gate_alpha_max", 0.25))
        if self.use_unimol_group_gate:
            hidden = int(cfg.get("unimol_group_gate_hidden", 128))
            self.unimol_group_gate = nn.Sequential(
                nn.LayerNorm(d_unimol, elementwise_affine=False),
                nn.Linear(d_unimol, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden, elementwise_affine=False),
                nn.Linear(hidden, len(self.unimol_group_gate_groups), bias=False),
            )
            nn.init.zeros_(self.unimol_group_gate[-1].weight)
            init = float(cfg.get("unimol_group_gate_alpha_init", 0.03))
            self.unimol_group_gate_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.unimol_group_gate_alpha_max))
            )
        else:
            self.unimol_group_gate = None
            self.register_buffer("unimol_group_gate_raw_alpha", torch.tensor(0.0), persistent=False)

        self.use_unimol_atom_gate = bool(cfg.get("use_unimol_atom_gate", False))
        self.unimol_atom_gate_alpha_max = float(cfg.get("unimol_atom_gate_alpha_max", 0.20))
        if self.use_unimol_atom_gate:
            hidden = int(cfg.get("unimol_atom_gate_hidden", 96))
            self.unimol_atom_local_proj = nn.Sequential(
                nn.LayerNorm(local_dim, elementwise_affine=False),
                nn.Linear(local_dim, hidden, bias=False),
                nn.GELU(),
            )
            self.unimol_atom_global_proj = nn.Sequential(
                nn.LayerNorm(d_unimol, elementwise_affine=False),
                nn.Linear(d_unimol, hidden, bias=False),
                nn.GELU(),
            )
            self.unimol_atom_gate = nn.Linear(hidden, 2, bias=False)
            nn.init.zeros_(self.unimol_atom_gate.weight)
            init = float(cfg.get("unimol_atom_gate_alpha_init", 0.03))
            self.unimol_atom_gate_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.unimol_atom_gate_alpha_max))
            )
        else:
            self.unimol_atom_local_proj = None
            self.unimol_atom_global_proj = None
            self.unimol_atom_gate = None
            self.register_buffer("unimol_atom_gate_raw_alpha", torch.tensor(0.0), persistent=False)

        self.use_unimol_residue_compat = bool(cfg.get("use_unimol_residue_compat", False))
        self.unimol_residue_compat_alpha_max = float(
            cfg.get("unimol_residue_compat_alpha_max", 0.20)
        )
        if self.use_unimol_residue_compat:
            hidden = int(cfg.get("unimol_residue_compat_hidden", 128))
            self.unimol_residue_prot_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden, bias=False),
                nn.GELU(),
            )
            self.unimol_residue_lig_proj = nn.Sequential(
                nn.LayerNorm(d_unimol, elementwise_affine=False),
                nn.Linear(d_unimol, hidden, bias=False),
                nn.GELU(),
            )
            self.unimol_residue_head = nn.Linear(hidden, n_classes, bias=False)
            nn.init.zeros_(self.unimol_residue_head.weight)
            init = float(cfg.get("unimol_residue_compat_alpha_init", 0.03))
            self.unimol_residue_compat_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.unimol_residue_compat_alpha_max))
            )
        else:
            self.unimol_residue_prot_proj = None
            self.unimol_residue_lig_proj = None
            self.unimol_residue_head = None
            self.register_buffer(
                "unimol_residue_compat_raw_alpha", torch.tensor(0.0), persistent=False
            )

        self.config["use_unimol_memory_kv"] = self.use_unimol_memory_kv
        self.config["unimol_memory_tokens"] = self.unimol_memory_tokens
        self.config["use_unimol_group_gate"] = self.use_unimol_group_gate
        self.config["unimol_group_gate_groups"] = list(self.unimol_group_gate_groups)
        self.config["use_unimol_atom_gate"] = self.use_unimol_atom_gate
        self.config["use_unimol_residue_compat"] = self.use_unimol_residue_compat
        self._cached_v11m12_stats = {}

    def _cached_unimol(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        unimol = getattr(self, "_cached_v11m10_unimol", None)
        if unimol is None:
            return torch.zeros(
                batch_size,
                int(self.config.get("d_unimol", 512)),
                device=device,
                dtype=dtype,
            )
        return unimol.to(device=device, dtype=dtype)

    @staticmethod
    def _unimol_valid_mask(unimol: torch.Tensor) -> torch.Tensor:
        return unimol.detach().float().abs().sum(dim=-1) > 1e-8

    def _build_unimol_memory(
        self,
        unimol: torch.Tensor,
        atom_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.unimol_memory_proj is None:
            return None, None, None
        raw = self.unimol_memory_proj(unimol).to(dtype=dtype)
        raw = raw.view(
            unimol.size(0),
            self.unimol_memory_tokens,
            self.local_delta_dim + int(self.config["d_model"]),
        )
        alpha = self._bounded_scale(
            self.unimol_memory_raw_alpha.to(dtype=dtype),
            self.unimol_memory_alpha_max,
        )
        mem_k = alpha * raw[..., : self.local_delta_dim]
        mem_v = alpha * raw[..., self.local_delta_dim :]
        valid = atom_mask.any(dim=1) & self._unimol_valid_mask(unimol)
        mem_mask = valid.unsqueeze(1).expand(-1, self.unimol_memory_tokens)

        zero = unimol.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m12_stats["unimol_memory_v_abs_mean"] = (
            mem_v.detach()[mem_mask].float().abs().mean() if mem_mask.any() else zero
        )
        return mem_k, mem_v, mem_mask

    def _unimol_group_gates(self, unimol: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.unimol_group_gate is None:
            return None
        raw = self.unimol_group_gate(unimol).to(dtype=dtype)
        raw = raw * self._unimol_valid_mask(unimol).to(dtype=dtype).unsqueeze(-1)
        gates = torch.tanh(raw)
        zero = unimol.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m12_stats["unimol_group_gate_abs_mean"] = (
            gates.detach().float().abs().mean() if gates.numel() else zero
        )
        return gates

    def _apply_interaction_kv_delta(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.local54_interaction_k is None or self.local54_interaction_v is None:
            return k, v

        k_res = torch.zeros_like(k)
        v_res = torch.zeros_like(v)
        unimol = self._cached_unimol(k.size(0), k.device, k.dtype)
        gates = self._unimol_group_gates(unimol, k.dtype)
        gate_alpha = self._bounded_scale(
            self.unimol_group_gate_raw_alpha.to(dtype=k.dtype),
            self.unimol_group_gate_alpha_max,
        )

        for group in self.local54_interaction_groups:
            feat = self._interaction_features(local54, group, k.dtype)
            group_k = self.local54_interaction_k[group](feat)
            group_v = self.local54_interaction_v[group](feat).to(dtype=v.dtype)
            if gates is not None and group in self.unimol_group_gate_groups:
                idx = self.unimol_group_gate_groups.index(group)
                mult = 1.0 + gate_alpha * gates[:, idx].view(-1, 1, 1)
                group_k = group_k * mult.to(dtype=group_k.dtype)
                group_v = group_v * mult.to(dtype=group_v.dtype)
            k_res = k_res + group_k
            v_res = v_res + group_v

        alpha = self._bounded_scale(
            self.local54_interaction_kv_raw_alpha.to(dtype=k.dtype),
            self.local54_interaction_kv_alpha_max,
        )
        valid = local54.abs().sum(dim=-1) > 0
        zero = k.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m10_stats["local54_interaction_kv_abs_mean"] = (
            v_res.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return k + alpha * k_res, v + alpha.to(dtype=v.dtype) * v_res

    def _apply_unimol_atom_gate(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
        unimol: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.unimol_atom_gate is None:
            return k, v
        local_h = self.unimol_atom_local_proj(local54.to(dtype=k.dtype))
        unimol_h = self.unimol_atom_global_proj(unimol.to(dtype=k.dtype)).unsqueeze(1)
        gates = torch.tanh(self.unimol_atom_gate(local_h * unimol_h)).to(dtype=k.dtype)
        valid_unimol = self._unimol_valid_mask(unimol).to(dtype=k.dtype).view(-1, 1, 1)
        gates = gates * valid_unimol
        alpha = self._bounded_scale(
            self.unimol_atom_gate_raw_alpha.to(dtype=k.dtype),
            self.unimol_atom_gate_alpha_max,
        )
        k = k * (1.0 + alpha * gates[..., :1])
        v = v * (1.0 + alpha.to(dtype=v.dtype) * gates[..., 1:2].to(dtype=v.dtype))

        valid = atom_mask.bool()
        zero = k.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m12_stats["unimol_atom_gate_abs_mean"] = (
            gates.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return k, v

    def _apply_unimol_residue_compat(
        self,
        delta_logits: torch.Tensor,
        hp: torch.Tensor,
        unimol: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.unimol_residue_head is None:
            return delta_logits
        hp_h = self.unimol_residue_prot_proj(hp.to(dtype=delta_logits.dtype))
        u_h = self.unimol_residue_lig_proj(unimol.to(dtype=delta_logits.dtype)).unsqueeze(1)
        compat = self.unimol_residue_head(hp_h * u_h).to(dtype=delta_logits.dtype)
        valid_unimol = self._unimol_valid_mask(unimol).to(dtype=delta_logits.dtype).view(-1, 1, 1)
        compat = compat * valid_unimol
        alpha = self._bounded_scale(
            self.unimol_residue_compat_raw_alpha.to(dtype=delta_logits.dtype),
            self.unimol_residue_compat_alpha_max,
        )
        valid = protein_mask.bool()
        zero = delta_logits.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m12_stats["unimol_residue_compat_abs_mean"] = (
            compat.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return delta_logits + alpha * compat

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """V11m10 local delta plus optional UniMol late-branch probes."""
        self._cached_v11m8_stats = {}
        self._cached_v11m10_stats = {}
        self._cached_v11m12_stats = {}
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
        ligand_context = self._apply_ligand_context_prior(ligand_context)
        hp_src = self._apply_local_ligand_film(hp_src, ligand_context, protein_mask)

        hp = self.local_delta_prot_norm(hp_src)
        hl = self.local_delta_lig_norm(hl_src)
        lf = self._encode_local_delta_features(local54, hl.dtype)
        atom_mask = ligand_mask.bool()
        unimol = self._cached_unimol(hp.size(0), hp.device, hp.dtype)

        q = self.local_delta_q(hp)
        k = torch.zeros((*hl.shape[:2], self.local_delta_dim), device=hl.device, dtype=hl.dtype)
        v = torch.zeros_like(hl)
        if self.local_delta_use_ligand_hidden:
            k = k + self.local_delta_k_lig(hl)
            v = v + self.local_delta_v_lig(hl)
        if self.local_delta_use_local_features:
            k = k + self.local_delta_k_local(lf)
            v = v + self.local_delta_v_local(lf).to(dtype=hl.dtype)
        k, v = self._apply_interaction_kv_delta(k, v, local54)
        k, v = self._apply_unimol_atom_gate(k, v, local54, unimol, atom_mask)

        kv_mask = atom_mask
        memory_score_bias = None
        mem_k, mem_v, mem_mask = self._build_unimol_memory(unimol, atom_mask, hl.dtype)
        if mem_k is not None and mem_v is not None and mem_mask is not None:
            k = torch.cat([k, mem_k.to(dtype=k.dtype)], dim=1)
            v = torch.cat([v, mem_v.to(dtype=v.dtype)], dim=1)
            kv_mask = torch.cat([kv_mask, mem_mask.bool()], dim=1)
            memory_score_bias = self.unimol_memory_score_bias.to(device=k.device, dtype=torch.float32)

        if self.local_delta_use_qk_norm:
            q_score = F.normalize(q.float(), dim=-1)
            k_score = F.normalize(k.float(), dim=-1)
            qk_scale = self.local_delta_qk_log_scale.exp().clamp(max=self.local_delta_score_clip)
            scores = torch.matmul(q_score, k_score.transpose(-2, -1)) * qk_scale
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.local_delta_dim)
        scores = scores.float().clamp(-self.local_delta_score_clip, self.local_delta_score_clip)
        if memory_score_bias is not None and self.unimol_memory_tokens > 0:
            scores = scores.clone()
            scores[:, :, -self.unimol_memory_tokens :] = (
                scores[:, :, -self.unimol_memory_tokens :] + memory_score_bias
            )

        valid_pair_mask = protein_mask.bool().unsqueeze(-1) & kv_mask.unsqueeze(1)
        valid_scores = scores[valid_pair_mask]
        score_mask = kv_mask.unsqueeze(1)
        scores = scores.masked_fill(~score_mask, -1.0e4)
        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))

        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        attn = attn * kv_mask.unsqueeze(1).to(dtype=attn.dtype)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = torch.where(has_atom, attn / denom, torch.zeros_like(attn))

        lig_read = torch.matmul(attn, v)
        attn_atoms = attn[:, :, :max_atoms]
        lig_read = self._apply_residue_chem_compat(
            hp=hp,
            lig_read=lig_read,
            local54=local54,
            ligand_context=ligand_context,
            attn=attn_atoms,
            protein_mask=protein_mask,
        )
        lig_read = self._apply_pocket_virtual_tokens(hp, lig_read, ligand_context, protein_mask)

        masked_scores = scores.masked_fill(~score_mask, -1.0e4)
        pair_max = masked_scores.max(dim=-1).values
        pair_lse = torch.logsumexp(masked_scores, dim=-1)
        n_valid = kv_mask.sum(dim=1).clamp(min=1).to(dtype=pair_lse.dtype).view(-1, 1)
        pair_lse = pair_lse - n_valid.log()
        entropy = -(attn.float() * attn.float().clamp(min=1e-8).log()).sum(dim=-1)
        pair_exp = (attn.float() * scores).sum(dim=-1)

        no_atom_rows = ~has_atom.squeeze(-1)
        pair_max = torch.where(no_atom_rows, torch.zeros_like(pair_max), pair_max)
        pair_lse = torch.where(no_atom_rows, torch.zeros_like(pair_lse), pair_lse)
        pair_exp = torch.where(no_atom_rows, torch.zeros_like(pair_exp), pair_exp)

        stats = torch.stack([pair_max, pair_lse, entropy, pair_exp], dim=-1).to(dtype=h_prot.dtype)
        delta_in = torch.cat(
            [hp_src.to(dtype=h_prot.dtype), lig_read.to(dtype=h_prot.dtype), stats],
            dim=-1,
        )
        delta_logits = self.local_delta_head(delta_in)
        delta_logits = self._apply_unimol_residue_compat(delta_logits, hp, unimol, protein_mask)

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
        outputs["unimol_memory_alpha"] = self._bounded_scale(
            self.unimol_memory_raw_alpha.detach(),
            self.unimol_memory_alpha_max,
        )
        outputs["unimol_memory_score_bias"] = self.unimol_memory_score_bias.detach()
        outputs["unimol_group_gate_alpha"] = self._bounded_scale(
            self.unimol_group_gate_raw_alpha.detach(),
            self.unimol_group_gate_alpha_max,
        )
        outputs["unimol_atom_gate_alpha"] = self._bounded_scale(
            self.unimol_atom_gate_raw_alpha.detach(),
            self.unimol_atom_gate_alpha_max,
        )
        outputs["unimol_residue_compat_alpha"] = self._bounded_scale(
            self.unimol_residue_compat_raw_alpha.detach(),
            self.unimol_residue_compat_alpha_max,
        )
        for key, value in getattr(self, "_cached_v11m12_stats", {}).items():
            outputs[key] = value
        return outputs
