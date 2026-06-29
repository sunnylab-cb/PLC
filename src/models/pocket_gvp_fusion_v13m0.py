"""Pocket-GVP-Fusion V13m0: residue-atom compatibility bias.

This is a lightweight ESMFold-inspired probe.  ESMFold uses pair states as
dynamic attention biases; V13m0 applies the same principle to PLC without
building a dense protein-protein pair trunk:

    local_delta_score(residue i, atom a)
      += alpha * bias(h_res_i, h_atom_a, local54_a)

The bias is bounded and zero-initialized so a warm-start checkpoint keeps its
original behavior at step 0.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import _LOCAL54_GROUP_RANGES, _atanh_ratio
from .pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13


def _parse_bias_groups(value) -> Tuple[str, ...]:
    if value is None:
        groups: Iterable[str] = ("base33", "electronic3", "sasa_hbond4", "flex_rot5")
    elif isinstance(value, str):
        groups = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        groups = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [g for g in groups if g not in _LOCAL54_GROUP_RANGES]
    if unknown:
        raise ValueError(f"unknown residue_atom_bias_local54_groups: {unknown}")
    return tuple(groups)


def _groups_dim(groups: Iterable[str]) -> int:
    total = 0
    for group in groups:
        for start, end in _LOCAL54_GROUP_RANGES[group]:
            total += end - start
    return total


class PocketGVPFusionV13m0(PocketGVPFusionV11m13):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m13.DEFAULT_CONFIG,
        "use_residue_atom_bias": False,
        "residue_atom_bias_hidden": 48,
        "residue_atom_bias_local54_groups": [
            "base33",
            "electronic3",
            "sasa_hbond4",
            "flex_rot5",
        ],
        "residue_atom_bias_alpha_init": 0.03,
        "residue_atom_bias_alpha_max": 0.18,
        "residue_atom_bias_clip": 6.0,
        "use_residue_atom_triangle_bias": False,
        "residue_atom_triangle_window": 7,
        "residue_atom_triangle_alpha_init": 0.02,
        "residue_atom_triangle_alpha_max": 0.12,
        "use_ligand_slots_delta": False,
        "ligand_slots_num": 16,
        "ligand_slots_hidden": 128,
        "ligand_slots_local54_groups": [
            "base33",
            "electronic3",
            "sasa_hbond4",
            "flex_rot5",
        ],
        "use_local54_qk_reparam": False,
        "local54_qk_reparam_hidden": 96,
        "local54_qk_reparam_alpha_init": 0.03,
        "local54_qk_reparam_alpha_max": 0.20,
        "local54_qk_reparam_local54_groups": [
            "base33",
            "electronic3",
            "sasa_hbond4",
            "flex_rot5",
        ],
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_residue_atom_bias = bool(cfg.get("use_residue_atom_bias", False))
        self.residue_atom_bias_alpha_max = float(cfg.get("residue_atom_bias_alpha_max", 0.18))
        self.residue_atom_bias_clip = float(cfg.get("residue_atom_bias_clip", 6.0))
        self.residue_atom_bias_groups = _parse_bias_groups(
            cfg.get("residue_atom_bias_local54_groups")
        )
        self.use_residue_atom_triangle_bias = bool(
            cfg.get("use_residue_atom_triangle_bias", False)
        )
        self.use_ligand_slots_delta = bool(cfg.get("use_ligand_slots_delta", False))
        self.use_local54_qk_reparam = bool(cfg.get("use_local54_qk_reparam", False))
        self.residue_atom_triangle_window = max(
            1, int(cfg.get("residue_atom_triangle_window", 7))
        )
        if self.residue_atom_triangle_window % 2 == 0:
            self.residue_atom_triangle_window += 1
        self.residue_atom_triangle_alpha_max = float(
            cfg.get("residue_atom_triangle_alpha_max", 0.12)
        )

        d_model = int(self.config["d_model"])
        hidden = int(cfg.get("residue_atom_bias_hidden", 48))
        chem_dim = _groups_dim(self.residue_atom_bias_groups)
        self.ligand_slots_groups = _parse_bias_groups(
            cfg.get("ligand_slots_local54_groups")
        )
        self.local54_qk_reparam_groups = _parse_bias_groups(
            cfg.get("local54_qk_reparam_local54_groups")
        )
        slot_chem_dim = _groups_dim(self.ligand_slots_groups)
        qk_chem_dim = _groups_dim(self.local54_qk_reparam_groups)
        if self.use_residue_atom_bias:
            self.residue_atom_bias_res = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.residue_atom_bias_atom = nn.Sequential(
                nn.LayerNorm(d_model + chem_dim),
                nn.Linear(d_model + chem_dim, hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.residue_atom_bias_out = nn.Sequential(
                nn.LayerNorm(hidden * 3),
                nn.Linear(hidden * 3, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.residue_atom_bias_out[-1].weight)
            nn.init.zeros_(self.residue_atom_bias_out[-1].bias)
            init = float(cfg.get("residue_atom_bias_alpha_init", 0.03))
            self.residue_atom_bias_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.residue_atom_bias_alpha_max))
            )
        else:
            self.residue_atom_bias_res = None
            self.residue_atom_bias_atom = None
            self.residue_atom_bias_out = None
            self.register_buffer(
                "residue_atom_bias_raw_alpha", torch.tensor(0.0), persistent=False
            )

        if self.use_residue_atom_triangle_bias:
            self.residue_atom_triangle_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.residue_atom_triangle_gate[-1].weight)
            nn.init.zeros_(self.residue_atom_triangle_gate[-1].bias)
            init = float(cfg.get("residue_atom_triangle_alpha_init", 0.02))
            self.residue_atom_triangle_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.residue_atom_triangle_alpha_max))
            )
        else:
            self.residue_atom_triangle_gate = None
            self.register_buffer(
                "residue_atom_triangle_raw_alpha", torch.tensor(0.0), persistent=False
            )

        if self.use_ligand_slots_delta:
            n_slots = int(cfg.get("ligand_slots_num", 16))
            slot_hidden = int(cfg.get("ligand_slots_hidden", 128))
            self.ligand_slot_queries = nn.Parameter(
                torch.empty(n_slots, self.local_delta_dim)
            )
            nn.init.normal_(self.ligand_slot_queries, mean=0.0, std=0.02)
            self.ligand_slot_atom_k = nn.Sequential(
                nn.LayerNorm(d_model + slot_chem_dim),
                nn.Linear(d_model + slot_chem_dim, slot_hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(slot_hidden),
                nn.Linear(slot_hidden, self.local_delta_dim, bias=False),
            )
            self.ligand_slot_atom_v = nn.Sequential(
                nn.LayerNorm(d_model + slot_chem_dim),
                nn.Linear(d_model + slot_chem_dim, slot_hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(slot_hidden),
                nn.Linear(slot_hidden, d_model, bias=False),
            )
            self.ligand_slot_to_k = nn.Linear(d_model, self.local_delta_dim, bias=False)
            self.ligand_slot_to_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.ligand_slot_queries = None
            self.ligand_slot_atom_k = None
            self.ligand_slot_atom_v = None
            self.ligand_slot_to_k = None
            self.ligand_slot_to_v = None

        self.local54_qk_reparam_alpha_max = float(
            cfg.get("local54_qk_reparam_alpha_max", 0.20)
        )
        if self.use_local54_qk_reparam:
            qk_hidden = int(cfg.get("local54_qk_reparam_hidden", 96))
            self.local54_q_reparam = nn.Sequential(
                nn.LayerNorm(d_model + self.local_delta_local_dim),
                nn.Linear(d_model + self.local_delta_local_dim, qk_hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(qk_hidden),
                nn.Linear(qk_hidden, self.local_delta_dim, bias=False),
            )
            self.local54_k_reparam = nn.Sequential(
                nn.LayerNorm(d_model + qk_chem_dim),
                nn.Linear(d_model + qk_chem_dim, qk_hidden, bias=False),
                nn.GELU(),
                nn.LayerNorm(qk_hidden),
                nn.Linear(qk_hidden, self.local_delta_dim, bias=False),
            )
            nn.init.zeros_(self.local54_q_reparam[-1].weight)
            nn.init.zeros_(self.local54_k_reparam[-1].weight)
            init = float(cfg.get("local54_qk_reparam_alpha_init", 0.03))
            self.local54_qk_reparam_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local54_qk_reparam_alpha_max))
            )
        else:
            self.local54_q_reparam = None
            self.local54_k_reparam = None
            self.register_buffer(
                "local54_qk_reparam_raw_alpha", torch.tensor(0.0), persistent=False
            )

        self.config["use_residue_atom_bias"] = self.use_residue_atom_bias
        self.config["residue_atom_bias_local54_groups"] = list(self.residue_atom_bias_groups)
        self.config["use_residue_atom_triangle_bias"] = self.use_residue_atom_triangle_bias
        self.config["use_ligand_slots_delta"] = self.use_ligand_slots_delta
        self.config["ligand_slots_local54_groups"] = list(self.ligand_slots_groups)
        self.config["use_local54_qk_reparam"] = self.use_local54_qk_reparam
        self.config["local54_qk_reparam_local54_groups"] = list(
            self.local54_qk_reparam_groups
        )
        self._cached_v13m0_stats: Dict[str, torch.Tensor] = {}

    def _bias_local54_features(self, local54: torch.Tensor) -> torch.Tensor:
        pieces = []
        for group in self.residue_atom_bias_groups:
            pieces.append(self._extract_local54_group(local54, group))
        return torch.cat(pieces, dim=-1)

    def _local54_features_for_groups(
        self,
        local54: torch.Tensor,
        groups: Iterable[str],
    ) -> torch.Tensor:
        pieces = [self._extract_local54_group(local54, group) for group in groups]
        return torch.cat(pieces, dim=-1)

    def _apply_local54_qk_reparam(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hl: torch.Tensor,
        local54: torch.Tensor,
        ligand_context: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local54_q_reparam is None or self.local54_k_reparam is None:
            return q, k

        chem = self._local54_features_for_groups(
            local54, self.local54_qk_reparam_groups
        ).to(dtype=hl.dtype)
        atom_in = torch.cat([hl, chem], dim=-1)
        q_shift = self.local54_q_reparam(ligand_context.float()).unsqueeze(1)
        k_shift = self.local54_k_reparam(atom_in.float()).to(dtype=k.dtype)
        alpha = self._bounded_scale(
            self.local54_qk_reparam_raw_alpha.to(dtype=q.dtype),
            self.local54_qk_reparam_alpha_max,
        )
        atom_valid = ligand_mask.bool()
        zero = q.new_tensor(0.0, dtype=torch.float32)
        self._cached_v13m0_stats["local54_qk_reparam_alpha"] = alpha.detach().float()
        self._cached_v13m0_stats["local54_qk_reparam_q_abs_mean"] = (
            q_shift.detach().float().abs().mean()
        )
        self._cached_v13m0_stats["local54_qk_reparam_k_abs_mean"] = (
            k_shift.detach()[atom_valid].float().abs().mean() if atom_valid.any() else zero
        )
        return q + alpha.to(dtype=q.dtype) * q_shift.to(dtype=q.dtype), k + alpha.to(dtype=k.dtype) * k_shift

    def _build_ligand_slots(
        self,
        hl: torch.Tensor,
        local54: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ligand_slot_queries is None:
            raise RuntimeError("_build_ligand_slots called with slots disabled")
        chem = self._local54_features_for_groups(local54, self.ligand_slots_groups).to(
            dtype=hl.dtype
        )
        atom_in = torch.cat([hl, chem], dim=-1)
        atom_k = self.ligand_slot_atom_k(atom_in.float())
        atom_v = self.ligand_slot_atom_v(atom_in.float()).to(dtype=hl.dtype)
        queries = self.ligand_slot_queries.to(device=hl.device, dtype=atom_k.dtype)
        slot_scores = torch.matmul(queries.unsqueeze(0), atom_k.transpose(-2, -1))
        slot_scores = slot_scores / math.sqrt(self.local_delta_dim)
        atom_mask = ligand_mask.bool()
        slot_scores = slot_scores.masked_fill(~atom_mask.unsqueeze(1), -1.0e4)
        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        slot_scores = torch.where(has_atom, slot_scores, torch.zeros_like(slot_scores))
        slot_attn = torch.softmax(slot_scores.float(), dim=-1).to(dtype=atom_v.dtype)
        slot_attn = slot_attn * atom_mask.unsqueeze(1).to(dtype=slot_attn.dtype)
        denom = slot_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        slot_attn = torch.where(has_atom, slot_attn / denom, torch.zeros_like(slot_attn))
        slots = torch.matmul(slot_attn, atom_v)
        slot_k = self.ligand_slot_to_k(slots)
        slot_v = self.ligand_slot_to_v(slots).to(dtype=hl.dtype)
        slot_mask = atom_mask.any(dim=1, keepdim=True).expand(-1, slots.size(1))
        entropy = -(slot_attn.float() * slot_attn.float().clamp(min=1e-8).log()).sum(dim=-1)
        self._cached_v13m0_stats["ligand_slots_attn_entropy"] = (
            entropy.detach()[slot_mask].mean() if slot_mask.any() else hl.new_tensor(0.0, dtype=torch.float32)
        )
        self._cached_v13m0_stats["ligand_slots_abs_mean"] = (
            slots.detach()[slot_mask].float().abs().mean() if slot_mask.any() else hl.new_tensor(0.0, dtype=torch.float32)
        )
        return slot_k, slot_v, slot_mask

    def _apply_residue_atom_bias(
        self,
        scores: torch.Tensor,
        hp: torch.Tensor,
        hl: torch.Tensor,
        local54: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.residue_atom_bias_res is None or self.residue_atom_bias_atom is None:
            return scores

        res_h = self.residue_atom_bias_res(hp.float())
        chem = self._bias_local54_features(local54).to(dtype=hl.dtype)
        atom_in = torch.cat([hl, chem], dim=-1)
        atom_h = self.residue_atom_bias_atom(atom_in.float())

        pair = torch.cat(
            [
                res_h.unsqueeze(2).expand(-1, -1, atom_h.size(1), -1),
                atom_h.unsqueeze(1).expand(-1, res_h.size(1), -1, -1),
                res_h.unsqueeze(2) * atom_h.unsqueeze(1),
            ],
            dim=-1,
        )
        bias = self.residue_atom_bias_out(pair).squeeze(-1)
        if self.residue_atom_bias_clip > 0:
            bias = bias.clamp(-self.residue_atom_bias_clip, self.residue_atom_bias_clip)
        alpha = self._bounded_scale(
            self.residue_atom_bias_raw_alpha.to(dtype=scores.dtype),
            self.residue_atom_bias_alpha_max,
        )
        valid = protein_mask.bool().unsqueeze(-1) & ligand_mask.bool().unsqueeze(1)
        zero = scores.new_tensor(0.0, dtype=torch.float32)
        self._cached_v13m0_stats["residue_atom_bias_abs_mean"] = (
            bias.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        self._cached_v13m0_stats["residue_atom_bias_alpha"] = alpha.detach().float()
        return scores + alpha.to(dtype=scores.dtype) * bias.to(dtype=scores.dtype)

    def _apply_residue_atom_triangle_bias(
        self,
        scores: torch.Tensor,
        hp: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Lightweight i-j-a triangle coherence over local residue neighborhoods.

        Full AlphaFold/ESMFold triangle attention would require a [L,L,C] pair
        trunk.  Here we use the existing residue-atom score field and add a
        bounded residual from neighboring residues that score the same ligand
        atom.  This tests whether pocket coherence helps assignment without
        changing the protein trunk.
        """
        if self.residue_atom_triangle_gate is None:
            return scores

        batch_size, n_res, n_atoms = scores.shape
        window = self.residue_atom_triangle_window
        pad = window // 2
        dtype = scores.dtype
        weight = torch.ones((1, 1, window), device=scores.device, dtype=scores.dtype)

        x = scores.transpose(1, 2).reshape(batch_size * n_atoms, 1, n_res)
        valid = (
            protein_mask.bool().unsqueeze(1)
            & ligand_mask.bool().unsqueeze(-1)
        )
        valid_f = valid.reshape(batch_size * n_atoms, 1, n_res).to(dtype=dtype)
        neigh_sum = F.conv1d(x * valid_f, weight, padding=pad)
        neigh_den = F.conv1d(valid_f, weight, padding=pad).clamp(min=1.0)
        neigh = (neigh_sum / neigh_den).reshape(batch_size, n_atoms, n_res).transpose(1, 2)

        gate = torch.tanh(self.residue_atom_triangle_gate(hp.float())).squeeze(-1)
        tri_bias = (neigh - scores) * gate.unsqueeze(-1).to(dtype=scores.dtype)
        alpha = self._bounded_scale(
            self.residue_atom_triangle_raw_alpha.to(dtype=scores.dtype),
            self.residue_atom_triangle_alpha_max,
        )
        pair_valid = protein_mask.bool().unsqueeze(-1) & ligand_mask.bool().unsqueeze(1)
        zero = scores.new_tensor(0.0, dtype=torch.float32)
        self._cached_v13m0_stats["residue_atom_triangle_abs_mean"] = (
            tri_bias.detach()[pair_valid].float().abs().mean() if pair_valid.any() else zero
        )
        self._cached_v13m0_stats["residue_atom_triangle_alpha"] = alpha.detach().float()
        return scores + alpha.to(dtype=scores.dtype) * tri_bias

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._cached_v11m8_stats = {}
        self._cached_v11m10_stats = {}
        self._cached_v13m0_stats = {}
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
        q, k = self._apply_local54_qk_reparam(
            q=q,
            k=k,
            hl=hl,
            local54=local54,
            ligand_context=ligand_context,
            ligand_mask=ligand_mask,
        )

        use_slots = self.use_ligand_slots_delta
        if use_slots:
            k, v, token_mask = self._build_ligand_slots(hl, local54, ligand_mask)
        else:
            token_mask = ligand_mask.bool()

        if self.local_delta_use_qk_norm:
            q_score = F.normalize(q.float(), dim=-1)
            k_score = F.normalize(k.float(), dim=-1)
            qk_scale = self.local_delta_qk_log_scale.exp().clamp(max=self.local_delta_score_clip)
            scores = torch.matmul(q_score, k_score.transpose(-2, -1)) * qk_scale
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.local_delta_dim)
        scores = scores.float().clamp(-self.local_delta_score_clip, self.local_delta_score_clip)
        if not use_slots:
            scores = self._apply_residue_atom_bias(scores, hp, hl, local54, protein_mask, ligand_mask)
            scores = self._apply_residue_atom_triangle_bias(scores, hp, protein_mask, ligand_mask)
        scores = scores.float().clamp(-self.local_delta_score_clip, self.local_delta_score_clip)

        atom_mask = token_mask.bool()
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
        if not use_slots:
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
        outputs["residue_atom_bias_alpha"] = self._bounded_scale(
            self.residue_atom_bias_raw_alpha.detach(),
            self.residue_atom_bias_alpha_max,
        )
        outputs["residue_atom_triangle_alpha"] = self._bounded_scale(
            self.residue_atom_triangle_raw_alpha.detach(),
            self.residue_atom_triangle_alpha_max,
        )
        for key, value in self._cached_v13m0_stats.items():
            outputs[key] = value
        return outputs
