"""
Pocket-GVP-Fusion V11m10: V11m8x co-attn with local54 group interaction.

V11m10 keeps the current strongest V11m8x co-attention trunk unchanged and adds
zero-leak, bounded local evidence probes:

- local54 interaction K/V: base33 atom identity acts as an anchor; electronic3,
  sasa_hbond4, and flex_rot5 provide conditional chemistry evidence residuals
  into the late residue-to-atom local_delta attention.
- residue-chem compatibility: residue hidden states gate group-specific local54
  readouts after the local_delta attention.
- optional ligand-context prior: global55 and/or UniMol are not broadcast into
  atom features. They only add a bounded residual to the pooled ligand context
  used by local evidence gates.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import _LOCAL54_GROUP_RANGES, _atanh_ratio
from .pocket_gvp_fusion_v11m8x import PocketGVPFusionV11m8x


_DEFAULT_CONDITION_GROUPS = ("electronic3", "sasa_hbond4", "flex_rot5")


def _group_dim(group: str) -> int:
    return sum(end - start for start, end in _LOCAL54_GROUP_RANGES[group])


def _parse_groups(value, default: Iterable[str] = _DEFAULT_CONDITION_GROUPS) -> Tuple[str, ...]:
    if value is None:
        groups = tuple(default)
    elif isinstance(value, str):
        groups = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        groups = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [group for group in groups if group not in _LOCAL54_GROUP_RANGES]
    if unknown:
        raise ValueError(f"unknown local54 interaction groups: {unknown}")
    return groups or tuple(default)


def _parse_sources(value) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        sources = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    else:
        sources = tuple(str(part).strip().lower() for part in value if str(part).strip())
    unknown = [source for source in sources if source not in {"global55", "unimol"}]
    if unknown:
        raise ValueError(f"unknown ligand context prior sources: {unknown}")
    return sources


class PocketGVPFusionV11m10(PocketGVPFusionV11m8x):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m8x.DEFAULT_CONFIG,
        "use_local54_interaction_kv": False,
        "use_local54_residue_chem_compat": False,
        "local54_interaction_groups": list(_DEFAULT_CONDITION_GROUPS),
        "local54_interaction_hidden": 96,
        "local54_interaction_kv_alpha_init": 0.05,
        "local54_interaction_kv_alpha_max": 0.30,
        "local54_compat_hidden": 128,
        "local54_compat_gate_bias_init": 0.05,
        "local54_compat_alpha_init": 0.05,
        "local54_compat_alpha_max": 0.30,
        "local54_compat_normalize_read": False,
        "local54_compat_read_scale": 1.0,
        "local54_compat_read_clip": 0.0,
        "use_ligand_context_prior": False,
        "ligand_context_prior_sources": [],
        "ligand_context_prior_hidden": 192,
        "ligand_context_prior_alpha_init": 0.03,
        "ligand_context_prior_alpha_max": 0.20,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        d_model = int(self.config["d_model"])
        local_dim = int(self.local_delta_local_dim)
        hidden = int(cfg.get("local54_interaction_hidden", 96))
        self.local54_interaction_groups = _parse_groups(cfg.get("local54_interaction_groups"))
        self.use_local54_interaction_kv = bool(cfg.get("use_local54_interaction_kv", False))
        self.use_local54_residue_chem_compat = bool(
            cfg.get("use_local54_residue_chem_compat", False)
        )
        self.local54_interaction_kv_alpha_max = float(
            cfg.get("local54_interaction_kv_alpha_max", 0.30)
        )
        self.local54_compat_alpha_max = float(cfg.get("local54_compat_alpha_max", 0.30))
        self.local54_compat_normalize_read = bool(
            cfg.get("local54_compat_normalize_read", False)
        )
        self.local54_compat_read_scale = float(cfg.get("local54_compat_read_scale", 1.0))
        self.local54_compat_read_clip = float(cfg.get("local54_compat_read_clip", 0.0))

        if self.use_local54_interaction_kv or self.use_local54_residue_chem_compat:
            self.local54_anchor_proj = nn.Sequential(
                nn.LayerNorm(_group_dim("base33"), elementwise_affine=False),
                nn.Linear(_group_dim("base33"), hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.local54_group_proj = nn.ModuleDict()
            for group in self.local54_interaction_groups:
                self.local54_group_proj[group] = nn.Sequential(
                    nn.LayerNorm(_group_dim(group), elementwise_affine=False),
                    nn.Linear(_group_dim(group), hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                )
        else:
            self.local54_anchor_proj = None
            self.local54_group_proj = None

        inter_dim = hidden * 3
        if self.use_local54_interaction_kv:
            self.local54_interaction_k = nn.ModuleDict()
            self.local54_interaction_v = nn.ModuleDict()
            for group in self.local54_interaction_groups:
                k_proj = nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, self.local_delta_dim, bias=False),
                )
                v_proj = nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, d_model, bias=False),
                )
                nn.init.zeros_(k_proj[-1].weight)
                nn.init.zeros_(v_proj[-1].weight)
                self.local54_interaction_k[group] = k_proj
                self.local54_interaction_v[group] = v_proj
            init = float(cfg.get("local54_interaction_kv_alpha_init", 0.05))
            self.local54_interaction_kv_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local54_interaction_kv_alpha_max))
            )
        else:
            self.local54_interaction_k = None
            self.local54_interaction_v = None
            self.register_buffer(
                "local54_interaction_kv_raw_alpha", torch.tensor(0.0), persistent=False
            )

        if self.use_local54_residue_chem_compat:
            self.local54_compat_value = nn.ModuleDict()
            for group in self.local54_interaction_groups:
                self.local54_compat_value[group] = nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, d_model, bias=False),
                )
                nn.init.zeros_(self.local54_compat_value[group][-1].weight)
            compat_hidden = int(cfg.get("local54_compat_hidden", 128))
            self.local54_compat_gate = nn.Sequential(
                nn.LayerNorm(d_model + d_model + local_dim),
                nn.Linear(d_model + d_model + local_dim, compat_hidden),
                nn.GELU(),
                nn.Linear(compat_hidden, len(self.local54_interaction_groups)),
            )
            nn.init.zeros_(self.local54_compat_gate[-1].weight)
            gate_bias_init = float(cfg.get("local54_compat_gate_bias_init", 0.05))
            nn.init.constant_(self.local54_compat_gate[-1].bias, gate_bias_init)
            init = float(cfg.get("local54_compat_alpha_init", 0.05))
            self.local54_compat_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local54_compat_alpha_max))
            )
            if self.local54_compat_normalize_read:
                self.local54_compat_read_norm = nn.LayerNorm(
                    d_model,
                    elementwise_affine=False,
                )
            else:
                self.local54_compat_read_norm = None
        else:
            self.local54_compat_value = None
            self.local54_compat_gate = None
            self.local54_compat_read_norm = None
            self.register_buffer("local54_compat_raw_alpha", torch.tensor(0.0), persistent=False)

        self.use_ligand_context_prior = bool(cfg.get("use_ligand_context_prior", False))
        self.ligand_context_prior_sources = _parse_sources(cfg.get("ligand_context_prior_sources"))
        self.ligand_context_prior_alpha_max = float(
            cfg.get("ligand_context_prior_alpha_max", 0.20)
        )
        prior_in = 0
        if "global55" in self.ligand_context_prior_sources:
            prior_in += int(self.delta_gate_global_dim)
        if "unimol" in self.ligand_context_prior_sources:
            prior_in += int(self.config.get("d_unimol", 512))
        if self.use_ligand_context_prior and prior_in > 0:
            prior_hidden = int(cfg.get("ligand_context_prior_hidden", 192))
            self.ligand_context_prior = nn.Sequential(
                nn.LayerNorm(prior_in),
                nn.Linear(prior_in, prior_hidden),
                nn.GELU(),
                nn.Linear(prior_hidden, d_model + local_dim),
            )
            nn.init.zeros_(self.ligand_context_prior[-1].weight)
            nn.init.zeros_(self.ligand_context_prior[-1].bias)
            init = float(cfg.get("ligand_context_prior_alpha_init", 0.03))
            self.ligand_context_prior_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.ligand_context_prior_alpha_max))
            )
        else:
            self.ligand_context_prior = None
            self.register_buffer(
                "ligand_context_prior_raw_alpha", torch.tensor(0.0), persistent=False
            )

        self.config["use_local54_interaction_kv"] = self.use_local54_interaction_kv
        self.config["use_local54_residue_chem_compat"] = (
            self.use_local54_residue_chem_compat
        )
        self.config["local54_interaction_groups"] = list(self.local54_interaction_groups)
        self.config["local54_compat_normalize_read"] = self.local54_compat_normalize_read
        self.config["local54_compat_read_scale"] = self.local54_compat_read_scale
        self.config["local54_compat_read_clip"] = self.local54_compat_read_clip
        self.config["use_ligand_context_prior"] = self.use_ligand_context_prior
        self.config["ligand_context_prior_sources"] = list(self.ligand_context_prior_sources)
        self._cached_v11m10_stats = {}
        self._cached_v11m10_global55 = None
        self._cached_v11m10_unimol = None

    def _local54_group(self, local54: torch.Tensor, group: str, dtype: torch.dtype) -> torch.Tensor:
        return self._extract_local54_group(local54, group).to(dtype=dtype)

    def _interaction_features(
        self,
        local54: torch.Tensor,
        group: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.local54_anchor_proj is None or self.local54_group_proj is None:
            raise RuntimeError("local54 interaction modules are not initialized")
        base = self._local54_group(local54, "base33", dtype)
        evidence = self._local54_group(local54, group, dtype)
        anchor_h = self.local54_anchor_proj(base)
        evidence_h = self.local54_group_proj[group](evidence)
        return torch.cat([anchor_h, evidence_h, anchor_h * evidence_h], dim=-1)

    def _apply_ligand_context_prior(
        self,
        ligand_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.ligand_context_prior is None:
            return ligand_context
        pieces = []
        if "global55" in self.ligand_context_prior_sources:
            global55 = self._cached_v11m10_global55
            if global55 is None:
                global55 = ligand_context.new_zeros(
                    (ligand_context.size(0), int(self.delta_gate_global_dim))
                )
            pieces.append(global55.to(device=ligand_context.device, dtype=ligand_context.dtype))
        if "unimol" in self.ligand_context_prior_sources:
            unimol = self._cached_v11m10_unimol
            if unimol is None:
                unimol = ligand_context.new_zeros(
                    (ligand_context.size(0), int(self.config.get("d_unimol", 512)))
                )
            pieces.append(unimol.to(device=ligand_context.device, dtype=ligand_context.dtype))
        if not pieces:
            return ligand_context
        prior_in = torch.cat(pieces, dim=-1)
        delta = self.ligand_context_prior(prior_in).to(dtype=ligand_context.dtype)
        alpha = self._bounded_scale(
            self.ligand_context_prior_raw_alpha.to(dtype=ligand_context.dtype),
            self.ligand_context_prior_alpha_max,
        )
        self._cached_v11m10_stats["ligand_context_prior_abs_mean"] = (
            delta.detach().float().abs().mean()
        )
        return ligand_context + alpha * delta

    def _apply_interaction_kv_delta(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        local54: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local54_interaction_k is None or self.local54_interaction_v is None:
            return k, v
        k_res = torch.zeros_like(k)
        v_res = torch.zeros_like(v)
        for group in self.local54_interaction_groups:
            feat = self._interaction_features(local54, group, k.dtype)
            k_res = k_res + self.local54_interaction_k[group](feat)
            v_res = v_res + self.local54_interaction_v[group](feat).to(dtype=v.dtype)
        alpha = self._bounded_scale(
            self.local54_interaction_kv_raw_alpha.to(dtype=k.dtype),
            self.local54_interaction_kv_alpha_max,
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
        if self.local54_compat_gate is None or self.local54_compat_value is None:
            return lig_read

        group_reads = []
        for group in self.local54_interaction_groups:
            feat = self._interaction_features(local54, group, lig_read.dtype)
            group_v = self.local54_compat_value[group](feat).to(dtype=lig_read.dtype)
            group_reads.append(torch.matmul(attn, group_v))
        reads = torch.stack(group_reads, dim=-2)

        context = ligand_context.to(dtype=hp.dtype).unsqueeze(1).expand(-1, hp.size(1), -1)
        gate_in = torch.cat([hp, context], dim=-1)
        gates = torch.tanh(self.local54_compat_gate(gate_in)).to(dtype=reads.dtype)
        compat_read = (reads * gates.unsqueeze(-1)).sum(dim=-2)
        raw_compat_read = compat_read
        if self.local54_compat_read_norm is not None:
            compat_read = self.local54_compat_read_norm(compat_read.to(dtype=hp.dtype)).to(
                dtype=lig_read.dtype
            )
        if self.local54_compat_read_scale != 1.0:
            compat_read = compat_read * compat_read.new_tensor(self.local54_compat_read_scale)
        if self.local54_compat_read_clip > 0.0:
            compat_read = compat_read.clamp(
                min=-self.local54_compat_read_clip,
                max=self.local54_compat_read_clip,
            )
        alpha = self._bounded_scale(
            self.local54_compat_raw_alpha.to(dtype=lig_read.dtype),
            self.local54_compat_alpha_max,
        )
        mask = protein_mask.unsqueeze(-1).to(dtype=lig_read.dtype)

        valid = protein_mask.bool()
        zero = hp.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m10_stats["local54_compat_gate_abs_mean"] = (
            gates.detach()[valid].abs().mean() if valid.any() else zero
        )
        self._cached_v11m10_stats["local54_compat_read_raw_abs_mean"] = (
            raw_compat_read.detach()[valid].abs().mean() if valid.any() else zero
        )
        self._cached_v11m10_stats["local54_compat_read_abs_mean"] = (
            compat_read.detach()[valid].abs().mean() if valid.any() else zero
        )
        return lig_read + alpha * compat_read * mask

    def _classify_local_delta(self, delta_in, protein_mask, ligand_mask):
        """Readout hook; the center auxiliary head retains its original input."""
        return self.local_delta_head(delta_in)

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """V11m8x local delta plus optional local54 group interaction residuals."""
        self._cached_v11m8_stats = {}
        self._cached_v11m10_stats = {}
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
        delta_logits = self._classify_local_delta(delta_in, protein_mask, ligand_mask)

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
        esm3_embeddings: Optional[torch.Tensor] = None,
        esm_valid_mask: Optional[torch.Tensor] = None,
        esm3_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._cached_v11m10_global55 = self._mean_ligand_global55(
            ligand_x, ligand_batch, ligand_mask
        )
        if unimol_embed is None:
            self._cached_v11m10_unimol = None
        else:
            self._cached_v11m10_unimol = unimol_embed.to(
                device=ligand_x.device, dtype=ligand_x.dtype
            )

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
            esm3_embeddings=esm3_embeddings,
            esm_valid_mask=esm_valid_mask,
            esm3_valid_mask=esm3_valid_mask,
        )

        outputs["local54_interaction_kv_alpha"] = self._bounded_scale(
            self.local54_interaction_kv_raw_alpha.detach(),
            self.local54_interaction_kv_alpha_max,
        )
        outputs["local54_compat_alpha"] = self._bounded_scale(
            self.local54_compat_raw_alpha.detach(),
            self.local54_compat_alpha_max,
        )
        outputs["ligand_context_prior_alpha"] = self._bounded_scale(
            self.ligand_context_prior_raw_alpha.detach(),
            self.ligand_context_prior_alpha_max,
        )
        for key, value in getattr(self, "_cached_v11m10_stats", {}).items():
            outputs[key] = value
        return outputs
