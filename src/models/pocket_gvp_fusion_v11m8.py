"""
Pocket-GVP-Fusion V11m8: light local-ligand residual probes on V11m6n.

V11m8 keeps the proven V11m6n path intact:
- stable V11 protein trunk;
- late raw local54/ligand-graph delta logits;
- no global/UniMol delta gate by default;
- fractional protein gradient protection.

The new switches are deliberately small residual probes:
- local ligand FiLM: pooled local54/graph context lightly modulates the delta
  branch protein hidden state;
- pocket virtual tokens: ligand-conditioned pocket queries aggregate protein
  evidence and write a small residual into the ligand readout;
- center auxiliary head: a bounded class-0 coherence residual, supervised by
  the normal residue labels through the existing loss;
- ligand-source dropout: regularizes local54/hidden source reliance during
  training without changing evaluation behavior.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m6 import PocketGVPFusionV11m6


_LOCAL54_GROUP_RANGES = {
    "base33": ((0, 33),),
    "electronic3": ((33, 36),),
    "sasa_hbond4": ((38, 42),),
    "flex_rot5": ((36, 38), (44, 47)),
    "metal4": ((47, 51),),
    "stereo5": ((42, 44), (51, 54)),
}


def _atanh_ratio(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    ratio = max(min(value / max_value, 0.999), -0.999)
    return 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))


class PocketGVPFusionV11m8(PocketGVPFusionV11m6):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m6.DEFAULT_CONFIG,
        "use_local_ligand_film": False,
        "local_ligand_film_hidden": 256,
        "local_ligand_film_alpha_init": 0.05,
        "local_ligand_film_alpha_max": 0.30,
        "use_pocket_virtual_tokens": False,
        "pocket_virtual_tokens": 4,
        "pocket_virtual_alpha_init": 0.05,
        "pocket_virtual_alpha_max": 0.30,
        "use_center_aux_head": False,
        "center_aux_hidden": 128,
        "center_aux_logit_scale_init": 0.05,
        "center_aux_logit_scale_max": 0.30,
        "ligand_source_dropout_p": 0.0,
        "ligand_dropout_local54_p": 0.0,
        "ligand_dropout_hidden_p": 0.0,
        "use_local54_chem_reweight": False,
        "local54_chem_base_scale": 1.0,
        "local54_chem_electronic_scale": 1.0,
        "local54_chem_sasa_hbond_scale": 1.0,
        "local54_chem_flex_rot_scale": 1.0,
        "local54_chem_metal_scale": 1.0,
        "local54_chem_stereo_scale": 1.0,
        "use_local54_chem_adapter": False,
        "local54_chem_adapter_hidden": 16,
        "local54_chem_adapter_scale_init": 0.03,
        "local54_chem_adapter_scale_max": 0.20,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        d_model = int(self.config["d_model"])
        n_classes = int(self.config["n_classes"])
        local_dim = int(self.local_delta_local_dim)

        self.use_local_ligand_film = bool(cfg.get("use_local_ligand_film", False))
        self.local_ligand_film_alpha_max = float(cfg.get("local_ligand_film_alpha_max", 0.30))
        if self.use_local_ligand_film:
            hidden = int(cfg.get("local_ligand_film_hidden", 256))
            self.local_ligand_film = nn.Sequential(
                nn.LayerNorm(d_model + local_dim),
                nn.Linear(d_model + local_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * d_model),
            )
            nn.init.zeros_(self.local_ligand_film[-1].weight)
            nn.init.zeros_(self.local_ligand_film[-1].bias)
            init = float(cfg.get("local_ligand_film_alpha_init", 0.05))
            self.local_ligand_film_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local_ligand_film_alpha_max))
            )
        else:
            self.local_ligand_film = None
            self.register_buffer("local_ligand_film_raw_alpha", torch.tensor(0.0), persistent=False)

        self.use_pocket_virtual_tokens = bool(cfg.get("use_pocket_virtual_tokens", False))
        self.pocket_virtual_tokens = int(cfg.get("pocket_virtual_tokens", 4))
        self.pocket_virtual_alpha_max = float(cfg.get("pocket_virtual_alpha_max", 0.30))
        if self.use_pocket_virtual_tokens:
            n_tokens = max(1, self.pocket_virtual_tokens)
            self.pocket_tokens = nn.Parameter(torch.empty(n_tokens, d_model))
            nn.init.normal_(self.pocket_tokens, mean=0.0, std=0.02)
            self.pocket_token_cond = nn.Sequential(
                nn.LayerNorm(d_model + local_dim),
                nn.Linear(d_model + local_dim, n_tokens * d_model),
            )
            nn.init.zeros_(self.pocket_token_cond[-1].weight)
            nn.init.zeros_(self.pocket_token_cond[-1].bias)
            self.pocket_token_q = nn.Linear(d_model, d_model, bias=False)
            self.pocket_prot_k = nn.Linear(d_model, d_model, bias=False)
            self.pocket_prot_v = nn.Linear(d_model, d_model, bias=False)
            self.pocket_res_q = nn.Linear(d_model, d_model, bias=False)
            self.pocket_token_k = nn.Linear(d_model, d_model, bias=False)
            self.pocket_token_v = nn.Linear(d_model, d_model, bias=False)
            init = float(cfg.get("pocket_virtual_alpha_init", 0.05))
            self.pocket_virtual_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.pocket_virtual_alpha_max))
            )
        else:
            self.pocket_tokens = None
            self.register_buffer("pocket_virtual_raw_alpha", torch.tensor(0.0), persistent=False)

        self.use_center_aux_head = bool(cfg.get("use_center_aux_head", False))
        self.center_aux_logit_scale_max = float(cfg.get("center_aux_logit_scale_max", 0.30))
        if self.use_center_aux_head:
            hidden = int(cfg.get("center_aux_hidden", 128))
            self.center_aux_head = nn.Sequential(
                nn.LayerNorm(2 * d_model + 4),
                nn.Linear(2 * d_model + 4, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.center_aux_head[-1].weight)
            nn.init.zeros_(self.center_aux_head[-1].bias)
            init = float(cfg.get("center_aux_logit_scale_init", 0.05))
            self.center_aux_raw_scale = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.center_aux_logit_scale_max))
            )
        else:
            self.center_aux_head = None
            self.register_buffer("center_aux_raw_scale", torch.tensor(0.0), persistent=False)

        self.ligand_source_dropout_p = float(cfg.get("ligand_source_dropout_p", 0.0))
        self.ligand_dropout_local54_p = float(cfg.get("ligand_dropout_local54_p", 0.0))
        self.ligand_dropout_hidden_p = float(cfg.get("ligand_dropout_hidden_p", 0.0))

        self.use_local54_chem_reweight = bool(cfg.get("use_local54_chem_reweight", False))
        scale = torch.ones(local_dim, dtype=torch.float32)
        if self.use_local54_chem_reweight:
            group_scales = {
                "base33": float(cfg.get("local54_chem_base_scale", 1.0)),
                "electronic3": float(cfg.get("local54_chem_electronic_scale", 1.0)),
                "sasa_hbond4": float(cfg.get("local54_chem_sasa_hbond_scale", 1.0)),
                "flex_rot5": float(cfg.get("local54_chem_flex_rot_scale", 1.0)),
                "metal4": float(cfg.get("local54_chem_metal_scale", 1.0)),
                "stereo5": float(cfg.get("local54_chem_stereo_scale", 1.0)),
            }
            for group, ranges in _LOCAL54_GROUP_RANGES.items():
                for start, end in ranges:
                    if start < local_dim:
                        scale[start:min(end, local_dim)] = group_scales[group]
        self.register_buffer("local54_chem_reweight_scale", scale, persistent=False)

        self.use_local54_chem_adapter = bool(cfg.get("use_local54_chem_adapter", False))
        self.local54_chem_adapter_scale_max = float(cfg.get("local54_chem_adapter_scale_max", 0.20))
        if self.use_local54_chem_adapter:
            hidden = int(cfg.get("local54_chem_adapter_hidden", 16))
            self.local54_chem_adapters = nn.ModuleDict()
            for group in ("electronic3", "sasa_hbond4", "flex_rot5"):
                group_dim = sum(end - start for start, end in _LOCAL54_GROUP_RANGES[group])
                adapter = nn.Sequential(
                    nn.LayerNorm(group_dim, elementwise_affine=False),
                    nn.Linear(group_dim, hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden, group_dim, bias=False),
                )
                nn.init.zeros_(adapter[-1].weight)
                self.local54_chem_adapters[group] = adapter
            init = float(cfg.get("local54_chem_adapter_scale_init", 0.03))
            self.local54_chem_adapter_raw_scale = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.local54_chem_adapter_scale_max))
            )
        else:
            self.local54_chem_adapters = None
            self.register_buffer("local54_chem_adapter_raw_scale", torch.tensor(0.0), persistent=False)

        self.config["use_local_ligand_film"] = self.use_local_ligand_film
        self.config["use_pocket_virtual_tokens"] = self.use_pocket_virtual_tokens
        self.config["use_center_aux_head"] = self.use_center_aux_head
        self.config["ligand_source_dropout_p"] = self.ligand_source_dropout_p
        self.config["ligand_dropout_local54_p"] = self.ligand_dropout_local54_p
        self.config["ligand_dropout_hidden_p"] = self.ligand_dropout_hidden_p
        self.config["use_local54_chem_reweight"] = self.use_local54_chem_reweight
        self.config["use_local54_chem_adapter"] = self.use_local54_chem_adapter

    def _extract_local54_group(self, local54: torch.Tensor, group: str) -> torch.Tensor:
        return torch.cat(
            [local54[..., start:end] for start, end in _LOCAL54_GROUP_RANGES[group]],
            dim=-1,
        )

    def _scatter_local54_group(
        self,
        out: torch.Tensor,
        group: str,
        values: torch.Tensor,
    ) -> None:
        offset = 0
        for start, end in _LOCAL54_GROUP_RANGES[group]:
            width = end - start
            out[..., start:end] = values[..., offset:offset + width]
            offset += width

    def _apply_local54_chemistry(self, local54: torch.Tensor) -> torch.Tensor:
        """Apply optional local54 chemistry reweighting/adapter before delta use."""
        out = local54
        if self.use_local54_chem_reweight:
            out = out * self.local54_chem_reweight_scale.to(device=out.device, dtype=out.dtype)
        if self.local54_chem_adapters is not None:
            residual = torch.zeros_like(out)
            for group, adapter in self.local54_chem_adapters.items():
                group_x = self._extract_local54_group(out, group)
                group_delta = adapter(group_x.to(dtype=out.dtype))
                self._scatter_local54_group(residual, group, group_delta)
            scale = self._bounded_scale(
                self.local54_chem_adapter_raw_scale.to(dtype=out.dtype),
                self.local54_chem_adapter_scale_max,
            )
            out = out + scale * residual
        return out

    def _bounded_scale(self, raw: torch.Tensor, max_value: float) -> torch.Tensor:
        if max_value <= 0.0:
            return raw
        return raw.tanh() * max_value

    def _masked_ligand_context(
        self,
        h_lig_atoms: torch.Tensor,
        local54: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        atom_mask = ligand_mask.unsqueeze(-1).to(dtype=h_lig_atoms.dtype)
        denom = atom_mask.sum(dim=1).clamp(min=1.0)
        h_pool = (h_lig_atoms * atom_mask).sum(dim=1) / denom
        local_pool = (local54.to(dtype=h_lig_atoms.dtype) * atom_mask).sum(dim=1) / denom
        return torch.cat([h_pool, local_pool], dim=-1)

    def _apply_source_dropout(
        self,
        hl_src: torch.Tensor,
        local54: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return hl_src, local54
        batch_size = hl_src.size(0)
        device = hl_src.device

        drop_all = self.ligand_source_dropout_p
        drop_local = self.ligand_dropout_local54_p
        drop_hidden = self.ligand_dropout_hidden_p

        if drop_all > 0.0:
            keep = (torch.rand(batch_size, 1, 1, device=device) >= drop_all).to(dtype=hl_src.dtype)
            hl_src = hl_src * keep
            local54 = local54 * keep.to(dtype=local54.dtype)
        if drop_hidden > 0.0:
            keep = (torch.rand(batch_size, 1, 1, device=device) >= drop_hidden).to(dtype=hl_src.dtype)
            hl_src = hl_src * keep
        if drop_local > 0.0:
            keep = (torch.rand(batch_size, 1, 1, device=device) >= drop_local).to(dtype=local54.dtype)
            local54 = local54 * keep
        return hl_src, local54

    def _apply_local_ligand_film(
        self,
        hp_src: torch.Tensor,
        ligand_context: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_ligand_film is None:
            return hp_src
        film = self.local_ligand_film(ligand_context.to(dtype=hp_src.dtype))
        gamma, beta = film.chunk(2, dim=-1)
        update = hp_src * torch.tanh(gamma).unsqueeze(1) + beta.unsqueeze(1)
        alpha = self._bounded_scale(
            self.local_ligand_film_raw_alpha.to(dtype=hp_src.dtype),
            self.local_ligand_film_alpha_max,
        )
        mask = protein_mask.unsqueeze(-1).to(dtype=hp_src.dtype)
        return hp_src + alpha * update * mask

    def _apply_pocket_virtual_tokens(
        self,
        hp: torch.Tensor,
        lig_read: torch.Tensor,
        ligand_context: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pocket_tokens is None:
            return lig_read

        batch_size = hp.size(0)
        n_tokens = self.pocket_tokens.size(0)
        cond = self.pocket_token_cond(ligand_context.to(dtype=hp.dtype)).view(
            batch_size, n_tokens, hp.size(-1)
        )
        tokens = self.pocket_tokens.unsqueeze(0).to(dtype=hp.dtype) + cond

        q_tok = self.pocket_token_q(tokens)
        k_prot = self.pocket_prot_k(hp)
        v_prot = self.pocket_prot_v(hp)
        scores = torch.matmul(q_tok, k_prot.transpose(-2, -1)) / math.sqrt(hp.size(-1))
        mask = protein_mask.bool().unsqueeze(1)
        scores = scores.float().masked_fill(~mask, -1.0e4)
        has_res = protein_mask.bool().any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_res, scores, torch.zeros_like(scores))
        token_attn = torch.softmax(scores, dim=-1).to(dtype=hp.dtype)
        token_attn = token_attn * protein_mask.unsqueeze(1).to(dtype=hp.dtype)
        token_attn = token_attn / token_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        token_state = torch.matmul(token_attn, v_prot)

        q_res = self.pocket_res_q(hp)
        k_tok = self.pocket_token_k(token_state)
        v_tok = self.pocket_token_v(token_state)
        res_scores = torch.matmul(q_res, k_tok.transpose(-2, -1)) / math.sqrt(hp.size(-1))
        res_attn = torch.softmax(res_scores.float(), dim=-1).to(dtype=hp.dtype)
        pocket_read = torch.matmul(res_attn, v_tok)

        alpha = self._bounded_scale(
            self.pocket_virtual_raw_alpha.to(dtype=lig_read.dtype),
            self.pocket_virtual_alpha_max,
        )
        self._cached_v11m8_stats["pocket_virtual_attn_entropy"] = (
            -(token_attn.float() * token_attn.float().clamp(min=1e-8).log()).sum(dim=-1).detach().mean()
        )
        return lig_read + alpha * pocket_read * protein_mask.unsqueeze(-1).to(dtype=lig_read.dtype)

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """V11m6n local delta plus optional V11m8 light residual probes."""
        self._cached_v11m8_stats = {}
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
        if getattr(self, "_cached_center_aux_logits", None) is not None:
            outputs["center_aux_logits"] = self._cached_center_aux_logits
        outputs["local_ligand_film_alpha"] = self._bounded_scale(
            self.local_ligand_film_raw_alpha.detach(),
            self.local_ligand_film_alpha_max,
        )
        outputs["pocket_virtual_alpha"] = self._bounded_scale(
            self.pocket_virtual_raw_alpha.detach(),
            self.pocket_virtual_alpha_max,
        )
        outputs["center_aux_scale"] = self._bounded_scale(
            self.center_aux_raw_scale.detach(),
            self.center_aux_logit_scale_max,
        )
        outputs["local54_chem_reweight_mean"] = self.local54_chem_reweight_scale.detach().mean()
        outputs["local54_chem_adapter_scale"] = self._bounded_scale(
            self.local54_chem_adapter_raw_scale.detach(),
            self.local54_chem_adapter_scale_max,
        )
        for key, value in getattr(self, "_cached_v11m8_stats", {}).items():
            outputs[key] = value
        return outputs
