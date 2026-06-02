"""
Pocket-GVP-Fusion V11m6: stable V11 trunk + local evidence delta-only branch.

V11m6 keeps the V11m4t conclusion but tightens the information path:
- no pair evidence is injected into cross-attention logits;
- the protein trunk/classifier still produce base_logits;
- local54 atom chemistry and ligand graph hidden states produce only a
  residue-level delta_logits correction;
- global55 + Uni-Mol CLS only calibrate the delta amplitude via V11m4t's gate.
- optional fractional protein-gradient flow lets the local delta tune the
  protein decision surface without taking the full no-detach gradient.

The branch is coordinate-free: it never uses protein-ligand cross distances,
angles, or shared-frame geometry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m4t import PocketGVPFusionV11m4t


class PocketGVPFusionV11m6(PocketGVPFusionV11m4t):
    """
    V11m6: local54/graph evidence -> late residue delta, not attention bias.

    Compared with V11m4t, this uses a separate `local_delta_*` branch instead
    of the earlier `pair_delta_*` branch.  The main practical differences are:
    - local54 contributes to both pair keys and values, so the delta can carry
      raw local chemistry evidence directly;
    - protein features are detached in the delta branch by default, protecting
      the strong V11 protein-side decision floor from the auxiliary correction;
    - the old pair_delta modules are not instantiated.
    """

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m4t.DEFAULT_CONFIG,
        "use_local_pair_delta": False,
        "use_local_delta_only": True,
        "local_delta_local_dim": 54,
        "local_delta_dim": 128,
        "local_delta_hidden": 256,
        "local_delta_dropout": 0.1,
        "local_delta_score_clip": 20.0,
        "local_delta_logit_scale_init": 0.5,
        "local_delta_logit_scale_max": 1.0,
        "local_delta_detach_protein": True,
        "local_delta_protein_grad_scale": 1.0,
        "local_delta_detach_ligand": False,
        "local_delta_use_ligand_hidden": True,
        "local_delta_use_local_features": True,
        "local_delta_zero_init_final": True,
        "local_delta_final_init_std": 0.0,
        "local_delta_use_qk_norm": False,
        "local_delta_qk_norm_scale_init": 11.313708498984761,
        "ligand_source_dropout_p": 0.0,
        "ligand_dropout_local54_p": 0.0,
        "ligand_dropout_hidden_p": 0.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        # Disable V11m4's old pair_delta modules; this class overrides
        # _compute_pair_delta and _delta_scale with local_delta equivalents.
        cfg["use_local_pair_delta"] = False
        super().__init__(cfg)

        self.use_local_delta_only = bool(cfg.get("use_local_delta_only", True))
        self.local_delta_local_dim = int(cfg.get("local_delta_local_dim", 54))
        self.local_delta_dim = int(cfg.get("local_delta_dim", 128))
        self.local_delta_score_clip = float(cfg.get("local_delta_score_clip", 20.0))
        self.local_delta_logit_scale_max = float(cfg.get("local_delta_logit_scale_max", 1.0))
        self.local_delta_detach_protein = bool(cfg.get("local_delta_detach_protein", True))
        self.local_delta_protein_grad_scale = float(cfg.get("local_delta_protein_grad_scale", 1.0))
        self.local_delta_detach_ligand = bool(cfg.get("local_delta_detach_ligand", False))
        self.local_delta_use_ligand_hidden = bool(cfg.get("local_delta_use_ligand_hidden", True))
        self.local_delta_use_local_features = bool(cfg.get("local_delta_use_local_features", True))
        self.local_delta_use_qk_norm = bool(cfg.get("local_delta_use_qk_norm", False))
        self.ligand_source_dropout_p = float(cfg.get("ligand_source_dropout_p", 0.0))
        self.ligand_dropout_local54_p = float(cfg.get("ligand_dropout_local54_p", 0.0))
        self.ligand_dropout_hidden_p = float(cfg.get("ligand_dropout_hidden_p", 0.0))

        d_model = int(self.config["d_model"])
        n_classes = int(self.config["n_classes"])
        hidden = int(cfg.get("local_delta_hidden", 256))
        dropout = float(cfg.get("local_delta_dropout", 0.1))

        if self.use_local_delta_only:
            self.local_delta_prot_norm = nn.LayerNorm(d_model)
            self.local_delta_lig_norm = nn.LayerNorm(d_model)
            self.local_delta_local_norm = nn.LayerNorm(self.local_delta_local_dim)

            self.local_delta_q = nn.Linear(d_model, self.local_delta_dim, bias=False)
            self.local_delta_k_lig = nn.Linear(d_model, self.local_delta_dim, bias=False)
            self.local_delta_k_local = nn.Linear(
                self.local_delta_local_dim, self.local_delta_dim, bias=False
            )
            self.local_delta_v_lig = nn.Linear(d_model, d_model, bias=False)
            self.local_delta_v_local = nn.Linear(self.local_delta_local_dim, d_model, bias=False)

            delta_in_dim = d_model + d_model + 4
            self.local_delta_head = nn.Sequential(
                nn.LayerNorm(delta_in_dim),
                nn.Linear(delta_in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            final = self.local_delta_head[-1]
            final_init_std = float(cfg.get("local_delta_final_init_std", 0.0) or 0.0)
            if final_init_std > 0.0:
                nn.init.normal_(final.weight, mean=0.0, std=final_init_std)
                nn.init.zeros_(final.bias)
            elif bool(cfg.get("local_delta_zero_init_final", True)):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

            init_scale = float(cfg.get("local_delta_logit_scale_init", 0.5))
            self.local_delta_logit_scale = nn.Parameter(torch.tensor(init_scale))
            qk_scale_init = float(cfg.get("local_delta_qk_norm_scale_init", math.sqrt(self.local_delta_dim)))
            self.local_delta_qk_log_scale = nn.Parameter(torch.tensor(math.log(max(qk_scale_init, 1e-6))))
        else:
            self.local_delta_head = None
            self.register_buffer("local_delta_logit_scale", torch.tensor(0.0), persistent=False)
            self.register_buffer("local_delta_qk_log_scale", torch.tensor(0.0), persistent=False)

        self.config["use_local_pair_delta"] = False
        self.config["use_local_delta_only"] = self.use_local_delta_only
        self.config["local_delta_local_dim"] = self.local_delta_local_dim
        self.config["local_delta_dim"] = self.local_delta_dim
        self.config["local_delta_protein_grad_scale"] = self.local_delta_protein_grad_scale
        self.config["local_delta_use_qk_norm"] = self.local_delta_use_qk_norm
        self.config["ligand_source_dropout_p"] = self.ligand_source_dropout_p
        self.config["ligand_dropout_local54_p"] = self.ligand_dropout_local54_p
        self.config["ligand_dropout_hidden_p"] = self.ligand_dropout_hidden_p

    def _pad_ligand_local_features(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter flat local54 atom features to [B, M, local_dim]."""
        batch_size, max_atoms = ligand_mask.shape
        local = ligand_x[:, : self.local_delta_local_dim]
        out = ligand_x.new_zeros((batch_size, max_atoms, self.local_delta_local_dim))

        n_atoms = int(local.size(0))
        if n_atoms == 0:
            return out
        if not bool((ligand_batch[1:] >= ligand_batch[:-1]).all()):
            raise ValueError("ligand_batch must be non-decreasing for V11m6 local scatter")

        counts = torch.bincount(ligand_batch, minlength=batch_size)
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=ligand_x.device)
        cumsum[1:] = counts.cumsum(0)
        local_idx = torch.arange(n_atoms, device=ligand_x.device) - cumsum[ligand_batch]
        valid = local_idx < max_atoms
        out[ligand_batch[valid], local_idx[valid]] = local[valid]
        return out

    def _delta_scale(self) -> torch.Tensor:
        if self.local_delta_logit_scale_max <= 0:
            return self.local_delta_logit_scale
        return self.local_delta_logit_scale.tanh() * self.local_delta_logit_scale_max

    def _encode_local_delta_features(self, local54: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.local_delta_local_norm(local54.to(dtype=dtype))

    def _apply_ligand_source_dropout(
        self,
        hl_src: torch.Tensor,
        local54: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-only sample-level dropout for local-delta ligand sources."""
        if not self.training:
            return hl_src, local54

        batch_size = hl_src.size(0)
        device = hl_src.device
        drop_all = max(0.0, self.ligand_source_dropout_p)
        drop_hidden = max(0.0, self.ligand_dropout_hidden_p)
        drop_local = max(0.0, self.ligand_dropout_local54_p)

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

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Late, coordinate-free local ligand evidence delta logits."""
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
        hl_src, local54 = self._apply_ligand_source_dropout(hl_src, local54)

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
        outputs["local_delta_logits"] = outputs.get("delta_logits")
        outputs["local_delta_scale"] = outputs.get("pair_delta_scale")
        for key, value in getattr(self, "_cached_local_delta_stats", {}).items():
            outputs[key] = value
        return outputs
