"""
Pocket-GVP-Fusion V11m11: V11m10 local54 K/V with BiMamba-style probes.

The HPC environment does not provide mamba_ssm, so these probes use a small
pure-PyTorch bidirectional gated scan block.  It is intentionally used only as
a bounded residual around the already working V11m10 late local-delta path:

- residue post: smooth protein residue hidden states before local_delta.
- ligand K/V: add bidirectional atom-context residuals to local_delta K/V.
- ligand memory: append a few low-prior ligand memory tokens to local_delta.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocket_gvp_fusion_v11m8 import _atanh_ratio
from .pocket_gvp_fusion_v11m10 import PocketGVPFusionV11m10


class _BiMambaLiteBlock(nn.Module):
    """Bidirectional gated scan block with depthwise local mixing.

    This is not a drop-in replacement for mamba_ssm.  It provides the two
    properties we want for a low-risk probe: bidirectional long-range context
    and bounded residual injection without adding a compiled dependency.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        dropout: float = 0.0,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("BiMambaLite kernel_size must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _masked_bi_scan(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * m
        fwd_count = torch.cumsum(m, dim=1).clamp(min=1.0)
        fwd = torch.cumsum(x, dim=1) / fwd_count

        x_rev = torch.flip(x, dims=[1])
        m_rev = torch.flip(m, dims=[1])
        bwd_count = torch.cumsum(m_rev, dim=1).clamp(min=1.0)
        bwd = torch.flip(torch.cumsum(x_rev, dim=1) / bwd_count, dims=[1])
        return 0.5 * (fwd + bwd) * m

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual_dtype = x.dtype
        h = self.norm(x.float()).to(dtype=residual_dtype)
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        scan = self._masked_bi_scan(u, mask.bool())
        out = self.out_proj(scan * torch.sigmoid(gate))
        out = self.dropout(out)
        return out * mask.unsqueeze(-1).to(dtype=out.dtype)


class _BiMambaLiteStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _BiMambaLiteBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dropout=dropout,
                zero_init_out=True,
            )
            for _ in range(max(1, int(n_layers)))
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = y + layer(y, mask)
        return (y - x) * mask.unsqueeze(-1).to(dtype=x.dtype)


class PocketGVPFusionV11m11(PocketGVPFusionV11m10):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m10.DEFAULT_CONFIG,
        "use_bimamba_residue_post": False,
        "bimamba_residue_layers": 1,
        "bimamba_residue_alpha_init": 0.03,
        "bimamba_residue_alpha_max": 0.20,
        "use_bimamba_ligand_kv": False,
        "bimamba_ligand_layers": 1,
        "bimamba_ligand_alpha_init": 0.03,
        "bimamba_ligand_alpha_max": 0.20,
        "use_mamba_lig_memory": False,
        "mamba_lig_memory_tokens": 4,
        "mamba_lig_memory_layers": 1,
        "mamba_lig_memory_alpha_init": 0.03,
        "mamba_lig_memory_alpha_max": 0.20,
        "mamba_lig_memory_score_bias_init": -4.0,
        "bimamba_kernel_size": 7,
        "bimamba_dropout": 0.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        d_model = int(self.config["d_model"])
        local_dim = int(self.local_delta_local_dim)
        kernel_size = int(cfg.get("bimamba_kernel_size", 7))
        dropout = float(cfg.get("bimamba_dropout", 0.0))

        self.use_bimamba_residue_post = bool(cfg.get("use_bimamba_residue_post", False))
        self.use_bimamba_ligand_kv = bool(cfg.get("use_bimamba_ligand_kv", False))
        self.use_mamba_lig_memory = bool(cfg.get("use_mamba_lig_memory", False))

        self.bimamba_residue_alpha_max = float(cfg.get("bimamba_residue_alpha_max", 0.20))
        self.bimamba_ligand_alpha_max = float(cfg.get("bimamba_ligand_alpha_max", 0.20))
        self.mamba_lig_memory_alpha_max = float(cfg.get("mamba_lig_memory_alpha_max", 0.20))

        if self.use_bimamba_residue_post:
            self.bimamba_residue_stack = _BiMambaLiteStack(
                d_model=d_model,
                n_layers=int(cfg.get("bimamba_residue_layers", 1)),
                kernel_size=kernel_size,
                dropout=dropout,
            )
            init = float(cfg.get("bimamba_residue_alpha_init", 0.03))
            self.bimamba_residue_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.bimamba_residue_alpha_max))
            )
        else:
            self.bimamba_residue_stack = None
            self.register_buffer("bimamba_residue_raw_alpha", torch.tensor(0.0), persistent=False)

        if self.use_bimamba_ligand_kv:
            self.bimamba_ligand_input = nn.Sequential(
                nn.LayerNorm(d_model + local_dim),
                nn.Linear(d_model + local_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.bimamba_ligand_stack = _BiMambaLiteStack(
                d_model=d_model,
                n_layers=int(cfg.get("bimamba_ligand_layers", 1)),
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.bimamba_ligand_k = nn.Linear(d_model, self.local_delta_dim, bias=False)
            self.bimamba_ligand_v = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.bimamba_ligand_k.weight)
            nn.init.zeros_(self.bimamba_ligand_v.weight)
            init = float(cfg.get("bimamba_ligand_alpha_init", 0.03))
            self.bimamba_ligand_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.bimamba_ligand_alpha_max))
            )
        else:
            self.bimamba_ligand_input = None
            self.bimamba_ligand_stack = None
            self.bimamba_ligand_k = None
            self.bimamba_ligand_v = None
            self.register_buffer("bimamba_ligand_raw_alpha", torch.tensor(0.0), persistent=False)

        if self.use_mamba_lig_memory:
            n_mem = int(cfg.get("mamba_lig_memory_tokens", 4))
            self.mamba_lig_memory_tokens = n_mem
            self.mamba_lig_memory_input = nn.Sequential(
                nn.LayerNorm(d_model + local_dim),
                nn.Linear(d_model + local_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.mamba_lig_memory_stack = _BiMambaLiteStack(
                d_model=d_model,
                n_layers=int(cfg.get("mamba_lig_memory_layers", 1)),
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.mamba_lig_memory_queries = nn.Parameter(torch.randn(n_mem, d_model) * 0.02)
            self.mamba_lig_memory_k = nn.Linear(d_model, self.local_delta_dim, bias=False)
            self.mamba_lig_memory_v = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.mamba_lig_memory_k.weight)
            nn.init.zeros_(self.mamba_lig_memory_v.weight)
            init = float(cfg.get("mamba_lig_memory_alpha_init", 0.03))
            self.mamba_lig_memory_raw_alpha = nn.Parameter(
                torch.tensor(_atanh_ratio(init, self.mamba_lig_memory_alpha_max))
            )
            self.mamba_lig_memory_score_bias = nn.Parameter(
                torch.tensor(float(cfg.get("mamba_lig_memory_score_bias_init", -4.0)))
            )
        else:
            self.mamba_lig_memory_tokens = 0
            self.mamba_lig_memory_input = None
            self.mamba_lig_memory_stack = None
            self.mamba_lig_memory_queries = None
            self.mamba_lig_memory_k = None
            self.mamba_lig_memory_v = None
            self.register_buffer("mamba_lig_memory_raw_alpha", torch.tensor(0.0), persistent=False)
            self.register_buffer("mamba_lig_memory_score_bias", torch.tensor(-4.0), persistent=False)

        self.config["use_bimamba_residue_post"] = self.use_bimamba_residue_post
        self.config["use_bimamba_ligand_kv"] = self.use_bimamba_ligand_kv
        self.config["use_mamba_lig_memory"] = self.use_mamba_lig_memory
        self.config["mamba_lig_memory_tokens"] = self.mamba_lig_memory_tokens
        self._cached_v11m11_stats = {}

    def _ligand_mamba_input(
        self,
        hl: torch.Tensor,
        lf: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        return module(torch.cat([hl, lf.to(dtype=hl.dtype)], dim=-1))

    def _apply_bimamba_residue_post(
        self,
        hp_src: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.bimamba_residue_stack is None:
            return hp_src
        delta = self.bimamba_residue_stack(hp_src, protein_mask.bool()).to(dtype=hp_src.dtype)
        alpha = self._bounded_scale(
            self.bimamba_residue_raw_alpha.to(dtype=hp_src.dtype),
            self.bimamba_residue_alpha_max,
        )
        valid = protein_mask.bool()
        zero = hp_src.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m11_stats["bimamba_residue_delta_abs_mean"] = (
            delta.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return hp_src + alpha * delta

    def _apply_bimamba_ligand_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        hl: torch.Tensor,
        lf: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bimamba_ligand_stack is None or self.bimamba_ligand_input is None:
            return k, v
        h = self._ligand_mamba_input(hl, lf, self.bimamba_ligand_input)
        delta = self.bimamba_ligand_stack(h, ligand_mask.bool()).to(dtype=hl.dtype)
        h_ctx = h + delta
        k_res = self.bimamba_ligand_k(h_ctx).to(dtype=k.dtype)
        v_res = self.bimamba_ligand_v(h_ctx).to(dtype=v.dtype)
        alpha = self._bounded_scale(
            self.bimamba_ligand_raw_alpha.to(dtype=k.dtype),
            self.bimamba_ligand_alpha_max,
        )
        valid = ligand_mask.bool()
        zero = hl.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m11_stats["bimamba_ligand_delta_abs_mean"] = (
            delta.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        self._cached_v11m11_stats["bimamba_ligand_k_abs_mean"] = (
            k_res.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        self._cached_v11m11_stats["bimamba_ligand_v_abs_mean"] = (
            v_res.detach()[valid].float().abs().mean() if valid.any() else zero
        )
        return k + alpha * k_res, v + alpha.to(dtype=v.dtype) * v_res

    def _build_mamba_lig_memory(
        self,
        hl: torch.Tensor,
        lf: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mamba_lig_memory_stack is None or self.mamba_lig_memory_input is None:
            return None, None, None
        h = self._ligand_mamba_input(hl, lf, self.mamba_lig_memory_input)
        delta = self.mamba_lig_memory_stack(h, ligand_mask.bool()).to(dtype=hl.dtype)
        h_mem_src = h + delta

        q = self.mamba_lig_memory_queries.to(device=hl.device, dtype=hl.dtype)
        scores = torch.einsum("md,bad->bma", q, h_mem_src) / math.sqrt(h_mem_src.size(-1))
        atom_mask = ligand_mask.bool()
        scores = scores.masked_fill(~atom_mask.unsqueeze(1), -1.0e4)
        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=hl.dtype)
        attn = attn * atom_mask.unsqueeze(1).to(dtype=attn.dtype)
        attn = torch.where(
            has_atom,
            attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6),
            torch.zeros_like(attn),
        )
        mem = torch.matmul(attn, h_mem_src)

        alpha = self._bounded_scale(
            self.mamba_lig_memory_raw_alpha.to(dtype=hl.dtype),
            self.mamba_lig_memory_alpha_max,
        )
        mem_k = alpha.to(dtype=hl.dtype) * self.mamba_lig_memory_k(mem).to(dtype=hl.dtype)
        mem_v = alpha.to(dtype=hl.dtype) * self.mamba_lig_memory_v(mem).to(dtype=hl.dtype)
        mem_mask = atom_mask.any(dim=1, keepdim=True).expand(-1, self.mamba_lig_memory_tokens)

        zero = hl.new_tensor(0.0, dtype=torch.float32)
        self._cached_v11m11_stats["mamba_lig_memory_attn_max"] = (
            attn.detach().float().amax(dim=-1)[mem_mask].mean() if mem_mask.any() else zero
        )
        self._cached_v11m11_stats["mamba_lig_memory_v_abs_mean"] = (
            mem_v.detach()[mem_mask].float().abs().mean() if mem_mask.any() else zero
        )
        return mem_k, mem_v, mem_mask

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """V11m10 local delta plus optional BiMamba-style residual probes."""
        self._cached_v11m8_stats = {}
        self._cached_v11m10_stats = {}
        self._cached_v11m11_stats = {}
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
        hp_src = self._apply_bimamba_residue_post(hp_src, protein_mask)

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
        k, v = self._apply_bimamba_ligand_kv(k, v, hl, lf, ligand_mask)

        atom_mask = ligand_mask.bool()
        kv_mask = atom_mask
        memory_score_bias = None
        mem_k, mem_v, mem_mask = self._build_mamba_lig_memory(hl, lf, ligand_mask)
        if mem_k is not None and mem_v is not None and mem_mask is not None:
            k = torch.cat([k, mem_k.to(dtype=k.dtype)], dim=1)
            v = torch.cat([v, mem_v.to(dtype=v.dtype)], dim=1)
            kv_mask = torch.cat([atom_mask, mem_mask.bool()], dim=1)
            memory_score_bias = self.mamba_lig_memory_score_bias.to(
                device=k.device,
                dtype=torch.float32,
            )

        if self.local_delta_use_qk_norm:
            q_score = F.normalize(q.float(), dim=-1)
            k_score = F.normalize(k.float(), dim=-1)
            qk_scale = self.local_delta_qk_log_scale.exp().clamp(max=self.local_delta_score_clip)
            scores = torch.matmul(q_score, k_score.transpose(-2, -1)) * qk_scale
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.local_delta_dim)
        scores = scores.float().clamp(-self.local_delta_score_clip, self.local_delta_score_clip)
        if memory_score_bias is not None and self.mamba_lig_memory_tokens > 0:
            scores = scores.clone()
            scores[:, :, -self.mamba_lig_memory_tokens :] = (
                scores[:, :, -self.mamba_lig_memory_tokens :] + memory_score_bias
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
        outputs["bimamba_residue_alpha"] = self._bounded_scale(
            self.bimamba_residue_raw_alpha.detach(),
            self.bimamba_residue_alpha_max,
        )
        outputs["bimamba_ligand_kv_alpha"] = self._bounded_scale(
            self.bimamba_ligand_raw_alpha.detach(),
            self.bimamba_ligand_alpha_max,
        )
        outputs["mamba_lig_memory_alpha"] = self._bounded_scale(
            self.mamba_lig_memory_raw_alpha.detach(),
            self.mamba_lig_memory_alpha_max,
        )
        outputs["mamba_lig_memory_score_bias"] = self.mamba_lig_memory_score_bias.detach()
        for key, value in getattr(self, "_cached_v11m11_stats", {}).items():
            outputs[key] = value
        return outputs
