"""
Pocket-GVP-Fusion V11m5: V11m4 + local pair evidence as P->L attention bias.

V11m4 showed ligand dependency comes primarily from local54 atom chemistry.
V11m5 moves that coordinate-free local pair evidence earlier:

    attn_logits(P_i, L_j) += beta * PairEvidence(hP_i, hL_j, local54_j)

Only the protein-query-ligand cross-attention path is biased.  Ligand->protein
and all shared V11/V11m3 modules are untouched.  No protein-ligand shared
coordinate, cross-distance, or holo geometry is used.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .pocket_gvp_fusion_v11m3 import PocketGVPFusionV11m3
from .transformer_layers import drop_path


def _bounded_raw(init_value: float, max_value: float) -> float:
    if max_value <= 0:
        return init_value
    ratio = max(min(init_value / max_value, 0.999), -0.999)
    return 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))


class LocalPairAttentionBias(nn.Module):
    """Coordinate-free local54 residue-atom compatibility bias."""

    def __init__(
        self,
        d_model: int,
        local_dim: int,
        pair_dim: int,
        score_clip: float,
        beta_init: float,
        beta_max: float,
    ):
        super().__init__()
        self.pair_dim = pair_dim
        self.score_clip = score_clip
        self.beta_max = beta_max

        self.prot_norm = nn.LayerNorm(d_model)
        self.lig_norm = nn.LayerNorm(d_model)
        self.local_norm = nn.LayerNorm(local_dim)
        self.q = nn.Linear(d_model, pair_dim, bias=False)
        self.k_lig = nn.Linear(d_model, pair_dim, bias=False)
        self.k_local = nn.Linear(local_dim, pair_dim, bias=False)
        self.raw_beta = nn.Parameter(torch.tensor(_bounded_raw(beta_init, beta_max)))

    def beta(self) -> torch.Tensor:
        if self.beta_max <= 0:
            return self.raw_beta
        return self.raw_beta.tanh() * self.beta_max

    def forward(
        self,
        h_prot: torch.Tensor,
        h_lig_atoms: torch.Tensor,
        local54: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        hp = self.prot_norm(h_prot)
        hl = self.lig_norm(h_lig_atoms)
        lf = self.local_norm(local54.to(dtype=hl.dtype))

        q = self.q(hp)
        k = self.k_lig(hl) + self.k_local(lf)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.pair_dim)
        scores = scores.float().clamp(-self.score_clip, self.score_clip)

        atom_mask = atom_mask.bool()
        scores = scores.masked_fill(~atom_mask.unsqueeze(1), 0.0)
        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))
        return self.beta().to(dtype=scores.dtype) * scores


class LocalPairBiasedCrossAttentionBlock(nn.Module):
    """
    Drop-in replacement for CrossAttentionBlock.prot_to_lig.

    The original norm/attention/FFN modules are reused, so V11 checkpoint keys
    stay compatible.  The only new parameters live under ``pair_bias``.
    """

    def __init__(
        self,
        base_block: nn.Module,
        d_model: int,
        local_dim: int,
        pair_dim: int,
        score_clip: float,
        beta_init: float,
        beta_max: float,
    ):
        super().__init__()
        self.drop_path_prob = base_block.drop_path_prob
        self.norm_q = base_block.norm_q
        self.norm_kv = base_block.norm_kv
        self.cross_attn = base_block.cross_attn
        self.norm2 = base_block.norm2
        self.ffn = base_block.ffn
        self.dropout = base_block.dropout
        self.pair_bias = LocalPairAttentionBias(
            d_model=d_model,
            local_dim=local_dim,
            pair_dim=pair_dim,
            score_clip=score_clip,
            beta_init=beta_init,
            beta_max=beta_max,
        )
        self._local54: Optional[torch.Tensor] = None
        self._atom_mask: Optional[torch.Tensor] = None

    def set_pair_context(self, local54: torch.Tensor, atom_mask: torch.Tensor) -> None:
        self._local54 = local54
        self._atom_mask = atom_mask.bool()

    def clear_pair_context(self) -> None:
        self._local54 = None
        self._atom_mask = None

    def _build_pair_bias(
        self,
        h_prot_raw: torch.Tensor,
        h_lig_raw: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self._local54 is None or self._atom_mask is None:
            return None

        # Ligand sequence is [global_token, atom_0, ..., atom_M-1].
        n_prefix = h_lig_raw.size(1) - self._local54.size(1)
        if n_prefix < 0:
            return None

        h_lig_atoms = h_lig_raw[:, n_prefix:, :]
        local54 = self._local54[:, : h_lig_atoms.size(1), :]
        atom_mask = self._atom_mask[:, : h_lig_atoms.size(1)]
        atom_bias = self.pair_bias(h_prot_raw, h_lig_atoms, local54, atom_mask)

        if n_prefix == 0:
            return atom_bias
        prefix_bias = atom_bias.new_zeros(atom_bias.size(0), atom_bias.size(1), n_prefix)
        return torch.cat([prefix_bias, atom_bias], dim=-1)

    def _cross_attention_with_bias(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor],
        pair_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_mod = self.cross_attn
        B, L, _ = x.shape
        S = kv.size(1)

        q = attn_mod.w_q(x).view(B, L, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
        k = attn_mod.w_k(kv).view(B, S, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
        v = attn_mod.w_v(kv).view(B, S, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * attn_mod.scale
        if pair_bias is not None:
            attn = attn + pair_bias.unsqueeze(1).to(dtype=attn.dtype)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        with torch.no_grad():
            attn_mod._last_attn_weights = attn.detach()
            attn_clamped = attn.clamp(min=1e-8)
            attn_mod._last_attn_entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1).mean()

        attn = attn_mod.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, attn_mod.d_model)
        out = attn_mod.w_o(out)

        if attn_mod.use_gate:
            gate = torch.sigmoid(attn_mod.gate(x))
            with torch.no_grad():
                attn_mod._last_gate_openness = gate.mean()
            out = out * gate
        return out

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = q
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        pair_bias = self._build_pair_bias(q, kv)

        out = self._cross_attention_with_bias(
            x=q_normed,
            kv=kv_normed,
            mask=kv_mask,
            pair_bias=pair_bias,
        )
        out = drop_path(self.dropout(out), self.drop_path_prob, self.training) + residual
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).float()

        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = drop_path(self.dropout(out), self.drop_path_prob, self.training) + residual
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).float()
        return out


class PocketGVPFusionV11m5(PocketGVPFusionV11m3):
    """
    V11m5: V11m4 stable delta branch plus local54 attention-bias injection.
    """

    DEFAULT_CONFIG = {
        **PocketGVPFusionV11m3.DEFAULT_CONFIG,
        "use_local_pair_delta": True,
        "pair_delta_local_dim": 54,
        "pair_delta_dim": 128,
        "pair_delta_hidden": 256,
        "pair_delta_dropout": 0.1,
        "pair_delta_score_clip": 20.0,
        "pair_delta_logit_scale_init": 0.5,
        "pair_delta_logit_scale_max": 1.0,
        "use_local_pair_bias": True,
        "pair_bias_dim": 128,
        "pair_bias_score_clip": 20.0,
        "pair_bias_beta_init": 0.1,
        "pair_bias_beta_max": 1.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

        self.use_local_pair_delta = bool(cfg.get("use_local_pair_delta", True))
        self.pair_delta_local_dim = int(cfg.get("pair_delta_local_dim", 54))
        self.pair_delta_dim = int(cfg.get("pair_delta_dim", 128))
        self.pair_delta_score_clip = float(cfg.get("pair_delta_score_clip", 20.0))
        self.pair_delta_logit_scale_max = float(cfg.get("pair_delta_logit_scale_max", 1.0))
        self.use_local_pair_bias = bool(cfg.get("use_local_pair_bias", True))

        d_model = int(self.config["d_model"])
        n_classes = int(self.config["n_classes"])
        hidden = int(cfg.get("pair_delta_hidden", 256))
        dropout = float(cfg.get("pair_delta_dropout", 0.1))

        if self.use_local_pair_delta:
            self.pair_delta_prot_norm = nn.LayerNorm(d_model)
            self.pair_delta_lig_norm = nn.LayerNorm(d_model)
            self.pair_delta_local_norm = nn.LayerNorm(self.pair_delta_local_dim)

            self.pair_delta_q = nn.Linear(d_model, self.pair_delta_dim, bias=False)
            self.pair_delta_k_lig = nn.Linear(d_model, self.pair_delta_dim, bias=False)
            self.pair_delta_k_local = nn.Linear(
                self.pair_delta_local_dim, self.pair_delta_dim, bias=False
            )
            self.pair_delta_v = nn.Linear(d_model, d_model, bias=False)

            delta_in_dim = d_model + d_model + 3
            self.pair_delta_head = nn.Sequential(
                nn.LayerNorm(delta_in_dim),
                nn.Linear(delta_in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            # Exact V11m3 behavior at initialization. The final layer learns the
            # delta first; earlier pair modules start receiving gradients after
            # the first update, avoiding a large initial branch perturbation.
            final = self.pair_delta_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

            init_scale = float(cfg.get("pair_delta_logit_scale_init", 0.5))
            self.pair_delta_logit_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.pair_delta_head = None
            self.register_buffer("pair_delta_logit_scale", torch.tensor(0.0), persistent=False)

        if self.use_local_pair_bias:
            # Plain list only: blocks are already registered under
            # cross_attention_layers[*].prot_to_lig.  Re-registering them in a
            # ModuleList would duplicate state_dict paths.
            self._local_pair_biased_blocks = []
            for cross_layer in self.cross_attention_layers:
                biased = LocalPairBiasedCrossAttentionBlock(
                    base_block=cross_layer.prot_to_lig,
                    d_model=d_model,
                    local_dim=self.pair_delta_local_dim,
                    pair_dim=int(cfg.get("pair_bias_dim", self.pair_delta_dim)),
                    score_clip=float(cfg.get("pair_bias_score_clip", self.pair_delta_score_clip)),
                    beta_init=float(cfg.get("pair_bias_beta_init", 0.1)),
                    beta_max=float(cfg.get("pair_bias_beta_max", 1.0)),
                )
                cross_layer.prot_to_lig = biased
                self._local_pair_biased_blocks.append(biased)
        else:
            self._local_pair_biased_blocks = []

        self.config["use_local_pair_delta"] = self.use_local_pair_delta
        self.config["pair_delta_local_dim"] = self.pair_delta_local_dim
        self.config["pair_delta_dim"] = self.pair_delta_dim
        self.config["use_local_pair_bias"] = self.use_local_pair_bias

    def _pad_ligand_local54(
        self,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter flat ligand local features to [B, M, local_dim]."""
        batch_size, max_atoms = ligand_mask.shape
        local = ligand_x[:, : self.pair_delta_local_dim]
        out = ligand_x.new_zeros((batch_size, max_atoms, self.pair_delta_local_dim))

        n_atoms = int(local.size(0))
        if n_atoms == 0:
            return out

        if not bool((ligand_batch[1:] >= ligand_batch[:-1]).all()):
            raise ValueError("ligand_batch must be non-decreasing for V11m5 local scatter")

        counts = torch.bincount(ligand_batch, minlength=batch_size)
        cumsum = torch.zeros(batch_size + 1, dtype=torch.long, device=ligand_x.device)
        cumsum[1:] = counts.cumsum(0)
        local_idx = torch.arange(n_atoms, device=ligand_x.device) - cumsum[ligand_batch]
        valid = local_idx < max_atoms
        out[ligand_batch[valid], local_idx[valid]] = local[valid]
        return out

    def _compute_pair_delta(
        self,
        h_prot: torch.Tensor,
        h_ligand_fused: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Coordinate-free residue-atom compatibility delta logits."""
        if not self.use_local_pair_delta:
            return h_prot.new_zeros((*h_prot.shape[:2], int(self.config["n_classes"])))

        # h_ligand_fused includes the dynamic global token at position 0.
        max_atoms = ligand_mask.size(1)
        h_lig_atoms = h_ligand_fused[:, 1 : 1 + max_atoms, :]
        local54 = self._pad_ligand_local54(ligand_x, ligand_batch, ligand_mask)

        hp = self.pair_delta_prot_norm(h_prot)
        hl = self.pair_delta_lig_norm(h_lig_atoms)
        lf = self.pair_delta_local_norm(local54.to(dtype=hl.dtype))

        q = self.pair_delta_q(hp)
        k = self.pair_delta_k_lig(hl) + self.pair_delta_k_local(lf)
        v = self.pair_delta_v(hl)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.pair_delta_dim)
        scores = scores.float().clamp(-self.pair_delta_score_clip, self.pair_delta_score_clip)

        atom_mask = ligand_mask.bool()
        score_mask = atom_mask.unsqueeze(1)  # [B, 1, M]
        scores = scores.masked_fill(~score_mask, -1.0e4)

        has_atom = atom_mask.any(dim=1).view(-1, 1, 1)
        scores = torch.where(has_atom, scores, torch.zeros_like(scores))

        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        attn = attn * atom_mask.unsqueeze(1).to(dtype=attn.dtype)

        # Keep no-atom rows exactly zero instead of uniform.
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = torch.where(has_atom, attn / denom, torch.zeros_like(attn))

        lig_read = torch.matmul(attn, v)

        masked_scores = scores.masked_fill(~score_mask, -1.0e4)
        pair_max = masked_scores.max(dim=-1).values
        pair_lse = torch.logsumexp(masked_scores, dim=-1)
        n_valid = atom_mask.sum(dim=1).clamp(min=1).to(dtype=pair_lse.dtype).view(-1, 1)
        pair_lse = pair_lse - n_valid.log()
        entropy = -(attn.float() * (attn.float().clamp(min=1e-8).log())).sum(dim=-1)

        stats = torch.stack([pair_max, pair_lse, entropy], dim=-1).to(dtype=h_prot.dtype)
        delta_in = torch.cat([h_prot, lig_read.to(dtype=h_prot.dtype), stats], dim=-1)
        delta_logits = self.pair_delta_head(delta_in)
        delta_logits = delta_logits * protein_mask.unsqueeze(-1).to(dtype=delta_logits.dtype)
        return delta_logits

    def _delta_scale(self) -> torch.Tensor:
        if self.pair_delta_logit_scale_max <= 0:
            return self.pair_delta_logit_scale
        return self.pair_delta_logit_scale.tanh() * self.pair_delta_logit_scale_max

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
    ) -> Dict[str, torch.Tensor]:
        local54_ctx = None
        if self.use_local_pair_bias:
            local54_ctx = self._pad_ligand_local54(ligand_x, ligand_batch, ligand_mask)
            for block in self._local_pair_biased_blocks:
                block.set_pair_context(local54_ctx, ligand_mask)

        try:
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
            )
        finally:
            for block in self._local_pair_biased_blocks:
                block.clear_pair_context()

        base_logits = outputs["logits"]
        delta_logits = self._compute_pair_delta(
            h_prot=outputs["h_fused"],
            h_ligand_fused=outputs["h_ligand_fused"],
            ligand_x=ligand_x,
            ligand_batch=ligand_batch,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
        )
        scale = self._delta_scale().to(dtype=base_logits.dtype)
        logits = base_logits + scale * delta_logits

        outputs["base_logits"] = base_logits
        outputs["delta_logits"] = delta_logits
        outputs["pair_delta_scale"] = scale.detach()
        outputs["pair_delta_abs_mean"] = delta_logits.detach().abs().mean()
        if self._local_pair_biased_blocks:
            betas = torch.stack([b.pair_bias.beta().detach() for b in self._local_pair_biased_blocks])
            outputs["pair_bias_beta_mean"] = betas.mean()
        outputs["logits"] = logits
        return outputs
