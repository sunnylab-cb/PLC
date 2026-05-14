"""
Pocket-GVP-Fusion V12: Interleaved Co-Attention (Evoformer-style) + Uni-Mol FiLM.

Architecture changes from V11:
  移除: 獨立 Protein Context Encoder (Module 4)
        獨立 Ligand Self-Attention (Module 5b)
        Cross-Attention Loop (Module 6)
  新增: N=6 個 CoAttentionBlock，每個 Block 包含:
          1. FiLM conditioning (Uni-Mol CLS → protein 全域化學調製)
          2. Protein Self-Attention (RoPE + 3D distance bias)
          3. Ligand Self-Attention (3D distance bias)
          4. Bidirectional Cross-Attention (P→L + L→P)

保留自 V11: Modules 1-3 (GVP + ESM Fusion), Module 5a (Ligand GNN), Module 7 (Classifier)
             DistanceBias, LigandDistanceBias, RoPE, GVP Output Dropout, Coord Noise

新增 Innovation:
  [1] FiLMLayer: 每個 Block 以 Uni-Mol 全域化學語義廣播調製蛋白質

重要限制:
  蛋白質坐標與配體坐標各自處於獨立坐標系（非 Holo 對齊態）。
  絕不能計算跨模態幾何距離（Cα ↔ 配體原子），此類特徵屬於偽特徵。
  合法幾何信息：蛋白質內 Cα-Cα 距離、配體原子間距離（各自坐標系內部）。

接受 unimol_embed [B, 512] 作為新輸入 (若不提供則退化為零向量，FiLM = identity)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple

from .gvp_layers import GVPEncoder, build_protein_graph, rbf_encode
from .ligand_encoder_3d import Ligand3DEncoder
from .chain_encoding import get_cumulative_gap_position_ids
from .transformer_layers import CrossAttentionBlock

# Re-use from V11 (avoid code duplication)
from .pocket_gvp_fusion_v11 import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    RoPEGatedMultiHeadAttention,
    DistanceBias,
    LigandDistanceBias,
    SwiGLU,
    RoPEGatedTransformerBlock,
    PairwiseBridge,
    InterBlockAttnRes,
    drop_path,
)


# ===========================================================================
# FiLM Conditioning Layer  [V12 Innovation #1]
# ===========================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation from Uni-Mol global CLS embedding.

    Residual delta form:
        h_prot ← h_prot + γ * LN(h_prot) + β

    γ, β are predicted from unimol_embed [B, d_unimol].
    Zero-init → γ=0, β=0 → delta=0 → true identity (output = h_prot exactly).

    Using LN inside delta (not on main path) keeps the conditioning numerically
    stable while leaving h_prot unmodified when unimol_embed is absent.
    """

    def __init__(self, d_unimol: int = 512, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_unimol, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_prot: torch.Tensor, unimol_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prot:       [B, L, d_model]
            unimol_embed: [B, d_unimol]
        Returns:
            [B, L, d_model]  — equals h_prot exactly when proj is zero-init
        """
        params = self.proj(unimol_embed)               # [B, 2*d_model]
        gamma, beta = params.chunk(2, dim=-1)          # [B, d_model] each
        delta = gamma.unsqueeze(1) * self.norm(h_prot) + beta.unsqueeze(1)
        return h_prot + delta


# ===========================================================================
# SASA Residual Gate  [V12 Innovation #2]
# ===========================================================================

class SASAResidualGate(nn.Module):
    """
    Learnable SASA residual gate applied to h_prot before the classifier.

    Form:  h_out = h_prot * (1 + amplitude * tanh(linear(rsa) * rsa_mask))

    - Zero-init linear → tanh(0) = 0 → gate = 1 → identity at training start.
    - After training, the gate direction and magnitude are fully learnable;
      it is NOT a fixed rule that surface residues get amplified.
    - amplitude bounds the modulation to ±amplitude (default 0.5 → [0.5×, 1.5×]).
    - rsa_mask (0/1 float): where 0, forces delta=0 (gate=identity) regardless of
      rsa_raw value. Prevents spurious bias when spatial_scalar_raw_features is
      absent in older .pt files (missing RSA ≠ RSA=0 = "fully buried").

    Inputs:
        h_prot:    [B, L, d_model]
        rsa_raw:   [B, L, 1]  raw RSA values in [0, 1]
        rsa_mask:  [B, L, 1]  float mask: 1=real RSA, 0=absent (gate forced to identity)
        prot_mask: [B, L]     bool mask: True = valid residue (padding zeroed)
    """

    def __init__(self, amplitude: float = 0.5):
        super().__init__()
        self.amplitude = amplitude
        self.proj = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        h_prot:    torch.Tensor,
        rsa_raw:   torch.Tensor,
        rsa_mask:  Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.amplitude * torch.tanh(self.proj(rsa_raw))   # [B, L, 1]
        if rsa_mask is not None:
            delta = delta * rsa_mask.to(dtype=delta.dtype)        # zero out absent RSA
        h_out = h_prot * (1.0 + delta)
        if prot_mask is not None:
            h_out = h_out * prot_mask.unsqueeze(-1).float()
        return h_out


# ===========================================================================
# Inter-Block AttnRes V2  [V12: partial-biased init]
# ===========================================================================

class InterBlockAttnResV2(nn.Module):
    """
    Weighted merge of anchor representations + current partial.

    Like V11's InterBlockAttnRes but with partial-biased initialization so that
    at training start the output ≈ partial (identity), satisfying V12's
    "zero-disruption" principle. This is in contrast to the original which
    starts as a uniform average over all sources.

    Mechanism:
        sources = stack(anchors + [partial])          # [K+1, B, L, d]
        logits  = proj(RMSNorm(sources)) + partial_bias  # partial gets +bias
        alpha   = softmax(logits, dim=0)              # [K+1, B, L]
        output  = (alpha[..., None] * sources).sum(0) # [B, L, d]

    With proj zero-init and partial_logit_bias=4.0:
        K=1 anchor: softmax([0, 4]) ≈ [0.018, 0.982] → ~98% partial
        K=2 anchors: softmax([0, 0, 4]) ≈ [0.009, 0.009, 0.982]
    Training gradually learns to increase anchor weights as needed.

    partial_logit_bias is a *learnable scalar* — the model can widen or narrow
    the residual window over training.
    """

    def __init__(self, d_model: int, init_partial_bias: float = 4.0):
        super().__init__()
        self.proj      = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.key_norm  = nn.RMSNorm(d_model)
        # Learnable bias that gives partial an initial logit advantage
        self.partial_logit_bias = nn.Parameter(torch.tensor(init_partial_bias))

    def forward(
        self,
        blocks:  List[torch.Tensor],   # anchor tensors  [B, L, d] each
        partial: torch.Tensor,          # current repr    [B, L, d]
    ) -> torch.Tensor:
        sources = torch.stack(blocks + [partial], dim=0)   # [K+1, B, L, d]
        keys    = self.key_norm(sources)
        logits  = self.proj(keys).squeeze(-1)              # [K+1, B, L]
        logits[-1] = logits[-1] + self.partial_logit_bias  # boost partial
        alpha   = logits.softmax(dim=0)                    # [K+1, B, L]
        return (alpha.unsqueeze(-1) * sources).sum(dim=0)  # [B, L, d]


# ===========================================================================
# Co-Attention Block  (V12 core building block)
# ===========================================================================

class CoAttentionBlock(nn.Module):
    """
    Interleaved Co-Attention Block (AlphaFold2 Evoformer style).

    Sub-operations per block (in order):
      1. FiLM:  Uni-Mol CLS embed → (γ, β) applied to h_prot
      2. Prot SA: protein residues attend to each other (RoPE + intra-protein 3D dist bias)
      3. Lig SA:  ligand atoms attend to each other (intra-ligand 3D dist bias)
      4. P→L CA: protein queries ligand atoms
      5. L→P CA: ligand atoms query updated protein residues

    Geometric distances used:
      - Protein self-attn: Cα-Cα distances (within protein coordinate system)
      - Ligand self-attn:  atom-atom distances (within ligand coordinate system)
      *** Cross-attention: NO distance bias — protein and ligand are in separate
          coordinate systems (NOT Holo state), so cross-modal distances are meaningless.

    After N=6 blocks, protein and ligand co-evolve in the same representation
    space; no separate alignment step is needed at classification.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,
        attn_dropout: Optional[float] = None,
        d_unimol: int = 512,
        use_film: bool = True,
        use_prot_dist_bias: bool = True,
        use_lig_dist_bias: bool = True,
    ):
        super().__init__()

        # 1. FiLM conditioning
        self.film = FiLMLayer(d_unimol, d_model) if use_film else None

        # 2. Protein Self-Attention (RoPE + intra-protein 3D dist bias)
        self.prot_self_attn = RoPEGatedTransformerBlock(
            d_model, n_heads, dropout,
            use_gated_attn=True, use_swiglu=True,
            drop_path_prob=drop_path_prob, attn_dropout=attn_dropout,
        )
        self.prot_dist_bias = DistanceBias(n_heads) if use_prot_dist_bias else None

        # 3. Ligand Self-Attention (intra-ligand 3D dist bias; no RoPE)
        self.lig_self_attn = RoPEGatedTransformerBlock(
            d_model, n_heads, dropout,
            use_gated_attn=True, use_swiglu=True,
            drop_path_prob=drop_path_prob, attn_dropout=attn_dropout,
        )
        self.lig_dist_bias = LigandDistanceBias(n_heads) if use_lig_dist_bias else None

        # 4 & 5. Bidirectional Cross-Attention (standard; no geometric bias)
        self.prot_to_lig = CrossAttentionBlock(d_model, n_heads, dropout, drop_path_prob, attn_dropout)
        self.lig_to_prot = CrossAttentionBlock(d_model, n_heads, dropout, drop_path_prob, attn_dropout)

    def forward(
        self,
        h_prot:       torch.Tensor,
        h_lig:        torch.Tensor,
        unimol_embed: torch.Tensor,
        prot_mask:    Optional[torch.Tensor],
        lig_mask:     Optional[torch.Tensor],     # [B, M+1] includes global token
        prot_coords:  Optional[torch.Tensor],     # [B, L, 3]  Cα (protein coord sys)
        lig_coords:   Optional[torch.Tensor],     # [B, M+1, 3] atom pos (ligand coord sys)
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. FiLM: global ligand chemistry conditions protein repr
        if self.film is not None:
            h_prot = self.film(h_prot, unimol_embed)

        # 2. Protein Self-Attention (intra-protein geometry only)
        prot_attn_bias = None
        if self.prot_dist_bias is not None and prot_coords is not None:
            prot_attn_bias = self.prot_dist_bias(prot_coords, mask=prot_mask)
        h_prot = self.prot_self_attn(
            h_prot, mask=prot_mask, attn_bias=prot_attn_bias, position_ids=position_ids
        )

        # 3. Ligand Self-Attention (intra-ligand geometry only)
        # lig_coords [B, M+1, 3]: position 0 = global token (zeros), 1..M = atoms.
        # LigandDistanceBias expects atom positions only (strip global token at idx 0).
        # Returns [B, n_heads, M+1, M+1] with zero bias for global token row/col.
        #
        # RoPE neutralization: pass all-zero position_ids so every atom gets
        # identical RoPE phase → pairwise relative encoding = 0 for all pairs
        # → RoPE has zero effect. Ligand atom order is arbitrary (RDKit enumeration),
        # so injecting sequential RoPE would be a spurious positional bias.
        lig_attn_bias = None
        if self.lig_dist_bias is not None and lig_coords is not None:
            atom_coords = lig_coords[:, 1:, :]                                     # [B, M, 3]
            atom_mask   = lig_mask[:, 1:] if lig_mask is not None else None        # [B, M]
            lig_attn_bias = self.lig_dist_bias(atom_coords, atom_mask=atom_mask)
        B_lig, M1, _ = h_lig.shape
        lig_zero_pos = torch.zeros(B_lig, M1, dtype=torch.long, device=h_lig.device)
        h_lig = self.lig_self_attn(
            h_lig, mask=lig_mask, attn_bias=lig_attn_bias, position_ids=lig_zero_pos
        )

        # 4 & 5. Parallel Bidirectional Cross-Attention (true Evoformer-style)
        # Both branches read from pre-update h_prot / h_lig so neither side
        # influences the other within the same block — gradient flows are symmetric.
        h_prot_new = self.prot_to_lig(
            q=h_prot, kv=h_lig, q_mask=prot_mask, kv_mask=lig_mask
        )
        h_lig_new = self.lig_to_prot(
            q=h_lig, kv=h_prot, q_mask=lig_mask, kv_mask=prot_mask   # note: h_prot not h_prot_new
        )

        return h_prot_new, h_lig_new


# ===========================================================================
# Main Model
# ===========================================================================

class PocketGVPFusionV12(nn.Module):
    """
    Pocket-GVP-Fusion V12: Interleaved Co-Attention + Uni-Mol FiLM.

    Config is backward-compatible with V11 (same key names for shared modules).
    V12-specific keys:
        n_co_attn_blocks:    int  = 6     # Co-Attention block count
        use_film:            bool = True  # FiLM from Uni-Mol
        use_prot_dist_bias:  bool = True  # intra-protein 3D dist bias in protein SA
        use_lig_dist_bias:   bool = True  # intra-ligand 3D dist bias in ligand SA
        d_unimol:            int  = 512   # Uni-Mol embedding dim
    """

    DEFAULT_CONFIG: Dict = {
        # ── Shared with V11 (same keys for checkpoint compatibility) ─────────
        'd_esm':              1280,
        'd_phys':             62,
        'd_spatial_scalar':   89,
        'd_phys_encoded':     128,
        'd_gvp_internal':     256,
        'd_model':            512,
        'd_ligand_atom':      109,
        'n_gvp_vectors':      12,
        'n_gvp_vector_hidden': 20,
        'n_gvp_layers':       3,
        'n_ligand_layers':    3,
        'n_heads':            8,
        'dropout':            0.1,
        'attn_dropout':       0.1,
        'classifier_dropout': 0.2,
        'stochastic_depth_prob': 0.1,
        'esm_dropout_prob':   0.1,
        'physchem_dropout_prob': 0.2,
        'gvp_dropout_prob':   0.1,
        'coord_noise_std':    0.5,
        'gvp_edge_rbf_bins':  16,
        'gvp_edge_rbf_dmax':  20.0,
        'ligand_edge_rbf_bins': 8,
        'gvp_radius':         10.0,
        'gvp_k_neighbors':    20,
        'edge_dim':           6,
        'use_gradient_checkpointing': False,
        'n_classes':          3,
        'use_chain_features': True,
        'use_chain_rope':     None,   # None → inherit from use_chain_features
        'use_chain_edge':     None,   # None → inherit from use_chain_features
        'chain_gap_size':     1000,
        'use_dist_aux':       False,
        # ── V12-specific ─────────────────────────────────────────────────────
        'd_unimol':             512,
        'n_co_attn_blocks':     6,
        'use_film':             True,
        'use_prot_dist_bias':   True,
        'use_lig_dist_bias':    True,
        'sasa_gate_amplitude':  0.5,   # tanh gate range: [1-amp, 1+amp] × h_prot
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

        # Resolve chain flags (None → inherit master switch)
        _master = c.get('use_chain_features', True)
        self._use_chain_rope = c['use_chain_rope'] if c.get('use_chain_rope') is not None else _master
        self._use_chain_edge = c['use_chain_edge'] if c.get('use_chain_edge') is not None else _master

        # ── Module 1: PhysChem MLP Encoder (identical to V11) ───────────────
        self.physchem_encoder = nn.Sequential(
            nn.Linear(c['d_phys'], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c['d_phys_encoded']),
            nn.LayerNorm(c['d_phys_encoded']),
            nn.GELU(),
        )

        # ── Module 2: Protein Geometric Encoder (GVP) — identical to V11 ────
        _rbf_bins   = c.get('gvp_edge_rbf_bins', 16)
        _edge_s_dim = _rbf_bins + 4 + (1 if self._use_chain_edge else 0)
        _edge_v_dim = 2
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
            node_h_dim =(c['d_gvp_internal'],   c['n_gvp_vector_hidden']),
            edge_in_dim=(_edge_s_dim,            _edge_v_dim),
            n_layers   = c['n_gvp_layers'],
            drop_rate  = c['dropout'],
        )

        # ── Module 3: Bridge & Gating Fusion (identical to V11) ─────────────
        self.pairwise_bridge = PairwiseBridge(n_vec_channels=c['n_gvp_vector_hidden'])
        n_pairwise = (c['n_gvp_vector_hidden'] * (c['n_gvp_vector_hidden'] + 1)) // 2
        d_geo  = c['d_gvp_internal'] + n_pairwise + c['d_phys_encoded']
        d_half = c['d_model'] // 2

        self.gvp_proj = nn.Sequential(
            nn.Linear(d_geo, d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout']),
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(c['d_esm'], d_half),
            nn.LayerNorm(d_half),
            nn.Dropout(c['dropout']),
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(d_half * 2, d_half),
            nn.ReLU(),
            nn.Linear(d_half, d_half),
            nn.Sigmoid(),
        )
        self.input_norm = nn.LayerNorm(c['d_model'])

        # ── Module 5a: Ligand 3D GNN Encoder (identical to V11) ─────────────
        _lig_edge_in = c['edge_dim'] + c.get('ligand_edge_rbf_bins', 8)
        self.ligand_encoder = Ligand3DEncoder(
            node_in_dim   = c['d_ligand_atom'],
            node_h_dim    = c['d_gvp_internal'] // 2,
            n_vec_channels = 4,
            n_vec_hidden  = 16,
            edge_in_dim   = _lig_edge_in,
            out_dim       = c['d_model'],
            n_layers      = c['n_ligand_layers'],
            dropout       = c['dropout'],
            radius        = 5.0,
            max_neighbors = 32,
        )

        # ── Module 6 (V12): N × CoAttentionBlock ────────────────────────────
        n_blocks  = c['n_co_attn_blocks']
        sdp_max   = c.get('stochastic_depth_prob', 0.0)
        dpp       = [sdp_max * i / max(n_blocks - 1, 1) for i in range(n_blocks)]
        attn_drop = c.get('attn_dropout', None)

        self.co_attn_blocks = nn.ModuleList([
            CoAttentionBlock(
                d_model          = c['d_model'],
                n_heads          = c['n_heads'],
                dropout          = c['dropout'],
                drop_path_prob   = dpp[i],
                attn_dropout     = attn_drop,
                d_unimol         = c['d_unimol'],
                use_film         = c['use_film'],
                use_prot_dist_bias = c['use_prot_dist_bias'],
                use_lig_dist_bias  = c['use_lig_dist_bias'],
            )
            for i in range(n_blocks)
        ])

        # ── RSA Input Projection  [V12 Innovation #3] ───────────────────────
        # Inject surface signal into h_prot BEFORE co-attention so that protein
        # SA and P→L cross-attn both see the RSA prior from block 0 onward.
        # Zero-init → identity at training start.  rsa_mask guards against
        # missing-RSA samples being treated as RSA=0 ("fully buried").
        self.rsa_input_proj = nn.Linear(1, c['d_model'], bias=False)
        nn.init.zeros_(self.rsa_input_proj.weight)

        # ── Uni-Mol → ligand global token  [V12 Innovation #4] ─────────────
        # Augments the mean-pool global token (h_lig[:,0]) with Uni-Mol info
        # so protein residues can selectively query it via P→L cross-attention,
        # not just receive it as a global FiLM broadcast.
        # Zero-init → starts as pure mean-pool; learns to expose Uni-Mol signal.
        self.unimol_to_global = nn.Linear(c['d_unimol'], c['d_model'], bias=False)
        nn.init.zeros_(self.unimol_to_global.weight)

        # Final LN before classifier
        self.final_norm = nn.LayerNorm(c['d_model'])

        # ── Inter-Block AttnRes V2 (one per co-attention block) ─────────────
        # Applied BEFORE each CoAttentionBlock so anchors influence every layer.
        # Anchors: [h_prot_0] for protein, [h_lig_0] for ligand (pre-blocks repr).
        # partial_logit_bias=4.0 → output ≈ 98% partial at init → identity-like.
        n_blocks = c.get('n_co_attn_blocks', 6)
        self.prot_attn_res = nn.ModuleList([
            InterBlockAttnResV2(d_model=c['d_model']) for _ in range(n_blocks)
        ])
        self.lig_attn_res = nn.ModuleList([
            InterBlockAttnResV2(d_model=c['d_model']) for _ in range(n_blocks)
        ])

        # ── SASA Residual Gate (soft surface weighting, identity-init) ───────
        self.sasa_gate = SASAResidualGate(amplitude=c.get('sasa_gate_amplitude', 0.5))

        # ── Module 7: Classifier (identical structure to V11) ────────────────
        d_cls = c['d_model'] + c['d_phys_encoded']
        self.classifier_input_norm = nn.LayerNorm(d_cls)
        self.classifier = nn.Sequential(
            nn.Linear(d_cls, 256),
            nn.ReLU(),
            nn.Dropout(c.get('classifier_dropout', 0.2)),
            nn.Linear(256, c['n_classes']),
        )

        # Optional distance regression aux head
        if c.get('use_dist_aux', False):
            self.dist_aux_head = nn.Sequential(
                nn.Linear(c['d_model'], 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Softplus(),
            )
        else:
            self.dist_aux_head = None

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        esm_embeddings:         torch.Tensor,           # [B, L, 1280]
        protein_physchem:       torch.Tensor,           # [B, L, d_phys]
        protein_spatial_scalar: torch.Tensor,           # [B, L, d_spatial_scalar]
        protein_spatial_vector: torch.Tensor,           # [B, L, n_gvp_vectors_spatial, 3]
        protein_gvp_vectors:    torch.Tensor,           # [B, L, n_gvp_vectors_backbone, 3]
        protein_coords:         torch.Tensor,           # [B, L, 3]  Cα (protein coord sys)
        protein_mask:           torch.Tensor,           # [B, L]     True=valid
        ligand_x:               torch.Tensor,           # [N_atoms, d_ligand_atom]
        ligand_pos:             torch.Tensor,           # [N_atoms, 3] ligand coord sys
        ligand_edge_index:      torch.Tensor,           # [2, N_edges]
        ligand_edge_attr:       torch.Tensor,           # [N_edges, edge_dim]
        ligand_batch:           torch.Tensor,           # [N_atoms]  sample index
        ligand_mask:            torch.Tensor,           # [B, M_max] True=valid
        unimol_embed:           Optional[torch.Tensor] = None,  # [B, 512]
        protein_rsa_raw:        Optional[torch.Tensor] = None,  # [B, L, 1] raw RSA in [0,1]
        protein_rsa_mask:       Optional[torch.Tensor] = None,  # [B, L, 1] 1=real RSA, 0=absent
        chain_ids:              Optional[torch.Tensor] = None,  # [B, L]; required when use_chain_edge=True
    ) -> Dict[str, torch.Tensor]:

        B, L, _ = esm_embeddings.shape
        device   = esm_embeddings.device
        c        = self.config

        # chain_ids is mandatory when use_chain_edge=True (GVP edge dim depends on it)
        if self._use_chain_edge and chain_ids is None:
            raise ValueError(
                "chain_ids must be provided when use_chain_edge=True "
                "(pass torch.zeros(B, L, dtype=torch.long) for single-chain proteins)"
            )

        # Chain-gap RoPE positions
        position_ids = None
        if self._use_chain_rope and chain_ids is not None:
            position_ids = get_cumulative_gap_position_ids(
                chain_ids, gap_size=c.get('chain_gap_size', 1000)
            )

        # ── Module 1: PhysChem ───────────────────────────────────────────────
        protein_physchem_encoded = self.physchem_encoder(protein_physchem)
        if self.training and c.get('physchem_dropout_prob', 0.0) > 0:
            pm = (torch.rand(B, 1, 1, device=device) > c['physchem_dropout_prob']).float()
            protein_physchem_encoded = protein_physchem_encoded * pm

        # ── Module 2: GVP Encoder ───────────────────────────────────────────
        protein_vectors_combined = torch.cat([protein_gvp_vectors, protein_spatial_vector], dim=2)
        protein_n_coords = protein_coords + protein_gvp_vectors[:, :, 1, :]
        protein_c_coords = protein_coords + protein_gvp_vectors[:, :, 3, :]

        mask_bool          = protein_mask.bool()
        n_valid_per_sample = mask_bool.sum(dim=1)
        total_valid        = n_valid_per_sample.sum().item()

        gvp_rbf_bins    = c.get('gvp_edge_rbf_bins', 16)
        gvp_rbf_dmax    = c.get('gvp_edge_rbf_dmax', 20.0)
        coord_noise_std = c.get('coord_noise_std', 0.0)

        if total_valid > 0:
            all_scalar  = protein_spatial_scalar[mask_bool]
            all_vectors = protein_vectors_combined[mask_bool]
            all_ca      = protein_coords[mask_bool]
            all_n       = protein_n_coords[mask_bool]
            all_cpos    = protein_c_coords[mask_bool]
            all_chain   = chain_ids[mask_bool] if chain_ids is not None else None

            if self.training and coord_noise_std > 0:
                noise    = torch.randn_like(all_ca) * coord_noise_std
                all_ca   = all_ca   + noise
                all_n    = all_n    + noise
                all_cpos = all_cpos + noise

            edge_index_list, edge_s_list, edge_v_list = [], [], []
            node_offset = 0
            for b in range(B):
                n_b = n_valid_per_sample[b].item()
                if n_b == 0:
                    continue
                chain_b = all_chain[node_offset:node_offset + n_b] if all_chain is not None else None
                ei_b, es_b, ev_b = build_protein_graph(
                    coords_ca = all_ca  [node_offset:node_offset + n_b],
                    coords_n  = all_n   [node_offset:node_offset + n_b],
                    coords_c  = all_cpos[node_offset:node_offset + n_b],
                    k         = c['gvp_k_neighbors'],
                    radius    = c['gvp_radius'],
                    num_rbf   = gvp_rbf_bins,
                    rbf_dmax  = gvp_rbf_dmax,
                    chain_ids = chain_b if self._use_chain_edge else None,
                )
                edge_index_list.append(ei_b + node_offset)
                edge_s_list.append(es_b)
                edge_v_list.append(ev_b)
                node_offset += n_b

            all_edge_index = torch.cat(edge_index_list, dim=1)
            all_edge_s     = torch.cat(edge_s_list, dim=0)
            all_edge_v     = torch.cat(edge_v_list, dim=0)

            h_s_flat, h_v_flat = self.gvp_encoder(
                node_s=all_scalar, node_v=all_vectors,
                edge_index=all_edge_index, edge_s=all_edge_s, edge_v=all_edge_v,
            )
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'],     device=device, dtype=h_s_flat.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=h_v_flat.dtype)
            h_s_gvp[mask_bool] = h_s_flat
            h_v_gvp[mask_bool] = h_v_flat
        else:
            h_s_gvp = torch.zeros(B, L, c['d_gvp_internal'],     device=device, dtype=protein_spatial_scalar.dtype)
            h_v_gvp = torch.zeros(B, L, c['n_gvp_vector_hidden'], 3, device=device, dtype=protein_vectors_combined.dtype)

        # GVP Output Dropout (training only)
        if self.training and c.get('gvp_dropout_prob', 0.0) > 0:
            gm = torch.bernoulli(torch.full((B, 1, 1), 1.0 - c['gvp_dropout_prob'], device=device))
            h_s_gvp = h_s_gvp * gm
            h_v_gvp = h_v_gvp * gm.unsqueeze(-1)

        # ── Module 3: Bridge & Gating Fusion ────────────────────────────────
        v_inv  = self.pairwise_bridge(h_v_gvp)
        h_geo  = torch.cat([h_s_gvp, v_inv, protein_physchem_encoded], dim=-1)

        current_esm = esm_embeddings
        if self.training and c.get('esm_dropout_prob', 0.0) > 0:
            em = torch.bernoulli(torch.full((B, 1, 1), 1.0 - c['esm_dropout_prob'], device=device))
            current_esm = current_esm * em

        h_gvp_proj = self.gvp_proj(h_geo)
        h_esm_proj = self.esm_proj(current_esm)
        h_concat   = torch.cat([h_gvp_proj, h_esm_proj], dim=-1)
        gate       = self.modality_gate(h_concat)
        h_prot     = self.input_norm(
            torch.cat([gate * h_gvp_proj, (1.0 - gate) * h_esm_proj], dim=-1)
        )  # [B, L, d_model]

        # ── Module 5a: Ligand GNN Encoder ───────────────────────────────────
        _lig_rbf_bins = c.get('ligand_edge_rbf_bins', 8)
        if ligand_edge_index.size(1) > 0:
            lig_row, lig_col = ligand_edge_index
            lig_dist = torch.norm(ligand_pos[lig_col] - ligand_pos[lig_row], dim=-1)
            lig_rbf  = rbf_encode(lig_dist, D_min=0.0, D_max=5.0, num_rbf=_lig_rbf_bins)
            lig_edge_enriched = torch.cat([ligand_edge_attr, lig_rbf], dim=-1)
        else:
            lig_edge_enriched = torch.cat([
                ligand_edge_attr,
                torch.zeros(0, _lig_rbf_bins, device=device, dtype=ligand_pos.dtype),
            ], dim=-1)

        h_ligand_nodes, _ = self.ligand_encoder(
            x=ligand_x, pos=ligand_pos,
            edge_index=ligand_edge_index, edge_attr=lig_edge_enriched,
            batch=ligand_batch,
        )  # [N_atoms, d_model]

        # Unpack PyG sparse → dense [B, M_max, d_model]
        B_dim, Max_M = ligand_mask.shape
        h_lig      = torch.zeros(B_dim, Max_M, c['d_model'], device=device, dtype=h_ligand_nodes.dtype)
        lig_coords = torch.zeros(B_dim, Max_M, 3,            device=device, dtype=ligand_pos.dtype)
        N_atoms = h_ligand_nodes.size(0)
        if N_atoms > 0:
            counts    = torch.bincount(ligand_batch, minlength=B_dim)
            cumsum    = torch.zeros(B_dim + 1, dtype=torch.long, device=device)
            cumsum[1:] = counts.cumsum(0)
            local_idx  = torch.arange(N_atoms, device=device) - cumsum[ligand_batch]
            valid_m    = local_idx < Max_M
            h_lig     [ligand_batch[valid_m], local_idx[valid_m]] = h_ligand_nodes[valid_m]
            lig_coords[ligand_batch[valid_m], local_idx[valid_m]] = ligand_pos[valid_m]

        # Dynamic ligand global token (masked mean pool)
        mask_f   = ligand_mask.float().unsqueeze(-1)                        # [B, M, 1]
        glob_tok = (h_lig * mask_f).sum(1, keepdim=True) \
                 / mask_f.sum(1, keepdim=True).clamp(min=1)                # [B, 1, d]
        h_lig    = torch.cat([glob_tok, h_lig], dim=1)                     # [B, M+1, d]
        g_mask   = torch.ones(B_dim, 1, device=device, dtype=ligand_mask.dtype)
        lig_mask_g  = torch.cat([g_mask, ligand_mask], dim=1)              # [B, M+1]
        # Pad lig_coords with zero row for global token at position 0
        lig_coords = torch.cat([
            torch.zeros(B_dim, 1, 3, device=device, dtype=lig_coords.dtype),
            lig_coords,
        ], dim=1)                                                           # [B, M+1, 3]

        # ── Uni-Mol global embedding ─────────────────────────────────────────
        if unimol_embed is None:
            unimol_embed = torch.zeros(B_dim, c['d_unimol'], device=device, dtype=h_prot.dtype)
        else:
            unimol_embed = unimol_embed.to(dtype=h_prot.dtype)

        # [Innovation #4] Augment ligand global token with Uni-Mol embedding.
        # h_lig[:, 0, :] starts as masked mean-pool; adding Uni-Mol projection
        # makes the chemical fingerprint directly queryable by protein residues
        # via P→L cross-attention (selective), not only broadcast via FiLM.
        # zero-init proj → no change at start of training.
        # Non-in-place cat to avoid autograd / torch.compile instability.
        h_lig_global = h_lig[:, 0:1, :] + self.unimol_to_global(unimol_embed).unsqueeze(1)
        h_lig = torch.cat([h_lig_global, h_lig[:, 1:, :]], dim=1)

        # [Innovation #3] RSA pre-injection: inject surface prior into h_prot
        # before co-attention so all 6 blocks see RSA.  rsa_mask zeroes out
        # the contribution for residues with missing raw RSA data.
        # Also multiply by protein_mask to ensure padding positions stay zero.
        if protein_rsa_raw is not None:
            rsa_f = protein_rsa_raw.to(dtype=h_prot.dtype)           # [B, L, 1]
            rsa_m = protein_rsa_mask.to(dtype=h_prot.dtype) \
                    if protein_rsa_mask is not None \
                    else torch.ones_like(rsa_f)                       # [B, L, 1]
            rsa_m = rsa_m * protein_mask.unsqueeze(-1).to(dtype=h_prot.dtype)
            h_prot = h_prot + rsa_m * self.rsa_input_proj(rsa_f)

        # ── Module 6 (V12): N × CoAttentionBlock with per-block AttnRes ─────
        # AttnRes anchor: h_prot_0 / h_lig_0 are the representations after
        # encoder fusion + RSA/UniMol injection, before any co-attention.
        # At each block i, AttnRes blends [anchor, current_partial] with learned
        # per-position weights (∂98% partial at init → does not disrupt baseline).
        h_prot_0 = h_prot   # protein anchor: post-encoders, pre-co-attn
        h_lig_0  = h_lig    # ligand  anchor: post-encoders, pre-co-attn

        for i, block in enumerate(self.co_attn_blocks):
            # AttnRes before each block: weighted merge of anchor + current repr.
            # Equivalent to V11's attn_res([b0], partial=h_prot_fused) before CA.
            h_prot = self.prot_attn_res[i](blocks=[h_prot_0], partial=h_prot)
            h_lig  = self.lig_attn_res[i](blocks=[h_lig_0],   partial=h_lig)

            if c.get('use_gradient_checkpointing', False) and self.training:
                def _run(_hp, _hl, _ue, _pm, _lm, _pc, _lc, _pid, _b=block):
                    return _b(_hp, _hl, _ue, _pm, _lm, _pc, _lc, _pid)
                h_prot, h_lig = checkpoint(
                    _run, h_prot, h_lig, unimol_embed,
                    protein_mask, lig_mask_g, protein_coords, lig_coords, position_ids,
                    use_reentrant=False,
                )
            else:
                h_prot, h_lig = block(
                    h_prot, h_lig, unimol_embed,
                    prot_mask=protein_mask, lig_mask=lig_mask_g,
                    prot_coords=protein_coords, lig_coords=lig_coords,
                    position_ids=position_ids,
                )

        # ── Module 7: Classifier ─────────────────────────────────────────────
        h_prot = self.final_norm(h_prot)

        # SASA Residual Gate: learnable modulation conditioned on RSA.
        # Identity at init (zero-init linear → delta = 0).
        # rsa_mask zeros out delta for residues with absent raw RSA (old .pt files),
        # preventing spurious bias from treating missing data as RSA=0 ("fully buried").
        # Skipped entirely if protein_rsa_raw is None (backward compat).
        if protein_rsa_raw is not None:
            rsa = protein_rsa_raw.to(dtype=h_prot.dtype)
            rsa_m = protein_rsa_mask.to(dtype=h_prot.dtype) if protein_rsa_mask is not None else None
            h_prot = self.sasa_gate(h_prot, rsa, rsa_mask=rsa_m, prot_mask=protein_mask)

        x_final = torch.cat([h_prot, protein_physchem_encoded], dim=-1)
        logits  = self.classifier(self.classifier_input_norm(x_final))

        result: Dict[str, torch.Tensor] = {
            'logits':         logits,
            'h_fused':        h_prot,
            'h_ligand_fused': h_lig,
        }
        if self.dist_aux_head is not None:
            result['dist_pred'] = self.dist_aux_head(h_prot).squeeze(-1)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
