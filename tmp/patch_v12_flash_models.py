from pathlib import Path

flash_import = """try:\n    from flash_attn import flash_attn_varlen_func\nexcept Exception:  # Optional HPC-only dependency\n    flash_attn_varlen_func = None\n\n"""
flash_method = """
    def _flash_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor],
        kv_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_flash_attn:
            return None
        if flash_attn_varlen_func is None:
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("use_flash_attn=True but flash-attn is not installed")
        if q.dtype not in (torch.float16, torch.bfloat16):
            if self.flash_attn_fallback:
                return None
            raise RuntimeError(f"flash-attn requires fp16/bf16, got {q.dtype}")

        B, H, L, D = q.shape
        S = k.size(2)
        if q_mask is None:
            q_mask = torch.ones(B, L, dtype=torch.bool, device=q.device)
        if kv_mask is None:
            kv_mask = torch.ones(B, S, dtype=torch.bool, device=q.device)
        if q_mask.dim() != 2 or kv_mask.dim() != 2:
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("flash-attn cross attention supports only 2D q/kv masks")

        q_mask = q_mask.bool()
        kv_mask = kv_mask.bool()
        q_lens = q_mask.sum(dim=1, dtype=torch.int32)
        kv_lens = kv_mask.sum(dim=1, dtype=torch.int32)
        if (q_lens == 0).any() or (kv_lens == 0).any():
            if self.flash_attn_fallback:
                return None
            raise RuntimeError("flash-attn received an empty q/kv sequence")

        q_b = q.transpose(1, 2).contiguous()  # [B, L, H, D]
        k_b = k.transpose(1, 2).contiguous()  # [B, S, H, D]
        v_b = v.transpose(1, 2).contiguous()
        q_unpad = q_b[q_mask]
        k_unpad = k_b[kv_mask]
        v_unpad = v_b[kv_mask]
        cu_q = F.pad(q_lens.cumsum(0), (1, 0)).to(torch.int32)
        cu_k = F.pad(kv_lens.cumsum(0), (1, 0)).to(torch.int32)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, cu_q, cu_k,
            int(q_lens.max().item()), int(kv_lens.max().item()),
            dropout_p=dropout_p, softmax_scale=self.scale, causal=False,
        )
        out = q_b.new_zeros(B, L, H, D)
        out[q_mask] = out_unpad
        return out.transpose(1, 2).reshape(B, L, self.d_model)
"""

def rep(s, old, new, path):
    if old not in s:
        raise SystemExit(f'MISSING in {path}: {old[:160]!r}')
    return s.replace(old, new, 1)

def patch_common(path_s):
    path=Path(path_s)
    s=path.read_text()
    if 'flash_attn_varlen_func' not in s:
        s=rep(s, "# Re-use from V11 (avoid code duplication)\nfrom .pocket_gvp_fusion_v11 import (\n", "# Re-use from V11 (avoid code duplication)\n"+flash_import+"from .pocket_gvp_fusion_v11 import (\n", path)
    s=rep(s,
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
""",
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
""", path)
    s=rep(s,
"""        self.scale = self.d_head ** -0.5
        self.attn_logit_clip = attn_logit_clip

        self.w_q = nn.Linear(d_model, d_model, bias=False)
""",
"""        self.scale = self.d_head ** -0.5
        self.attn_logit_clip = attn_logit_clip
        self.use_flash_attn = bool(use_flash_attn)
        self.flash_attn_fallback = bool(flash_attn_fallback)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
""", path)
    s=rep(s,
"""    def forward(
        self,
        q_in: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
""",
flash_method+"""
    def forward(
        self,
        q_in: torch.Tensor,
        kv: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
""", path)
    s=rep(s,
"""        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
""",
"""        v = self.w_v(kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        flash_out = self._flash_forward(q, k, v, q_mask=q_mask, kv_mask=kv_mask)
        if flash_out is not None:
            out = self.w_o(flash_out)
            out = out * torch.sigmoid(self.gate(q_in))
            if q_mask is not None:
                out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
            return out

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
""", path)
    s=rep(s,
"""        out = self.w_o(out)
        out = out * torch.sigmoid(self.gate(q_in))
        return out
""",
"""        out = self.w_o(out)
        out = out * torch.sigmoid(self.gate(q_in))
        if q_mask is not None:
            out = out.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return out
""", path)
    s=rep(s,
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
""",
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
""", path)
    s=rep(s,
"""            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
""",
"""            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
""", path)
    s=rep(s, """        out = self.cross_attn(q_normed, kv_normed, kv_mask=kv_mask)
""", """        out = self.cross_attn(q_normed, kv_normed, q_mask=q_mask, kv_mask=kv_mask)
""", path)
    s=rep(s,
"""        p2l_length_ref: float = 256.0,
        p2l_rezero_gate_max: Optional[float] = None,
    ):
""",
"""        p2l_length_ref: float = 256.0,
        p2l_rezero_gate_max: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
""", path)
    s=rep(s,
"""        self.prot_to_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
        )
        self.lig_to_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
        )
""",
"""        self.prot_to_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.lig_to_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=cross_attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
""", path)
    s=rep(s, """        'debug_asserts':        False,
""", """        'debug_asserts':        False,
        # Optional HPC-only FlashAttention backend for cross-attention.
        # Self-attention keeps the standard path because it uses distance bias.
        'use_flash_attn':      False,
        'flash_attn_fallback': True,
""", path)
    s=rep(s,
"""                p2l_rezero_gate_max = c.get('p2l_rezero_gate_max', None),
            )
""",
"""                p2l_rezero_gate_max = c.get('p2l_rezero_gate_max', None),
                use_flash_attn = c.get('use_flash_attn', False),
                flash_attn_fallback = c.get('flash_attn_fallback', True),
            )
""", path)
    path.write_text(s)
    print(f'patched common {path}')

def patch_m1_extra():
    path=Path('src/models/pocket_gvp_fusion_v12m1.py')
    s=path.read_text()
    s=rep(s,
"""        gate_init: float = 0.05,
        gate_max: Optional[float] = 0.2,
    ):
""",
"""        gate_init: float = 0.05,
        gate_max: Optional[float] = 0.2,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
""", path)
    s=rep(s,
"""        self.int_from_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
        self.int_from_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
        self.prot_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
        self.lig_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
""",
"""        self.int_from_prot = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.int_from_lig = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.prot_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
        self.lig_from_int = V12CrossAttentionBlock(
            d_model, n_heads, dropout, drop_path_prob, attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
""", path)
    s=rep(s,
"""                    gate_init=c.get('interaction_gate_init', 0.05),
                    gate_max=c.get('interaction_gate_max', 0.2),
                )
""",
"""                    gate_init=c.get('interaction_gate_init', 0.05),
                    gate_max=c.get('interaction_gate_max', 0.2),
                    use_flash_attn=c.get('use_flash_attn', False),
                    flash_attn_fallback=c.get('flash_attn_fallback', True),
                )
""", path)
    path.write_text(s)
    print('patched v12m1 interaction flash')

for f in ['src/models/pocket_gvp_fusion_v12.py','src/models/pocket_gvp_fusion_v12m1.py']:
    patch_common(f)
patch_m1_extra()
