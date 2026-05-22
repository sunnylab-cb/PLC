from pathlib import Path

def rep(s, old, new, path, count=1):
    if old not in s:
        raise SystemExit(f'MISSING in {path}: {old[:140]!r}')
    return s.replace(old, new, count)

def fix(path_s):
    path=Path(path_s)
    s=path.read_text()
    # Remove mistakenly added self-attn args from V12RoPEGatedMultiHeadAttention.
    s=rep(s,
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
""",
"""        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
""", path, 1)
    s=rep(s,
"""            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
""",
"""            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
""", path, 1)
    # Add args to V12GatedCrossAttention only.
    s=rep(s,
"""class V12GatedCrossAttention(nn.Module):
    \"\"\"V12-local cross attention with all-masked-row protection.\"\"\"

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
    ):
""",
"""class V12GatedCrossAttention(nn.Module):
    \"\"\"V12-local cross attention with all-masked-row protection.\"\"\"

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        attn_logit_clip: Optional[float] = None,
        use_flash_attn: bool = False,
        flash_attn_fallback: bool = True,
    ):
""", path, 1)
    s=rep(s,
"""            dropout=dropout,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
        )
""",
"""            dropout=dropout,
            attn_dropout=attn_dropout,
            attn_logit_clip=attn_logit_clip,
            use_flash_attn=use_flash_attn,
            flash_attn_fallback=flash_attn_fallback,
        )
""", path, 1)
    path.write_text(s)
    print(f'fixed {path}')

for f in ['src/models/pocket_gvp_fusion_v12.py','src/models/pocket_gvp_fusion_v12m1.py']:
    fix(f)
