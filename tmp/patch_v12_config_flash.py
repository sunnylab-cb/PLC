from pathlib import Path
p=Path('configs/pocket_gvp_v12_hpc_unimol_ligand.yaml')
s=p.read_text()
def rep(old,new):
    global s
    if old not in s:
        raise SystemExit(f'MISSING config snippet: {old[:120]!r}')
    s=s.replace(old,new,1)
if 'ligand_encoder_lr_mult' not in s:
    rep(
"""  zero_open_lr_mult: 0.1
  zero_open_adamw_eps: 1.0e-3

  scheduler: "cosine"
""",
"""  zero_open_lr_mult: 0.1
  zero_open_adamw_eps: 1.0e-3
  # Ligand-path optimization experiment: let ligand GNN / direct Uni-Mol atom
  # path learn faster while keeping protein frontend gradients bounded.
  ligand_encoder_lr_mult: 5.0
  ligand_lr_param_tokens: ["ligand_encoder", "unimol_ligand_proj", "unimol_ligand_alpha"]

  scheduler: "cosine"
""")
if 'grad_value_clip_groups' not in s:
    rep(
"""  grad_value_clip: 1000.0
  grad_value_clip_param_tokens: ["esm_proj.0.weight"]
  enable_profiling: false
""",
"""  grad_value_clip: 1000.0
  grad_value_clip_param_tokens: ["esm_proj.0.weight"]
  grad_value_clip_groups:
    - name: "protein_frontend"
      tokens: ["esm_proj", "gvp_proj", "physchem_encoder", "modality_gate"]
      value: 0.2
  enable_profiling: false
""")
if 'use_flash_attn' not in s:
    rep(
"""  cross_attn_logit_clip: 30.0
  self_attn_logit_clip: 30.0
""",
"""  cross_attn_logit_clip: 30.0
  self_attn_logit_clip: 30.0
  # Optional HPC-only FlashAttention backend for cross-attention.  Leave false
  # on local ROCm unless flash-attn is installed and validated.
  use_flash_attn: false
  flash_attn_fallback: true
""")
p.write_text(s)
print('patched config')
