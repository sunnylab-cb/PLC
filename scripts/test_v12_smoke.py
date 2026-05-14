"""
V12 smoke test: identity-init checks + full forward pass.
Run with: conda run -n AIAA python scripts/test_v12_smoke.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

# ── Test 1: SASAResidualGate identity init ───────────────────────────────────
print("=== Test 1: SASAResidualGate identity init ===")
from src.models.pocket_gvp_fusion_v12 import SASAResidualGate
gate = SASAResidualGate(amplitude=0.5)
B, L, D = 2, 10, 512
h   = torch.randn(B, L, D)
rsa = torch.rand(B, L, 1)
mask = torch.ones(B, L, 1)
with torch.no_grad():
    out = gate(h, rsa, rsa_mask=mask)
diff = (out - h).abs().max().item()
status = "PASS" if diff < 1e-6 else "FAIL"
print(f"  max diff={diff:.2e}  -> {status}")
assert diff < 1e-6, "SASA gate identity FAILED"

# ── Test 2: RSA input proj identity init ────────────────────────────────────
print("=== Test 2: rsa_input_proj identity init ===")
from src.models.pocket_gvp_fusion_v12 import PocketGVPFusionV12
model = PocketGVPFusionV12()
model.eval()

with torch.no_grad():
    x = torch.randn(B, L, 1)
    out2 = model.rsa_input_proj(x)
diff2 = out2.abs().max().item()
status2 = "PASS" if diff2 < 1e-6 else "FAIL"
print(f"  max output (should be 0)={diff2:.2e}  -> {status2}")
assert diff2 < 1e-6, "rsa_input_proj identity FAILED"

# ── Test 3: unimol_to_global identity init ───────────────────────────────────
print("=== Test 3: unimol_to_global identity init ===")
with torch.no_grad():
    u = torch.randn(B, 512)
    out3 = model.unimol_to_global(u)
diff3 = out3.abs().max().item()
status3 = "PASS" if diff3 < 1e-6 else "FAIL"
print(f"  max output (should be 0)={diff3:.2e}  -> {status3}")
assert diff3 < 1e-6, "unimol_to_global identity FAILED"

# ── Test 4: FiLM identity init ───────────────────────────────────────────────
print("=== Test 4: FiLM identity init ===")
from src.models.pocket_gvp_fusion_v12 import FiLMLayer
film = FiLMLayer(d_unimol=512, d_model=512)
hp = torch.randn(B, L, D)
ue = torch.randn(B, 512)
with torch.no_grad():
    out4 = film(hp, ue)
diff4 = (out4 - hp).abs().max().item()
status4 = "PASS" if diff4 < 1e-6 else "FAIL"
print(f"  max diff={diff4:.2e}  -> {status4}")
assert diff4 < 1e-6, "FiLM identity FAILED"

# ── Test 5: param count ──────────────────────────────────────────────────────
print("=== Test 5: param count ===")
info = model.count_parameters()
print(f"  total={info['total']/1e6:.1f}M  trainable={info['trainable']/1e6:.1f}M")

# ── Test 6: full forward pass (CPU, small tensors) ───────────────────────────
print("=== Test 6: full forward pass ===")
torch.manual_seed(42)
B2, L2, M2 = 2, 30, 8
N_atoms = B2 * M2  # 16 atoms total

esm     = torch.randn(B2, L2, 1280)
phys    = torch.randn(B2, L2, 62)
sp_sc   = torch.randn(B2, L2, 89)
sp_vec  = torch.randn(B2, L2, 8, 3)
gvp_vec = torch.randn(B2, L2, 4, 3)
coords  = torch.randn(B2, L2, 3)
pmask   = torch.ones(B2, L2, dtype=torch.bool)
chain_ids = torch.zeros(B2, L2, dtype=torch.long)

# protein_rsa_raw and mask
rsa_raw_f  = torch.rand(B2, L2, 1)
rsa_mask_f = torch.ones(B2, L2, 1)

# Ligand (sparse PyG format)
lig_x     = torch.randn(N_atoms, 109)
lig_pos   = torch.randn(N_atoms, 3)
lig_ei    = torch.zeros(2, 0, dtype=torch.long)
lig_ea    = torch.zeros(0, 6)
lig_batch = torch.cat([torch.full((M2,), i, dtype=torch.long) for i in range(B2)])
lig_mask  = torch.ones(B2, M2, dtype=torch.bool)

# Uni-Mol embed
unimol = torch.randn(B2, 512)

with torch.no_grad():
    out = model(
        esm_embeddings=esm,
        protein_physchem=phys,
        protein_spatial_scalar=sp_sc,
        protein_spatial_vector=sp_vec,
        protein_gvp_vectors=gvp_vec,
        protein_coords=coords,
        protein_mask=pmask,
        ligand_x=lig_x,
        ligand_pos=lig_pos,
        ligand_edge_index=lig_ei,
        ligand_edge_attr=lig_ea,
        ligand_batch=lig_batch,
        ligand_mask=lig_mask,
        unimol_embed=unimol,
        protein_rsa_raw=rsa_raw_f,
        protein_rsa_mask=rsa_mask_f,
        chain_ids=chain_ids,
    )

logits = out['logits']
print(f"  logits shape: {logits.shape}  (expected [{B2}, {L2}, 3])")
assert logits.shape == (B2, L2, 3), f"Wrong logits shape: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN in logits"
print("  PASS")

# ── Test 7: forward with rsa=None (backward compat) ─────────────────────────
print("=== Test 7: forward with protein_rsa_raw=None ===")
with torch.no_grad():
    out7 = model(
        esm_embeddings=esm,
        protein_physchem=phys,
        protein_spatial_scalar=sp_sc,
        protein_spatial_vector=sp_vec,
        protein_gvp_vectors=gvp_vec,
        protein_coords=coords,
        protein_mask=pmask,
        ligand_x=lig_x,
        ligand_pos=lig_pos,
        ligand_edge_index=lig_ei,
        ligand_edge_attr=lig_ea,
        ligand_batch=lig_batch,
        ligand_mask=lig_mask,
        chain_ids=chain_ids,
    )
assert out7['logits'].shape == (B2, L2, 3)
assert not torch.isnan(out7['logits']).any()
print("  PASS")

print("\n=== All tests PASSED ===")
