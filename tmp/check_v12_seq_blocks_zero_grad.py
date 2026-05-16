#!/usr/bin/env python3
"""Check two V12 CoAttentionBlocks in sequence under zero-init."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pocket_gvp_fusion_v12 import CoAttentionBlock, zero_init_co_attention_residuals


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([
        CoAttentionBlock(
            512, 8, dropout=0.0, drop_path_prob=0.0, attn_dropout=0.0,
            d_unimol=512, use_film=True, use_prot_dist_bias=True,
            use_lig_dist_bias=True, cross_attn_logit_clip=30.0,
            self_attn_logit_clip=30.0, inner_residual_scale=0.05,
        )
        for _ in range(2)
    ]).to(device)
    for block in blocks:
        zero_init_co_attention_residuals(block)

    hp = torch.randn(15, 331, 512, device=device, requires_grad=True)
    hl = torch.randn(15, 87, 512, device=device, requires_grad=True)
    unimol = torch.randn(15, 512, device=device)
    pm = torch.ones(15, 331, dtype=torch.bool, device=device)
    lm = torch.ones(15, 87, dtype=torch.bool, device=device)
    pc = torch.randn(15, 331, 3, device=device)
    lc = torch.randn(15, 87, 3, device=device)
    hp0, hl0 = hp, hl

    for block in blocks:
        hpn, hln = block(hp, hl, unimol, pm, lm, pc, lc)
        hp = hp + 0.25 * (hpn - hp)
        hl = hl + 0.25 * (hln - hl)

    print("delta", (hp - hp0).abs().max().item(), (hl - hl0).abs().max().item())
    hp.float().sum().backward()

    rows = []
    for name, param in blocks.named_parameters():
        if param.grad is None:
            continue
        if any(tok in name for tok in ("norm", "w_q", "w_k", "w_v", "gate", "w_o", "w_2")):
            rows.append((float(param.grad.detach().float().norm().cpu()), name))
    rows.sort(reverse=True)
    for norm, name in rows[:30]:
        print(f"{norm:.6e} {name}")


if __name__ == "__main__":
    main()
