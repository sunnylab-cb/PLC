#!/usr/bin/env python3
"""Isolated zero-init gradient checks for V12 blocks."""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pocket_gvp_fusion_v12 import (
    CoAttentionBlock,
    FiLMLayer,
    V12CrossAttentionBlock,
    zero_init_co_attention_residuals,
)


def print_selected_grads(prefix, module):
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if any(token in name for token in ("norm", "w_q", "w_k", "w_v", "gate", "w_o", "w_2")):
            norm = param.grad.detach().float().norm().item()
            print(prefix, name, f"{norm:.6e}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("device", device)

    blk = V12CrossAttentionBlock(
        512, 8, dropout=0.0, drop_path_prob=0.0,
        attn_dropout=0.0, attn_logit_clip=30.0,
    ).to(device)
    torch.nn.init.zeros_(blk.cross_attn.w_o.weight)
    torch.nn.init.zeros_(blk.ffn.w_2.weight)
    q = torch.randn(2, 5, 512, device=device, requires_grad=True)
    kv = torch.randn(2, 4, 512, device=device, requires_grad=True)
    qm = torch.ones(2, 5, dtype=torch.bool, device=device)
    km = torch.ones(2, 4, dtype=torch.bool, device=device)
    out = blk(q, kv, qm, km)
    print("cross_delta_max", f"{(out - q).abs().max().item():.6e}")
    out.sum().backward()
    print_selected_grads("cross_grad", blk)

    film = FiLMLayer(512, 512).to(device)
    h = torch.randn(2, 5, 512, device=device, requires_grad=True)
    u = torch.randn(2, 512, device=device)
    out = film(h, u, prot_mask=qm)
    print("film_delta_max", f"{(out - h).abs().max().item():.6e}")
    out.sum().backward()
    print_selected_grads("film_grad", film)

    cb = CoAttentionBlock(
        512, 8, dropout=0.0, drop_path_prob=0.0,
        attn_dropout=0.0, d_unimol=512,
        cross_attn_logit_clip=30.0,
        self_attn_logit_clip=30.0,
        inner_residual_scale=0.05,
    ).to(device)
    zero_init_co_attention_residuals(cb)
    hp = torch.randn(2, 6, 512, device=device, requires_grad=True)
    hl = torch.randn(2, 5, 512, device=device, requires_grad=True)
    unimol = torch.randn(2, 512, device=device)
    pm = torch.ones(2, 6, dtype=torch.bool, device=device)
    lm = torch.ones(2, 5, dtype=torch.bool, device=device)
    pc = torch.randn(2, 6, 3, device=device)
    lc = torch.randn(2, 5, 3, device=device)
    hpo, hlo = cb(hp, hl, unimol, pm, lm, pc, lc)
    print("co_delta_max", f"{(hpo - hp).abs().max().item():.6e}", f"{(hlo - hl).abs().max().item():.6e}")
    (hpo.sum() + hlo.sum()).backward()
    print_selected_grads("co_grad", cb)


if __name__ == "__main__":
    main()
