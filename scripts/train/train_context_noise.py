#!/usr/bin/env python3
"""Original scratch GFGF + input-context noise, with UNCHANGED original losses.

No teacher, KL, self-distillation, resume, frozen trunk or inference modification.
The train-loop insertion logs noise only; augmentation is before checkpointing.
"""

import ast
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v11m13 as trainer
from src.models.protein_context_noise import ProteinContextNoise, log_context_noise

_original_gradient_stats = trainer.base.compute_gradient_stats


def validate_config(config):
    t, m = config["training"], config["model"]
    if "self_distillation" in config:
        raise ValueError("Context-noise trial must not involve self-distillation")
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "resume",
                              "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Context noise trial is scratch-only with no freezing")
    if m.get("use_compile") or m.get("protein_hybrid_pattern") != "GFGF":
        raise ValueError("Expected original GFGF without torch.compile")
    if m.get("use_gradient_checkpointing"):
        raise ValueError("Keep original checkpointing=False; stateful co-attention replay is unsupported")


def create_model(config, device, use_gradient_checkpointing=None, use_compile=None):
    validate_config(config)
    if use_compile or use_gradient_checkpointing:
        raise ValueError("Keep original compile=False and checkpointing=False")
    model = trainer.create_model(config, device, use_gradient_checkpointing, use_compile)
    model._protein_context_noise = ProteinContextNoise(config["context_noise"])
    return model


def context_gradient_stats(model):
    stats = _original_gradient_stats(model)
    base = getattr(model, "module", model)
    for name in ("protein_context_encoder", "cross_attention_layers", "local_delta_head"):
        module = getattr(base, name, None)
        if module is None:
            continue
        grads = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
        if grads:
            squares = torch.stack([g.square().sum() for g in grads]).sum()
            stats[f"context_noise_{name}_norm"] = squares.sqrt().item()
            stats[f"context_noise_{name}_rms"] = (squares / sum(g.numel() for g in grads)).sqrt().item()
    return stats


def logging_epoch_ast(source):
    tree = ast.parse(source)
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_epoch"]

    class InsertLog(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if ast.unparse(node) == "logits = outputs['logits']":
                self.count += 1
                added = ast.parse("""
if writer is not None and is_main_process() and (batch_idx + 1) % log_every == 0:
    log_context_noise(model, labels, mask, writer, epoch * len(dataloader) + batch_idx)
""").body[0]
                return [node, added]
            return node

    patch = InsertLog()
    function = patch.visit(function)
    if patch.count != 1:
        raise RuntimeError(f"Expected exactly one TRAIN logging insertion, got {patch.count}")
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def install_training_hook():
    source = (ROOT / "scripts/train/train_v11m3.py").read_text()
    source = trainer.shared._patch_v11m3_source_for_v11m4(source)
    trainer.base.log_context_noise = log_context_noise
    exec(compile(logging_epoch_ast(source), str(__file__), "exec"), trainer.base.__dict__)
    trainer.base.create_model = create_model
    trainer.base.compute_gradient_stats = context_gradient_stats
    return trainer.base


if __name__ == "__main__":
    if any(arg.startswith("--resume") for arg in sys.argv[1:]):
        raise SystemExit("Noise trials must start from scratch")
    install_training_hook().main()
