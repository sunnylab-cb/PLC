#!/usr/bin/env python3
"""Scratch GFGF self-distillation; unchanged V11m13 model and VAL selection.

Reuse the exact production train_epoch, inserting one additive TRAIN loss before
gradient accumulation. Do not copy a stale epoch loop or modify shared trainers.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_v11m13 as trainer
from scripts.diagnostics.strict_assignment_common import read_strict_splits
from src.losses.self_distillation import TeacherCache, add_self_distillation, attach_distillation


def distillation_epoch_ast(source):
    tree = ast.parse(source)
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_epoch"]

    class InsertLoss(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if ast.unparse(node) == "loss = loss / accumulation_steps":
                self.count += 1
                addition = ast.parse("""
loss = add_self_distillation(model, outputs['logits'], batch, labels, mask,
    loss, epoch, writer, epoch * len(dataloader) + batch_idx,
    (batch_idx + 1) % log_every == 0)
""").body[0]
                return [addition, node]
            return node

    patch = InsertLoss()
    function = patch.visit(function)
    if patch.count != 1:
        raise RuntimeError(f"Expected one TRAIN loss insertion point, got {patch.count}")
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def validate_scratch(config):
    t, m, data = config["training"], config["model"], config["data"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "resume", "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Self-distillation students must start from scratch, unfrozen")
    if m.get("use_compile") or m.get("protein_hybrid_pattern") != "GFGF":
        raise ValueError("Expected GFGF without torch.compile")
    if data.get("nma_n_confs") != 1 or data.get("nma_max_conf") != 0:
        raise ValueError("Offline cache requires exactly conf0, no NMA expansion")


def create_model(config, device, use_gradient_checkpointing=None, use_compile=None):
    validate_scratch(config)
    if use_compile:
        raise ValueError("torch.compile is disabled for this experiment")
    opts = config["self_distillation"]
    splits = read_strict_splits(config["data"]["split_file"])
    cache = TeacherCache.load(opts["teacher_cache"], config, splits, opts)
    model = trainer.create_model(config, device, use_gradient_checkpointing, use_compile)
    attach_distillation(model, cache, opts)
    return model


def install_training_hook():
    source = (ROOT / "scripts/train/train_v11m3.py").read_text()
    source = trainer.shared._patch_v11m3_source_for_v11m4(source)
    trainer.base.add_self_distillation = add_self_distillation
    exec(compile(distillation_epoch_ast(source), str(__file__), "exec"), trainer.base.__dict__)
    trainer.base.create_model = create_model
    return trainer.base


if __name__ == "__main__":
    if any(arg.startswith("--resume") for arg in sys.argv[1:]):
        raise SystemExit("This first-generation self-distillation experiment is scratch-only")
    install_training_hook().main()
