#!/usr/bin/env python3
"""Star-only continuation from archived best-F1 states, never implicit scratch.

Prepare on CPU, smoke in a Slurm GPU allocation, then resume with unchanged
model/loss/split/LR schedule. RNG was not saved by the historical trainer, so
this is a best-epoch continuation branch, not bitwise replay of the HPC run.
"""
from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(ROOT / "scripts/diagnostics")]
from prepare_chain_boundary_controls import FLAGS, VARIANTS, sha, source_hashes
from scripts.sync.sync_chainwise_core_to_star import relocate_config, verify_core


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint_info(checkpoint):
    required = ("epoch", "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "scaler_state_dict", "early_stopping_state",
                "best_val_f1_c0", "config", "val_metrics")
    if any(k not in checkpoint for k in required):
        raise ValueError("Incomplete resume checkpoint")
    epoch = checkpoint["epoch"]
    f1 = float(checkpoint["val_metrics"]["f1_c0"])
    stop = checkpoint["early_stopping_state"]
    if not isinstance(epoch, int) or epoch < 0 or not math.isfinite(f1):
        raise ValueError("Invalid checkpoint epoch/F1")
    if stop["metric"] != "f1_c0" or stop["bad_epochs"] != 0:
        raise ValueError("Expected the saved best-F1 stopping state")
    if not math.isclose(f1, checkpoint["best_val_f1_c0"], abs_tol=1e-10):
        raise ValueError("Saved metric does not match best-F1 checkpoint")
    optimizer = checkpoint["optimizer_state_dict"]
    scheduler = checkpoint["scheduler_state_dict"]
    lrs = [g["lr"] for g in optimizer["param_groups"]]
    if not optimizer["state"] or scheduler["last_epoch"] != epoch + 1:
        raise ValueError("Missing optimizer state or inconsistent scheduler epoch")
    if lrs != scheduler["_last_lr"]:
        raise ValueError("Optimizer/scheduler learning rates disagree")
    return dict(saved_epoch=epoch + 1, resume_epoch=epoch + 2, f1_c0=f1,
                lr=lrs, early_stopping=stop,
                rng_state_present=any("rng" in k.lower() for k in checkpoint))


def resume_config(original, core_original, bundle, root, name):
    expected = copy.deepcopy(core_original)
    expected["model"].update(dict(zip(FLAGS, VARIANTS[name])))
    # Only experiment naming/output paths differed in the frozen controls.
    left, right = copy.deepcopy(original), copy.deepcopy(expected)
    left.pop("experiment"); right.pop("experiment")
    if left != right or original["experiment"]["seed"] != 44:
        raise ValueError("Checkpoint scientific recipe differs from the frozen core")
    cfg = relocate_config(original, bundle)
    cfg["experiment"].update(name=f"G0_chainesm_{name}_seed44_star_resume",
                             output_dir=str(root / "outputs" / name))
    return cfg


def prepare(root, bundle, archive, model_source):
    import torch
    if (root / "prepared/provenance.json").exists():
        raise ValueError("Do not overwrite a prepared continuation")
    print("RESUME_PREPARE verify_existing_core", flush=True)
    core = verify_core(bundle, smoke=True)
    original = yaml.safe_load((bundle / "source/hpc_config.yaml").read_text())
    source_manifest = model_source.parent / "prepared/provenance.json"
    expected_model = json.loads(source_manifest.read_text())["model_hashes"]
    if source_hashes(model_source) != expected_model:
        raise ValueError("Archived fixed model source changed")
    shutil.copytree(model_source, root / "model_code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "prepared").mkdir(exist_ok=True)
    runs = {}
    for name in VARIANTS:
        candidates = []
        for path in sorted((archive / "outputs" / name).glob("*/best_f1_c0_model.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            info = checkpoint_info(ckpt)
            resume_config(ckpt["config"], original, bundle, root, name)
            candidates.append(dict(path=str(path), sha256=sha(path), **info))
            del ckpt
        if not candidates:
            raise FileNotFoundError(f"No best-F1 checkpoint for {name}")
        chosen = max(candidates, key=lambda c: (c["f1_c0"], c["saved_epoch"]))
        source = Path(chosen["path"])
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        cfg = resume_config(checkpoint["config"], original, bundle, root, name)
        initial = root / "initial" / name
        initial.mkdir(parents=True)
        shutil.copy2(source, initial / "resume.pt")
        if sha(initial / "resume.pt") != chosen["sha256"]:
            raise ValueError("Copied resume checkpoint differs")
        loss_source = source.parent / "best_model.pt"
        loss_sha = None
        if loss_source.exists():
            shutil.copy2(loss_source, initial / "best_model.pt")
            loss_sha = sha(initial / "best_model.pt")
        path = root / "prepared" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        runs[name] = dict(selected=chosen, candidates=candidates, config_sha256=sha(path),
                          best_loss_sha256=loss_sha)
        del checkpoint
        gc.collect()
        print("RESUME_SELECTED", name, json.dumps(chosen), flush=True)
    assets = {str(bundle / rel): digest for rel, digest in core["assets"].items()
              if Path(rel).suffix != ".pt"}
    assets[str(bundle / "prepared/provenance.json")] = sha(bundle / "prepared/provenance.json")
    report = dict(complete=True, host_policy="star_only", no_test=True,
                  bundle=str(bundle), archive=str(archive), runs=runs, core_assets=assets,
                  model_hashes=source_hashes(root / "model_code"),
                  code_hashes=source_hashes(root / "code_v1"),
                  rng_exact_replay=False, fresh_optimizer=False)
    dump(root / "prepared/provenance.json", report)


def verify(root, name, require_smoke=False):
    p = json.loads((root / "prepared/provenance.json").read_text())
    if not p["complete"] or p["host_policy"] != "star_only" or not p["no_test"]:
        raise ValueError("Invalid continuation provenance")
    for folder, key in (("model_code", "model_hashes"), ("code_v1", "code_hashes")):
        if source_hashes(root / folder) != p[key]:
            raise ValueError(f"Frozen source changed: {folder}")
    for path, digest in p["core_assets"].items():
        if sha(path) != digest:
            raise ValueError(f"Core metadata changed: {path}")
    r = p["runs"][name]
    if sha(root / "initial" / name / "resume.pt") != r["selected"]["sha256"]:
        raise ValueError("Initial checkpoint changed")
    if sha(root / "prepared" / f"{name}.yaml") != r["config_sha256"]:
        raise ValueError("Prepared resume config changed")
    if r["best_loss_sha256"] and sha(root / "initial" / name / "best_model.pt") != r["best_loss_sha256"]:
        raise ValueError("Initial best-loss checkpoint changed")
    if require_smoke:
        smoke = json.loads((root / "smoke" / f"{name}.json").read_text())
        if (not smoke["passed"] or smoke["provenance_sha256"] != sha(root / "prepared/provenance.json")
                or smoke["initial_sha256"] != r["selected"]["sha256"]):
            raise ValueError("Matching GPU resume smoke required")
    return p


def install_trainer(root, completed_epochs):
    from train_chainwise_esm_complex import install
    trainer = install(root, training_code=root / "model_code", start_epoch=completed_epochs)
    import src.models.gvp_layers as graph
    if not Path(graph.__file__).resolve().is_relative_to((root / "model_code").resolve()):
        raise RuntimeError("Wrong model snapshot imported")
    return trainer


def restore_optimizer(model, cfg, checkpoint, optimizer_class):
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    t = cfg["training"]
    no_decay = ("bias", "norm.weight", "norm.bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [dict(params=[p for n, p in model.named_parameters()
                           if p.requires_grad and not any(k in n for k in no_decay)], weight_decay=t["weight_decay"]),
              dict(params=[p for n, p in model.named_parameters()
                           if p.requires_grad and any(k in n for k in no_decay)], weight_decay=0.0)]
    optimizer = optimizer_class(groups, lr=t["learning_rate"], betas=t["betas"])
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=t["warmup_epochs"]),
        CosineAnnealingLR(optimizer, T_max=t["num_epochs"] - t["warmup_epochs"], eta_min=t["min_lr"])],
        milestones=[t["warmup_epochs"]])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if [g["lr"] for g in optimizer.param_groups] != scheduler.get_last_lr():
        raise RuntimeError("Restored LR mismatch")
    for param, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != param.shape or not state[key].isfinite().all():
                raise RuntimeError("Optimizer tensor mismatch/non-finite state")
    return optimizer, scheduler


def smoke(root, name, steps):
    import csv
    import torch
    p = verify(root, name)
    if (os.environ.get("SLURMD_NODENAME") != "star" or not os.environ.get("SLURM_JOB_ID")
            or steps < 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Smoke requires >=2 steps and exactly one allocated GPU")
    checkpoint = torch.load(root / "initial" / name / "resume.pt", map_location="cpu", weights_only=True)
    info = checkpoint_info(checkpoint)
    trainer = install_trainer(root, info["saved_epoch"])
    from train_chainwise_esm_complex import load_smoke
    helpers = load_smoke()
    cfg = yaml.safe_load((root / "prepared" / f"{name}.yaml").read_text())
    trainer.base.set_seed(cfg["experiment"]["seed"])
    torch.set_num_threads(4)
    with open(cfg["scan_metadata_file"]) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    def balanced(split, n):
        groups = [[r["sample_id"] for r in rows if r["source"] == s and r["split"] == split]
                  for s in ("pdbbind", "biolip")]
        return [groups[i % 2][i // 2] for i in range(n)]
    t, device = cfg["training"], torch.device("cuda")
    print("RESUME_SMOKE load_data", name, flush=True)
    batch = helpers.training_batch(cfg, balanced("train", t["batch_size"]))
    model = trainer.create_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer, scheduler = restore_optimizer(model, cfg, checkpoint, helpers.CheckedAdamW)
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    before = {param: float(state["step"]) for param, state in optimizer.state.items()}
    reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
    kwargs = {k: v for k, v in t.items() if k in inspect.signature(trainer.base.train_epoch).parameters and k not in reserved}
    kwargs.update(gamma=t["focal_gamma"], scaler=scaler, log_every=5)
    kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
    kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
    writer = helpers.ScalarRecorder()
    weights = torch.tensor(t["class_weights"], device=device)
    def train(batches):
        return trainer.base.train_epoch(model=model, dataloader=batches, optimizer=optimizer,
            device=device, epoch=info["saved_epoch"], class_weights=weights, writer=writer, **kwargs)
    del checkpoint
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    metrics = train([batch] * steps)
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise RuntimeError("Non-finite resumed train metrics")
    if not any(float(optimizer.state[param]["step"]) > value for param, value in before.items()):
        raise RuntimeError("No optimizer updates during resumed smoke")
    stress_metrics = train([helpers.cap_batch(batch)])
    if not all(math.isfinite(float(v)) for v in stress_metrics.values()):
        raise RuntimeError("Non-finite resumed cap1800 metrics")
    ids = balanced("val", 4)
    allowed = inspect.signature(helpers.PocketDatasetV8).parameters
    ds = helpers.PocketDatasetV8(**{k: v for k, v in cfg["data"].items() if k in allowed},
                               pdb_ids=ids, use_nma=False, split="val", verbose=False)
    if ds.pdb_ids != ids:
        raise RuntimeError("Smoke VAL rows dropped")
    vm = trainer.base.evaluate(model, [helpers.pocket_collate_fn_v10([ds[i] for i in range(len(ds))])],
                              class_weights=weights, gamma=t["focal_gamma"], device=device, use_amp=True)
    for key in ("f1_c0", "auprc_c0_global", "mcc_c0_at_0_5_global"):
        if not math.isfinite(vm[key]):
            raise RuntimeError(f"Non-finite resumed VAL metric: {key}")
    torch.cuda.synchronize()
    report = dict(passed=True, variant=name, provenance_sha256=sha(root / "prepared/provenance.json"),
                  initial_sha256=p["runs"][name]["selected"]["sha256"], restored=info,
                  train_ids=batch["pdb_ids"], val_ids=ids, train_metrics=metrics,
                  val_metrics={k: vm[k] for k in ("f1_c0", "auprc_c0_global", "mcc_c0_at_0_5_global")},
                  steps=steps, seconds=time.monotonic()-start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  gpu=torch.cuda.get_device_name(0), test_used=False, updates_discarded=True)
    verify(root, name)
    dump(root / "smoke" / f"{name}.json", report)
    print("RESUME_SMOKE_PASSED", json.dumps(report), flush=True)


def train(root, name):
    import torch
    p = verify(root, name, require_smoke=True)
    if (os.environ.get("SLURMD_NODENAME") != "star" or not os.environ.get("SLURM_JOB_ID")
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Training requires exactly one allocated GPU")
    initial = root / "initial" / name / "resume.pt"
    checkpoint = torch.load(initial, map_location="cpu", weights_only=True)
    info = checkpoint_info(checkpoint)
    trainer = install_trainer(root, info["saved_epoch"])
    out = root / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    # A previous/partially started run must not be silently reset to its input.
    with (out / "started.json").open("x") as f:
        json.dump(dict(initial=str(initial), initial_sha256=p["runs"][name]["selected"]["sha256"],
                       restored=info, rng_exact_replay=False), f, indent=2)
    shutil.copy2(initial, out / "best_f1_c0_model.pt")
    loss = root / "initial" / name / "best_model.pt"
    if loss.exists():
        shutil.copy2(loss, out / "best_model.pt")
    def setup(config, rank, resume_path=None):
        if Path(resume_path).resolve() != initial.resolve():
            raise ValueError("Unexpected resume path")
        if rank != 0:
            raise RuntimeError("This continuation is single-GPU only")
        (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        return out
    trainer.base.setup_experiment = setup
    del checkpoint
    sys.argv = [sys.argv[0], "--config", str(root / "prepared" / f"{name}.yaml"),
                "--resume", str(initial), "--amp", "--accumulation-steps", "1"]
    print("STAR_RESUME_START", name, json.dumps(info), flush=True)
    trainer.base.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "train"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.phase == "prepare":
        if any(v is None for v in (args.bundle, args.archive, args.model_source)):
            parser.error("prepare requires bundle, archive, and model-source")
        prepare(args.root, args.bundle, args.archive, args.model_source)
    else:
        if args.variant is None:
            parser.error("smoke/train require variant")
        (smoke(args.root, args.variant, args.steps) if args.phase == "smoke"
         else train(args.root, args.variant))


if __name__ == "__main__":
    main()
