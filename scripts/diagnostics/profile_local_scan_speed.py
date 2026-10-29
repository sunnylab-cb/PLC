#!/usr/bin/env python3
"""TRAIN-only paired timing: chunk128/512/1024 x legacy/batched statistics.

One GPU runs every arm sequentially. No checkpoints are read or saved, no
ongoing job is changed, and no TEST/VAL scores select a speed configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
import train_local_spatial_scan as trainer
from prepare_local_spatial_scan import verify, VARIANTS
from prepare_libre_transfer import sha256, source_hashes
from smoke_local_spatial_scan import cap_batch
from smoke_libre_transfer import ScalarRecorder, training_batch
from src.data.collate import prepare_model_inputs_v11
from src.utils.batched_training_stats import local_gradient_report_batched


CASES = [(chunk, fast) for chunk in (128, 512, 1024) for fast in (False, True)]


def seed():
    torch.manual_seed(44)
    np.random.seed(44)
    random.seed(44)


def case_config(cfg, chunk, fast):
    cfg = copy.deepcopy(cfg)
    cfg["model"]["scan_chunk"] = chunk
    cfg["training"]["batched_local_stats"] = fast
    return cfg


def run_train(model, optimizer, scaler, batches, cfg, writer):
    t = cfg["training"]
    reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
    kwargs = {k: v for k, v in t.items()
              if k in inspect.signature(trainer.base.train_epoch).parameters and k not in reserved}
    kwargs.update(gamma=t["focal_gamma"], scaler=scaler, log_every=10,
                  use_amp=True, use_bf16=True)
    kwargs.update({"cnc_" + k: v for k, v in t["cnc"].items()})
    kwargs.update({"dist_aux_" + k: v for k, v in t["dist_aux"].items()})
    result = trainer.base.train_epoch(model=model, dataloader=batches, optimizer=optimizer,
        device=torch.device("cuda"), epoch=6, writer=writer,
        class_weights=torch.tensor(t["class_weights"], device="cuda"), **kwargs)
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Nonfinite TRAIN metrics in speed probe")
    return result


class CountingAdamW(torch.optim.AdamW):
    """Count actual updates without per-parameter synchronization in timing."""
    successful_steps = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.successful_steps += 1
        return result


def setup(cfg):
    seed()
    model = trainer.create_model(cfg, torch.device("cuda"))
    t = cfg["training"]
    opt = CountingAdamW(model.parameters(), lr=t["learning_rate"],
                        weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
    scaler = torch.amp.GradScaler("cuda")
    return model, opt, scaler


def parity(cfg, batch):
    """Check chunk operators without randomness, then actual stochastic TRAIN."""
    deterministic = copy.deepcopy(cfg)
    for key in ("dropout", "classifier_dropout", "coord_noise_std", "esm_dropout_prob",
                "physchem_dropout_prob", "gvp_dropout_prob", "ligand_source_dropout_p",
                "ligand_dropout_hidden_p", "ligand_dropout_local54_p"):
        deterministic["model"][key] = 0.0
    model, opt, scaler = setup(case_config(deterministic, 128, False))
    inputs = prepare_model_inputs_v11(batch, torch.device("cuda"))
    expected, expected_grad = None, None
    differences = {}
    for chunk in (128, 512, 1024):
        model.chunk = chunk
        model.zero_grad(set_to_none=True)
        out = model(**inputs)
        loss = out["logits"].float().square().mean() + out["pred_dist"].float().mean()
        loss.backward()
        current = {k: out[k].detach().cpu() for k in ("logits", "pred_dist")}
        grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
        if expected is None:
            expected, expected_grad = current, grads
        for key in expected:
            torch.testing.assert_close(current[key], expected[key], rtol=2e-4, atol=2e-5)
        if grads.keys() != expected_grad.keys():
            raise RuntimeError("Chunk changed gradient connectivity")
        for key in grads:
            torch.testing.assert_close(grads[key], expected_grad[key], rtol=2e-3, atol=1e-5)
        differences[str(chunk)] = dict(
            max_output_abs=max((current[k]-expected[k]).abs().max().item() for k in current),
            max_gradient_abs=max((grads[k]-expected_grad[k]).abs().max().item() for k in grads))
        _, fast_stats = local_gradient_report_batched(model)
        legacy_stats = trainer.compute_gradient_stats(model)
        if fast_stats.keys() != legacy_stats.keys():
            raise RuntimeError("Gradient diagnostic keys changed")
        for key in fast_stats:
            if not math.isclose(fast_stats[key], legacy_stats[key], rel_tol=2e-6, abs_tol=1e-8):
                raise RuntimeError(f"Gradient diagnostic mismatch: {key}")
    del model, opt, scaler, out, loss, inputs, grads, expected_grad
    gc.collect()
    torch.cuda.empty_cache()
    states, scalars, metric_rows = [], [], []
    for fast in (False, True):
        config = case_config(cfg, 128, fast)
        model, opt, scaler = setup(config)
        writer = ScalarRecorder()
        metrics = run_train(model, opt, scaler, [batch] * 10, config, writer)
        if opt.successful_steps != 10:
            raise RuntimeError("AMP skipped an update during statistics parity")
        states.append({k: p.detach().cpu().clone() for k, p in model.state_dict().items()})
        scalars.append(writer.scalars)
        metric_rows.append(metrics)
        del model, opt, scaler, writer
        gc.collect()
        torch.cuda.empty_cache()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=1e-5, atol=1e-6)
    if scalars[0].keys() != scalars[1].keys():
        raise RuntimeError("TRAIN scalar keys changed")
    for key in scalars[0]:
        np.testing.assert_allclose(scalars[0][key], scalars[1][key], rtol=2e-5, atol=1e-6)
    for key in metric_rows[0]:
        np.testing.assert_allclose(metric_rows[0][key], metric_rows[1][key], rtol=2e-5, atol=1e-6)
    return dict(passed=True, chunk_fp32=differences, stochastic_statistics_steps=10,
                statistics_max_weight_abs=max((states[0][k]-states[1][k]).abs().max().item() for k in states[0]))


def summarize(rows):
    summary = []
    for variant in VARIANTS[1:]:
        ref = [r["seconds"] for r in rows if r["variant"] == variant and r["chunk"] == 128
               and not r["batched_stats"] and r["status"] == "passed"]
        for chunk, fast in CASES:
            selected = [r for r in rows if r["variant"] == variant and r["chunk"] == chunk
                        and r["batched_stats"] == fast and r["status"] == "passed"]
            if selected and ref:
                seconds = statistics.median(r["seconds"] for r in selected)
                summary.append(dict(variant=variant, chunk=chunk, batched_stats=fast,
                    n_repeats=len(selected), median_seconds=seconds,
                    median_pairs_s=statistics.median(r["pairs_s"] for r in selected),
                    speedup_vs_128_legacy=statistics.median(ref)/seconds,
                    peak_allocated_gib=max(r["peak_allocated_gib"] for r in selected)))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 10 or args.steps % 10 or args.warmup < 1 or args.repeats < 1:
        p.error("steps must be a positive multiple of10; warmup/repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit through Slurm with one GPU")
    verify(args.reference)
    hashes = source_hashes(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    configs = {v: yaml.safe_load((args.reference / "prepared" / f"{v}.yaml").read_text())
               for v in VARIANTS[1:]}
    cfg0 = configs[VARIANTS[1]]
    if any(c["data"] != cfg0["data"] or c["training"] != cfg0["training"] for c in configs.values()):
        raise RuntimeError("Expected identical production data/training across LocalScan arms")
    if cfg0["training"]["batch_size"] != 16 or cfg0["data"]["max_seq_len"] != 1800:
        raise RuntimeError("Unexpected frozen batch/cap")
    with open(cfg0["scan_metadata_file"]) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sources = [[r["sample_id"] for r in rows if r["source"] == source and r["split"] == "train"]
               for source in ("pdbbind", "biolip")]
    rng = random.Random(44)
    batches = [training_batch(cfg0, [sources[i % 2][i // 2] for i in range(16)])]
    for _ in range(2):
        selections = [rng.sample(source, 8) for source in sources]
        batches.append(training_batch(cfg0, [selections[i % 2][i // 2] for i in range(16)]))
    report = dict(passed=False, reference=str(args.reference),
        reference_manifest_sha256=sha256(args.reference / "prepared/provenance.json"),
        source_hashes=hashes, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        seed=44, batch_size=16, steps=args.steps, warmup=args.warmup, repeats=args.repeats,
        train_batches=[dict(ids=b["pdb_ids"], shape=list(b["protein_mask"].shape)) for b in batches],
        test_forward_used=False, val_forward_used=False, configurations=configs,
        note="Cached CPU TRAIN batches; includes H2D/loss/backward/optimizer/metrics. "
             "Excludes disk loading and TensorBoard file I/O. Same batch16/checkpointing/loss/AMP. "
             "No throughput result implies a full-epoch or accuracy improvement.",
        parity={}, runs=[], cap_stress=[], summary=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["summary"] = summarize(report["runs"])
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    save()
    for variant, cfg in configs.items():
        print("SPEED_PARITY", variant, flush=True)
        report["parity"][variant] = parity(cfg, batches[0])
        save()
        for repeat in range(args.repeats):
            order = list(CASES)
            random.Random(44 + repeat).shuffle(order)
            for chunk, fast in order:
                config = case_config(cfg, chunk, fast)
                row = dict(variant=variant, chunk=chunk, batched_stats=fast, repeat=repeat)
                print("SPEED_CASE", json.dumps(row), flush=True)
                model, opt, scaler = setup(config)
                stage = "timing"
                try:
                    run_train(model, opt, scaler, [batches[i % len(batches)] for i in range(args.warmup)],
                              config, ScalarRecorder())
                    before_steps = opt.successful_steps
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    metrics = run_train(model, opt, scaler,
                        [batches[i % len(batches)] for i in range(args.steps)], config, ScalarRecorder())
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    updates = opt.successful_steps - before_steps
                    if updates != args.steps:
                        raise RuntimeError("AMP skipped updates; timing is not comparable")
                    row.update(status="passed", seconds=elapsed, pairs_s=16*args.steps/elapsed,
                               optimizer_updates=updates, train_metrics=metrics,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    report["runs"].append(row)
                    print("SPEED_RESULT", json.dumps(row), flush=True)
                    save()
                    if repeat == 0:
                        stage = "cap_stress"
                        torch.cuda.reset_peak_memory_stats()
                        before_steps = opt.successful_steps
                        stress_metrics = run_train(model, opt, scaler, [cap_batch(batches[0])], config, ScalarRecorder())
                        if opt.successful_steps - before_steps != 1:
                            raise RuntimeError("AMP skipped the cap-sized update")
                        report["cap_stress"].append(dict(variant=variant, chunk=chunk, batched_stats=fast,
                            length=1800, passed=True, metrics=stress_metrics,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
                except torch.cuda.OutOfMemoryError:
                    if stage == "timing":
                        report["runs"].append({**row, "status": "oom"})
                    else:
                        report["cap_stress"].append(dict(variant=variant, chunk=chunk,
                            batched_stats=fast, length=1800, passed=False, reason="oom"))
                    print("SPEED_OOM", json.dumps(row), flush=True)
                finally:
                    del model, opt, scaler
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
    verify(args.reference)
    if hashes != source_hashes(ROOT):
        raise RuntimeError("Profile source changed during execution")
    report["passed"] = (len(report["cap_stress"]) == len(CASES)*len(configs)
                        and all(r["passed"] for r in report["cap_stress"])
                        and len(report["runs"]) == len(CASES)*len(configs)*args.repeats
                        and all(r["status"] == "passed" for r in report["runs"]))
    save()
    print("SPEED_PROBE_COMPLETE", json.dumps(dict(passed=report["passed"], summary=report["summary"])), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
