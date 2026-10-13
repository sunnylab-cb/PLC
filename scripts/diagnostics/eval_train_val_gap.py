#!/usr/bin/env python3
"""Fixed best-F1 TRAIN/VAL evaluation; no optimization, TEST, or threshold search.

Baseline: full TRAIN plus full VAL, with optional TRAIN-only teacher logits.
Controls: the same deterministic source-stratified TRAIN subset plus full VAL.
Both sides use eval mode, identical precision/crops, hard-label Focal/Tversky,
and no CNC, sample weighting, smoothing, or auxiliary loss.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

CODE = Path(os.environ.get("PLC_FROZEN_CODE", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
from scripts.diagnostics.strict_assignment_common import read_strict_splits
from src.metrics.metrics import compute_binary_metrics_at_threshold, compute_binary_ranking_metrics

SPLIT_SHA = "170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc"
LABEL_SHA = "c1b56f25dc15f80c81c3321ec48ad2c67fd87ec9aca02eca76d09f240ff980b6"
ARMS = {
    "product": "plc_product_readout",
    "additive": "plc_additive_readout_control",
    "cnc_off": "plc_cnc_off",
    "classmass": "plc_residue_classmass",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        return x
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def stratified_ids(ids, sources, n, seed=44):
    """Largest-remainder proportional sampling, independent of labels/order."""
    if n <= 0 or n > len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected positive subset size and unique available IDs")
    groups = defaultdict(list)
    for sample in ids:
        groups[sources[sample]].append(sample)
    exact = {k: len(v) * n / len(ids) for k, v in groups.items()}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(groups, key=lambda k: (-(exact[k] - quotas[k]), k))[:n - sum(quotas.values())]:
        quotas[k] += 1
    selected = []
    for k in sorted(groups):
        ranked = sorted(groups[k], key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
        selected.extend(ranked[:quotas[k]])
    return sorted(selected)


def expected_calibration_error(confidence, correct, bins=15):
    idx = (confidence * bins).long().clamp(0, bins - 1)
    result = 0.0
    for b in range(bins):
        keep = idx == b
        if keep.any():
            result += float(keep.float().mean() * (confidence[keep].mean() - correct[keep].float().mean()).abs())
    return result


def common_losses(logits, labels, training):
    """One row, hard labels, FP32, exactly the shared unmodified VAL objective."""
    logp = logits.float().log_softmax(-1)
    p = logp.exp()
    log_true = logp.gather(1, labels[:, None]).squeeze(1)
    weights = logits.new_tensor(training["class_weights"], dtype=torch.float32)
    focal = (-log_true * (1 - log_true.exp()).pow(training["focal_gamma"]) * weights[labels]).mean()
    y = (labels == 0).float()
    tp, fp, fn = (p[:, 0] * y).sum(), (p[:, 0] * (1 - y)).sum(), ((1 - p[:, 0]) * y).sum()
    smooth = training["tversky_smooth"]
    tv = 1 - (tp + smooth) / (tp + training["tversky_alpha"] * fp + training["tversky_beta"] * fn + smooth)
    return {"focal": float(focal), "tversky": float(tv),
            "total": float(training["focal_weight"] * focal + training["tversky_weight"] * tv)}


def summarize(rows, gate, training):
    if not rows:
        raise ValueError("Cannot summarize an empty selection")
    z = torch.cat([r["logits"] for r in rows]).float()
    y = torch.cat([r["labels"] for r in rows]).long()
    if z.ndim != 2 or z.shape != (len(y), 3) or not torch.isfinite(z).all():
        raise ValueError("Expected finite valid-residue logits [N,3]")
    logp = z.log_softmax(-1)
    p = logp.exp()
    pred, true_contact = p.argmax(-1), y == 0
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    confidence = p.max(-1).values
    correct = pred == y
    result = {"n_rows": len(rows), "n_residues": len(y),
              "class_counts": torch.bincount(y, minlength=3).tolist(),
              "accuracy_global": float(correct.float().mean()),
              "nll_global": float(nll.mean()),
              "brier3_global": float((p - F.one_hot(y, 3)).square().sum(-1).mean()),
              "brier_contact_global": float((p[:, 0] - true_contact.float()).square().mean()),
              "ece15_toplabel_global": expected_calibration_error(confidence, correct),
              "ece15_contact_global": expected_calibration_error(p[:, 0], true_contact),
              "wrong_confidence_mean": float(confidence[~correct].mean()) if (~correct).any() else None,
              "wrong_confidence_ge_0_9_fraction": float((confidence[~correct] >= .9).float().mean()) if (~correct).any() else None,
              "nll_p50_p90_p99": torch.quantile(nll, torch.tensor([.5, .9, .99])).tolist(),
              "nll_by_true_class": {str(c): float(nll[y == c].mean()) if (y == c).any() else None for c in range(3)},
              "nll_sum_fraction_by_class": {str(c): float(nll[y == c].sum() / nll.sum()) if nll.sum() > 0 else 0.0 for c in range(3)},
              "accuracy_by_true_class": {str(c): float(correct[y == c].float().mean()) if (y == c).any() else None for c in range(3)},
              "contact_ranking_global": compute_binary_ranking_metrics(p[:, 0], true_contact),
              "contact_argmax_global": compute_binary_metrics_at_threshold((pred == 0).float(), true_contact, .5),
              "contact_at_0_5_global": compute_binary_metrics_at_threshold(p[:, 0], true_contact, .5),
              "contact_at_checkpoint_val_gate_global": compute_binary_metrics_at_threshold(p[:, 0], true_contact, gate)}
    for temperature in (1, 2, 4):
        lp = (z / temperature).log_softmax(-1)
        entropy = -(lp.exp() * lp).sum(-1)
        result[f"entropy_T{temperature}_global"] = float(entropy.mean())
        result[f"entropy_T{temperature}_by_class"] = {
            str(c): float(entropy[y == c].mean()) if (y == c).any() else None for c in range(3)}
    losses, acc, f1, gate_f1, top = [], [], [], [], defaultdict(list)
    for row in rows:
        labels, probs = row["labels"].long(), row["logits"].float().softmax(-1)
        target, predictions = labels == 0, probs.argmax(-1)
        losses.append(common_losses(row["logits"], labels, training))
        acc.append(float((predictions == labels).float().mean()))
        f1.append(compute_binary_metrics_at_threshold((predictions == 0).float(), target, .5)["f1"])
        gate_f1.append(compute_binary_metrics_at_threshold(probs[:, 0], target, gate)["f1"])
        for divisor in (20, 10, 5):
            k = max(1, math.ceil(len(labels) / divisor))
            hits = int(target[probs[:, 0].topk(k).indices].sum())
            top[f"top_l{divisor}_precision_macro"].append(hits / k)
            top[f"top_l{divisor}_hit_rate_macro"].append(float(hits > 0))
            if target.any():
                top[f"top_l{divisor}_recall_positive_rows_macro"].append(hits / int(target.sum()))
    result.update(accuracy_sample_macro=float(np.mean(acc)), contact_f1_sample_macro=float(np.mean(f1)),
                  contact_f1_val_gate_sample_macro=float(np.mean(gate_f1)),
                  common_loss_sample_macro={key: float(np.mean([v[key] for v in losses])) for key in losses[0]},
                  top_l={key: float(np.mean(values)) for key, values in top.items()})
    return result


def grouped_summary(rows, sources, gate, training):
    return {"all": summarize(rows, gate, training), "by_source": {
        source: summarize([r for r in rows if sources[r["sample_id"]] == source], gate, training)
        for source in sorted({sources[r["sample_id"]] for r in rows})}}


def tensor_digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous().numpy()
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def collect(model, cfg, ids, split, batch_size, workers, precision, output):
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in cfg["data"].items() if k in allowed}
    kwargs.update(pdb_ids=ids, split=split, use_nma=False, nma_n_confs=1, verbose=False)
    ds = PocketDatasetV8(**kwargs)
    if len(ds) != len(ids) or set(ds.pdb_ids) != set(ids):
        raise RuntimeError("Dataset silently dropped or changed requested IDs")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=pocket_collate_fn_v10, pin_memory=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    rows, seen = [], set()
    start = time.monotonic()
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            inputs = prepare_model_inputs_v11(batch, torch.device("cuda"))
            with torch.autocast("cuda", enabled=precision != "fp32", dtype=dtype):
                logits = model(**inputs)["logits"]
            for i, sample in enumerate(batch["pdb_ids"]):
                valid = batch["protein_mask"][i].bool()
                z = logits[i][valid.to(logits.device)].detach().float().cpu()
                y = batch["labels"][i][valid].cpu().long()
                if sample in seen or not len(y) or not torch.isfinite(z).all():
                    raise RuntimeError(f"Invalid/duplicate row or nonfinite logits: {sample}")
                seen.add(sample)
                rows.append(dict(sample_id=sample, logits=z, labels=y,
                    residue_axis_sha256=tensor_digest(batch["protein_coords"][i][valid], batch["chain_ids"][i][valid], y)))
            if step % 25 == 0 or len(rows) == len(ids):
                progress = dict(split=split, completed=len(rows), requested=len(ids), seconds=time.monotonic() - start)
                write_json(output / "progress.json", progress)
                print("PROGRESS", json.dumps(progress), flush=True)
    if seen != set(ids):
        raise RuntimeError("Evaluation coverage mismatch")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--transfer-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", default="baseline,product,additive,cnc_off,classmass")
    ap.add_argument("--train-subset", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--save-teacher-train", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run under a Slurm GPU allocation")
    random.seed(44); np.random.seed(44); torch.manual_seed(44)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or not set(arms) <= {"baseline", *ARMS}:
        raise ValueError("Unknown/duplicate arms")
    args.output.mkdir(parents=True, exist_ok=False)
    base_runs = list((args.baseline_root / "outputs").glob("protein_hybrid_GFGF_scratch_seed44_v11m13_*"))
    if len(base_runs) != 1:
        raise RuntimeError("Ambiguous/missing baseline run")
    base = yaml.safe_load((base_runs[0] / "config.yaml").read_text())
    data = base["data"]
    if sha256(data["split_file"]) != SPLIT_SHA or sha256(data["distances_file"]) != LABEL_SHA:
        raise RuntimeError("Unexpected split or label overlay")
    groups = read_strict_splits(data["split_file"])
    if {k: len(v) for k, v in groups.items()} != {"train": 10317, "val": 647, "test": 647}:
        raise RuntimeError("Unexpected strict split size")
    sources = {}
    source_file = Path(data["data_root"]) / "provenance.jsonl"
    for line in source_file.read_text().splitlines():
        r = json.loads(line)
        sample, source = r["sample_id"], r["source"]
        if sample in sources and sources[sample] != source:
            raise RuntimeError(f"Conflicting source for {sample}")
        sources[sample] = source
    subset = stratified_ids(groups["train"], sources, args.train_subset)
    write_json(args.output / "selection.json", dict(train_subset=subset, val_ids=groups["val"],
        train_source_counts=dict(Counter(sources[s] for s in groups["train"])),
        subset_source_counts=dict(Counter(sources[s] for s in subset)),
        split_sha256=SPLIT_SHA, label_sha256=LABEL_SHA, source_manifest_sha256=sha256(source_file),
        script_sha256=sha256(__file__), frozen_code=str(CODE), precision=args.precision,
        batch_size=args.batch_size, test_used=False, eval_mode=True,
        note="No TRAIN gate search. Apply stored checkpoint VAL gate to both splits."))
    from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
    results = {}
    for arm in arms:
        matches = base_runs if arm == "baseline" else list((args.transfer_root / "outputs").glob(ARMS[arm] + "_GFGF_scratch_seed44_v11m13_*"))
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous/missing {arm} run")
        run = matches[0]
        out = args.output / arm
        out.mkdir()
        checkpoint = run / "best_f1_c0_model.pt"
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        if cfg["model"] != ckpt["config"]["model"] or cfg["data"] != base["data"]:
            raise RuntimeError("Model or data config mismatch")
        gate = float(ckpt["val_contact_gate"])
        if not math.isfinite(gate) or not 0 <= gate <= 1:
            raise RuntimeError("Missing/invalid VAL-selected gate")
        if cfg["training"]["weighted_dice_weight"] != 0:
            raise RuntimeError("Unimplemented shared weighted-dice objective")
        model_cfg = dict(cfg["model"], use_compile=False, use_gradient_checkpointing=False)
        model = PocketGVPFusionV11m13(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.requires_grad_(False).cuda().eval()
        metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            epoch_one_based=int(ckpt["epoch"]) + 1, gate=gate, data_config=cfg["data"],
            val_legacy_f1=float(ckpt["val_metrics"]["f1_c0"]), precision=args.precision,
            model_config_sha256=hashlib.sha256(json.dumps(cfg["model"], sort_keys=True).encode()).hexdigest())
        write_json(out / "provenance.json", metadata)
        del ckpt
        results[arm] = dict(provenance=metadata)
        # VAL first provides an early result; it is never stored in the teacher cache.
        for split, ids in (("val", groups["val"]), ("train", groups["train"] if arm == "baseline" else subset)):
            rows = collect(model, cfg, ids, split, args.batch_size, args.workers, args.precision, out)
            result = grouped_summary(rows, sources, gate, cfg["training"])
            if arm == "baseline" and split == "train":
                wanted = set(subset)
                results[arm]["train_subset"] = grouped_summary([r for r in rows if r["sample_id"] in wanted], sources, gate, cfg["training"])
                if args.save_teacher_train:
                    cache = dict(format_version=1, split="train", teacher=metadata,
                        split_sha256=SPLIT_SHA, label_sha256=LABEL_SHA, n_rows=len(rows), rows=rows,
                        note="TRAIN real-ligand logits only; check crop/axis/label hashes before KD use.")
                    torch.save(cache, out / "teacher_train_logits.pt.tmp")
                    (out / "teacher_train_logits.pt.tmp").replace(out / "teacher_train_logits.pt")
            results[arm][split] = result
            write_json(out / f"{split}_metrics.json", result)
            write_json(args.output / "results.json", results)
            print("RESULT", arm, split, json.dumps(result["all"]), flush=True)
            del rows
        del model
        gc.collect(); torch.cuda.empty_cache()
    write_json(args.output / "complete.json", dict(complete=True, arms=arms, test_used=False))


if __name__ == "__main__":
    main()
