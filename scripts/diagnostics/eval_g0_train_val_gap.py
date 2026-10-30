#!/usr/bin/env python3
"""Frozen G0 TRAIN-subset/strict-VAL gap plus paired full/chain diagnostics.

No gradient, augmentation, gate search, TEST or benchmark selection. Both sets
use identical eval-mode inputs, BF16, caps and the saved VAL-selected gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# In the actual job, import the exact G0 model/data/helpers, not today's code.
# This occurs before any project import (unit tests use the current helpers).
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--training", required=True, type=Path)
    known, _ = bootstrap.parse_known_args()
    sys.path.insert(0, str(known.training / "code_v1"))
from scripts.benchmark.eval_holo_chain_context import chain_indices, stitch, load_npz
from scripts.benchmark.eval_joint_holo_reference import code_hashes
from scripts.diagnostics.eval_joint_val_chain_context import (
    ValAxisDataset, axis_digest, split_val_sample, identity_collate, summarize_records,
)
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json
from scripts.diagnostics.strict_assignment_common import read_strict_splits
from src.contact_labels import decode_contact_labels


def stratified_train_ids(metadata, n=4096):
    groups = {s: [sid for sid, row in metadata.items() if row["split"] == "train" and row["source"] == s]
              for s in ("pdbbind", "biolip")}
    total = sum(map(len, groups.values()))
    n = min(n, total)
    quotas = {s: int(n * len(ids) / total) for s, ids in groups.items()}
    for s in sorted(groups, key=lambda s: -(n * len(groups[s]) / total - quotas[s]))[:n - sum(quotas.values())]:
        quotas[s] += 1
    selected = []
    for s, ids in groups.items():
        ids.sort(key=lambda sid: hashlib.sha256(("frozen_train_gap_v1/" + sid).encode()).hexdigest())
        selected.extend(ids[:quotas[s]])
    return sorted(selected)


def spec_for(label, member, cap):
    y, _ = decode_contact_labels(label)
    keys = np.asarray(label["residue_keys"][:cap], dtype=str)
    y = y[:cap].numpy()
    chains = chain_indices(keys)
    return dict(source=member["source"], protein_id=member["pdb_id"].lower(),
        component_id=member["component_id"], n_residues=len(y), raw_n_residues=len(label["residue_keys"]),
        n_chains=len(chains), n_contact_chains=sum(bool((y[idx] == 0).any()) for idx in chains.values()),
        axis_sha256=axis_digest(keys, y))


@torch.inference_mode()
def run(args):
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
    from scripts.infer.infer_v11_internal_ligdep import create_v11_model, model_needs_unimol

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a BF16-capable Slurm GPU allocation")
    torch.set_num_threads(4)
    work = args.work
    work.mkdir(parents=True, exist_ok=False)
    cfg_path = args.training / "prepared/G0_small_global.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    provenance = json.loads((args.training / "prepared/provenance.json").read_text())
    current = code_hashes()
    for rel, expected in provenance["source_hashes"].items():
        if rel.startswith(("src/models/", "src/data/")) and current.get(rel) != expected:
            raise ValueError(f"G0 model/input code changed: {rel}")
    assets = dict(provenance["asset_hashes"])
    for path, expected in assets.items():
        if sha(path) != expected:
            raise ValueError(f"Audited asset changed: {path}")
    if sha(cfg_path) != provenance["config_hashes"]["G0_small_global"]:
        raise ValueError("G0 config changed")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt["config"] != cfg or args.checkpoint.name != "best_f1_c0_model.pt":
        raise ValueError("Expected unchanged G0 VAL-best checkpoint")
    gate = float(ckpt["val_contact_gate"])
    if not np.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("Invalid stored VAL gate")
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    with (args.upstream / "split/candidate_split.tsv").open() as f:
        meta = {r["sample_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    for split, ids in groups.items():
        if any(meta[sid]["split"] != split for sid in ids):
            raise ValueError("Split metadata mismatch")
    selected = dict(train=stratified_train_ids(meta, args.train_samples), val=groups["val"])
    if set(selected["train"]) & set(selected["val"]) or any(set(ids) & set(groups["test"]) for ids in selected.values()):
        raise ValueError("Development cohort overlaps TEST")
    protocol = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        epoch=int(ckpt["epoch"]) + 1, config_sha256=sha(cfg_path), val_gate=gate,
        populations=selected, original_counts={k: len(v) for k, v in groups.items()},
        train_sampling="source-proportional hash sample independent of labels/predictions",
        evaluation="same frozen eval-mode model; no NMA/no dropout/BF16; all chains",
        labels="legacy audited per-copy <5A; NOT new union targets",
        test_forward=False, tune_threshold=False, chain_cap="original parent1800 axis, then stitch",
        source_hashes=current, model_source=str(args.training / "code_v1"),
        diagnostic_runner_sha256=sha(__file__), assets=assets)
    write_json(work / "protocol.json", protocol)
    device = torch.device("cuda")
    torch.manual_seed(44)
    model, version = create_v11_model(ckpt, cfg, device)
    del ckpt
    model.eval()
    if model_needs_unimol(model):
        raise ValueError("Unexpected UniMol dependency")

    def predict(samples):
        batch = pocket_collate_fn_v10(samples)
        inputs = prepare_model_inputs_v11(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs)["logits"]
        valid = inputs["protein_mask"].bool()
        if not torch.isfinite(logits[valid]).all():
            raise ValueError("Nonfinite valid-residue logits")
        probabilities = logits.float().softmax(-1).cpu().numpy()
        return [probabilities[i, :s["n_residues"]] for i, s in enumerate(samples)]

    summaries = {}
    start = time.monotonic()
    for split, ids in selected.items():
        allowed = inspect.signature(PocketDatasetV8).parameters
        kwargs = {k: v for k, v in cfg["data"].items() if k in allowed}
        kwargs.update(pdb_ids=ids, split=split, use_nma=False, nma_n_confs=1, verbose=False)
        base = PocketDatasetV8(**kwargs)
        specs = {sid: spec_for(base.distances_dict[sid], meta[sid], cfg["data"]["max_seq_len"]) for sid in ids}
        dataset = ValAxisDataset(base, specs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
            num_workers=args.workers, collate_fn=identity_collate)
        dest = work / split
        dest.mkdir()
        hashes, records, nll, residues = {}, [], Counter(), Counter()
        for samples in loader:
            full = predict(samples)
            parts = [(i, idx, part) for i, sample in enumerate(samples) for _, idx, part in split_val_sample(sample)]
            collected = [[] for _ in samples]
            for off in range(0, len(parts), 2):
                chunk = parts[off:off + 2]
                pred = predict([p[2] for p in chunk])
                for (owner, idx, _), prob in zip(chunk, pred):
                    collected[owner].append((idx, prob))
            for owner, sample in enumerate(samples):
                sid, y = sample["pdb_id"], sample["labels"].numpy()
                fp = stitch(len(y), [(np.arange(len(y)), full[owner])])
                cp = stitch(len(y), collected[owner])
                path = dest / f"{sid}.npz"
                np.savez_compressed(path, full_probs=fp, chain_probs=cp, labels=y,
                    residue_keys=np.asarray(sample["residue_keys"], dtype=str))
                hashes[sid] = sha(path)
                records.append(dict(spec=specs[sid], labels=y, keys=np.asarray(sample["residue_keys"], dtype=str),
                                    full=fp[:, 0], chain=cp[:, 0]))
                for mode, p in (("full", fp), ("chain", cp)):
                    for source in ("all", specs[sid]["source"]):
                        key = source + "/" + mode
                        nll[key] += float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).sum())
                        residues[key] += len(y)
            if len(records) % 100 == 0 or len(records) == len(dataset):
                progress = dict(split=split, processed=len(records), expected=len(dataset), seconds=time.monotonic() - start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                write_json(work / "progress.json", progress)
                print("G0_GAP_PROGRESS", json.dumps(progress), flush=True)
        if list(hashes) != ids:
            raise ValueError("Incomplete or reordered inference")
        summaries[split] = summarize_records(records, gate)
        summaries[split]["unweighted_three_class_nll"] = {k: nll[k] / residues[k] for k in residues}
        write_json(dest / "prediction_audit.json", hashes)
        write_json(dest / "summary.json", summaries[split])
    gap = {}
    for source in ("all", "pdbbind", "biolip"):
        train = summaries["train"]["results"][source]["full"]
        val = summaries["val"]["results"][source]["full"]
        gap[source] = dict(auprc=train["auprc"] - val["auprc"], auroc=train["auroc"] - val["auroc"],
            fixed_threshold={rule: {agg: {m: train["rules"][rule][agg][m] - val["rules"][rule][agg][m]
                for m in ("f1", "mcc")} for agg in ("sample_macro", "pooled", "protein_mean_of_ligand_mean")}
                for rule in ("at_0_5", "at_val_gate", "top20")})
    write_json(work / "summary.json", dict(complete=True, version=version, train_minus_val=gap,
        protocol_sha256=sha(work / "protocol.json"), val_gate=gate, seconds=time.monotonic() - start,
        note="TRAIN is a fixed subset; gap is descriptive, not proof that capacity alone causes transfer failure."))
    print("G0_GAP_COMPLETE", json.dumps(gap), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--training", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--workers", type=int, default=8)
    run(p.parse_args())
