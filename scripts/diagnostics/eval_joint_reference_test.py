#!/usr/bin/env python3
"""Locked, reference-only TEST evaluation; never selects or tunes anything.

Prepare locks the VAL-best checkpoint, its existing VAL gate, the independent
TEST population and source-stratified donor mapping before inference. Evaluate
runs real/shuffle/zero once under that contract. No TEST threshold search,
optimization, automatic numerical fallback, or checkpoint comparison exists.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from prepare_internal_gfgf import split_members, verify_prepared
from prepare_libre_transfer import sha256, source_hashes
from strict_assignment_common import read_strict_splits
from scripts.infer.infer_v11_internal_ligdep import (
    create_v11_model, model_needs_unimol, predict_v11_logits,
)
from scripts.infer.internal_ligand_dependency import (
    LigandAblationDataset, batch_unimol, collate_with_ligand_source, load_unimol_cache,
)
from src.metrics.metrics import compute_binary_metrics_at_threshold, compute_binary_ranking_metrics

MODES = ("real", "shuffle", "zero")
POLICY = dict(reference_only=True, test_used_for_selection=False,
              test_driven_iteration_allowed=False, selection_metric="val_f1_c0",
              batch_size=8, num_workers=4, precision="bf16", shuffle_seed=42,
              shuffle_scope="within_source_bijective_no_self",
              nonfinite_policy="fail_without_dropping_rows", modes=list(MODES))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def donor_indices(ids, sources, seed=42):
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate TEST IDs")
    groups = defaultdict(list)
    for i, sid in enumerate(ids):
        groups[sources[sid]].append(i)
    result = list(range(len(ids)))
    rng = random.Random(seed)
    for source in sorted(groups):
        order = groups[source].copy()
        if len(order) < 2:
            raise ValueError("At least two rows per source are needed for shuffle")
        rng.shuffle(order)
        for i, target in enumerate(order):
            result[target] = order[(i + 1) % len(order)]
    return result


def checkpoint_gate(ckpt):
    if ckpt.get("epoch") != 24:
        raise ValueError("This reference evaluation is locked to VAL-best epoch25")
    gate = float(ckpt["val_contact_gate"])
    if not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("Invalid VAL-selected gate")
    if gate != float(ckpt["val_metrics"]["best_gate_c0"]):
        raise ValueError("Gate differs from the selected checkpoint's VAL gate")
    return gate


def prepare(experiment, checkpoint, output):
    experiment, checkpoint = experiment.resolve(), checkpoint.resolve()
    if checkpoint.name != "best_f1_c0_model.pt" or experiment not in checkpoint.parents:
        raise ValueError("Expected the experiment's VAL-best F1 checkpoint")
    if (output / "protocol.json").exists():
        raise FileExistsError("Do not overwrite an already locked TEST protocol")
    provenance = verify_prepared(experiment, require_smoke=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    gate = checkpoint_gate(ckpt)
    cfg = ckpt["config"]
    if cfg != yaml.safe_load((experiment / "prepared/GFGF.yaml").read_text()):
        raise ValueError("Checkpoint and audited data/config differ")
    if cfg["experiment"]["seed"] != 44 or not cfg["training"].get("use_bf16"):
        raise ValueError("Expected the audited BF16 seed44 run")
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    members = split_members(experiment / "split/candidate_split.tsv", groups, "pdbbind_biolip_joint")
    ids = groups["test"]
    sources = {sid: members[sid][1] for sid in ids}
    donors = donor_indices(ids, sources, POLICY["shuffle_seed"])
    evaluation_sources = source_hashes(ROOT)
    for rel, digest in provenance["source_hashes"].items():
        if rel.startswith(("src/models/", "src/data/")) and evaluation_sources.get(rel) != digest:
            raise ValueError(f"Evaluation model/data source differs from training: {rel}")
    artifacts = dict(provenance["asset_hashes"])
    artifacts[str(checkpoint)] = sha256(checkpoint)
    artifacts[str(experiment / "prepared/provenance.json")] = sha256(experiment / "prepared/provenance.json")
    protocol = dict(policy=POLICY, checkpoint=str(checkpoint), epoch=25, val_gate=gate,
                    checkpoint_sha256=artifacts[str(checkpoint)], artifact_sha256=artifacts,
                    evaluation_source_sha256=evaluation_sources,
                    samples=[dict(sample_id=sid, source=sources[sid], donor_id=ids[donors[i]])
                             for i, sid in enumerate(ids)],
                    counts_by_source=dict(Counter(sources.values())),
                    zero_definition="zero atom features/coords/bond attributes/UniMol; retain atom count and edge indices",
                    coordinate_contract="no shared protein-ligand coordinates supplied to model",
                    normalization="unchanged training-only fitted features",
                    caps=dict(max_atoms=cfg["data"]["max_atoms"], max_seq_len=cfg["data"]["max_seq_len"],
                              atom_truncation_policy=cfg["data"].get("atom_truncation_policy", "head")))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", protocol)
    print("LOCKED_REFERENCE_TEST", json.dumps(dict(rows=len(ids), by_source=protocol["counts_by_source"],
          epoch=25, gate=gate, protocol_sha256=sha256(output / "protocol.json"))), flush=True)


def validate_logits(logits, labels, sample_id):
    if logits.shape != (len(labels), 3) or not len(labels) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Invalid/nonfinite valid-residue logits: {sample_id}")
    if not bool(((labels >= 0) & (labels < 3)).all()):
        raise RuntimeError(f"Invalid labels: {sample_id}")


def summarize(rows, gate):
    if not rows:
        raise ValueError("Empty reference population")
    scores = torch.cat([r["scores"] for r in rows])
    targets = torch.cat([r["labels"] == 0 for r in rows])
    predictions = torch.cat([r["argmax"] == 0 for r in rows])
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite scores must never be silently filtered")
    result = dict(n_samples=len(rows), n_residues=len(scores), n_positive=int(targets.sum()),
                  ranking_global=compute_binary_ranking_metrics(scores, targets),
                  argmax_global=compute_binary_metrics_at_threshold(predictions.float(), targets, .5),
                  at_0_5_global=compute_binary_metrics_at_threshold(scores, targets, .5),
                  at_val_gate_global=compute_binary_metrics_at_threshold(scores, targets, gate))
    sample_metrics = defaultdict(list)
    for row in rows:
        target = row["labels"] == 0
        for label, values, threshold in [("argmax", (row["argmax"] == 0).float(), .5),
                                          ("at_0_5", row["scores"], .5),
                                          ("at_val_gate", row["scores"], gate)]:
            metrics = compute_binary_metrics_at_threshold(values, target, threshold)
            for key in ("f1", "mcc", "precision", "recall"):
                sample_metrics[f"{label}_{key}"].append(metrics[key])
        for divisor in (20, 10, 5):
            k = max(1, math.ceil(len(target) / divisor))
            hits = int(target[row["scores"].topk(k).indices].sum())
            sample_metrics[f"top_l{divisor}_precision"].append(hits / k)
            sample_metrics[f"top_l{divisor}_recall"].append(hits / max(int(target.sum()), 1))
    result["sample_macro"] = {key: sum(values) / len(values) for key, values in sample_metrics.items()}
    return result


def evaluate(output, expected_protocol_sha):
    protocol_path = output / "protocol.json"
    if sha256(protocol_path) != expected_protocol_sha:
        raise ValueError("Changed preregistered reference protocol")
    protocol = json.loads(protocol_path.read_text())
    if protocol["policy"] != POLICY or source_hashes(ROOT) != protocol["evaluation_source_sha256"]:
        raise ValueError("Changed evaluation policy/code")
    if (output / "started.json").exists():
        raise FileExistsError("Reference run already started; inspect it rather than automatically rerunning")
    for path, digest in protocol["artifact_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Changed locked artifact: {path}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This fixed protocol requires a BF16-capable GPU")
    ckpt = torch.load(protocol["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if checkpoint_gate(ckpt) != protocol["val_gate"]:
        raise ValueError("Changed checkpoint gate")
    cfg = ckpt["config"]
    ids = [r["sample_id"] for r in protocol["samples"]]
    if ids != read_strict_splits(Path(cfg["data"]["split_file"]))["test"]:
        raise ValueError("Locked population is not exactly TEST")
    sources = {r["sample_id"]: r["source"] for r in protocol["samples"]}
    index = {sid: i for i, sid in enumerate(ids)}
    donors = [index[r["donor_id"]] for r in protocol["samples"]]
    if donors != donor_indices(ids, sources, POLICY["shuffle_seed"]):
        raise ValueError("Changed shuffle assignment")
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    kwargs = {k: v for k, v in cfg["data"].items() if k in inspect.signature(PocketDatasetV8).parameters}
    kwargs.update(pdb_ids=ids, split="test", use_nma=False, nma_n_confs=1, verbose=True)
    dataset = PocketDatasetV8(**kwargs)
    if dataset.pdb_ids != ids:
        raise RuntimeError("Production loader changed TEST coverage/order")
    device = torch.device("cuda")
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    model, version = create_v11_model(ckpt, cfg, device)
    del ckpt
    needs_unimol = model_needs_unimol(model)
    d_unimol = int(cfg["model"].get("d_unimol", 512))
    cache_args = SimpleNamespace(no_unimol=False, data_root=None, unimol_embeddings_dir=None)
    cache = load_unimol_cache(cfg, cache_args, d_unimol) if needs_unimol else {}
    write_json(output / "started.json", dict(protocol_sha256=expected_protocol_sha,
               job_id=__import__("os").environ.get("SLURM_JOB_ID"), model_version=version,
               torch_version=torch.__version__, device=torch.cuda.get_device_name()))
    summaries, baseline_axis = {}, None
    for mode in MODES:
        ablated = LigandAblationDataset(dataset, mode, seed=POLICY["shuffle_seed"])
        ablated.perm = donors
        loader = torch.utils.data.DataLoader(ablated, batch_size=POLICY["batch_size"],
            shuffle=False, num_workers=POLICY["num_workers"], pin_memory=True,
            collate_fn=collate_with_ligand_source)
        rows, seen, axis = [], [], hashlib.sha256()
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                ue = batch_unimol(batch["unimol_source_pdb_ids"], cache, d_unimol, device, POLICY["shuffle_seed"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = predict_v11_logits(model, batch, device, ue)
                for i, sid in enumerate(batch["pdb_ids"]):
                    valid = batch["protein_mask"][i].bool()
                    z = logits[i][valid.to(device)].float().cpu()
                    y = batch["labels"][i][valid].long().cpu()
                    validate_logits(z, y, sid)
                    scores = z.softmax(-1)[:, 0].contiguous()
                    if not torch.isfinite(scores).all():
                        raise RuntimeError(f"Nonfinite probabilities: {sid}")
                    axis.update(sid.encode() + b"\0" + y.numpy().tobytes())
                    rows.append(dict(sample_id=sid, source=sources[sid],
                        donor_id=batch["ligand_source_pdb_ids"][i],
                        scores=scores, labels=y.to(torch.uint8), argmax=z.argmax(-1).to(torch.uint8)))
                    seen.append(sid)
                if step % 25 == 0 or len(rows) == len(ids):
                    progress = dict(mode=mode, completed=len(rows), requested=len(ids),
                                    elapsed_seconds=time.monotonic()-start)
                    write_json(output / "progress.json", progress)
                    print("PROGRESS", json.dumps(progress), flush=True)
        if seen != ids:
            raise RuntimeError("Inference population/order mismatch")
        if baseline_axis is not None and baseline_axis != axis.hexdigest():
            raise RuntimeError("Residue labels/axis differ between ligand modes")
        baseline_axis = axis.hexdigest()
        summary = {"all": summarize(rows, protocol["val_gate"])}
        for source in sorted(set(sources.values())):
            summary[source] = summarize([r for r in rows if r["source"] == source], protocol["val_gate"])
        summary["runtime"] = dict(seconds=time.monotonic()-start,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
             residue_axis_sha256=baseline_axis, nonfinite_residues=0)
        torch.save(rows, output / f"predictions_{mode}.pt")
        write_json(output / f"summary_{mode}.json", summary)
        summaries[mode] = summary
        print("MODE_COMPLETE", mode, flush=True)
    gaps = {}
    for source in ("all", *sorted(set(sources.values()))):
        gaps[source] = {}
        for mode in ("shuffle", "zero"):
            gaps[source][f"real_minus_{mode}"] = {
                decision: {metric: summaries["real"][source][decision][metric] - summaries[mode][source][decision][metric]
                           for metric in ("f1", "mcc")}
                for decision in ("argmax_global", "at_0_5_global", "at_val_gate_global")}
    write_json(output / "reference_test_summary.json", dict(complete=True,
        reference_only=True, test_driven_iteration_allowed=False, protocol_sha256=expected_protocol_sha,
        epoch=25, val_gate=protocol["val_gate"], summaries=summaries, gaps=gaps))
    print("REFERENCE_TEST_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--experiment", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--protocol-sha256")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        if args.experiment is None or args.checkpoint is None:
            ap.error("prepare requires experiment and checkpoint")
        prepare(args.experiment, args.checkpoint, args.output)
    else:
        if not args.protocol_sha256:
            ap.error("evaluation requires the pre-inference protocol hash")
        evaluate(args.output, args.protocol_sha256)


if __name__ == "__main__":
    main()
