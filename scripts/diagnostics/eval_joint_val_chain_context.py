#!/usr/bin/env python3
"""Strict VAL only: frozen full-context versus independently inferred chains.

Keep production VAL caps/normalization, checkpoint and both fixed thresholds.
Never infer TRAIN/TEST, select GT chains, tune a gate, or replace the baseline.
BioLiP is already chain-sliced: report source and multichain strata explicitly.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark.analyze_holo_nonshared import require
from scripts.benchmark.eval_holo_chain_context import chain_indices, split_sample, stitch, load_npz
from scripts.benchmark.eval_joint_holo_reference import code_hashes
from scripts.diagnostics.recompute_joint_topk_reference import Aggregate, METRICS, metrics, score_row, sha, write_json
from scripts.diagnostics.strict_assignment_common import read_strict_splits
from scripts.infer.benchmark_integrity import safe_tensor_load
from src.contact_labels import decode_contact_labels, fixed5_protocol

CHECKPOINT_SHA = "234ba442325a6fb512ba54670ca233006e1004999f95322ae24c5a61f65dad05"
POLICY = dict(
    split="val", epoch=25, checkpoint_selection="existing VAL-best F1, unchanged",
    modes=["full", "chain"], precision="bf16", batch_size=2, seed=44,
    max_atoms=256, max_seq_len=1800, atom_truncation_policy="alternating_ends",
    normalization="already materialized with joint TRAIN-only statistics; no second normalization",
    feature_context="slice existing cached descriptors, rebuild downstream graph/attention per chain",
    chain_selection="all observed chains including zero-contact chains; labels only score/stratify",
    primary="same-axis global Top20 and frozen VAL gate; source/multichain stratified",
    thresholds=["0.5", "checkpoint.val_contact_gate"], tune_threshold=False,
    train_forward=False, test_forward=False, external_forward=False, automatic_policy_selection=False,
)


def val_members(split_path, groups):
    selected = set(groups["val"])
    require(not selected & set(groups["train"]) and not selected & set(groups["test"]), "VAL overlaps other sets")
    result = {}
    with split_path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["sample_id"] not in selected:
                continue
            sid, source = row["sample_id"], row["source"]
            require(sid not in result and row["split"] == "val", "Wrong/duplicate VAL member")
            require(source in ("pdbbind", "biolip") and row["sample_key"] == source + ":" + sid,
                    "Invalid source-aware member")
            result[sid] = dict(source=source, protein_id=row["pdb_id"].lower(), component_id=row["component_id"])
    require(set(result) == selected, "Incomplete VAL metadata")
    return result


def axis_digest(keys, labels):
    keys = np.asarray(keys, dtype=str)
    chain_indices(keys)
    y = np.asarray(labels)
    require(y.shape == (len(keys),) and np.isin(y, [0, 1, 2]).all(), "Invalid label axis")
    digest = hashlib.sha256(json.dumps(keys.tolist(), separators=(",", ":")).encode())
    digest.update(y.astype(np.uint8).tobytes())
    return digest.hexdigest()


def axis_spec(label, member, cap=1800):
    require(label["contact_label_protocol"] == fixed5_protocol(), "Not the audited native <5A labels")
    y, _ = decode_contact_labels(label)
    keys = np.asarray(label["residue_keys"], dtype=str)
    require(len(y) == len(keys), "Label/key mismatch")
    raw_n = len(y)
    keys, y = keys[:cap], y[:cap].numpy()
    groups = chain_indices(keys)
    require(int((y == 0).sum()) >= 3, "Audited >=3 contacts changed")
    return dict(**member, n_residues=len(y), raw_n_residues=raw_n, n_chains=len(groups),
        chain_sizes={c: len(idx) for c, idx in groups.items()},
        n_contact=int((y == 0).sum()),
        n_contact_chains=sum(bool((y[idx] == 0).any()) for idx in groups.values()),
        axis_sha256=axis_digest(keys, y))


def stratum_names(spec):
    source = spec["source"]
    size = "single_chain" if spec["n_chains"] == 1 else "multichain"
    contact = "one_contact_chain" if spec["n_contact_chains"] == 1 else "multiple_contact_chains"
    return ["all", source, size, f"{source}/{size}", contact, f"{source}/{contact}"]


def split_val_sample(sample):
    parts = split_sample(sample)
    # These production-loader fields are absent from the benchmark adapter.
    for _, idx, part in parts:
        for name in ("protein_rsa_raw", "protein_rsa_mask", "labels_original"):
            if name in sample:
                require(len(sample[name]) == sample["n_residues"], f"Bad residue axis: {name}")
                part[name] = sample[name].index_select(0, torch.from_numpy(idx))
    return parts


class ValAxisDataset(torch.utils.data.Dataset):
    def __init__(self, base, specs):
        require(base.pdb_ids == list(specs), "Production loader changed VAL coverage/order")
        self.base, self.specs = base, specs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sid = sample["pdb_id"]
        require(sid == self.base.pdb_ids[index] and sid in self.specs, "Non-VAL or reordered row")
        spec = self.specs[sid]
        require(sample["n_residues"] == spec["n_residues"], "Unexpected residue truncation")
        label = self.base.distances_dict[sid]
        keys = np.asarray(label["residue_keys"][:sample["n_residues"]], dtype=str)
        require(axis_digest(keys, sample["labels"].numpy()) == spec["axis_sha256"], "Changed VAL label/key axis")
        all_chains = sorted({str(k[0]) for k in label["residue_keys"]})
        expected = torch.tensor([all_chains.index(c) for c in keys[:, 0]], dtype=torch.long)
        require(torch.equal(sample["chain_ids"], expected), "Feature and label chain IDs differ")
        for name, value in sample.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                require(bool(torch.isfinite(value).all()), f"Nonfinite VAL input: {sid}/{name}")
        sample["residue_keys"] = keys.tolist()
        return sample


def identity_collate(samples):
    return samples


def prepare(args):
    root, work = args.training.resolve(), args.work.resolve()
    require(not (work / "protocol.json").exists(), "Do not overwrite a frozen evaluation")
    provenance = json.loads((root / "prepared/provenance.json").read_text())
    assets = dict(provenance["asset_hashes"])
    for path, digest in assets.items():
        require(sha(path) == digest, f"Changed audited training asset: {path}")
    cp = args.checkpoint.resolve()
    require(cp.name == "best_f1_c0_model.pt" and root in cp.parents and sha(cp) == CHECKPOINT_SHA,
            "Not the frozen joint seed44 epoch25 checkpoint")
    cfg_path = root / "prepared/GFGF.yaml"
    require(sha(cfg_path) == provenance["config_sha256"], "Changed production config")
    cfg = yaml.safe_load(cfg_path.read_text())
    ckpt = safe_tensor_load(cp)
    require(ckpt["epoch"] == 24 and ckpt["config"] == cfg, "Changed epoch/config")
    gate = float(ckpt["val_contact_gate"])
    require(gate == 0.5535550117492676 and gate == float(ckpt["val_metrics"]["best_gate_c0"]), "Changed fixed VAL gate")
    saved_val_metrics = {k: float(ckpt["val_metrics"][k]) for k in
                         ("f1_c0", "precision_c0", "recall_c0", "best_gate_c0") if k in ckpt["val_metrics"]}
    require(all(np.isfinite(v) for v in saved_val_metrics.values()), "Nonfinite selected VAL score/gate")
    del ckpt
    for key in ("max_atoms", "max_seq_len", "atom_truncation_policy"):
        require(cfg["data"][key] == POLICY[key], f"Changed production input cap: {key}")
    require(not cfg["model"].get("use_esm3") and not cfg["model"].get("use_compile"), "Unexpected model options")
    current = code_hashes()
    for rel, digest in provenance["source_hashes"].items():
        if rel.startswith(("src/models/", "src/data/")):
            require(current.get(rel) == digest, f"Model/data code differs from training: {rel}")
    si = json.loads((root / "split/summary.json").read_text())
    require(si["passed"] and si["cross_set_audit"]["edges"] == 0 and
            si["cross_set_audit"]["max_global_si"] < .3 and si["contract"]["edge_rule"] == "global_only",
            "Strict global SI30 audit missing/failed")
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    require({s: len(ids) for s, ids in groups.items()} == si["counts"], "Audited split counts changed")
    members = val_members(root / "split/candidate_split.tsv", groups)
    labels = safe_tensor_load(cfg["data"]["distances_file"])
    specs = {sid: axis_spec(labels[sid], members[sid], cfg["data"]["max_seq_len"]) for sid in groups["val"]}
    del labels
    coverage = defaultdict(Counter)
    for spec in specs.values():
        for group in stratum_names(spec):
            coverage[group].update(pairs=1, chains=spec["n_chains"], residues=spec["n_residues"],
                contact_residues=spec["n_contact"], contact_chains=spec["n_contact_chains"])
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "val_manifest.json", specs)
    paths = [cfg_path, cp, root / "prepared/provenance.json", work / "val_manifest.json"]
    assets.update({str(path): sha(path) for path in paths})
    write_json(work / "protocol.json", dict(policy=POLICY, training=str(root), checkpoint=str(cp),
        config=str(cfg_path), val_gate=gate, checkpoint_val_metrics=saved_val_metrics,
        assets_sha256=assets, source_sha256=current,
        counts={k: dict(v) for k, v in coverage.items()}, si_max=si["cross_set_audit"]["max_global_si"]))
    write_json(work / "prepared.json", dict(complete=True, protocol_sha256=sha(work / "protocol.json"),
                                           job_id=os.environ.get("SLURM_JOB_ID")))
    print("VAL_CHAIN_PREPARED", json.dumps({k: dict(v) for k, v in coverage.items()}), flush=True)


def verify(work):
    p = json.loads((work / "protocol.json").read_text())
    done = json.loads((work / "prepared.json").read_text())
    require(done["complete"] and done["protocol_sha256"] == sha(work / "protocol.json"), "Preparation changed/incomplete")
    require(p["policy"] == POLICY and p["source_sha256"] == code_hashes(), "Changed code/policy")
    for path, digest in p["assets_sha256"].items():
        require(sha(path) == digest, f"Changed frozen input: {path}")
    return p


@torch.inference_mode()
def infer(args):
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
    from scripts.infer.infer_v11_internal_ligdep import create_v11_model, model_needs_unimol

    work = args.work
    p = verify(work)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    require(not (work / "inference_started.json").exists(), "Inspect previous attempt before retry")
    write_json(work / "inference_started.json", dict(job_id=os.environ.get("SLURM_JOB_ID")))
    specs = json.loads((work / "val_manifest.json").read_text())
    cfg = yaml.safe_load(Path(p["config"]).read_text())
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    require(list(specs) == groups["val"], "Locked population is not exactly VAL")
    kwargs = {k: v for k, v in cfg["data"].items() if k in inspect.signature(PocketDatasetV8).parameters}
    kwargs.update(pdb_ids=list(specs), split="val", use_nma=False, nma_n_confs=1, verbose=True)
    dataset = ValAxisDataset(PocketDatasetV8(**kwargs), specs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=args.workers,
                                         collate_fn=identity_collate)
    ckpt = safe_tensor_load(p["checkpoint"])
    require(ckpt["config"] == cfg and float(ckpt["val_contact_gate"]) == p["val_gate"], "Changed checkpoint")
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    device = torch.device("cuda")
    model, version = create_v11_model(ckpt, cfg, device)
    del ckpt
    model.eval()
    require(not model_needs_unimol(model), "Unexpected UniMol model")

    def predict(samples):
        batch = pocket_collate_fn_v10(samples)
        inputs = prepare_model_inputs_v11(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        logits = output["logits"] if isinstance(output, dict) else output
        require(bool(torch.isfinite(logits[inputs["protein_mask"].bool()]).all()), "Nonfinite VAL logits")
        probs = logits.float().softmax(-1).cpu().numpy()
        return [probs[b, :s["n_residues"]].copy() for b, s in enumerate(samples)]

    dest = work / "predictions"
    dest.mkdir(exist_ok=False)
    hashes, seen, chains = {}, [], 0
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for samples in loader:
        full = predict(samples)
        tasks, collected = [], [[] for _ in samples]
        for owner, sample in enumerate(samples):
            tasks.extend((owner, idx, part) for _, idx, part in split_val_sample(sample))
        for offset in range(0, len(tasks), 2):
            chunk = tasks[offset:offset + 2]
            scores = predict([s for _, _, s in chunk])
            for (owner, idx, _), prob in zip(chunk, scores):
                collected[owner].append((idx, prob))
            chains += len(chunk)
        for owner, sample in enumerate(samples):
            sid = sample["pdb_id"]
            n = sample["n_residues"]
            full_probs = stitch(n, [(np.arange(n), full[owner])])
            chain_probs = stitch(n, collected[owner])
            path = dest / f"{sid}.npz"
            np.savez_compressed(path, full_probs=full_probs, chain_probs=chain_probs,
                labels=sample["labels"].numpy(), residue_keys=np.asarray(sample["residue_keys"], dtype=str),
                ca_coords=sample["gvp_coords"].numpy(), n_atoms=np.asarray(sample["n_atoms"]))
            hashes[sid] = sha(path)
            seen.append(sid)
        if len(seen) % 100 == 0 or len(seen) == len(dataset):
            progress = dict(pairs=len(seen), total_pairs=len(dataset), chain_forwards=chains,
                seconds=time.monotonic() - start, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            write_json(work / "progress.json", progress)
            print("VAL_CHAIN_PROGRESS", json.dumps(progress), flush=True)
    require(seen == list(specs) and chains == p["counts"]["all"]["chains"], "Incomplete VAL inference")
    write_json(work / "prediction_audit.json", hashes)
    write_json(work / "inference_complete.json", dict(complete=True, version=version, **progress,
        protocol_sha256=sha(work / "protocol.json"), prediction_audit_sha256=sha(work / "prediction_audit.json"),
        job_id=os.environ.get("SLURM_JOB_ID")))


def summarize_records(records, gate):
    from sklearn.metrics import average_precision_score, roc_auc_score

    aggregate, protein_means = defaultdict(Aggregate), defaultdict(list)
    scores, targets, zero_counts = defaultdict(list), defaultdict(list), defaultdict(Counter)
    control = []
    for record in records:
        spec, y = record["spec"], record["labels"] == 0
        groups = stratum_names(spec)
        if spec["n_chains"] == 1:
            control.append(float(np.abs(record["full"] - record["chain"]).mean()))
        for mode in POLICY["modes"]:
            prob = record[mode]
            rules = score_row(prob, y, gate)
            for group in groups:
                scores[(group, mode)].append(prob)
                targets[(group, mode)].append(y)
                for rule, counts in rules.items():
                    aggregate[(group, mode, rule)].add(counts)
                    protein_means[(group, mode, rule, spec["protein_id"])].append(
                        metrics(counts["n"], counts["positive"], counts["predicted"], counts["tp"]))
            for idx in chain_indices(record["keys"]).values():
                if y[idx].any():
                    continue
                for rule, threshold in (("at_0_5", .5), ("at_val_gate", gate)):
                    n_pred = int((prob[idx].astype(np.float64) >= threshold).sum())
                    for group in groups:
                        zero_counts[(group, mode, rule)].update(chains=1, residues=len(idx), predicted=n_pred)
    protein_groups = defaultdict(list)
    for (group, mode, rule, _), values in protein_means.items():
        protein_groups[(group, mode, rule)].append(
            {k: float(np.mean([v[k] for v in values])) for k in METRICS})
    results = {}
    for (group, mode), parts in scores.items():
        prob, y = np.concatenate(parts), np.concatenate(targets[(group, mode)])
        rules = {}
        for key, a in aggregate.items():
            if key[:2] != (group, mode):
                continue
            rule = key[2]
            entry = a.finish()
            balanced = protein_groups[key]
            entry["protein_mean_of_ligand_mean"] = {k: float(np.mean([v[k] for v in balanced])) for k in METRICS}
            entry["n_proteins"] = len(balanced)
            rules[rule] = entry
        zero = {}
        for rule in ("at_0_5", "at_val_gate"):
            c = zero_counts[(group, mode, rule)]
            zero[rule] = dict(chains=c["chains"], residues=c["residues"], predicted=c["predicted"],
                fpr=c["predicted"] / c["residues"] if c["residues"] else None,
                mean_predicted=c["predicted"] / c["chains"] if c["chains"] else None)
        results.setdefault(group, {})[mode] = dict(rules=rules, zero_contact_chains=zero,
            auprc=float(average_precision_score(y, prob)) if y.any() else None,
            auroc=float(roc_auc_score(y, prob)) if y.any() and not y.all() else None)
    gaps = {g: {rule: {agg: {m: result["chain"]["rules"][rule][agg][m] - result["full"]["rules"][rule][agg][m]
                for m in METRICS} for agg in ("sample_macro", "pooled", "protein_mean_of_ligand_mean")}
                for rule in result["full"]["rules"]} for g, result in results.items()}
    return dict(results=results, chain_minus_full=gaps, single_chain_control=dict(n=len(control),
        mean_abs_probability_difference=float(np.mean(control)) if control else None,
        max_sample_mean_abs_difference=max(control) if control else None))


def summarize(args):
    work = args.work
    p = verify(work)
    done = json.loads((work / "inference_complete.json").read_text())
    require(done["complete"] and done["protocol_sha256"] == sha(work / "protocol.json") and
            done["prediction_audit_sha256"] == sha(work / "prediction_audit.json"), "Inference incomplete/changed")
    specs = json.loads((work / "val_manifest.json").read_text())
    hashes = json.loads((work / "prediction_audit.json").read_text())
    require(set(hashes) == set(specs), "VAL prediction coverage changed")
    records = []
    for sid, spec in specs.items():
        path = work / "predictions" / f"{sid}.npz"
        require(sha(path) == hashes[sid], f"Changed prediction: {sid}")
        data = load_npz(path)
        require(axis_digest(data["residue_keys"], data["labels"]) == spec["axis_sha256"], "Prediction axis differs")
        n = spec["n_residues"]
        record = dict(spec=spec, labels=data["labels"], keys=data["residue_keys"])
        for mode in POLICY["modes"]:
            record[mode] = stitch(n, [(np.arange(n), data[mode + "_probs"])])[:, 0]
        records.append(record)
    result = summarize_records(records, p["val_gate"])
    write_json(work / "summary.json", dict(complete=True, policy=POLICY, protocol_sha256=sha(work / "protocol.json"),
        val_gate=p["val_gate"], coverage=p["counts"], runtime=done, **result))
    for group in ("all", "pdbbind", "biolip", "multichain"):
        if group in result["results"]:
            print("VAL_CHAIN_SUMMARY", group, json.dumps({m: {r: v["rules"][r]["sample_macro"]
                  for r in ("top20", "at_0_5", "at_val_gate")} for m, v in result["results"][group].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "infer", "summarize"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    globals()[args.phase](args)
