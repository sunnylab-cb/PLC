#!/usr/bin/env python3
"""Frozen HOLO4K chain-context inference, not a new trained/selected model.

Predict EVERY observed chain for EVERY ligand, then stitch onto the parent's
1800-residue axis. Primary Top20 is global, not twenty predictions per chain.
Positive-chain-only scores are explicitly oracle/conditional diagnostics.
Cached receptor features are sliced, not re-extracted in isolated chains.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark.analyze_holo_nonshared import (
    PARENT_SHA, add_protein_means, require, same_numbers,
)
from scripts.benchmark.diagnose_libre_ranking import csv_rows, load_pair
from scripts.benchmark.eval_joint_holo_reference import code_hashes
from scripts.benchmark.run_libre_joint_test import compare
from scripts.diagnostics.recompute_joint_topk_reference import (
    Aggregate, METRICS, metrics, score_row, sha, write_json,
)

POLICY = dict(
    reference_only=True, test_driven_iteration_allowed=False,
    checkpoint="unchanged joint GFGF seed44 VAL-best epoch25",
    precision="bf16", chain_microbatch=2, max_atoms=-1, max_seq_len=1800,
    primary="all chains independently inferred, stitched, global Top20 per original pair",
    secondary="GT-positive chain conditional; cropped full-context control on identical axes",
    chain_selection="all chains from residue keys only, never labels/ligand chain names",
    feature_context="slice existing full-receptor V4/ESM; do not recompute descriptors",
    normalization="unchanged joint TRAIN-only statistics and spatial clamp",
    threshold="unchanged PLC VAL gate; released LiBRe FP32 p>0.6",
    external_si30_certified=False, retrain=False, tune_threshold=False,
)
RESIDUE_FIELDS = (
    "esm", "esm3", "protein_physchem", "protein_spatial_scalar", "protein_spatial_vector",
    "gvp_coords", "gvp_vectors", "chain_ids", "labels", "raw_distances",
    "esm_valid_mask", "esm3_valid_mask",
)


def chain_indices(keys):
    keys = np.asarray(keys, dtype=str)
    require(keys.ndim == 2 and keys.shape[1] == 3 and len(keys) > 0, "Invalid residue keys")
    require(len(set(map(tuple, keys))) == len(keys), "Duplicate residue keys")
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[str(key[0])].append(i)
    return {chain: np.asarray(idx, dtype=np.int64) for chain, idx in groups.items()}


def split_sample(sample):
    """Whitelist residue axes: ligand atom count can equal residue count."""
    require(len(sample["residue_keys"]) == sample["n_residues"], "Residue axis mismatch")
    result = []
    for chain, idx in chain_indices(sample["residue_keys"]).items():
        part = dict(sample)
        index = torch.from_numpy(idx)
        for field in RESIDUE_FIELDS:
            if field in sample:
                require(sample[field].shape[0] == sample["n_residues"], f"Bad axis: {field}")
                part[field] = sample[field].index_select(0, index)
        part["chain_ids"] = torch.zeros(len(idx), dtype=torch.long)
        part["residue_keys"] = [sample["residue_keys"][i] for i in idx]
        part["n_residues"] = len(idx)
        result.append((chain, idx, part))
    return result


def stitch(n, pieces):
    result = np.empty((n, 3), dtype=np.float32)
    covered = np.zeros(n, dtype=np.int32)
    for idx, probs in pieces:
        idx = np.asarray(idx, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float32)
        require(idx.ndim == 1 and len(idx) > 0 and ((idx >= 0) & (idx < n)).all(), "Invalid chain indices")
        require(len(np.unique(idx)) == len(idx), "Duplicate chain indices")
        require(probs.shape == (len(idx), 3) and np.isfinite(probs).all(), "Bad chain probabilities")
        require(((probs >= 0) & (probs <= 1)).all() and
                np.allclose(probs.sum(-1), 1, atol=1e-5), "Invalid probability simplex")
        covered[idx] += 1
        result[idx] = probs
    require(np.all(covered == 1), "Chains must cover the original axis exactly once")
    return result


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def parent_predictions(parent):
    return parent / "inference/predictions/holo4k"


def prepare(args):
    parent, work = args.parent, args.work
    require(sha(parent / "protocol.json") == PARENT_SHA, "Wrong frozen HOLO parent")
    p = json.loads((parent / "protocol.json").read_text())
    summary = json.loads((parent / "matched_summary.json").read_text())
    require(summary["complete"] and summary["protocol_sha256"] == PARENT_SHA, "Incomplete parent")
    current = code_hashes()
    for rel, digest in p["code_sha256"].items():
        if rel.startswith(("src/models/", "src/data/", "scripts/infer/")):
            require(current.get(rel) == digest, f"Changed model/input code: {rel}")
    for path, digest in p["asset_sha256"].items():
        require(sha(path) == digest, f"Changed parent asset: {path}")
    cohorts = json.loads((args.cohorts / "cohorts.json").read_text())
    nonshared = json.loads((args.cohorts / "summary.json").read_text())
    require(nonshared["complete"] and nonshared["all_cohort_reproduces_parent"], "Unverified cohorts")
    require(nonshared["parent_protocol_sha256"] == PARENT_SHA, "Wrong cohort parent")
    cp = json.loads((args.cohorts / "protocol.json").read_text())
    require(sha(args.cohorts / "cohorts.json") == cp["input_sha256"][str(args.cohorts / "cohorts.json")],
            "Changed cohort membership")
    plc = csv_rows(parent / "inference/holo4k_all_results.csv")
    audit = {r["sample_id"]: r for r in json.loads((parent / "matched_prediction_audit.json").read_text())}
    require(len(plc) == p["expected_all"] and set(audit) == set(cohorts["cohorts"]["all_matched"]["sample_ids"]),
            "Changed population")
    manifest, counts = {}, Counter()
    for sid in sorted(plc):
        path = parent_predictions(parent) / f"{sid}.npz"
        digest = sha(path)
        if sid in audit:
            require(digest == audit[sid]["plc_sha256"], f"Changed parent prediction: {sid}")
        data = load_npz(path)
        chains = chain_indices(data["residue_keys"])
        manifest[sid] = dict(sha256=digest, chains={c: idx.tolist() for c, idx in chains.items()})
        counts["pairs"] += 1
        counts["chain_ligand_rows"] += len(chains)
        counts["single_chain_pairs"] += int(len(chains) == 1)
        counts["multichain_pairs"] += int(len(chains) > 1)
        counts["short_chains_lt3"] += sum(len(idx) < 3 for idx in chains.values())
    work.mkdir(parents=True, exist_ok=True)
    require(not (work / "protocol.json").exists(), "Protocol exists; do not overwrite")
    write_json(work / "chains.json", manifest)
    write_json(work / "cohorts.json", cohorts)
    paths = [parent / name for name in ("protocol.json", "matched_summary.json", "matched_prediction_audit.json",
             "inference/holo4k_all_results.csv")]
    paths += [work / "chains.json", work / "cohorts.json"]
    contract = dict(policy=POLICY, parent=str(parent), parent_protocol=p, counts=dict(counts),
                    source_sha256=current, input_sha256={str(path): sha(path) for path in paths})
    write_json(work / "protocol.json", contract)
    write_json(work / "prepared.json", dict(complete=True, protocol_sha256=sha(work / "protocol.json"),
                                           job_id=os.environ.get("SLURM_JOB_ID")))
    print("CHAIN_CONTEXT_PREPARED", json.dumps(dict(counts)), flush=True)


def verify(work):
    p = json.loads((work / "protocol.json").read_text())
    done = json.loads((work / "prepared.json").read_text())
    require(done["complete"] and done["protocol_sha256"] == sha(work / "protocol.json"), "Unprepared evaluation")
    require(p["policy"] == POLICY and p["source_sha256"] == code_hashes(), "Changed evaluation code/policy")
    for path, digest in {**p["input_sha256"], **p["parent_protocol"]["asset_sha256"]}.items():
        require(sha(path) == digest, f"Changed input: {path}")
    return p


def check_sample(sample, baseline):
    for field, array in (("labels", "labels"), ("gvp_coords", "ca_coords")):
        require(np.array_equal(sample[field].numpy(), baseline[array]), f"Changed sample axis: {field}")
    require(np.array_equal(np.asarray(sample["residue_keys"], dtype=str), baseline["residue_keys"]),
            "Changed residue keys")


@torch.inference_mode()
def infer(args):
    from torch.utils.data import DataLoader
    import yaml
    from scripts.infer.benchmark_integrity import safe_tensor_load
    from scripts.infer.infer_v11_benchmark import (
        BenchmarkV11Dataset, _load_cohort_norm, benchmark_collate, identity_collate,
    )
    from scripts.infer.infer_v11_internal_ligdep import create_v11_model, model_needs_unimol
    from src.data.collate import prepare_model_inputs_v11
    from src.metrics.metrics import contact_gate_predictions

    work = args.work
    contract = verify(work)
    p, parent = contract["parent_protocol"], Path(contract["parent"])
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    require(not (work / "inference_started.json").exists(), "Inference exists; inspect before retry")
    write_json(work / "inference_started.json", dict(job_id=os.environ.get("SLURM_JOB_ID")))
    cfg = yaml.safe_load(Path(p["config"]).read_text())
    ckpt = safe_tensor_load(p["checkpoint"])
    require(ckpt["epoch"] == 24 and ckpt["config"] == cfg and
            float(ckpt["val_contact_gate"]) == p["val_gate"], "Changed checkpoint/config/gate")
    device = torch.device("cuda")
    model, version = create_v11_model(ckpt, cfg, device)
    del ckpt
    require(not model_needs_unimol(model), "Unexpected UniMol model")
    model.eval()
    norm, spatial = _load_cohort_norm(Path(p["stats"]))
    dc = cfg["data"]
    require(dc["use_spatial_clamp"] and dc["spatial_clamp_dims"] == [62, 76, 78, 79] and
            dc["spatial_clamp_range"] == [-5., 5.], "Unexpected clamp")
    root = Path(p["benchmark"])
    dataset = BenchmarkV11Dataset(
        benchmark_root=root, dataset="holo4k",
        tag_file=root / "residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt",
        protein_feature_root=root / "features_fixed_conservative/protein_v4",
        esm_root=root / "esm2", esm3_root=None, ligand_feature_root=root / "ligand_features_v5",
        max_seq_len=1800, max_atoms=-1, tag_mode="per_ligand", ligand_mode="real",
        ligand_norm=norm, spatial_norm=spatial, spatial_clamp_dims=dc["spatial_clamp_dims"],
        spatial_clamp_range=dc["spatial_clamp_range"], atom_truncation_policy="alternating_ends",
        residue_alignment="keys", residue_coverage="protein",
        tag_protein_root=root / "p2rank-datasets-fixed_atom_h", nonfinite_policy="sasa_zero",
    )
    manifest = json.loads((work / "chains.json").read_text())
    require(dataset.keys == sorted(manifest), "Changed inference population")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=args.workers,
                        collate_fn=identity_collate)
    dest = work / "predictions"
    dest.mkdir(exist_ok=False)
    start = time.monotonic()
    seen, total_chains, output_hashes = 0, 0, {}
    torch.cuda.reset_peak_memory_stats()
    for samples in loader:
        tasks, collected = [], [[] for _ in samples]
        for owner, sample in enumerate(samples):
            sid = sample["pdb_id"]
            baseline_path = parent_predictions(parent) / f"{sid}.npz"
            require(sha(baseline_path) == manifest[sid]["sha256"], f"Changed baseline: {sid}")
            check_sample(sample, load_npz(baseline_path))
            parts = split_sample(sample)
            require({c: idx.tolist() for c, idx, _ in parts} == manifest[sid]["chains"], "Chain plan changed")
            tasks.extend((owner, idx, part) for _, idx, part in parts)
        for offset in range(0, len(tasks), 2):
            chunk = tasks[offset:offset + 2]
            batch = benchmark_collate([part for _, _, part in chunk])
            inputs = prepare_model_inputs_v11(batch, device)
            for field in ("esm_valid_mask", "esm3_valid_mask"):
                if field in batch:
                    inputs[field] = batch[field].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(**inputs)
            logits = output["logits"] if isinstance(output, dict) else output
            require(bool(torch.isfinite(logits[inputs["protein_mask"].bool()]).all()), "Nonfinite chain logits")
            probs = logits.float().softmax(-1).cpu().numpy()
            for b, (owner, idx, _) in enumerate(chunk):
                collected[owner].append((idx, probs[b, :len(idx)]))
            total_chains += len(chunk)
            del output, logits, inputs
        for owner, sample in enumerate(samples):
            sid = sample["pdb_id"]
            probs = stitch(sample["n_residues"], collected[owner])
            path = dest / f"{sid}.npz"
            np.savez_compressed(path, probs=probs, labels=sample["labels"].numpy(),
                preds=contact_gate_predictions(torch.from_numpy(probs), p["val_gate"]).numpy(),
                ca_coords=sample["gvp_coords"].numpy(), residue_keys=np.asarray(sample["residue_keys"], dtype=str),
                protein_key=np.asarray(sample["protein_key"]), ligand_key=np.asarray(sample["ligand_key"]))
            output_hashes[sid] = sha(path)
            seen += 1
        if seen % 100 == 0 or seen == len(dataset):
            progress = dict(pairs=seen, total_pairs=len(dataset), chains=total_chains,
                seconds=time.monotonic() - start, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            write_json(work / "progress.json", progress)
            print("CHAIN_PROGRESS", json.dumps(progress), flush=True)
    require(seen == contract["counts"]["pairs"] and total_chains == contract["counts"]["chain_ligand_rows"],
            "Incomplete inference")
    write_json(work / "prediction_audit.json", output_hashes)
    write_json(work / "inference_complete.json", dict(complete=True, version=version,
        protocol_sha256=sha(work / "protocol.json"), audit_sha256=sha(work / "prediction_audit.json"),
        **progress, job_id=os.environ.get("SLURM_JOB_ID")))


def conditional_scores(records, gate):
    """Metrics restricted to GT-positive chains; zero-contact leakage separately."""
    aggregate, proteins = defaultdict(Aggregate), defaultdict(list)
    zeros = {m: Counter() for m in ("full", "chain", "libre")}
    positive_chains, all_chains = 0, 0
    for sid, record in records.items():
        for idx in chain_indices(record["keys"]).values():
            y = record["target"][idx]
            all_chains += 1
            if not y.any():
                for model in zeros:
                    scores = record[model][idx]
                    pred = scores > np.float32(.6) if model == "libre" else scores.astype(np.float64) >= gate
                    zeros[model].update(chains=1, residues=len(y), predicted=int(pred.sum()))
                continue
            positive_chains += 1
            for model in zeros:
                scores = record[model][idx]
                rules = score_row(scores, y, gate)
                native = rules.pop("at_val_gate")
                if model == "libre":
                    pred = scores.astype(np.float32) > np.float32(.6)
                    tp = int(y[pred].sum())
                    native = dict(n=len(y), positive=int(y.sum()), predicted=int(pred.sum()), tp=tp,
                                  cutoff_tie=0, expected_tp=float(tp))
                rules["native_threshold"] = native
                for rule in ("top20", "native_threshold"):
                    c = rules[rule]
                    aggregate[(model, rule)].add(c)
                    proteins[(model, rule, sid.split("_", 1)[0])].append(
                        metrics(c["n"], c["positive"], c["predicted"], c["tp"]))
    result = {}
    for (model, rule), a in aggregate.items():
        entry = a.finish()
        groups = [v for (m, r, _), v in proteins.items() if m == model and r == rule]
        entry["protein_mean_of_chain_ligand_mean"] = {
            k: float(np.mean([np.mean([v[k] for v in group]) for group in groups])) for k in METRICS}
        result.setdefault(model, {})[rule] = entry
    return dict(oracle_binding_chain=True, budget="Top20 within each positive chain, NOT primary global Top20",
        n_all_chain_pairs=all_chains, n_positive_chain_pairs=positive_chains, models=result,
        zero_contact_chains={m: dict(**c, fpr=c["predicted"] / c["residues"] if c["residues"] else None,
            mean_predicted=c["predicted"] / c["chains"] if c["chains"] else None) for m, c in zeros.items()})


def summarize(args):
    work = args.work
    contract = verify(work)
    p, parent = contract["parent_protocol"], Path(contract["parent"])
    completed = json.loads((work / "inference_complete.json").read_text())
    require(completed["complete"] and completed["protocol_sha256"] == sha(work / "protocol.json") and
            completed["audit_sha256"] == sha(work / "prediction_audit.json"), "Incomplete/changed inference")
    manifest = json.loads((work / "chains.json").read_text())
    cohorts = json.loads((work / "cohorts.json").read_text())["cohorts"]
    output_hashes = json.loads((work / "prediction_audit.json").read_text())
    require(set(output_hashes) == set(manifest), "Incomplete output coverage")
    libre = Path(p["libre"])
    lib_rows, plc_rows = csv_rows(libre / "samples.csv"), csv_rows(parent / "inference/holo4k_all_results.csv")
    audit = {r["sample_id"]: r for r in json.loads((parent / "matched_prediction_audit.json").read_text())}
    records, single_control = {}, []
    for sid in sorted(output_hashes):
        path = work / "predictions" / f"{sid}.npz"
        require(sha(path) == output_hashes[sid], f"Changed chain output: {sid}")
        if sid not in lib_rows:
            continue
        pp, lp = parent_predictions(parent) / f"{sid}.npz", libre / "predictions" / f"{sid}.npz"
        require(sha(pp) == audit[sid]["plc_sha256"] and sha(lp) == audit[sid]["libre_sha256"], "Changed comparator")
        pred, _, y, coords, keys, _ = load_pair(lp, pp, lib_rows[sid], plc_rows[sid])
        new, baseline = load_npz(path), load_npz(pp)
        for field in ("labels", "residue_keys", "ca_coords", "protein_key", "ligand_key"):
            require(np.array_equal(new[field], baseline[field]), f"Changed output axis: {sid}/{field}")
        chain = stitch(len(y), [(np.arange(len(y)), new["probs"])])[:, 0]
        records[sid] = dict(full=pred["plc"], chain=chain, libre=pred["libre"], target=y, keys=keys)
        if len(manifest[sid]["chains"]) == 1:
            single_control.append(float(np.abs(chain - pred["plc"]).mean()))
    results = {}
    for name, cohort in cohorts.items():
        ids = cohort["sample_ids"]
        require(bool(ids), "Empty frozen cohort")
        lib = {sid: records[sid]["libre"] for sid in ids}
        models = {}
        for mode in ("full", "chain"):
            rows = [dict(sample_id=sid, source="holo4k", scores=torch.from_numpy(records[sid][mode]),
                         labels=torch.from_numpy(np.where(records[sid]["target"], 0, 2).astype(np.uint8))) for sid in ids]
            result = compare(rows, lib, p["val_gate"])
            add_protein_means(result, rows, lib, p["val_gate"])
            if mode == "full" and name == "all_matched":
                same_numbers(result["models"], json.loads((parent / "matched_summary.json").read_text())["models"])
            models[mode] = result["models"]["plc"]["all"]
            models["libre"] = result["models"]["libre"]["all"]
        conditional = conditional_scores({sid: records[sid] for sid in ids}, p["val_gate"])
        results[name] = dict(n_samples=len(ids), models=models, conditional_chain=conditional)
        print("CHAIN_COHORT", name, json.dumps({m: models[m]["rules"]["top20"]["sample_macro"] for m in models}), flush=True)
    write_json(work / "summary.json", dict(complete=True, policy=POLICY, results=results,
        protocol_sha256=sha(work / "protocol.json"), runtime=completed,
        single_chain_control=dict(n=len(single_control), mean_abs_probability_difference=float(np.mean(single_control))
            if single_control else None, caveat="BF16 batch/padding shapes can cause numerical differences")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "infer", "summarize"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    globals()[args.phase](args)
