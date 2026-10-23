#!/usr/bin/env python3
"""Frozen joint epoch25 HOLO4K inference and matched released-LiBRe comparison.

Neither COACH420 nor HOLO4K was excluded from joint training. This is an
overlap-uncertified reference evaluation, not an external SI30 certificate.
Use existing VAL gate and prespecified budgets; never tune on benchmark labels.
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark.diagnose_libre_ranking import csv_rows, load_pair
from scripts.benchmark.run_libre_joint_test import compare
from scripts.diagnostics.recompute_joint_topk_reference import REFERENCE_SHA, METRICS, metrics, score_row, sha, write_json
from scripts.infer.benchmark_integrity import safe_tensor_load

POLICY = dict(reference_only=True, test_driven_iteration_allowed=False,
              external_homology_excluded=False, coach420_isolated=False,
              holo4k_isolated=False, overlap_filter=False,
              checkpoint_selection="same VAL-best epoch25 as internal TEST",
              precision="bf16", batch_size=2, max_atoms=-1, max_seq_len=1800,
              normalization="latest joint TRAIN-only ligand and raw spatial stats",
              contact="native heavy-atom d_min <5 A", primary="top20",
              fixed_k=[10, 20, 40, 80], length_divisors=[20, 10, 5],
              threshold="existing PLC VAL gate; released LiBRe p>0.6",
              nonfinite_policy="previously audited SASA-only input repair; fail otherwise")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def code_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ("src", "scripts")
            for p in sorted((ROOT / folder).rglob("*.py"))}


def overlap_by_pdb(split_path, keys_by_dataset):
    groups = defaultdict(set)
    with Path(split_path).open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            groups[row["split"]].add(row["pdb_id"].lower())
    out = {}
    for dataset, keys in keys_by_dataset.items():
        proteins = {key.split("_", 1)[0] for key in keys}
        out[dataset] = {}
        for split, ids in groups.items():
            matched = sorted(p for p in proteins if p[:4].lower() in ids)
            matched_set = set(matched)
            out[dataset][split] = dict(shared_protein_keys=len(matched),
                shared_pair_rows=sum(key.split("_", 1)[0] in matched_set for key in keys),
                total_protein_keys=len(proteins), total_pair_rows=len(keys),
                examples=matched[:20])
    return dict(method="exact PDB accession only, NOT SI/sequence/pair-equivalence audit",
                external_isolation_guaranteed=False, counts=out)


def command(args, p, output, preflight=False):
    cmd = [sys.executable, "-u", str(ROOT / "scripts/infer/infer_v11_benchmark.py"),
           "--checkpoint", p["checkpoint"], "--config", p["config"],
           "--train-cohort-stats", p["stats"], "--benchmark-root", p["benchmark"],
           "--datasets", "holo4k", "--tag-mode", "per_ligand", "--ligand-mode", "real",
           "--max-atoms", "-1", "--max-seq-len", "1800", "--atom-truncation-policy", "alternating_ends",
           "--residue-alignment", "keys", "--residue-coverage", "protein",
           "--tag-protein-root", "p2rank-datasets-fixed_atom_h", "--nonfinite-policy", "sasa_zero",
           "--precision", "bf16", "--batch-size", "2", "--num-workers", str(args.workers),
           "--output", str(output)]
    return cmd + (["--cpu", "--preflight-only"] if preflight else ["--use-checkpoint-gate", "--save-predictions"])


def prepare(args):
    protocol_path = args.work / "protocol.json"
    require(not protocol_path.exists(), "Protocol already exists; inspect before retry")
    ref_path = args.reference / "protocol.json"
    require(sha(ref_path) == REFERENCE_SHA, "Wrong frozen internal reference")
    ref = json.loads(ref_path.read_text())
    ckpt_path = Path(ref["checkpoint"])
    require(sha(ckpt_path) == ref["checkpoint_sha256"], "Changed selected checkpoint")
    cfg_path = args.training / "prepared/GFGF.yaml"
    stats_path = args.training / "dataset/train_only_stats.json"
    split_path = args.training / "split/candidate_split.tsv"
    for path in (stats_path, split_path):
        require(sha(path) == ref["artifact_sha256"][str(path)], f"Changed training asset: {path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    require(not cfg["model"].get("use_compile") and not cfg["model"].get("use_esm3"), "Unexpected model")
    ckpt = safe_tensor_load(ckpt_path)
    require(ckpt["epoch"] == 24 and ckpt["config"] == cfg and
            float(ckpt["val_contact_gate"]) == ref["val_gate"], "Changed epoch/config/gate")
    del ckpt
    current = code_hashes()
    for path, digest in ref["evaluation_source_sha256"].items():
        if path.startswith(("src/models/", "src/data/")):
            require(current.get(path) == digest, f"Model/data code differs from training: {path}")
    keys = {ds: sorted(safe_tensor_load(args.benchmark / "residue_ligand_tags/atom_h" /
                    f"{ds}_per_ligand_dmin.pt")) for ds in ("coach420", "holo4k")}
    libre = csv_rows(args.libre / "samples.csv")
    failed = json.loads((args.libre / "failures.json").read_text())
    require(len(keys["holo4k"]) == 7245 and len(libre) == 7238 and len(failed) == 7,
            "Changed external benchmark population")
    require(set(libre) <= set(keys["holo4k"]), "LiBRe keys outside HOLO4K")
    failed_keys = {r["sample_key"] for r in failed}
    require(not (failed_keys & set(libre)) and failed_keys | set(libre) == set(keys["holo4k"]),
            "LiBRe failures/successes do not partition the benchmark")
    paths = [ref_path, cfg_path, stats_path, split_path, ckpt_path,
             args.libre / "samples.csv", args.libre / "protocol.json", args.libre / "summary.json",
             args.libre / "failures.json"]
    paths += [args.benchmark / "residue_ligand_tags/atom_h" / f"{ds}_per_ligand_dmin.pt" for ds in keys]
    p = dict(policy=POLICY, checkpoint=str(ckpt_path), config=str(cfg_path), stats=str(stats_path),
             benchmark=str(args.benchmark), libre=str(args.libre), epoch=25, val_gate=ref["val_gate"],
             expected_all=7245, expected_matched=7238, code_sha256=current,
             asset_sha256={str(path): sha(path) for path in paths},
             overlap_audit=overlap_by_pdb(split_path, keys))
    write_json(protocol_path, p)
    write_json(args.work / "external_overlap.json", p["overlap_audit"])
    subprocess.run(command(args, p, args.work / "preflight", preflight=True), check=True)
    qc = json.loads((args.work / "preflight/benchmark_summary_all.json").read_text())
    require(qc["summaries"]["holo4k"]["n_failed"] == 0, "Benchmark input QC failed")
    write_json(args.work / "prepared.json", dict(complete=True, protocol_sha256=sha(protocol_path),
               job_id=os.environ.get("SLURM_JOB_ID"), preflight_sha256=sha(args.work / "preflight/benchmark_summary_all.json")))


def matched_summary(output, libre, gate):
    plc_rows, lib_rows = csv_rows(output / "holo4k_all_results.csv"), csv_rows(libre / "samples.csv")
    require(len(plc_rows) == 7245 and len(lib_rows) == 7238 and set(lib_rows) <= set(plc_rows), "Wrong completed cohort")
    rows, lib_scores, protein_aggregates, protein_best = [], {}, defaultdict(list), {}
    audit = []
    for sid, lr in sorted(lib_rows.items()):
        pp = output / "predictions/holo4k" / (sid + ".npz")
        lp = libre / "predictions" / (sid + ".npz")
        p, native, y, _, _, _ = load_pair(lp, pp, lr, plc_rows[sid])
        rows.append(dict(sample_id=sid, source="holo4k", scores=torch.from_numpy(p["plc"]),
                         labels=torch.from_numpy(np.where(y, 0, 2).astype(np.uint8))))
        lib_scores[sid] = p["libre"]
        protein = plc_rows[sid]["protein_key"]
        for model in ("plc", "libre"):
            rules = score_row(p[model], y, gate)
            rules.pop("at_val_gate")
            selected = native[model]
            rules["native_threshold"] = dict(n=len(y), positive=int(y.sum()), predicted=int(selected.sum()),
                                               tp=int(y[selected].sum()))
            for rule, c in rules.items():
                m = metrics(c["n"], c["positive"], c["predicted"], c["tp"])
                protein_aggregates[(model, rule, protein)].append(m)
                key = (model, rule, protein)
                protein_best[key] = max(protein_best.get(key, 0.), m["f1"])
        audit.append(dict(sample_id=sid, plc_sha256=sha(pp), libre_sha256=sha(lp)))
    result = compare(rows, lib_scores, gate)
    for model in ("plc", "libre"):
        for rule, record in result["models"][model]["all"]["rules"].items():
            grouped = [v for (m, r, _), v in protein_aggregates.items() if m == model and r == rule]
            record["protein_mean_of_ligand_mean"] = {k: float(np.mean([np.mean([x[k] for x in v]) for v in grouped])) for k in METRICS}
            record["protein_mean_of_ligand_max_f1"] = float(np.mean([v for (m, r, _), v in protein_best.items() if m == model and r == rule]))
            record["n_proteins"] = len(grouped)
    result.update(complete=True, plc_all_rows=7245, matched_rows=7238, unmatched_libre_rows=7,
                  external_isolation_guaranteed=False, reference_only=True, test_driven_iteration_allowed=False)
    write_json(output.parent / "matched_prediction_audit.json", audit)
    return result


def infer(args):
    protocol_path = args.work / "protocol.json"
    prepared = json.loads((args.work / "prepared.json").read_text())
    require(prepared["complete"] and sha(protocol_path) == prepared["protocol_sha256"], "Preparation changed/incomplete")
    p = json.loads(protocol_path.read_text())
    require(p["policy"] == POLICY and p["code_sha256"] == code_hashes(), "Changed evaluation code/policy")
    for path, digest in p["asset_sha256"].items():
        require(sha(path) == digest, f"Changed asset: {path}")
    require(not (args.work / "inference_started.json").exists(), "Inference already started; inspect before restart")
    write_json(args.work / "inference_started.json", dict(job_id=os.environ.get("SLURM_JOB_ID")))
    subprocess.run(command(args, p, args.work / "inference"), check=True)
    meta = json.loads((args.work / "inference/benchmark_summary_all.json").read_text())
    require(meta["contact_threshold"] == p["val_gate"] and meta["max_atoms"] == -1 and
            meta["spatial_normalization"] == "raw_to_train_only_zscore" and
            meta["checkpoint_sha256"] == p["asset_sha256"][p["checkpoint"]] and
            meta["feature_stats_sha256"] == p["asset_sha256"][p["stats"]], "Changed inference recipe")
    qc = meta["summaries"]["holo4k"]["input_integrity"]
    require(qc["n_evaluated"] == 7245 and qc["n_skipped"] == 0 and qc["all_evaluated_logits_finite"], "Incomplete predictions")
    result = matched_summary(args.work / "inference", Path(p["libre"]), p["val_gate"])
    result.update(policy=POLICY, protocol_sha256=sha(protocol_path), job_id=os.environ.get("SLURM_JOB_ID"),
                  overlap_audit=p["overlap_audit"])
    write_json(args.work / "matched_summary.json", result)
    print("MATCHED_HOLO_COMPLETE", json.dumps({m: result["models"][m]["all"]["rules"]["top20"]["sample_macro"] for m in ("plc", "libre")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("prepare", "infer"))
    for name in ("work", "training", "reference", "benchmark", "libre"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(2)
    (prepare if args.phase == "prepare" else infer)(args)
