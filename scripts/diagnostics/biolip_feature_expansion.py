#!/usr/bin/env python3
"""Complete missing BioLiP features before building a NEW independent joint pool.

Preparation only: no old holdout/external exclusion, no split, no model changes.
The full legacy regular-organic metadata scope is considered, not just existing
feature rows. Raw, atomic outputs live outside the production BioLiP tree.
Final admission still requires native <5 A contacts, >=3 positives, exact axes,
chemical/feature QC, deduplication, SI audit and TRAIN-only normalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

try:
    from .prepare_internal_codataset_sequences import AA3, bio_eligible, digest
except ImportError:
    from prepare_internal_codataset_sequences import AA3, bio_eligible, digest


CODE = Path(__file__).resolve().parents[2]
METADATA = "qc_training_manifest_allclean_20260811/biolip_pairs_metadata.tsv.gz"


def safe_name(value):
    if not value or value in (".", "..") or any(c in value for c in "/\\\x00\r\n"):
        raise ValueError(f"unsafe identifier: {value!r}")
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_tsv(path, rows, fields):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def file_index(path, suffix):
    return {p.name[:-len(suffix)] for p in path.iterdir()
            if p.name.endswith(suffix) and p.is_file()} if path.is_dir() else set()


def paths_for(root, prepared, row):
    p, c, s = row["pdb_id"], row["protein_chain"], row["feature_stem"]
    result = {}
    for kind, rel in {
        "protein": Path("protein_features") / f"{p}_protein.pt",
        "esm": Path("esm2_features") / f"{p}_{c}_esm2.pt",
        "ligand": Path("ligand_features") / p / f"{s}.pt",
    }.items():
        new, old = prepared / rel, root / rel
        result[kind] = new if new.is_file() else old
    return result


def plan(args):
    root, out = args.biolip_root, args.experiment / "plan"
    out.mkdir(exist_ok=True)
    contract = {"metadata_sha256": digest(root / METADATA),
                "qc_pool_sha256": digest(args.qc_pool / "pool_summary.json"),
                "policy": "unchanged bio_eligible; no external or old split exclusions",
                "script_sha256": digest(Path(__file__)), "root": str(root),
                "qc_pool": str(args.qc_pool)}
    if (out / "summary.json").exists():
        saved = json.loads((out / "summary.json").read_text())
        if saved["contract"] != contract:
            raise ValueError("expansion contract changed; use a new experiment")
        for name, sha in saved["output_sha256"].items():
            if digest(out / name) != sha:
                raise ValueError(f"changed plan artifact: {name}")
        print("PLAN_ALREADY_COMPLETE", json.dumps(saved), flush=True)
        return
    qc_summary = json.loads((args.qc_pool / "pool_summary.json").read_text())
    if digest(args.qc_pool / "samples.jsonl") != qc_summary["output_sha256"]["samples.jsonl"]:
        raise ValueError("QC sample ledger hash differs from its summary")
    with (args.qc_pool / "samples.jsonl").open() as f:
        accepted = {r["sample_id"] for r in map(json.loads, f) if r["source"] == "biolip"}
    proteins = file_index(root / "protein_features", "_protein.pt")
    esms = file_index(root / "esm2_features", "_esm2.pt")
    ligands = set()
    for folder in (root / "ligand_features").iterdir():
        if folder.is_dir():
            ligands.update((folder.name, s) for s in file_index(folder, ".pt"))
    counts, seen_obs, seen_ids = Counter(), set(), set()
    candidates, ready, remaining, aliases = [], [], [], []
    chains, missing_pdbs = {}, set()
    with gzip.open(root / METADATA, "rt", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames
        for row in reader:
            counts["raw_metadata"] += 1
            if not bio_eligible(row):
                counts["outside_unchanged_metadata_policy"] += 1
                continue
            counts["metadata_eligible"] += 1
            p, c, s = row["pdb_id"].lower(), row["protein_chain"], row["feature_stem"]
            for name in (p, c, s):
                safe_name(name)
            if s != f"{p}_{c}_{row['binding_site_id']}_{row['ligand_id']}":
                raise ValueError(f"metadata identity differs from feature stem: {s}")
            obs = (p, c, row["ligand_id"], row["ligand_resseq"],
                   tuple(sorted(row["binding_residues"].split())))
            if obs in seen_obs:
                aliases.append(row)
                continue
            if s in seen_ids:
                raise ValueError(f"conflicting sample id: {s}")
            seen_obs.add(obs)
            seen_ids.add(s)
            row["pdb_id"] = p
            candidates.append(row)
            has_p, has_e, has_l = p in proteins, f"{p}_{c}" in esms, (p, s) in ligands
            counts[f"features_P{int(has_p)}_E{int(has_e)}_L{int(has_l)}"] += 1
            if not has_p:
                missing_pdbs.add(p)
            if not has_e or not has_p:
                # A newly extracted V4 axis must not inherit an unchecked old
                # CIF-derived ESM axis, even when a chain-named file exists.
                chains[(p, c)] = {"pdb_id": p, "protein_chain": c,
                                  "v4_at_plan": str(int(has_p))}
            if s in accepted:
                counts["previously_qc_passed"] += 1
            else:
                # Re-extract old invalid/legacy tensors into the new namespace,
                # as well as missing tensors. Never overwrite the original.
                (ready if has_p and has_e else remaining).append(row)
                counts["ligand_reextract_existing" if has_l else "ligand_extract_missing"] += 1
    for name, rows in (("candidates.tsv", candidates), ("ligand_ready.tsv", ready),
                       ("ligand_remaining.tsv", remaining), ("observation_aliases.tsv", aliases)):
        write_tsv(out / name, rows, fields)
    chain_rows = [chains[k] for k in sorted(chains)]
    write_tsv(out / "esm_missing_chains.tsv", chain_rows,
              ["pdb_id", "protein_chain", "v4_at_plan"])
    (out / "protein_missing_ids.txt").write_text("".join(p + "\n" for p in sorted(missing_pdbs)))
    raw = Counter()
    for p in missing_pdbs:
        if (root / "fixed_receptors" / p / "receptor_fixed.cif").is_file():
            raw["fixed_cif"] += 1
        elif (root / "shared_receptors" / p / "receptor_full.cif").is_file():
            raw["original_cif"] += 1
        else:
            raw["no_declared_cif"] += 1
    names = ["candidates.tsv", "ligand_ready.tsv", "ligand_remaining.tsv",
             "observation_aliases.tsv", "esm_missing_chains.tsv", "protein_missing_ids.txt"]
    summary = dict(contract=contract, counts=dict(counts), candidates=len(candidates),
                   ligand_ready=len(ready), ligand_remaining=len(remaining),
                   missing_unique_proteins=len(missing_pdbs), chains_to_extract=len(chains),
                   missing_protein_sources=dict(raw), training_approved=False,
                   output_sha256={n: digest(out / n) for n in names})
    atomic_json(out / "summary.json", summary)
    print("PLAN_COMPLETE", json.dumps(summary), flush=True)


def verified_plan(args):
    out = args.experiment / "plan"
    result = json.loads((out / "summary.json").read_text())
    if result["contract"]["root"] != str(args.biolip_root):
        raise ValueError("BioLiP input root differs from plan")
    for name, sha in result["output_sha256"].items():
        if digest(out / name) != sha:
            raise ValueError(f"changed plan artifact: {name}")
    return result


def extract_cpu(args):
    verified_plan(args)
    exp, root = args.experiment, args.biolip_root
    prep = exp / "prepared"
    prep.mkdir(exist_ok=True)
    if args.stage.startswith("ligand_"):
        manifest = exp / "plan" / f"{args.stage}.tsv"
        rows = read_tsv(manifest)
        if args.limit:
            rows = rows[:args.limit]
            manifest = exp / f"{args.stage}_smoke.tsv"
            with (exp / "plan" / f"{args.stage}.tsv").open() as f:
                original_fields = next(csv.reader(f, delimiter="\t"))
            write_tsv(manifest, rows, original_fields)
        sys.path.insert(0, str(CODE / "preparation/ligand"))
        import biolip_ligand_extract as extractor
        import extract_ligand_features_v5 as v5
        if not (extractor.V5_AVAILABLE and extractor.V6_TYPES_AVAILABLE and
                v5.PYPROPEL_AVAILABLE and v5.V4_AVAILABLE):
            raise RuntimeError("ligand dependency preflight failed; refusing fallback")
        cache = root / "pairs_cache_v6.pkl"
        if not cache.is_file():
            raise FileNotFoundError("No existing pair cache; do not silently rediscover/overwrite")
        output = prep / "ligand_features"
        command = [sys.executable, "-u", str(CODE / "preparation/ligand/biolip_ligand_extract.py"),
                   "--biolip_root", str(root), "--output", str(output),
                   "--pairs-cache", str(cache), "--manifest", str(manifest),
                   "--workers", str(args.workers), "--num_conformers", "50",
                   "--rdkit-threads", "1", "--per-pair-timeout", "600",
                   "--failures-file", str(exp / "logs" / f"{args.stage}_failures.tsv")]
    else:
        ids = (exp / "plan/protein_missing_ids.txt").read_text().split()
        if args.limit:
            ids = ids[:args.limit]
        id_file = exp / ("protein_smoke_ids.txt" if args.limit else "protein_stage_ids.txt")
        id_file.write_text("".join(p + "\n" for p in ids))
        rows = [{"pdb_id": p} for p in ids]
        output = prep / "protein_features"
        command = [sys.executable, "-u", str(CODE / "preparation/protein/biolip_protein_struct_extract.py"),
                   "--biolip_root", str(root), "--fixed_root", str(root / "fixed_receptors"),
                   "--output", str(output), "--pdbid_list_file", str(id_file),
                   "--workers", str(args.workers)]
    tag = args.stage + ("_smoke" if args.limit else "")
    print("EXTRACT_START", tag, len(rows), json.dumps(command), flush=True)
    started = time.monotonic()
    log = exp / "logs" / f"{tag}_extract.log"
    with log.open("a") as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode if rows else 0
    missing = []
    for r in rows:
        path = (output / r["pdb_id"] / f"{r['feature_stem']}.pt" if args.stage.startswith("ligand_")
                else output / f"{r['pdb_id']}_protein.pt")
        if not path.is_file():
            missing.append(r.get("feature_stem", r["pdb_id"]))
    summary = dict(stage=tag, requested=len(rows), files_present=len(rows)-len(missing),
                   missing_ids=missing, extractor_returncode=rc, command=command,
                   elapsed_seconds=time.monotonic()-started, training_approved=False,
                   status="complete" if not missing and rc == 0 else "partial_or_failed")
    atomic_json(exp / f"{tag}_summary.json", summary)
    print("EXTRACT_END", json.dumps({k:v for k,v in summary.items() if k != "missing_ids"}), flush=True)
    if rc not in (0, 1) or (rows and len(missing) == len(rows)):
        raise RuntimeError(f"{tag} did not complete; inspect {log}")


def chain_axis(protein, chain):
    names = protein["residue_names"]
    columns = [protein[k] for k in ("chain_ids", "residue_indices", "insertion_codes")]
    if any(len(c) != len(names) for c in columns):
        raise ValueError("protein axis length mismatch")
    indices = [i for i, c in enumerate(columns[0]) if str(c) == chain]
    if not indices:
        raise ValueError(f"chain missing from V4: {chain}")
    keys = [(str(columns[0][i]), int(columns[1][i]), str(columns[2][i] or "").strip()) for i in indices]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate protein residue keys")
    letters = [AA3.get(str(names[i]).upper(), "X") for i in indices]
    return keys, letters


def atomic_torch(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_embedding(data, keys, letters):
    import torch
    cols = [data[k] for k in ("chain_id", "residue_index", "insertion_code")]
    if any(len(col) != len(keys) for col in cols) or list(zip(*cols)) != keys:
        raise ValueError("cached ESM residue axis differs")
    if data["residue_name_1"] != letters:
        raise ValueError("cached ESM residue identity differs")
    x = data["embeddings"]
    if x.shape != (len(keys), 1280) or not torch.isfinite(x).all():
        raise ValueError("invalid ESM shape/values")


def extract_esm(args):
    import torch
    import esm
    from preparation.protein.biolip_esm2_extract import _run_esm2_windowed
    verified_plan(args)
    torch.set_num_threads(args.workers)
    if not torch.cuda.is_available():
        raise RuntimeError("ESM extraction requires an allocated Slurm GPU")
    rows = read_tsv(args.experiment / "plan/esm_missing_chains.tsv")
    want_ready = args.stage == "esm_ready"
    rows = [r for r in rows if (r["v4_at_plan"] == "1") == want_ready]
    if args.limit:
        rows = rows[:args.limit]
    if not args.esm_checkpoint.is_file():
        raise FileNotFoundError("Offline ESM checkpoint absent; no downloads authorized by this script")
    checkpoint_sha = digest(args.esm_checkpoint)
    model = alphabet = None
    if rows:
        # A missing/incompatible installation must fail once, not be swallowed
        # as thousands of apparent per-chain extraction failures.
        # These are the user-provided, trusted local fair-esm weights. Explicit
        # weights_only=False also supports PyTorch >=2.6 without a global override.
        model_data = torch.load(args.esm_checkpoint, map_location="cpu", weights_only=False)
        regression = args.esm_checkpoint.with_name(args.esm_checkpoint.stem + "-contact-regression.pt")
        regression_data = torch.load(regression, map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            args.esm_checkpoint.stem, model_data, regression_data)
        del model_data, regression_data
        model = model.cuda().eval()
        print("ESM_MODEL_READY", str(args.esm_checkpoint), flush=True)
    prep = args.experiment / "prepared"
    counts, failures = Counter(), []
    tag = args.stage + ("_smoke" if args.limit else "")
    started = time.monotonic()
    previous_path, protein = None, None
    with (args.experiment / f"{tag}_records.jsonl").open("a") as record:
        for number, r in enumerate(rows, 1):
            p, c = r["pdb_id"], r["protein_chain"]
            bundle = f"{p}_{c}"
            path = paths_for(args.biolip_root, prep, dict(r, feature_stem="unused"))["protein"]
            target = prep / "esm2_features" / f"{bundle}_esm2.pt"
            try:
                if path != previous_path:
                    protein = torch.load(path, map_location="cpu", weights_only=False)
                    previous_path = path
                keys, letters = chain_axis(protein, c)
                sequence = "".join(letters)
                sha = hashlib.sha256(sequence.encode()).hexdigest()
                if target.is_file():
                    data = torch.load(target, map_location="cpu", weights_only=False)
                    validate_embedding(data, keys, letters)
                    if data.get("checkpoint_sha256") != checkpoint_sha:
                        raise ValueError("cached ESM checkpoint differs")
                    counts["resumed"] += 1
                else:
                    cache = prep / "esm2_unique" / f"{sha}.pt"
                    if cache.is_file():
                        latent = torch.load(cache, map_location="cpu", weights_only=False)
                        if latent["sequence"] != sequence or latent["checkpoint_sha256"] != checkpoint_sha:
                            raise ValueError("unique ESM provenance differs")
                        counts["sequence_cache_hit"] += 1
                    else:
                        embedding, window = _run_esm2_windowed(model, alphabet, sequence)
                        latent = dict(sequence=sequence, embeddings=embedding.float(), window=window,
                                      checkpoint_sha256=checkpoint_sha)
                        if embedding.shape != (len(sequence), 1280) or not torch.isfinite(embedding).all():
                            raise ValueError("nonfinite/invalid model output")
                        atomic_torch(cache, latent)
                        counts["unique_sequences_computed"] += 1
                    data = dict(embeddings=latent["embeddings"], residue_name_1=letters,
                                chain_id=[k[0] for k in keys], residue_index=[k[1] for k in keys],
                                insertion_code=[k[2] for k in keys], sequence=sequence,
                                sequence_sha256=sha, checkpoint_sha256=checkpoint_sha,
                                layer=33, protein_feature_path=str(path), window=latent["window"],
                                cif_source=protein.get("cif_source"), normalization="pretrained_final_layernorm",
                                extraction_axis="exact V4 observed residues; no truncation")
                    validate_embedding(data, keys, letters)
                    atomic_torch(target, data)
                counts["success"] += 1
                record.write(json.dumps(dict(bundle=bundle, status="ok", sequence_sha256=sha)) + "\n")
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                error = f"{type(exc).__name__}: {exc}"
                failures.append(dict(bundle=bundle, error=error))
                counts["failed"] += 1
                record.write(json.dumps(dict(bundle=bundle, status="failed", error=error)) + "\n")
            record.flush()
            if number % 25 == 0 or number == len(rows):
                progress = dict(stage=tag, processed=number, requested=len(rows), counts=dict(counts),
                                elapsed_seconds=time.monotonic()-started, training_approved=False)
                atomic_json(args.experiment / f"{tag}_progress.json", progress)
                print("ESM_PROGRESS", json.dumps(progress), flush=True)
    summary = dict(stage=tag, requested=len(rows), counts=dict(counts), failures=failures,
                   checkpoint_sha256=checkpoint_sha, elapsed_seconds=time.monotonic()-started,
                   training_approved=False, status="complete" if not failures else "partial_or_failed")
    atomic_json(args.experiment / f"{tag}_summary.json", summary)
    if rows and not counts["success"]:
        raise RuntimeError("No ESM chains extracted; inspect records")


def link_checked(source, target):
    source = source.resolve(strict=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise ValueError(f"refusing to replace existing overlay link: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def collect(args):
    verified_plan(args)
    exp, root = args.experiment, args.biolip_root
    rows = read_tsv(exp / "plan/candidates.tsv")
    prep, view = exp / "prepared", exp / "view"
    view.mkdir(exist_ok=True)
    for name in ("fixed_receptors", "shared_receptors", "qc_training_manifest_allclean_20260811"):
        link_checked(root / name, view / name)
    counts = Counter()
    with (exp / "coverage.jsonl").open("w") as log:
        for r in rows:
            paths = paths_for(root, prep, r)
            present = {k: p.is_file() for k, p in paths.items()}
            counts["all_files_present"] += all(present.values())
            for k, p in paths.items():
                if p.is_file():
                    rel = p.relative_to(prep if p.is_relative_to(prep) else root)
                    link_checked(p, view / rel)
                else:
                    counts[f"missing_{k}"] += 1
            log.write(json.dumps(dict(feature_stem=r["feature_stem"], present=present)) + "\n")
    stages = {}
    for stage in ("ligand_ready", "ligand_remaining", "protein", "esm_ready", "esm_new"):
        path = exp / f"{stage}_summary.json"
        stages[stage] = json.loads(path.read_text()).get("status") if path.is_file() else "unfinished"
    summary = dict(candidates=len(rows), counts=dict(counts), stages=stages, view=str(view),
                   training_approved=False, next="Run fresh geometry/axis QC, deduplication, joint SI and TRAIN stats")
    atomic_json(exp / "coverage_summary.json", summary)
    print("COVERAGE", json.dumps(summary), flush=True)


def geometry_qc(args):
    collect(args)
    command = [sys.executable, "-u", str(CODE / "scripts/diagnostics/prepare_internal_codataset_sequences.py"),
               "--pdbbind-root", str(args.pdbbind_root),
               "--pdbbind-ids", str(CODE / "preparation/dataset_split/mmseqs_splits_strict.txt"),
               "--biolip-root", str(args.experiment / "view"),
               "--output-dir", str(args.experiment / "qc_pool"),
               "--workers", str(args.workers), "--contact-rule", "fixed5"]
    print("GEOMETRY_QC_START", json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("plan", "ligand_ready", "ligand_remaining",
                    "protein", "esm_ready", "esm_new", "collect", "qc"))
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--biolip-root", type=Path, required=True)
    ap.add_argument("--qc-pool", type=Path)
    ap.add_argument("--pdbbind-root", type=Path,
                    default=Path("/hpc2ssd/JH_DATA/spooler/shuang886/PLC/data"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="Smoke only; never marks a full stage complete")
    ap.add_argument("--esm-checkpoint", type=Path,
                    default=Path.home()/".cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")
    args = ap.parse_args()
    args.experiment, args.biolip_root = args.experiment.resolve(), args.biolip_root.resolve()
    if args.experiment == args.biolip_root or args.experiment.is_relative_to(args.biolip_root):
        ap.error("Use an isolated experiment outside production BioLiP")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        ap.error("workers/limit must be positive")
    if args.stage == "plan" and args.qc_pool is None:
        ap.error("plan requires --qc-pool")
    args.experiment.mkdir(parents=True, exist_ok=True)
    (args.experiment / "logs").mkdir(exist_ok=True)
    with (args.experiment / f"{args.stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "plan":
            plan(args)
        elif args.stage.startswith("esm_"):
            extract_esm(args)
        elif args.stage == "collect":
            collect(args)
        elif args.stage == "qc":
            geometry_qc(args)
        else:
            extract_cpu(args)


if __name__ == "__main__":
    main()
