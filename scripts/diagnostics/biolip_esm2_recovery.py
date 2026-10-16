#!/usr/bin/env python3
"""Recover failed ESM2 chains directly from CIF without requiring a V4 file.

Outputs are quarantined sequence features, NOT training-admitted samples.
No chain renaming, duplicate-key repair, residue truncation or zero filling.
The expansion's production/overlay paths are never modified. V4 alignment,
native-contact QC, deduplication and an independent SI audit remain required.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from scripts.diagnostics.biolip_feature_expansion import (
    atomic_json, atomic_torch, chain_axis, digest, read_tsv, safe_name,
    validate_embedding, verified_plan,
)

# Match V4's amino-acid policy, not the broader legacy ESM modified-AA map.
CANONICAL = dict(HID="HIS", HIE="HIS", HIP="HIS", ASH="ASP", GLH="GLU",
                 LYN="LYS", MSE="MET", SEC="CYS")
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
              "ARNDCQEGHILKMFPSTWYV"))


def cif_chain_axis(model, chain_id):
    chains = [c for c in model if c.name.strip() == chain_id]
    if not chains:
        raise ValueError("chain_absent_in_cif")
    if len(chains) != 1:
        raise ValueError("duplicate_chain_identifier")
    keys, letters, coords = [], [], []
    for residue in chains[0]:
        name = CANONICAL.get(residue.name.strip(), residue.name.strip())
        if name not in AA:
            continue
        ca = residue.find_atom("CA", "*")
        if ca is None:
            continue
        key = (chain_id, int(residue.seqid.num), residue.seqid.icode.strip())
        xyz = list(ca.pos)
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError("nonfinite_ca_coordinates")
        keys.append(key)
        letters.append(AA[name])
        coords.append(xyz)
    if not keys:
        raise ValueError("no_v4_compatible_amino_acid_residues")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate_residue_keys")
    return dict(keys=keys, letters=letters, ca_coords=coords,
                sequence="".join(letters))


def compare_v4(axis, protein, chain):
    import torch
    keys, letters = chain_axis(protein, chain)
    if keys != [tuple(k) for k in axis["keys"]] or letters != axis["letters"]:
        raise ValueError("v4_axis_or_identity_mismatch")
    indices = [i for i, c in enumerate(protein["chain_ids"]) if str(c) == chain]
    ca = torch.as_tensor(protein["ca_coords"])[indices].float()
    expected = torch.tensor(axis["ca_coords"], dtype=torch.float32)
    if ca.shape != expected.shape or not torch.isfinite(ca).all() or not torch.allclose(ca, expected, atol=0.002, rtol=0):
        raise ValueError("v4_ca_mismatch")


def audit_protein(task):
    import gemmi
    import torch
    torch.set_num_threads(1)
    pid, requests, root, expansion = task
    root, expansion = Path(root), Path(expansion)
    paths = [("fixed", root / "fixed_receptors" / pid / "receptor_fixed.cif"),
             ("original", root / "shared_receptors" / pid / "receptor_full.cif")]
    protein_path = expansion / "prepared/protein_features" / f"{pid}_protein.pt"
    if not protein_path.is_file():
        protein_path = root / "protein_features" / f"{pid}_protein.pt"
    protein = None
    v4_error = "missing_v4"
    if protein_path.is_file():
        try:
            protein = torch.load(protein_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            v4_error = f"v4_load_error: {type(exc).__name__}: {exc}"
    pending = {r["chain"]: dict(r, attempts=[]) for r in requests}
    resolved = []
    inputs = []
    for source, path in paths:
        if not pending:
            break
        if not path.is_file():
            for row in pending.values():
                row["attempts"].append(dict(source=source, error="cif_not_found"))
            continue
        sha = digest(path)
        inputs.append(dict(path=str(path), sha256=sha))
        try:
            structure = gemmi.read_structure(str(path))
            if not len(structure):
                raise ValueError("cif_has_no_models")
            model = structure[0]
        except Exception as exc:
            for row in pending.values():
                row["attempts"].append(dict(source=source, error=f"{type(exc).__name__}: {exc}"))
            continue
        for chain, row in list(pending.items()):
            try:
                axis = cif_chain_axis(model, chain)
            except Exception as exc:
                row["attempts"].append(dict(source=source, error=f"{type(exc).__name__}: {exc}"))
                continue
            alignment = v4_error
            if protein is not None:
                try:
                    compare_v4(axis, protein, chain)
                    alignment = "exact_axis_identity_and_ca_match"
                except Exception as exc:
                    alignment = f"{type(exc).__name__}: {exc}"
            resolved.append(dict(row, **axis, status="sequence_ready", cif_source=source,
                                 cif_path=str(path), cif_sha256=sha,
                                 sequence_sha256=hashlib.sha256(axis["sequence"].encode()).hexdigest(),
                                 v4_path=str(protein_path), v4_alignment=alignment,
                                 training_approved=False))
            del pending[chain]
        del model, structure
    resolved.extend(dict(row, status="unresolved", training_approved=False)
                    for row in pending.values())
    return resolved, inputs


def plan(args):
    expansion, out = args.expansion, args.output
    verified_plan(argparse.Namespace(experiment=expansion, biolip_root=args.biolip_root))
    summary_path = out / "plan_summary.json"
    inputs = [expansion / "plan/summary.json", expansion / "plan/candidates.tsv",
              expansion / "plan/esm_missing_chains.tsv",
              expansion / "esm_ready_summary.json", expansion / "esm_new_summary.json"]
    contract = dict(expansion=str(expansion), biolip_root=str(args.biolip_root),
                    input_sha256={str(p): digest(p) for p in inputs},
                    script_sha256=digest(Path(__file__)))
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["contract"] != contract or digest(out / "axes.jsonl") != saved["axes_sha256"]:
            raise ValueError("recovery plan changed; use a new output directory")
        print("RECOVERY_PLAN_ALREADY_COMPLETE", json.dumps(saved), flush=True)
        return
    planned = {f"{r['pdb_id']}_{r['protein_chain']}": r
               for r in read_tsv(expansion / "plan/esm_missing_chains.tsv")}
    pair_counts = Counter(f"{r['pdb_id']}_{r['protein_chain']}"
                          for r in read_tsv(expansion / "plan/candidates.tsv"))
    failures = {}
    for stage in ("esm_ready", "esm_new"):
        data = json.loads((expansion / f"{stage}_summary.json").read_text())
        for failure in data["failures"]:
            bundle = failure["bundle"]
            if bundle not in planned or bundle in failures:
                raise ValueError(f"unexpected or duplicate failed bundle: {bundle}")
            failures[bundle] = failure
    groups = defaultdict(list)
    for bundle, failure in sorted(failures.items()):
        r = planned[bundle]
        pid, chain = safe_name(r["pdb_id"]), safe_name(r["protein_chain"])
        groups[pid].append(dict(bundle=safe_name(bundle), pdb_id=pid, chain=chain,
                                previous_error=failure["error"], candidate_pairs=pair_counts[bundle]))
    tasks = [(pid, rows, str(args.biolip_root), str(expansion)) for pid, rows in groups.items()]
    records, cif_inputs = [], []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, (rows, sources) in enumerate(pool.map(audit_protein, tasks), 1):
            records.extend(rows)
            cif_inputs.extend(sources)
            print("CIF_AUDIT_PROGRESS", i, "/", len(tasks), flush=True)
    records.sort(key=lambda r: r["bundle"])
    path = out / "axes.jsonl"
    tmp = out / ".axes.jsonl.tmp"
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    counts = Counter(r["status"] for r in records)
    pair_status = Counter()
    for row in records:
        pair_status[row["status"]] += row["candidate_pairs"]
    summary = dict(contract=contract, requested_chains=len(failures), proteins=len(groups),
                   counts=dict(counts), candidate_pairs_by_status=dict(pair_status),
                   source_counts=dict(Counter(r.get("cif_source", "unresolved") for r in records)),
                   v4_alignment_counts=dict(Counter(r.get("v4_alignment", "unresolved") for r in records)),
                   input_cifs=cif_inputs, axes_sha256=digest(path), training_approved=False)
    atomic_json(summary_path, summary)
    print("RECOVERY_PLAN_COMPLETE", json.dumps({k:v for k,v in summary.items()
                                                if k not in ("contract", "input_cifs")}), flush=True)


def validate_latent(data, row, checkpoint_sha):
    import torch
    if data["sequence"] != row["sequence"] or data["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("ESM sequence/checkpoint provenance mismatch")
    embedding = data["embeddings"]
    if embedding.shape != (len(row["sequence"]), 1280) or not torch.isfinite(embedding).all():
        raise ValueError("invalid ESM embedding shape/values")


def extract(args):
    import esm
    import torch
    from preparation.protein.biolip_esm2_extract import _run_esm2_windowed
    torch.set_num_threads(args.workers)
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated Slurm GPU is required")
    summary = json.loads((args.output / "plan_summary.json").read_text())
    if summary["contract"]["expansion"] != str(args.expansion):
        raise ValueError("expansion differs from recovery plan")
    axes = args.output / "axes.jsonl"
    if digest(axes) != summary["axes_sha256"]:
        raise ValueError("changed recovery axis manifest")
    with axes.open() as f:
        rows = [r for r in map(json.loads, f) if r["status"] == "sequence_ready"]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no unambiguous CIF protein chains to recover")
    checkpoint_sha = digest(args.esm_checkpoint)
    for stage in ("esm_ready", "esm_new"):
        old = json.loads((args.expansion / f"{stage}_summary.json").read_text())
        if old["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("checkpoint differs from original ESM extraction")
    regression = args.esm_checkpoint.with_name(args.esm_checkpoint.stem + "-contact-regression.pt")
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        args.esm_checkpoint.stem,
        torch.load(args.esm_checkpoint, map_location="cpu", weights_only=False),
        torch.load(regression, map_location="cpu", weights_only=False))
    model = model.cuda().eval()
    tag = "smoke" if args.limit else "extract"
    counts, failures = Counter(), []
    started = time.monotonic()
    with (args.output / f"{tag}_records.jsonl").open("a") as record:
        for index, row in enumerate(rows, 1):
            bundle, sha = row["bundle"], row["sequence_sha256"]
            target = args.output / "esm2_features" / f"{safe_name(bundle)}_esm2.pt"
            try:
                if hashlib.sha256(row["sequence"].encode()).hexdigest() != sha:
                    raise ValueError("sequence digest differs from plan")
                keys = [tuple(k) for k in row["keys"]]
                if target.is_file():
                    data = torch.load(target, map_location="cpu", weights_only=False)
                    validate_embedding(data, keys, row["letters"])
                    validate_latent(data, row, checkpoint_sha)
                    if data.get("recovery_axes_sha256") != summary["axes_sha256"]:
                        raise ValueError("output belongs to another recovery plan")
                    counts["resumed"] += 1
                else:
                    existing = args.expansion / "prepared/esm2_unique" / f"{sha}.pt"
                    cache = args.output / "unique" / f"{sha}.pt"
                    source = existing if existing.is_file() else cache
                    if source.is_file():
                        latent = torch.load(source, map_location="cpu", weights_only=False)
                        validate_latent(latent, row, checkpoint_sha)
                        counts["sequence_cache_hit"] += 1
                    else:
                        embedding, window = _run_esm2_windowed(model, alphabet, row["sequence"])
                        latent = dict(sequence=row["sequence"], checkpoint_sha256=checkpoint_sha,
                                      embeddings=embedding.float(), window=window)
                        validate_latent(latent, row, checkpoint_sha)
                        atomic_torch(cache, latent)
                        counts["unique_sequences_computed"] += 1
                    data = dict(latent, sequence_sha256=sha, layer=33,
                                chain_id=[k[0] for k in keys], residue_index=[k[1] for k in keys],
                                insertion_code=[k[2] for k in keys], residue_name_1=row["letters"],
                                ca_coords=torch.tensor(row["ca_coords"], dtype=torch.float32),
                                cif_source=row["cif_source"], cif_path=row["cif_path"],
                                cif_sha256=row["cif_sha256"], v4_alignment=row["v4_alignment"],
                                normalization="pretrained_final_layernorm",
                                extraction_axis="CIF observed V4-compatible residues; no truncation",
                                recovery_axes_sha256=summary["axes_sha256"], training_approved=False)
                    validate_embedding(data, keys, row["letters"])
                    atomic_torch(target, data)
                counts["success"] += 1
                record.write(json.dumps(dict(bundle=bundle, status="ok")) + "\n")
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                error = f"{type(exc).__name__}: {exc}"
                counts["failed"] += 1
                failures.append(dict(bundle=bundle, error=error))
                record.write(json.dumps(failures[-1]) + "\n")
            record.flush()
            if index % 25 == 0 or index == len(rows):
                progress = dict(processed=index, requested=len(rows), counts=dict(counts),
                                elapsed_seconds=time.monotonic()-started, training_approved=False)
                atomic_json(args.output / f"{tag}_progress.json", progress)
                print("RECOVERY_ESM_PROGRESS", json.dumps(progress), flush=True)
    atomic_json(args.output / f"{tag}_summary.json", dict(progress, failures=failures,
                checkpoint_sha256=checkpoint_sha, axes_sha256=summary["axes_sha256"],
                status="complete" if not failures else "partial_or_failed"))
    if not counts["success"] or (args.limit and failures):
        raise RuntimeError("ESM recovery failed; inspect the recorded errors")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("plan", "extract"), required=True)
    ap.add_argument("--expansion", type=Path, required=True)
    ap.add_argument("--biolip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--esm-checkpoint", type=Path, default=Path.home() /
                    ".cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")
    args = ap.parse_args()
    args.expansion, args.biolip_root, args.output = [p.resolve() for p in
            (args.expansion, args.biolip_root, args.output)]
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        ap.error("workers/limit must be positive")
    if (args.output == args.expansion or args.output.is_relative_to(args.biolip_root)
            or any(args.output.is_relative_to(args.expansion / part)
                   for part in ("prepared", "view", "plan", "qc_pool"))):
        ap.error("recovery output must be isolated from production/expansion feature paths")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (plan if args.stage == "plan" else extract)(args)


if __name__ == "__main__":
    main()
