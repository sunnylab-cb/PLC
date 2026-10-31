#!/usr/bin/env python3
"""CPU-only strict-VAL background audit, NEVER chain-only model inference.

Compare protein-heavy-atom SASA in the cached full input and isolated chains.
Relate geometry to saved FULL-COMPLEX G0 probabilities. No native ligand pose
is used for features; existing labels are only diagnostic annotations.
Input complexes are not automatically certified biological assemblies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.protein.esm2_chainwise_complex import locate, residue_keys
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json


def atom_element(name):
    name = str(name).strip().upper().lstrip("0123456789")
    if not name or name[0] in "HD":
        return None
    if name == "SE":
        return "SE"
    if name[0] not in "CNOSP":
        raise ValueError(f"Unsupported protein atom name: {name}")
    return name[0]


def build_structure(data):
    from Bio.PDB import Structure, Model, Chain, Residue, Atom
    keys = residue_keys(data)
    coords, names = data["all_atom_coords"], data["all_atom_names"]
    if len(coords) != len(keys) or len(names) != len(keys):
        raise ValueError("Atomic/residue axis mismatch")
    structure = Structure.Structure("protein_only")
    model = Model.Model(0)
    structure.add(model)
    residues, xyz, owners, chains = [], [], [], []
    for i, (key, points, atoms) in enumerate(zip(keys, coords, names)):
        chain, number, insertion = key
        if chain not in model:
            model.add(Chain.Chain(chain))
        res = Residue.Residue((" ", number, insertion or " "), str(data["residue_names"][i]), " ")
        model[chain].add(res)
        points = np.asarray(points, dtype=float)
        if points.shape != (len(atoms), 3) or not np.isfinite(points).all():
            raise ValueError("Invalid protein atom coordinates")
        for name, point in zip(atoms, points):
            element = atom_element(name)
            if element is None:
                continue
            atom = Atom.Atom(str(name), point, 0., 1., " ", str(name).ljust(4), len(xyz), element=element)
            res.add(atom)
            xyz.append(point)
            owners.append(i)
            chains.append(chain)
        if not len(res):
            raise ValueError("Residue without usable heavy atoms")
        residues.append(res)
    return structure, residues, np.asarray(xyz), np.asarray(owners), np.asarray(chains), keys


def geometry_background(data, points=100):
    from Bio.PDB.SASA import ShrakeRupley
    from scipy.spatial import cKDTree
    structure, residues, xyz, owners, chains, keys = build_structure(data)
    # Selenium is explicit rather than BioPython's unknown-element fallback.
    sr = ShrakeRupley(probe_radius=1.4, n_points=points, radii_dict={"SE": 1.90})
    sr.compute(structure, level="R")
    full = np.asarray([r.sasa for r in residues], dtype=float)
    for chain in structure[0]:
        sr.compute(chain, level="R")
    isolated = np.asarray([r.sasa for r in residues], dtype=float)
    delta = isolated - full
    if (delta < -1e-6).any():
        raise ValueError("Removing atoms unexpectedly decreased SASA")
    nearest = np.full(len(keys), np.inf)
    for chain in dict.fromkeys(chains):
        query = chains == chain
        if query.all():
            continue
        distance = cKDTree(xyz[~query]).query(xyz[query], k=1, workers=1)[0]
        np.minimum.at(nearest, owners[query], distance)
    return dict(sasa_full=full, sasa_isolated=isolated, delta_sasa=np.maximum(delta, 0),
                nearest_other_chain=nearest, residue_keys=np.asarray(keys, dtype=str))


def bucket_metrics(prob, labels, arrays, gate):
    y = labels == 0
    nearest, delta = arrays["nearest_other_chain"], arrays["delta_sasa"]
    masks = dict(all=np.ones(len(y), dtype=bool), no_other_chain_6A=nearest > 6,
                 other_chain_within_6A=nearest <= 6, delta_sasa_gt10=delta > 10,
                 delta_sasa_le10=delta <= 10)
    out = {}
    for name, mask in masks.items():
        py, yy = prob[mask], y[mask]
        item = dict(n=int(mask.sum()), positives=int(yy.sum()), negatives=int((~yy).sum()),
                    probability_sum=float(py.sum()), delta_sasa_sum=float(delta[mask].sum()))
        for rule, threshold in (("at_0_5", .5), ("at_val_gate", gate)):
            pred = py >= threshold
            item[rule] = dict(tp=int((pred & yy).sum()), fp=int((pred & ~yy).sum()),
                              fn=int((~pred & yy).sum()), tn=int((~pred & ~yy).sum()))
        out[name] = item
    return out


def process_one(task):
    torch.set_num_threads(1)
    row, feature, prediction, output, gate, points = task
    try:
        data = torch.load(feature, map_location="cpu", weights_only=False)
        arrays = geometry_background(data, points)
        with np.load(prediction, allow_pickle=False) as p:
            keys, labels = p["residue_keys"], p["labels"]
            prob = p["full_probs"][:, 0].astype(float)
        n = len(labels)
        if arrays["residue_keys"][:n].tolist() != keys.tolist() or len(prob) != n or not np.isfinite(prob).all():
            raise ValueError("Frozen G0 prediction/geometry axis mismatch")
        full_arrays = arrays
        arrays = {k: v[:n] for k, v in arrays.items()}
        destination = output / (row["sample_id"] + ".npz")
        np.savez_compressed(destination, **arrays, labels=labels, full_context_probability=prob)
        return dict(sample_id=row["sample_id"], pdb_id=row["pdb_id"], source=row["source"], passed=True,
            n_chains=len(set(k[0] for k in residue_keys(data))), full_residues=len(full_arrays["sasa_full"]),
            evaluated_residues=n, features_sha256=sha(feature), prediction_sha256=sha(prediction),
            output_sha256=sha(destination), buckets=bucket_metrics(prob, labels, arrays, gate))
    except Exception as exc:
        return dict(sample_id=row["sample_id"], passed=False, error=repr(exc))


def run(args):
    root = args.experiment
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    cfg = yaml.safe_load(Path(manifest["original_config"]).read_text())
    prediction_root = args.predictions
    protocol = json.loads((prediction_root.parent / "protocol.json").read_text())
    original_cfg_hash = sha(manifest["original_config"])
    if protocol["config_sha256"] != original_cfg_hash or protocol["test_forward"]:
        raise ValueError("Need original G0 development predictions")
    gate = float(protocol["val_gate"])
    hashes = json.loads((prediction_root / "prediction_audit.json").read_text())
    with (root / "prepared/metadata.tsv").open() as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "val"]
    multi = [r for r in rows if r["source"] == "pdbbind" and int(r["n_chains"]) > 1]
    single = [r for r in rows if int(r["n_chains"]) == 1]
    selected = list(multi)
    for source in ("pdbbind", "biolip"):
        controls = [r for r in single if r["source"] == source]
        controls.sort(key=lambda r: hashlib.sha256(("background_audit_v1/"+r["sample_id"]).encode()).hexdigest())
        selected.extend(controls[:args.controls_per_source])
    output = root / "background_audit"
    output.mkdir(exist_ok=False)
    (output / "residues").mkdir()
    feature_dir = Path(cfg["data"]["data_root"]) / cfg["data"]["nma_features_dir"]
    tasks = []
    for row in selected:
        sid = row["sample_id"]
        prediction = prediction_root / f"{sid}.npz"
        if sid not in hashes or sha(prediction) != hashes[sid]:
            raise ValueError(f"Missing/changed frozen prediction: {sid}")
        tasks.append((row, locate(feature_dir, sid, "_protein_conf0.pt"), prediction,
                      output / "residues", gate, args.sasa_points))
    write_json(output / "protocol.json", dict(selected_ids=[r["sample_id"] for r in selected],
        selection="ALL strict-VAL multichain PDBbind + hash-selected single-chain controls",
        frozen_prediction_protocol_sha256=sha(prediction_root.parent / "protocol.json"),
        input_manifest_sha256=sha(root / "prepared/manifest.json"),
        val_gate=gate, n_sasa_points=args.sasa_points, sasa_probe_radius=1.4, heavy_atoms_only=True,
        native_ligand_as_feature=False, train_input_modified=False, no_model_inference=True,
        assembly_status="cached input context, NOT a verified biological assembly",
        bins="6A other-chain heavy-atom neighborhood and deltaSASA10A^2 are descriptive, not model gates"))
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, r in enumerate(executor.map(process_one, tasks, chunksize=1), 1):
            records.append(r)
            if i % 25 == 0 or i == len(tasks):
                print("BACKGROUND_AUDIT", i, len(tasks), sum(not x["passed"] for x in records), flush=True)
                write_json(output / "progress.json", dict(done=i, expected=len(tasks), failures=[x for x in records if not x["passed"]]))
    pooled = defaultdict(Counter)
    for record in records:
        if not record["passed"]:
            continue
        for bucket, values in record["buckets"].items():
            for stratum in ("all", record["source"], "multichain" if record["n_chains"] > 1 else "single_chain"):
                target = pooled[stratum + "/" + bucket]
                target.update({k: v for k, v in values.items() if not isinstance(v, dict)})
                for rule in ("at_0_5", "at_val_gate"):
                    target.update({rule+"/"+k: v for k, v in values[rule].items()})
    result = {}
    for name, values in pooled.items():
        values = dict(values)
        for rule in ("at_0_5", "at_val_gate"):
            values[rule+"/fpr"] = values[rule+"/fp"] / values["negatives"] if values["negatives"] else None
            values[rule+"/fnr"] = values[rule+"/fn"] / values["positives"] if values["positives"] else None
        result[name] = values
    write_json(output / "records.json", records)
    failed = [r for r in records if not r["passed"]]
    write_json(output / "summary.json", dict(complete=not failed, expected=len(tasks), passed=len(records)-len(failed),
        failures=failed, residue_pooled=result, no_training_or_chain_only_inference=True,
        limitations=["Associations are descriptive, not causal proof of context benefit",
            "Original input is not certified biological assembly; no missing symmetry mates added",
            "Unobserved atoms/residues are not reconstructed; same cached protein heavy atoms in both SASA modes",
            "Ligand copies weight pooled statistics; per-query data retained for protein-balanced analysis"]))
    if failed:
        raise RuntimeError(f"{len(failed)} audited queries failed; retained in summary, not silently excluded")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--controls-per-source", type=int, default=50)
    p.add_argument("--sasa-points", type=int, default=100)
    torch.set_num_threads(1)
    run(p.parse_args())
