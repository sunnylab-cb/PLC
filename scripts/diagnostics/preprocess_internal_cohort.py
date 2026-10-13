#!/usr/bin/env python3
"""Fit this cohort's TRAIN-only statistics and materialize an isolated dataset.

Requires a complete, passing exhaustive SI audit. Never reads old stats or model
checkpoints. Geometry labels already passed the actual >=3-contact QC gate.
This is preprocessing, not a GPU training smoke-test approval.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from internal_cohort_qc import finite_shape
from prepare_internal_codataset_sequences import digest, save_json
from run_internal_pairwise_split import load_pool


class Moments:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.minimum = np.full(dim, np.inf)
        self.maximum = np.full(dim, -np.inf)
        self.all_integer = np.ones(dim, dtype=bool)

    def update(self, value):
        x = np.asarray(value, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not len(x) or not np.isfinite(x).all():
            raise ValueError("invalid raw features in TRAIN stats")
        n, mean = len(x), x.mean(0)
        delta = mean-self.mean
        total = self.n+n
        self.m2 += ((x-mean)**2).sum(0) + delta**2 * self.n*n/total
        self.mean += delta*n/total
        self.n = total
        self.minimum = np.minimum(self.minimum,x.min(0))
        self.maximum = np.maximum(self.maximum,x.max(0))
        self.all_integer &= (np.abs(x-np.round(x)) < 1e-4).all(0)

    def report(self, ddof):
        if not self.n:
            raise ValueError("empty training statistics")
        std = np.sqrt(np.maximum(self.m2/max(self.n-ddof,1),0))
        return dict(n_rows=self.n,mean=self.mean.tolist(),std=std.tolist(),
            min=self.minimum.tolist(),max=self.maximum.tolist(),all_integer=self.all_integer.tolist())


def norm_dims(stats):
    return np.flatnonzero(~np.array(stats["all_integer"]) & (
        (np.array(stats["max"])-np.array(stats["min"]) > 5) | (np.array(stats["std"]) > 2))).tolist()


def load(path):
    return torch.load(path,map_location="cpu",weights_only=False)


def atomic_save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(value,tmp)
    tmp.replace(path)


def fit_training_stats(samples, assignment):
    ligand, spatial = Moments(109), Moments(89)
    train_keys, seen_proteins, seen_ligands = [], set(), set()
    for key in sorted(samples):
        if assignment[key] != "train":
            continue
        row = samples[key]
        train_keys.append(key)
        lp = row["ligand_path"]
        if lp not in seen_ligands:
            data = load(lp)
            if data.get("feature_normalization"):
                raise ValueError("do not fit new stats on already normalized features")
            ligand.update(data["atom_features"])
            seen_ligands.add(lp)
        pp = row["qc"]["aligned_protein_path"]
        # Same BioLiP input chain can occur under many ligand-row sidecars.
        signature = (row["v4_path"], row["selected_chain"], row["qc"]["axis_sha256"])
        if signature not in seen_proteins:
            spatial.update(load(pp)["spatial_scalar_raw_features"])
            seen_proteins.add(signature)
    return dict(ligand=ligand.report(ddof=1),spatial=spatial.report(ddof=0),
        train_keys=train_keys, unique_ligand_files=len(seen_ligands), unique_protein_inputs=len(seen_proteins),
        fit_scope="this cohort TRAIN only; unique source inputs; no VAL/TEST in fitting",
        esm2="pretrained features unchanged",physchem="fixed per-residue encodings unchanged")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir",type=Path,required=True)
    ap.add_argument("--split-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    report = json.loads((args.split_dir/"summary.json").read_text())
    if report["passed"] is not True or report["cross_set_audit"]["edges"] != 0:
        raise ValueError("SI audit has not passed")
    split_file = args.split_dir/"candidate_split.tsv"
    if digest(split_file) != report["split_sha256"]:
        raise ValueError("audited split changed")
    if digest(args.pool_dir/"pool_summary.json") != report["contract"]["pool_sha256"]:
        raise ValueError("split belongs to a different sequence pool")
    samples, _, pool = load_pool(args.pool_dir)
    assignment = {}
    with split_file.open() as f:
        for row in csv.DictReader(f,delimiter="\t"):
            if row["sample_key"] in assignment or row["split"] not in ("train","val","test"):
                raise ValueError("invalid/duplicate split row")
            assignment[row["sample_key"]] = row["split"]
    if set(assignment) != set(samples):
        raise ValueError("split/pool coverage mismatch")
    if len({row["sample_id"] for row in samples.values()}) != len(samples):
        raise ValueError("sample IDs collide across sources")
    for row in samples.values():
        for path, fingerprint in row["qc"]["sources"].items():
            st = Path(path).stat()
            if st.st_size != fingerprint["size"] or st.st_mtime_ns != fingerprint["mtime_ns"]:
                raise ValueError(f"source changed after QC: {path}")
        if digest(row["qc"]["labels_path"]) != row["qc"]["labels_sha256"]:
            raise ValueError("QC label artifact changed")
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    contract = dict(split_sha256=report["split_sha256"],pool_sha256=digest(args.pool_dir/"pool_summary.json"),
                    script_sha256=digest(Path(__file__)),cohort=pool["cohort"])
    contract_path = out/"preprocessing_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("output root belongs to another cohort/split/preprocessor")
    save_json(contract_path,contract)
    stats = fit_training_stats(samples,assignment)
    stats.update(contract)
    stats["ligand"]["norm_dims"] = norm_dims(stats["ligand"])
    save_json(out/"train_only_stats.json",stats)
    stats_hash = digest(out/"train_only_stats.json")
    lm = torch.tensor(stats["ligand"]["mean"],dtype=torch.float32)
    ls = torch.tensor(stats["ligand"]["std"],dtype=torch.float32).clamp_min(1e-6)
    dims = stats["ligand"]["norm_dims"]
    pm = torch.tensor(stats["spatial"]["mean"],dtype=torch.float32)
    ps = torch.tensor(stats["spatial"]["std"],dtype=torch.float32)
    ps[ps < 1e-8] = 1.0
    distances, output_rows, contact_counts = {}, [], {s:[] for s in ("train","val","test")}
    for index,(key,row) in enumerate(sorted(samples.items()),1):
        sid = row["sample_id"]
        ligand = dict(load(row["ligand_path"]))
        x = ligand["atom_features"].float().clone()
        x[:,dims] = (x[:,dims]-lm[dims])/ls[dims]
        finite_shape(x,(len(x),109),"normalized ligand")
        ligand["atom_features"] = x
        ligand["feature_normalization"] = dict(stats_sha256=stats_hash,fit="TRAIN only",norm_dims=dims)
        atomic_save(ligand,out/"ligand_features_v5/2001-2010"/sid/(sid+"_ligand.pt"))
        protein = dict(load(row["qc"]["aligned_protein_path"]))
        scalar = (torch.as_tensor(protein["spatial_scalar_raw_features"]).float()-pm)/ps
        scalar[:,[62,76,78,79]] = scalar[:,[62,76,78,79]].clamp(-5,5)
        finite_shape(scalar,(len(scalar),89),"normalized spatial")
        protein["spatial_scalar_features"] = scalar
        protein["normalization_provenance"] = dict(stats_sha256=stats_hash,fit="TRAIN only")
        atomic_save(protein,out/"complete_features_v4_nma/2001-2010"/sid/(sid+"_protein_conf0.pt"))
        esm_dst = out/"esm2_latents/2001-2010"/sid/(sid+"_protein.pt")
        esm_dst.parent.mkdir(parents=True,exist_ok=True)
        esm_src = Path(row["esm_path"]).resolve()
        if esm_dst.exists() or esm_dst.is_symlink():
            if not esm_dst.is_symlink() or esm_dst.resolve() != esm_src:
                raise ValueError(f"conflicting ESM target: {esm_dst}")
        else:
            esm_dst.symlink_to(esm_src)
        label = load(row["qc"]["labels_path"])
        n_contact = int((label["distances"] < 5).sum())
        if n_contact < 3 or len(label["distances"]) != len(scalar):
            raise ValueError("label/feature QC changed during preprocessing")
        distances[sid] = label
        output_rows.append((assignment[key],sid))
        contact_counts[assignment[key]].append(n_contact)
        if index % 500 == 0:
            print("PREPROCESS",pool["cohort"],index,"/",len(samples),flush=True)
    atomic_save(distances,out/"residue_ligand_distances_dcenter.pt")
    (out/"splits.txt").write_text("".join(f"{s}\t{k}\n" for s,k in output_rows))
    summary = dict(**contract,preprocessing_passed=True,training_smoke_passed=False,
        dataset_root=str(out.resolve()),stats_sha256=stats_hash,training_split_sha256=digest(out/"splits.txt"),
        dataset_counts=dict(Counter(s for s,_ in output_rows)),by_source=report["by_source"],
        actual_contact_counts={s:dict(min=min(v),p50=float(np.median(v)),p90=float(np.quantile(v,.9)))
                               for s,v in contact_counts.items()},
        next_model="V11m13 group_rescompat + GFGF protein hybrid, seed44, random initialization",
        init_from_checkpoint=None,freeze_v11_base=False,use_compile=False,
        limitations=["No production loader/GPU smoke test yet; do not blindly launch training",
                     "Scaffold split and PLM pretraining overlap are not certified by protein SI"])
    save_json(out/"preprocessing_summary.json",summary)
    print("PREPROCESS_COMPLETE",json.dumps(summary,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
