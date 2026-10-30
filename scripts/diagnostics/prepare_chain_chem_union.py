#!/usr/bin/env python3
"""CPU-only label migration/QC. Preserve strict membership and TRAIN statistics.

Retain all chain-query rows in the audit, including zero-contact chains;
only >=3-contact rows enter supervised splits. No TEST/HOLO model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import inspect
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.contact_labels import decode_contact_labels, fixed5_protocol
from src.data.chain_chem_union import (
    ChainChemUnionDataset, TASK_ID, UnionBuilder, chemical_identity, geometry_digest, normalized_keys,
)
from scripts.diagnostics.prepare_libre_transfer import source_hashes
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json
from scripts.diagnostics.strict_assignment_common import read_strict_splits

VARIANTS = ("union_base", "union_wd010", "union_drop020")


def inspect_parent(row):
    torch.set_num_threads(1)
    lig = torch.load(row["ligand_path"], map_location="cpu", weights_only=False)
    try:
        chem = chemical_identity(lig.get("ligand_path", "__missing__"), lig.get("num_heavy_atoms"))
    except ValueError as exc:
        chem = dict(status="unresolved_singleton", reason=str(exc))
    protein = torch.load(row["qc"]["aligned_protein_path"], map_location="cpu", weights_only=False)
    keys = normalized_keys(zip(protein.get("chain_ids", protein.get("chain_id")),
        protein.get("residue_indices", protein.get("residue_index")),
        protein.get("insertion_codes", protein.get("insertion_code"))))
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[0]].append(i)
    chains = []
    for chain, idx in sorted(groups.items()):
        part_keys = [keys[i] for i in idx]
        chains.append(dict(chain=chain, indices=idx, residue_keys=part_keys,
                           geometry_sha256=geometry_digest(part_keys, protein["ca_coords"][idx])))
    return dict(sample_id=row["sample_id"], chemistry=chem, keys=keys, chains=chains)


def make_configs(base, root):
    result = {}
    if base["model"]["scan_variant"] != "G0_small_global" or base["experiment"]["seed"] != 44:
        raise ValueError("Expected scratch width-256 G0 seed44 reference")
    if base["model"]["dropout"] != .15 or base["training"]["weight_decay"] != .05:
        raise ValueError("Unexpected regularization baseline")
    for name in VARIANTS:
        c = copy.deepcopy(base)
        c["experiment"].update(name=f"{TASK_ID}_{name}_seed44", version_suffix="chainunion",
                                output_dir=str(root / "outputs" / name))
        c["chain_union_manifest"] = str(root / "prepared/union.pt")
        c["scan_metadata_file"] = str(root / "prepared/metadata.tsv")
        c["data"].update(split_file=str(root / "prepared/splits.txt"), max_seq_len=1800,
                          nma_n_confs=1, nma_max_conf=0, distances_file_original=None)
        c["training"].update(init_from_checkpoint=None, init_from_v11m3=None, freeze_v11_base=False,
            freeze_v11m3_base=False, early_stopping_metric="f1_c0", early_stopping_patience=30,
            save_every=0, save_latest_checkpoint=False, batch_size=16, accumulation_steps=1)
        c["model"].update(use_gradient_checkpointing=True, use_compile=False)
        if name == "union_wd010":
            c["training"]["weight_decay"] = .10
        if name == "union_drop020":
            c["model"]["dropout"] = .20
        result[name] = c
    return result


def verify(root, variant=None, smoke=False):
    p = json.loads((root / "prepared/provenance.json").read_text())
    if not p["complete"] or p["task_id"] != TASK_ID:
        raise ValueError("Incomplete or wrong task preparation")
    for path, expected in p["assets"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if source_hashes(root / "code_v1") != p["source_hashes"]:
        raise ValueError("Immutable source snapshot changed")
    for name, expected in p["configs"].items():
        if sha(root / "prepared" / f"{name}.yaml") != expected:
            raise ValueError(f"Configuration changed: {name}")
    if smoke:
        report = json.loads((root / "smoke" / f"{variant}.json").read_text())
        if not report["passed"] or report["provenance_sha256"] != sha(root / "prepared/provenance.json"):
            raise ValueError("Missing matching GPU smoke")
    return p


def prepare(args):
    root, upstream = args.experiment.resolve(), args.upstream.resolve()
    output = root / "prepared"
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((upstream / "prepared/provenance.json").read_text())
    assets = {k: v for k, v in prior["asset_hashes"].items()
              if k.startswith(str(upstream / "dataset")) or k.startswith(str(upstream / "split"))}
    assets[str(args.base_config)] = sha(args.base_config)
    assets[str(args.pool)] = sha(args.pool)
    for path, expected in assets.items():
        if sha(path) != expected:
            raise ValueError(f"Upstream audited input changed: {path}")
    si = json.loads((upstream / "split/summary.json").read_text())
    if not si["passed"] or si["cross_set_audit"]["edges"] != 0 or si["cross_set_audit"]["max_global_si"] >= .3:
        raise ValueError("Missing strict global-SI30 certificate")
    cfg = yaml.safe_load(args.base_config.read_text())
    if cfg["data"]["label_contact_thresh"] != 5 or cfg["data"]["label_near_thresh"] != 8:
        raise ValueError("This migration only supports audited <5/<8A targets")
    splits = read_strict_splits(Path(cfg["data"]["split_file"]))
    split_sets = {s: set(ids) for s, ids in splits.items()}
    with (upstream / "split/candidate_split.tsv").open() as f:
        metadata = {r["sample_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    if set(metadata) != set().union(*split_sets.values()):
        raise ValueError("Parent split/metadata coverage mismatch")
    pool = {}
    for line in args.pool.open():
        r = json.loads(line)
        if r["sample_id"] in metadata:
            m = metadata[r["sample_id"]]
            if m["source"] != r["source"] or m["pdb_id"].lower() != r["pdb_id"].lower():
                raise ValueError("Pool and materialized parent metadata mismatch")
            if r["sample_id"] in pool:
                raise ValueError("Duplicate parent sample")
            pool[r["sample_id"]] = r
    if set(pool) != set(metadata):
        raise ValueError("Missing parent QC provenance")
    # The upstream exhaustive SI audit operates on these original chain sequences.
    sequence_splits = {}
    for sid, row in pool.items():
        split = metadata[sid]["split"]
        if sid not in split_sets[split]:
            raise ValueError("Parent moved to another split")
        for seq in row["qc"]["chains"].values():
            if sequence_splits.setdefault(seq, split) != split:
                raise ValueError("Exact original chain sequence crosses splits")
    labels = torch.load(cfg["data"]["distances_file"], map_location="cpu", weights_only=False)
    builder = UnionBuilder()
    rows = [pool[sid] for sid in sorted(pool)]
    chemistry = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(inspect_parent, rows, chunksize=16), 1):
            sid = result["sample_id"]
            label = labels[sid]
            if label["contact_label_protocol"] != fixed5_protocol():
                raise ValueError("Wrong upstream target protocol")
            _, d = decode_contact_labels(label)
            if normalized_keys(label["residue_keys"]) != result["keys"]:
                raise ValueError(f"Label/geometry axis mismatch: {sid}")
            chemistry[result["chemistry"]["status"]] += 1
            for chain in result["chains"]:
                builder.add(dict(**metadata[sid], **chain, parent_n=len(d),
                    chemistry=result["chemistry"], distances=d[chain["indices"]]))
            if i % 1000 == 0:
                print("UNION_PREP_PROGRESS", i, len(rows), dict(chemistry), flush=True)
    manifest = builder.finish()
    manifest.update(parent_split_sha256=sha(cfg["data"]["split_file"]),
                    parent_label_sha256=sha(cfg["data"]["distances_file"]))
    torch.save(manifest, output / "union.pt")
    counts, source_counts, observations = Counter(), Counter(), Counter()
    with (output / "splits.txt").open("w") as sf, (output / "metadata.tsv").open("w") as mf:
        fields = ["split", "sample_id", "source", "pdb_id", "chain", "component_id", "representative", "n_members", "n_contact"]
        writer = csv.DictWriter(mf, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in manifest["rows"]:
            observations["all_chain_queries"] += 1
            observations["merged_queries"] += int(r["n_members"] > 1)
            observations["zero_contact_queries"] += int(r["n_contact"] == 0)
            observations["one_two_contact_queries"] += int(0 < r["n_contact"] < 3)
            observations["cropped_chains"] += int(len(r["distances"]) > manifest["cap"])
            if r["eligible"]:
                sf.write(f"{r['split']}\t{r['sample_id']}\n")
                writer.writerow(r)
                counts[r["split"]] += 1
                source_counts[r["split"] + "/" + r["source"]] += 1
    configs = make_configs(cfg, root)
    for name, c in configs.items():
        (output / f"{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=True))
    # TEST is relabelled/partition-audited but never forwarded or used for choices.
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    allowed = inspect.signature(PocketDatasetV8).parameters
    loader_qc = {}
    for split in ("train", "val"):
        eligible = [r for r in manifest["rows"] if r["split"] == split and r["eligible"]]
        chosen = []
        for source in ("pdbbind", "biolip"):
            source_rows = [r for r in eligible if r["source"] == source]
            chosen.extend(sorted(source_rows, key=lambda r: (-r["n_members"], -r["n_residues"], r["sample_id"]))[:8])
        kwargs = {k: v for k, v in configs[VARIANTS[0]]["data"].items() if k in allowed}
        ds = ChainChemUnionDataset(output / "union.pt", [r["sample_id"] for r in chosen], split, **kwargs)
        for i in range(len(ds)):
            sample = ds[i]
            for k, v in sample.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
                    raise ValueError(f"Nonfinite normalized union input: {sample['pdb_id']}/{k}")
            if int((sample["labels"] == 0).sum()) < 3:
                raise ValueError("Eligible query lost contacts")
        loader_qc[split] = [r["sample_id"] for r in chosen]
    assets.update({str(output / name): sha(output / name) for name in ("union.pt", "splits.txt", "metadata.tsv")})
    report = dict(complete=True, task_id=TASK_ID, assets=assets, source_hashes=source_hashes(root / "code_v1"),
        configs={name: sha(output / f"{name}.yaml") for name in VARIANTS}, counts=dict(counts),
        source_counts=dict(source_counts), audit=dict(observations), chemistry=dict(chemistry),
        parent_counts={s: len(ids) for s, ids in splits.items()}, loader_qc=loader_qc,
        native_chain_si_inherited=True, exact_chain_cross_split_overlap=0,
        si_max=si["cross_set_audit"]["max_global_si"], test_forward_used=False,
        normalization="unchanged upstream TRAIN-only stats; no second normalization",
        limitations=["Union covers observed/QC-approved ligand copies only, not missing annotations",
            "Cross-source or nonidentical receptor coordinate frames are not merged",
            "ESM/V4 context remains cached full-receptor context, not re-extracted isolated chains",
            "Unknown SDF chemistry retained as singleton; report unresolved coverage",
            "New chain target/cap/weighting is not numerically comparable with old per-copy task"])
    write_json(output / "provenance.json", report)
    print("UNION_PREPARED", json.dumps({k: report[k] for k in ("counts", "source_counts", "audit", "chemistry")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--base-config", type=Path)
    p.add_argument("--pool", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--require-smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.verify:
        verify(args.experiment, args.variant, args.require_smoke)
        print("CHAIN_UNION_VERIFIED", flush=True)
    else:
        if not all((args.upstream, args.base_config, args.pool)):
            p.error("Preparation requires --upstream, --base-config and --pool")
        prepare(args)


if __name__ == "__main__":
    main()
