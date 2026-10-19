#!/usr/bin/env python3
"""Freeze the exact GFGF seed44 recipe on an independently audited cohort.

Changes are dataset/split/label and output paths, plus explicit F1-based early
stopping for new runs. No checkpoint loading, old normalization, external
benchmark filtering, or TEST-based model selection.
"""

import argparse
import copy
import csv
import inspect
import json
from pathlib import Path

import torch
import yaml

from prepare_libre_transfer import BASE_SHA, sha256, source_hashes
from prepare_internal_codataset_sequences import save_json
from strict_assignment_common import read_strict_splits
from src.contact_labels import decode_contact_labels, fixed5_protocol


COHORTS = ("pdbbind_only", "pdbbind_biolip_joint")


def make_config(base, experiment, dataset, cohort="pdbbind_only"):
    if cohort not in COHORTS:
        raise ValueError("Unknown cohort")
    cfg = copy.deepcopy(base)
    m, t = cfg["model"], cfg["training"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "resume",
                              "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Expected scratch, fully unfrozen baseline")
    if (m["protein_hybrid_pattern"] != "GFGF" or m["use_compile"]
            or not m["use_local54_residue_chem_compat"] or not m["use_v12_coattention_block"]
            or cfg["experiment"]["seed"] != 44):
        raise ValueError("Expected exact seed44 group_rescompat + GFGF recipe")
    d = cfg["data"]
    if d.get("distances_file_original"):
        raise ValueError("Do not retain legacy label sidecars")
    d.update(data_root=str(dataset), split_file=str(dataset/"splits.txt"),
             distances_file=str(dataset/"residue_ligand_distances_dcenter.pt"))
    prefix = "pdbbind" if cohort == "pdbbind_only" else "pdbbind_biolip_joint"
    cfg["experiment"].update(name=prefix+"_global_si30_GFGF_scratch_seed44",
                             output_dir=str(experiment/"outputs"))
    t["early_stopping_metric"] = "f1_c0"
    return cfg


def split_members(path, groups, cohort):
    """Bind source-prefixed QC keys to the IDs consumed by the training loader."""
    if cohort not in COHORTS:
        raise ValueError("Unknown cohort")
    allowed = {"pdbbind"} if cohort == "pdbbind_only" else {"pdbbind", "biolip"}
    members = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid, source, split = row["sample_id"], row["source"], row["split"]
            if (sid in members or source not in allowed or split not in groups
                    or row["sample_key"] != source+":"+sid):
                raise ValueError("Invalid/colliding source-aware split member")
            members[sid] = (row["sample_key"], source, split)
    expected = [(sid, split) for split, ids in groups.items() for sid in ids]
    if (len(expected) != len(members) or len({sid for sid, _ in expected}) != len(expected)
            or any(sid not in members or members[sid][2] != split for sid, split in expected)
            or {item[1] for item in members.values()} != allowed):
        raise ValueError("Audited cohort and training membership differ")
    return members


def source_smoke_ids(groups, members, split, count):
    buckets = [[sid for sid in groups[split] if members[sid][1] == source]
               for source in sorted({v[1] for v in members.values()})]
    selected = []
    for index in range(count):
        for bucket in buckets:
            if index < len(bucket):
                selected.append(bucket[index])
            if len(selected) == count:
                return selected
    return selected


def validate_dataset(experiment, cfg, pool, cohort="pdbbind_only"):
    from src.data.pocket_dataset_v8 import PocketDatasetV8

    data = experiment/"dataset"
    si = json.loads((experiment/"split/summary.json").read_text())
    prep = json.loads((data/"preprocessing_summary.json").read_text())
    if (si["passed"] is not True or si["contract"]["edge_rule"] != "global_only"
            or si["cross_set_audit"]["edges"] != 0 or si["cross_set_audit"]["max_global_si"] >= .3
            or si["cohort"] != cohort or si["external_benchmarks_used"]):
        raise ValueError("New independent cohort global SI audit has not passed")
    if (prep["preprocessing_passed"] is not True or prep["cohort"] != cohort
            or prep["contact_label_protocol"] != fixed5_protocol()):
        raise ValueError("Expected newly prepared native heavy-atom <5A labels")
    checks = {
        experiment/"split/candidate_split.tsv": si["split_sha256"],
        pool/"pool_summary.json": si["contract"]["pool_sha256"],
        data/"splits.txt": prep["training_split_sha256"],
        data/"train_only_stats.json": prep["stats_sha256"],
        Path(cfg["data"]["distances_file"]): prep["labels_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"Changed audited artifact: {path}")
    if (prep["split_sha256"] != si["split_sha256"] or prep["pool_sha256"] != si["contract"]["pool_sha256"]
            or Path(prep["dataset_root"]).resolve() != data.resolve()
            or Path(prep["labels_file"]).resolve() != Path(cfg["data"]["distances_file"]).resolve()):
        raise ValueError("Preprocessing belongs to another split, pool or directory")
    groups = read_strict_splits(data/"splits.txt")
    counts = {s:len(ids) for s,ids in groups.items()}
    if counts != si["counts"] or counts != prep["dataset_counts"]:
        raise ValueError("SI/preprocessing/training split counts differ")
    stats = json.loads((data/"train_only_stats.json").read_text())
    members = split_members(experiment/"split/candidate_split.tsv", groups, cohort)
    if (set(stats["train_keys"]) != {members[sid][0] for sid in groups["train"]}
            or len(stats["train_keys"]) != len(groups["train"])
            or stats["split_sha256"] != si["split_sha256"]):
        raise ValueError("Normalization was not fitted on exactly this TRAIN")
    ids = sum(groups.values(), [])
    kwargs = {k:v for k,v in cfg["data"].items() if k in inspect.signature(PocketDatasetV8).parameters}
    loader = PocketDatasetV8(**kwargs, pdb_ids=ids, use_nma=False, verbose=False)
    if loader.pdb_ids != ids or set(loader.distances_dict) != set(ids):
        raise ValueError("Production loader silently dropped samples or labels")
    contact_min = {}
    for s, split_ids in groups.items():
        minimum = float("inf")
        for sid in split_ids:
            label = loader.distances_dict[sid]
            if label["contact_label_protocol"] != fixed5_protocol():
                raise ValueError(f"Mixed contact protocol: {sid}")
            target, _ = decode_contact_labels(label)
            n = int((target[:cfg["data"]["max_seq_len"]] == 0).sum())
            if n < 3:
                raise ValueError(f"Fewer than 3 positives after training cap: {sid}")
            minimum = min(minimum, n)
        contact_min[s] = minimum
    # Metadata/label QC above covers TEST; no TEST forward pass or model selection.
    read_ids = groups["train"][:4]+groups["train"][-2:]+groups["val"][:4]+groups["val"][-2:]
    for split in ("train", "val"):
        for source in sorted({v[1] for v in members.values()}):
            source_ids = [sid for sid in groups[split] if members[sid][1] == source]
            read_ids.extend(source_ids[:2])
    read_ids = list(dict.fromkeys(read_ids))
    for sid in read_ids:
        item = loader[ids.index(sid)]
        for name, value in item.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite production input: {sid}/{name}")
        if int((item["labels"] == 0).sum()) < 3:
            raise ValueError(f"Production label mismatch: {sid}")
    return dict(passed=True, counts=counts, by_source=si["by_source"], loader_ids_match=True,
        minimum_contact_count_after_residue_cap=contact_min,
        train_val_item_smoke=read_ids, test_forward_used=False), checks


def verify_prepared(experiment, require_smoke=False):
    root = Path(experiment)
    manifest = json.loads((root/"prepared/provenance.json").read_text())
    for path, expected in manifest["asset_hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if source_hashes(root/"code_v1") != manifest["source_hashes"]:
        raise ValueError("Source snapshot changed")
    if sha256(root/"prepared/GFGF.yaml") != manifest["config_sha256"]:
        raise ValueError("Frozen config changed")
    if require_smoke:
        smoke = json.loads((root/"smoke_pass.json").read_text())
        if smoke.get("passed") is not True or smoke.get("provenance_sha256") != sha256(root/"prepared/provenance.json"):
            raise ValueError("GPU smoke missing or from a different preparation")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--pool-dir", type=Path)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--cohort", choices=COHORTS, default="pdbbind_only")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--require-smoke", action="store_true")
    args = ap.parse_args()
    root = args.experiment.resolve()
    if args.verify:
        verify_prepared(root, args.require_smoke)
        print("INTERNAL_GFGF_INPUTS_VERIFIED", flush=True)
        return
    if args.base is None or args.pool_dir is None or sha256(args.base) != BASE_SHA:
        raise ValueError("Use the exact audited GFGF baseline config and QC pool")
    if (root/"prepared/provenance.json").exists():
        existing = verify_prepared(root)
        if existing.get("cohort", "pdbbind_only") != args.cohort:
            raise ValueError("Prepared artifact belongs to another cohort")
        print("ALREADY_PREPARED", flush=True)
        return
    torch.set_num_threads(1)
    base = yaml.safe_load(args.base.read_text())
    cfg = make_config(base, root, root/"dataset", args.cohort)
    qc, checks = validate_dataset(root, cfg, args.pool_dir, args.cohort)
    checks.update({p:sha256(p) for p in (Path(__file__).resolve(), args.base, root/"split/summary.json",
        root/"dataset/preprocessing_summary.json", root/"dataset/preprocessing_contract.json")})
    prepared = root/"prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared/"GFGF.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    save_json(prepared/"loader_qc.json", qc)
    manifest = dict(asset_hashes={str(p.resolve()):h for p,h in checks.items()},
        source_hashes=source_hashes(root/"code_v1"), config_sha256=sha256(prepared/"GFGF.yaml"),
        loader_qc=qc, cohort=args.cohort, seed=44, start="scratch", test_used_for_selection=False,
        model_and_training_unchanged=cfg["model"] == base["model"] and cfg["training"] == base["training"],
        changed_fields=["data.data_root", "data.split_file", "data.distances_file",
                        "experiment.name", "experiment.output_dir", "training.early_stopping_metric"],
        protocol="native heavy-atom d_min <5A; global chain SI <.30 across all three sets; TRAIN-only stats")
    save_json(prepared/"provenance.json", manifest)
    verify_prepared(root)
    print("INTERNAL_GFGF_PREPARED", json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
