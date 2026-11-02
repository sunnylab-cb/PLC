#!/usr/bin/env python3
"""HPC -> Star core-training bundle: cached features only, NEVER model weights.

Base copies normalized conf0/ligand features and already ready ESM caches.
Final requires successful HPC chainwise extraction, fills ESM, checksum-checks
all three feature trees, and enables the portable core-training verifier.
No training, GPU allocation, checkpoint copy, deletion, or remote normalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "G0_chainwise_esm_complex"
FAMILIES = {"complete_features_v4_nma": "_protein_conf0.pt",
            "ligand_features_v5": "_ligand.pt", "esm2_latents": "_protein.pt"}
YEARS = ("1981-2000", "2001-2010", "2011-2019")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def paths_for(root, suffix, ids):
    found = {}
    for year in YEARS:
        folder = root / year
        if not folder.is_dir():
            continue
        for entry in os.scandir(folder):
            sid = entry.name
            if sid not in ids:
                continue
            rel = f"{year}/{sid}/{sid}{suffix}"
            p = root / rel
            if sid in found or not p.is_file() or p.stat().st_size == 0:
                raise ValueError(f"Ambiguous/missing/empty feature: {p}")
            found[sid] = rel
    if set(found) != set(ids):
        raise ValueError(f"Missing {len(set(ids)-set(found))} {root} samples")
    return found


def relocate_config(cfg, destination):
    new = copy.deepcopy(cfg)
    t = new["training"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Core bundle is scratch-only")
    if new["model"]["scan_variant"] != "G0_small_global" or new["model"].get("use_unimol_ligand_input"):
        raise ValueError("Unexpected core model")
    if new["data"].get("esm3_features_dir") or new["data"]["nma_max_conf"] != 0:
        raise ValueError("Only ESM2 and conf0 are in this bundle")
    data = destination / "data"
    new["data"].update(data_root=str(data), esm_features_dir="esm2_latents",
                       distances_file=str(data / "residue_ligand_distances_dcenter.pt"),
                       split_file=str(data / "splits.txt"))
    new["scan_metadata_file"] = str(destination / "prepared/metadata.tsv")
    new["experiment"]["output_dir"] = str(destination / "outputs")
    assert new["model"] == cfg["model"] and new["training"] == cfg["training"]
    return new


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def copy_code(source, target):
    for folder in ("src", "scripts", "tests", "preparation"):
        for p in (source / folder).rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".slurm", ".sh") or "__pycache__" in p.parts:
                continue
            dest = target / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)


def prepare(args):
    src, work = args.source, args.work
    plan_path = work / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["source"] != str(src) or plan["destination"] != str(args.destination):
            raise ValueError("Different existing transfer plan")
        return plan
    manifest = json.loads((src / "prepared/manifest.json").read_text())
    joint = Path(manifest["upstream"])
    cfg = yaml.safe_load((src / "prepared" / f"{VARIANT}.yaml").read_text())
    work.mkdir(parents=True, exist_ok=True)
    payload = work / "payload"
    payload.mkdir(exist_ok=False)
    for folder in ("prepared", "data", "source"):
        (payload / folder).mkdir()
    with (src / "prepared/metadata.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = {r["sample_id"] for r in rows}
    if len(ids) != len(rows) or any(Path(s).name != s for s in ids):
        raise ValueError("Duplicate/unsafe sample IDs")
    counts = dict(Counter(r["split"] for r in rows))
    if sum(manifest["all_counts"].values()) != len(ids):
        raise ValueError("Different source cohort")
    for name in ("preprocessing_contract.json", "preprocessing_summary.json",
                 "residue_ligand_distances_dcenter.pt", "splits.txt", "train_only_stats.json"):
        p = joint / "dataset" / name
        if sha(p) != manifest["assets"][str(p)]:
            raise ValueError(f"Changed frozen input {p}")
        shutil.copy2(p, payload / "data" / name)
    shutil.copy2(src / "prepared/metadata.tsv", payload / "prepared/metadata.tsv")
    for source, name in ((src / "prepared/manifest.json", "chainwise_manifest.json"),
                         (src / "prepared" / f"{VARIANT}.yaml", "hpc_config.yaml"),
                         (joint / "split/summary.json", "si_summary.json"),
                         (joint / "prepared/provenance.json", "joint_provenance.json"),
                         (Path(manifest["training"]) / "prepared/provenance.json", "g0_provenance.json")):
        shutil.copy2(source, payload / "source" / name)
    (payload / "prepared" / f"{VARIANT}.yaml").write_text(
        yaml.safe_dump(relocate_config(cfg, args.destination), sort_keys=True))
    copy_code(ROOT, payload / "code_v1")
    model_source = Path(manifest["training"]) / "code_v1"
    copy_code(model_source, payload / "model_code")
    for rel, digest in manifest["original_source_hashes"].items():
        if rel.startswith(("src/", "scripts/", "tests/")) and sha(payload / "model_code" / rel) != digest:
            raise ValueError(f"Original G0 source mismatch: {rel}")
    shutil.copytree(joint / "deps", payload / "deps", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    forbidden = [p for folder in ("code_v1", "model_code", "deps") for p in (payload / folder).rglob("*")
                 if p.suffix in (".pt", ".pth", ".safetensors", ".ckpt")]
    if forbidden:
        raise ValueError(f"Weights forbidden in code/dependencies: {forbidden[:3]}")
    sources, indices = {}, {}
    unique_inodes = set()
    estimated = 0
    for family, suffix in FAMILIES.items():
        original = joint / "dataset" / family
        index = paths_for(original, suffix, ids)
        indices[family] = index
        sources[family] = str(src / "esm2_latents" if family == "esm2_latents" else original)
        (work / f"{family}.txt").write_text("".join(index[s] + "\n" for s in sorted(ids)))
        for sid, rel in index.items():
            st = (original / rel).stat()
            # Reassembled complex ESM files are new independent files.
            key = (family, sid) if family == "esm2_latents" and sid in manifest["changed"] else (st.st_dev, st.st_ino)
            if key not in unique_inodes:
                estimated += st.st_size
                unique_inodes.add(key)
        print("CORE_INVENTORY", family, len(index), flush=True)
    base_ids = ids - set(manifest["changed"])
    (work / "esm_base.txt").write_text("".join(indices["esm2_latents"][s] + "\n" for s in sorted(base_ids)))
    plan = dict(source=str(src), destination=str(args.destination), sources=sources,
                indices=indices, counts=counts, total=len(ids), base_esm=len(base_ids),
                pending_esm=len(manifest["changed"]), estimated_unique_bytes=estimated,
                source_manifest_sha256=sha(src / "prepared/manifest.json"), weights_transferred=False)
    write_json(plan_path, plan)
    print("CORE_PLAN", json.dumps({k:v for k,v in plan.items() if k not in ("indices", "sources")}), flush=True)
    return plan


def ssh_args(args):
    return ["ssh", "-i", str(args.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=10"]


def remote(args, command):
    return subprocess.run(ssh_args(args) + [args.host, command], check=True, text=True, capture_output=True).stdout


def sync(args, source, destination, file_list=None, checksum=False, dry_run=False):
    cmd = ["rsync", "-rltHL", "--partial", "--partial-dir=.rsync-partial", "--stats",
           "-e", shlex.join(ssh_args(args))]
    if file_list is not None:
        cmd += ["--files-from=" + str(file_list)]
    if checksum:
        cmd += ["--checksum"]
    if dry_run:
        cmd.remove("--stats")
        cmd += ["--dry-run", "--itemize-changes"]
    else:
        cmd += ["--info=progress2"]
    cmd += [str(source) + "/", args.host + ":" + str(destination) + "/"]
    if dry_run:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        # Directory mtimes are not training data; any file delta invalidates the receipt.
        changes = [s for s in result.stdout.splitlines() if len(s) > 1 and s[1] != "d"]
        if changes:
            raise ValueError(f"Transfer checksum mismatch: {changes[:10]}")
    else:
        subprocess.run(cmd, check=True)


def check_inventory(root, path, total):
    counts = Counter()
    seen = set()
    with (root / "prepared/metadata.tsv").open() as f:
        ids = [r["sample_id"] for r in csv.DictReader(f, delimiter="\t")]
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError("Metadata coverage differs from frozen cohort")
    expected = set(ids)
    covered = {family: set() for family in FAMILIES}
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rel = Path(r["path"])
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 4
                    or rel.parts[0] not in FAMILIES or str(rel) in seen):
                raise ValueError("Unsafe/duplicate feature inventory")
            family, year, sid, filename = rel.parts
            if (year not in YEARS or sid not in expected or sid in covered[family]
                    or filename != sid + FAMILIES[family]):
                raise ValueError("Feature inventory sample identity mismatch")
            covered[family].add(sid)
            seen.add(str(rel))
            p = root / "data" / rel
            if p.is_symlink() or not p.is_file() or p.stat().st_size != int(r["bytes"]) or p.stat().st_size == 0:
                raise ValueError(f"Missing/changed/linked feature: {p}")
            counts[rel.parts[0]] += 1
    if dict(counts) != {k: total for k in FAMILIES}:
        raise ValueError(f"Incomplete core feature coverage: {counts}")
    if any(sids != expected for sids in covered.values()):
        raise ValueError("Feature/sample coverage differs across modalities")
    return dict(counts)


def verify_core(root, smoke=False):
    p = json.loads((root / "prepared/provenance.json").read_text())
    if not p["complete"] or p["kind"] != "fullcomplex_core_no_weights" or p["weights_transferred"]:
        raise ValueError("Core training bundle is not ready")
    for rel, digest in p["assets"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"Changed core asset: {rel}")
    for folder, expected in p["code_hashes"].items():
        if tree_hashes(root / folder) != expected:
            raise ValueError(f"Changed code/dependencies: {folder}")
    check_inventory(root, root / "prepared/features.tsv", p["total"])
    cfg = yaml.safe_load((root / "prepared" / f"{VARIANT}.yaml").read_text())
    original = yaml.safe_load((root / "source/hpc_config.yaml").read_text())
    if cfg != relocate_config(original, root):
        raise ValueError("Scientific configuration changed during relocation")
    if smoke:
        report = json.loads((root / "smoke" / f"{VARIANT}.json").read_text())
        if not report["passed"] or report["provenance_sha256"] != sha(root / "prepared/provenance.json"):
            raise ValueError("Matching single-GPU smoke required before training")
    return p


def finish_on_star(root):
    receipt = json.loads((root / "prepared/transfer_receipt.json").read_text())
    if not receipt["checksum_verified"] or receipt["destination"] != str(root):
        raise ValueError("Matching checksum transfer receipt required")
    check_inventory(root, root / "prepared/features.tsv", receipt["total"])
    assets = {}
    for folder in ("prepared", "source"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and p.name != "provenance.json":
                assets[str(p.relative_to(root))] = sha(p)
    for p in (root / "data").iterdir():
        if p.is_file():
            assets[str(p.relative_to(root))] = sha(p)
    p = dict(kind="fullcomplex_core_no_weights", complete=True, weights_transferred=False,
             total=receipt["total"], counts=receipt["counts"], assets=assets,
             code_hashes={d: tree_hashes(root / d) for d in ("code_v1", "model_code", "deps")},
             source_completion_sha256=receipt["source_completion_sha256"],
             checkpoint_initialization=None, model_context="FULL COMPLEX", training_launched=False)
    write_json(root / "prepared/provenance.json", p)
    verify_core(root)
    print("CORE_TRAINING_READY_NO_WEIGHTS", json.dumps({k:v for k,v in p.items() if k not in ("assets", "code_hashes")}), flush=True)


def transfer(args):
    plan = prepare(args)
    dest, work, src = args.destination, args.work, args.source
    payload = work / "payload"
    status = work / f"{args.phase}_status.json"
    def progress(stage, **extra):
        write_json(status, dict(phase=args.phase, stage=stage, time=time.time(), **extra))
        print("CORE_SYNC", args.phase, stage, flush=True)
    remote(args, "mkdir -p " + shlex.quote(str(dest / "data")))
    free = int(remote(args, "/usr/bin/python3 -c " + shlex.quote(
        f"import shutil;print(shutil.disk_usage({str(dest)!r}).free)" )).strip())
    if args.phase == "base" and free < plan["estimated_unique_bytes"] * 1.15 + 10 * 2**30:
        raise ValueError("Insufficient free space for conservative core transfer estimate")
    if args.phase == "base":
        progress("metadata_and_code")
        sync(args, payload, dest)
        for family in ("complete_features_v4_nma", "ligand_features_v5", "esm2_latents"):
            progress(family)
            listing = work / ("esm_base.txt" if family == "esm2_latents" else family + ".txt")
            remote(args, "mkdir -p " + shlex.quote(str(dest / "data" / family)))
            sync(args, Path(plan["sources"][family]), dest / "data" / family, listing)
        progress("base_complete_waiting_for_new_esm", complete=True, pending_esm=plan["pending_esm"])
        return
    source_provenance = src / "prepared/provenance.json"
    p = json.loads(source_provenance.read_text())
    if not p["complete"] or not p["no_chain_only_samples"] or p["recipe"]["protein_context"] != "unchanged full complex":
        raise ValueError("HPC full-complex ESM extraction is not complete")
    if sha(src / "prepared/manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("Source manifest changed during transfer")
    for name in ("provenance.json", "materialized.json", "extraction_complete.json", "cache_parity.json"):
        source = src / "prepared" / name
        if name in ("materialized.json", "extraction_complete.json") and sha(source) != p["assets"][str(source)]:
            raise ValueError(f"Changed source completion evidence: {name}")
        shutil.copy2(source, payload / "source" / ("esm_" + name))
    progress("final_esm")
    sync(args, Path(plan["sources"]["esm2_latents"]), dest / "data/esm2_latents", work / "esm2_latents.txt", checksum=True)
    with (payload / "prepared/features.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "bytes"), delimiter="\t")
        writer.writeheader()
        for family, index in plan["indices"].items():
            for sid, rel in sorted(index.items()):
                path = Path(plan["sources"][family]) / rel
                writer.writerow(dict(path=family + "/" + rel, bytes=path.stat().st_size))
    for family in FAMILIES:
        progress("checksum_" + family)
        sync(args, Path(plan["sources"][family]), dest / "data" / family, work / (family + ".txt"), checksum=True, dry_run=True)
    write_json(payload / "prepared/transfer_receipt.json", dict(checksum_verified=True,
        destination=str(dest), source=str(src), source_completion_sha256=sha(source_provenance),
        total=plan["total"], counts=plan["counts"], weights_transferred=False,
        method="sequential rsync checksum dry-run: no feature-file differences"))
    sync(args, payload, dest, checksum=True)
    progress("star_coverage_verification")
    command = shlex.join([str(args.star_python), str(dest / "code_v1/scripts/sync/sync_chainwise_core_to_star.py"),
                          "finish", "--destination", str(dest)])
    print(remote(args, command), flush=True)
    progress("complete", complete=True, training_launched=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("base", "final", "finish", "verify"))
    p.add_argument("--source", type=Path)
    p.add_argument("--work", type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--host", default="husrcf@10.123.4.78")
    p.add_argument("--key", type=Path, default=Path.home() / ".ssh/id_ed25519_star")
    p.add_argument("--star-python", type=Path, default=Path("/home/husrcf/anaconda3/envs/gpsite_full/bin/python"))
    args = p.parse_args()
    if args.phase == "finish":
        finish_on_star(args.destination)
    elif args.phase == "verify":
        verify_core(args.destination)
    else:
        transfer(args)
