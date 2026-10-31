#!/usr/bin/env python3
"""Encode chains independently, then restore the UNCHANGED full-complex axis.

No chain-only model samples, label unions, split changes, or new normalization.
Only multichain PDBbind ESM caches change; existing single-chain caches remain.
Long chains use the BioLiP 1022/768 overlap-mean policy, without truncation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.prepare_libre_transfer import source_hashes
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json
from scripts.diagnostics.strict_assignment_common import read_strict_splits

VARIANT = "G0_chainwise_esm_complex"
RECIPE = dict(id="esm2_l33_chainwise_fullcomplex_v1", model="esm2_t33_650M_UR50D",
              layer=33, dtype="float32", window=1022, overlap=768,
              merge="uniform overlapping-window mean", extra_normalization=False,
              protein_context="unchanged full complex", token_mapping="unchanged cached sequence")


def seq_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def residue_keys(data):
    values = [data.get(a, data.get(b)) for a, b in (
        ("chain_id", "chain_ids"), ("residue_index", "residue_indices"),
        ("insertion_code", "insertion_codes"))]
    if any(v is None for v in values) or len({len(v) for v in values}) != 1:
        raise ValueError("Missing/inconsistent residue metadata")
    keys = [(str(c), int(i), str(code or "").strip()) for c, i, code in zip(*values)]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Empty/duplicate residue axis")
    return keys


def chain_parts(data):
    keys = residue_keys(data)
    sequence = data["sequence"]
    emb = data["embeddings"]
    if not isinstance(sequence, str) or len(sequence) != len(keys):
        raise ValueError("Sequence/residue-axis mismatch")
    if emb.shape != (len(keys), 1280) or not torch.isfinite(emb).all():
        raise ValueError("Expected finite ESM2 L33 [N,1280]")
    if any(x not in "ACDEFGHIKLMNPQRSTVWYXBZUO" for x in sequence):
        raise ValueError("Unsupported token: never silently remap the original sequence")
    parts = []
    for chain in dict.fromkeys(k[0] for k in keys):
        indices = [i for i, key in enumerate(keys) if key[0] == chain]
        seq = "".join(sequence[i] for i in indices)
        parts.append(dict(chain=chain, indices=indices, sequence=seq, sequence_sha256=seq_hash(seq)))
    return parts


def window_starts(n, width=1022, overlap=768):
    if n < 1 or not 0 <= overlap < width:
        raise ValueError("Invalid sequence/window length")
    if n <= width:
        return [0]
    starts = list(range(0, n - width + 1, width - overlap))
    if starts[-1] != n - width:
        starts.append(n - width)
    return starts


def stitch_embeddings(original, parts, cache):
    result = torch.empty((len(original["sequence"]), 1280), dtype=torch.float32)
    seen = torch.zeros(len(result), dtype=torch.int32)
    for part in parts:
        idx = torch.tensor(part["indices"], dtype=torch.long)
        emb = cache(part)
        if emb.shape != (len(idx), 1280) or emb.dtype != torch.float32 or not torch.isfinite(emb).all():
            raise ValueError("Invalid chain embeddings")
        if "".join(original["sequence"][i] for i in idx.tolist()) != part["sequence"]:
            raise ValueError("Changed sequence during reassembly")
        result[idx] = emb
        seen.index_add_(0, idx, torch.ones(len(idx), dtype=torch.int32))
    if not (seen == 1).all():
        raise ValueError("Reassembly dropped or duplicated residues")
    output = dict(original)
    output["embeddings"] = result
    output["esm_context_policy"] = dict(RECIPE)
    if residue_keys(output) != residue_keys(original):
        raise ValueError("Reassembly changed residue metadata")
    return output


def locate(directory, sid, suffix="_protein.pt"):
    paths = [directory / year / sid / (sid + suffix)
             for year in ("1981-2000", "2001-2010", "2011-2019")]
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise ValueError(f"Missing/ambiguous input: {sid} under {directory}")
    return found[0]


def inspect_esm(item):
    torch.set_num_threads(1)
    row, path = item
    d = torch.load(path, map_location="cpu", weights_only=False)
    parts = chain_parts(d)
    return dict(**row, original=str(path), original_sha256=sha(path), parts=parts,
                n_residues=len(d["sequence"]),
                n_chains=len({k[0] for k in residue_keys(d)[:1800]}))


def prepare(args):
    root = args.experiment.resolve()
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = args.training / "prepared/G0_small_global.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg["model"]["scan_variant"] != "G0_small_global" or cfg["experiment"]["seed"] != 44:
        raise ValueError("Expected unchanged G0 seed44 baseline")
    if cfg["data"]["label_contact_thresh"] != 5 or cfg["data"]["max_seq_len"] != 1800:
        raise ValueError("Unexpected target/cap")
    parent = json.loads((args.training / "prepared/provenance.json").read_text())
    assets = dict(parent["asset_hashes"])
    assets[str(cfg_path)] = sha(cfg_path)
    assets[str(args.training / "prepared/provenance.json")] = sha(args.training / "prepared/provenance.json")
    for path, expected in assets.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen G0 asset changed: {path}")
    if sha(cfg_path) != parent["config_hashes"]["G0_small_global"]:
        raise ValueError("G0 configuration hash mismatch")
    si = json.loads((args.upstream / "split/summary.json").read_text())
    if not si["passed"] or si["cross_set_audit"]["edges"] or si["cross_set_audit"]["max_global_si"] >= .3:
        raise ValueError("Strict SI30 audit missing")
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    with (args.upstream / "split/candidate_split.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    membership = {sid: split for split, ids in groups.items() for sid in ids}
    if len(rows) != len(membership) or {r["sample_id"] for r in rows} != set(membership):
        raise ValueError("Split metadata coverage mismatch")
    if any(membership[r["sample_id"]] != r["split"] for r in rows):
        raise ValueError("Split changed")
    if any(r["source"] not in ("pdbbind", "biolip") for r in rows):
        raise ValueError("Unknown cohort source")
    esm = Path(cfg["data"]["data_root"]) / cfg["data"]["esm_features_dir"]
    output = root / "esm2_latents"
    pdb_rows = [(r, locate(esm, r["sample_id"])) for r in rows if r["source"] == "pdbbind"]
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, r in enumerate(executor.map(inspect_esm, pdb_rows, chunksize=8), 1):
            records[r["sample_id"]] = r
            if i % 1000 == 0:
                print("CHAINWISE_PREP", i, len(pdb_rows), flush=True)
    changed, unique, counts = {}, {}, Counter()
    fields = list(rows[0]) + ["n_chains"]
    with (out / "metadata.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            sid = row["sample_id"]
            src = locate(esm, sid)
            dest = output / src.relative_to(esm)
            r = records.get(sid)
            row["n_chains"] = r["n_chains"] if r else 1
            writer.writerow(row)
            counts[row["split"] + "/" + row["source"]] += 1
            if r is not None and len(r["parts"]) > 1:
                r["output"] = str(dest)
                changed[sid] = r
                dest.parent.mkdir(parents=True, exist_ok=True)
                for p in r["parts"]:
                    key = p["sequence_sha256"]
                    if key in unique and unique[key]["sequence"] != p["sequence"]:
                        raise ValueError("Sequence hash collision")
                    u = unique.setdefault(key, dict(sequence=p["sequence"], splits=[]))
                    if row["split"] not in u["splits"]:
                        u["splits"].append(row["split"])
            else:
                # Symlink the unchanged per-sample folder, not another data tree.
                dest.parent.parent.mkdir(parents=True, exist_ok=True)
                dest.parent.symlink_to(src.parent.resolve(), target_is_directory=True)
    weight_hash = sha(args.weights)
    assets[str(args.weights)] = weight_hash
    manifest = dict(recipe=RECIPE, weights=str(args.weights), weights_sha256=weight_hash,
                    libre_cache=str(args.libre_cache), changed=changed, sequences=unique,
                    all_counts=dict(counts), original_config=str(cfg_path),
                    training=str(args.training), upstream=str(args.upstream),
                    source_hashes=source_hashes(root / "code_v1"), assets=assets,
                    original_source_hashes=parent["source_hashes"],
                    extraction_script_sha256=sha(__file__),
                    protein_context="FULL COMPLEX", labels="unchanged original per-copy <5A")
    write_json(out / "manifest.json", manifest)
    new = copy.deepcopy(cfg)
    new["data"]["esm_features_dir"] = str(output)
    new["scan_metadata_file"] = str(out / "metadata.tsv")
    new["experiment"].update(name="joint_si30_G0_chainwise_esm_fullcomplex_seed44",
                              version_suffix="chainesm", output_dir=str(root / "outputs"))
    for key in ("init_from_checkpoint", "init_from_v11m3"):
        if new["training"].get(key) is not None:
            raise ValueError("Not a scratch baseline")
    if any(new["training"].get(k) for k in ("freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Frozen trunk is forbidden")
    (out / f"{VARIANT}.yaml").write_text(yaml.safe_dump(new, sort_keys=True))
    write_json(out / "inventory.json", dict(complete=True, changed_complexes=len(changed),
        changed_by_split=dict(Counter(r["split"] for r in changed.values())),
        unique_chains=len(unique), unique_residues=sum(len(v["sequence"]) for v in unique.values()),
        long_unique_chains=sum(len(v["sequence"]) > 1022 for v in unique.values()),
        model_context="full complex", no_single_chain_training=True,
        scope="Only existing multichain PDBbind caches replaced; BioLiP/single-chain inputs unchanged"))
    print("CHAINWISE_PREPARED", json.dumps(json.loads((out / "inventory.json").read_text())), flush=True)


def load_manifest(root):
    p = json.loads((root / "prepared/manifest.json").read_text())
    if p["recipe"] != RECIPE or sha(__file__) != p["extraction_script_sha256"]:
        raise ValueError("Changed extraction recipe/code")
    for rel, digest in p["original_source_hashes"].items():
        if rel.startswith(("src/models/", "src/data/", "scripts/train/")):
            if sha(Path(p["training"]) / "code_v1" / rel) != digest:
                raise ValueError(f"Original G0 code changed: {rel}")
    return p


def atomic_save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(data, tmp)
    tmp.replace(path)


@torch.inference_mode()
def extract(args):
    import esm
    root = args.experiment
    manifest = load_manifest(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm GPU allocation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    weight = Path(manifest["weights"])
    if sha(weight) != manifest["weights_sha256"]:
        raise ValueError("Changed ESM2 weights")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weight))
    model = model.float().eval().to("cuda")
    converter = alphabet.get_batch_converter()

    def encode(seq):
        _, _, tokens = converter([("chain", seq)])
        value = model(tokens.to("cuda"), repr_layers=[33], return_contacts=False)["representations"][33]
        return value[0, 1:len(seq) + 1].float().cpu().clone()

    def cached(seq):
        path = Path(manifest["libre_cache"]) / (seq_hash(seq) + ".pt")
        if len(seq) > 1022 or not path.exists():
            return None
        d = torch.load(path, map_location="cpu", weights_only=True)
        e = d["embeddings"]
        if d.get("sequence") != seq or d.get("policy") != "fp32_l33_residue_only_v1":
            raise ValueError(f"Incompatible cache metadata: {path}")
        if e.shape != (len(seq), 1280) or e.dtype != torch.float32 or not torch.isfinite(e).all():
            raise ValueError(f"Invalid cached tensor: {path}")
        return e

    sequences = sorted(manifest["sequences"].items(), key=lambda x: ("val" not in x[1]["splits"], x[0]))
    checks = []
    for key, rec in sequences:
        seq = rec["sequence"]
        old = cached(seq)
        if old is None:
            continue
        new = encode(seq)
        passed = bool(torch.allclose(old, new, rtol=2e-4, atol=2e-4))
        checks.append(dict(sequence_sha256=key, max_abs=float((old-new).abs().max()), passed=passed))
        if len(checks) == 8:
            break
    reuse = bool(checks) and all(c["passed"] for c in checks)
    write_json(root / "prepared/cache_parity.json", dict(reuse_enabled=reuse, checks=checks,
        fallback="fresh extraction for all chains if parity fails", weights_sha256=sha(weight)))
    counts = Counter()
    for i, (key, rec) in enumerate(sequences, 1):
        dest = root / "chain_cache" / f"{key}.pt"
        seq = rec["sequence"]
        if dest.exists():
            d = torch.load(dest, map_location="cpu", weights_only=True)
            validate_cache(d, seq, manifest)
            counts["resume"] += 1
            continue
        value = cached(seq) if reuse else None
        origin = "verified_libre_cache" if value is not None else "fresh"
        if value is None:
            value = torch.zeros(len(seq), 1280)
            coverage = torch.zeros(len(seq))
            for start in window_starts(len(seq)):
                part = encode(seq[start:start + 1022])
                value[start:start + len(part)] += part
                coverage[start:start + len(part)] += 1
            if not (coverage > 0).all():
                raise ValueError("Uncovered residues")
            value /= coverage[:, None]
        d = dict(sequence=seq, embeddings=value, recipe=RECIPE,
                 weights_sha256=manifest["weights_sha256"], origin=origin)
        if origin == "verified_libre_cache":
            d["input_cache_sha256"] = sha(Path(manifest["libre_cache"]) / (key + ".pt"))
        validate_cache(d, seq, manifest)
        atomic_save(d, dest)
        counts[origin] += 1
        if i % 50 == 0 or i == len(sequences):
            write_json(root / "prepared/extraction_progress.json", dict(done=i, total=len(sequences), counts=dict(counts)))
            print("CHAINWISE_EXTRACT", i, len(sequences), dict(counts), flush=True)
    write_json(root / "prepared/extraction_complete.json", dict(complete=True, total=len(sequences),
        manifest_sha256=sha(root / "prepared/manifest.json"), counts=dict(counts)))


def validate_cache(d, seq, manifest):
    e = d["embeddings"]
    if (d.get("recipe") != RECIPE or d.get("sequence") != seq
            or d.get("weights_sha256") != manifest["weights_sha256"]
            or e.shape != (len(seq), 1280) or e.dtype != torch.float32 or not torch.isfinite(e).all()):
        raise ValueError("Chain cache provenance/shape mismatch")


def materialize_one(item):
    root, record, weights_hash = item
    torch.set_num_threads(1)
    original = torch.load(record["original"], map_location="cpu", weights_only=False)
    if sha(record["original"]) != record["original_sha256"]:
        raise ValueError("Original cache changed")
    def get(part):
        d = torch.load(root / "chain_cache" / (part["sequence_sha256"] + ".pt"),
                       map_location="cpu", weights_only=True)
        validate_cache(d, part["sequence"], {"weights_sha256": weights_hash})
        return d["embeddings"]
    output = stitch_embeddings(original, record["parts"], get)
    cos = torch.nn.functional.cosine_similarity(original["embeddings"].float(), output["embeddings"], dim=-1)
    atomic_save(output, Path(record["output"]))
    return record["sample_id"], dict(sha256=sha(record["output"]), n_residues=len(cos),
        median_residue_cosine=float(cos.median()), split=record["split"])


def finalize(args):
    root = args.experiment
    manifest = load_manifest(root)
    done = json.loads((root / "prepared/extraction_complete.json").read_text())
    if not done["complete"] or done["manifest_sha256"] != sha(root / "prepared/manifest.json"):
        raise ValueError("Extraction incomplete/changed")
    tasks = [(root, r, manifest["weights_sha256"]) for r in manifest["changed"].values()]
    output = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (sid, info) in enumerate(executor.map(materialize_one, tasks, chunksize=4), 1):
            output[sid] = info
            if i % 500 == 0:
                print("CHAINWISE_REASSEMBLE", i, len(tasks), flush=True)
    write_json(root / "prepared/materialized.json", output)
    assets = dict(manifest["assets"])
    for name in ("manifest.json", "metadata.tsv", "materialized.json", "extraction_complete.json"):
        assets[str(root / "prepared" / name)] = sha(root / "prepared" / name)
    report = dict(complete=True, recipe=RECIPE, assets=assets, source_hashes=manifest["source_hashes"],
        configs={VARIANT: sha(root / "prepared" / f"{VARIANT}.yaml")},
        counts=manifest["all_counts"], changed_complexes=len(output),
        train_from_scratch=True, model_context="unchanged FULL COMPLEX",
        no_chain_only_samples=True, no_label_or_split_changes=True, test_forward_used=False)
    write_json(root / "prepared/provenance.json", report)
    verify(root)
    print("CHAINWISE_COMPLEX_READY", len(output), flush=True)


def verify(root, smoke=False):
    load_manifest(root)
    p = json.loads((root / "prepared/provenance.json").read_text())
    if not p["complete"] or p["recipe"] != RECIPE or not p["no_chain_only_samples"]:
        raise ValueError("Wrong/incomplete full-complex experiment")
    for path, expected in p["assets"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if source_hashes(root / "code_v1") != p["source_hashes"]:
        raise ValueError("Source snapshot changed")
    for name, expected in p["configs"].items():
        if sha(root / "prepared" / f"{name}.yaml") != expected:
            raise ValueError("Changed training configuration")
    if smoke:
        s = json.loads((root / "smoke" / f"{VARIANT}.json").read_text())
        if not s["passed"] or s["provenance_sha256"] != sha(root / "prepared/provenance.json"):
            raise ValueError("Matching GPU smoke required")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "finalize", "verify"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--libre-cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "verify":
        verify(args.experiment, args.require_smoke)
    else:
        globals()[args.phase](args)
