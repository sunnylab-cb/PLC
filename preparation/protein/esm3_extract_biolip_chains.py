#!/usr/bin/env python3
"""Extract, normalize, and materialize ESM3 features for audited BioLiP chains."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from esm.sdk.api import ESMProtein, LogitsConfig

from esm3_extract_latents import _load_model, _resolve_device, _set_default_cache


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "ASX": "B", "GLX": "Z", "SEC": "U", "PYL": "O", "MSE": "M",
    "HID": "H", "HIE": "H", "HIP": "H", "ASH": "D", "GLH": "E",
    "CYX": "C", "CYM": "C", "LYN": "K",
}
YEAR_BUCKETS = ("1981-2000", "2001-2010", "2011-2019")


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def load_manifest(path: Path, selected: set[str]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample_id = row["sample_id"].strip()
            if sample_id not in selected:
                continue
            key = (row["pdb_id"].lower(), row["protein_chain"].strip())
            groups[key].append(sample_id)
            seen.add(sample_id)
    missing = sorted(selected - seen)
    if missing:
        raise RuntimeError(f"manifest missing {len(missing)} selected IDs: {missing[:20]}")
    return dict(groups)


def find_reference(root: Path, sample_id: str) -> tuple[str, Path]:
    for bucket in YEAR_BUCKETS:
        path = root / "esm2_latents" / bucket / sample_id / f"{sample_id}_protein.pt"
        if path.is_file():
            return bucket, path
    raise FileNotFoundError(f"ESM2 reference missing for {sample_id}")


def residue_keys(obj: dict) -> list[tuple[str, int, str]]:
    chains = obj.get("chain_id", obj.get("chain_ids"))
    indices = obj.get("residue_index", obj.get("residue_indices"))
    insertions = obj.get("insertion_code", obj.get("insertion_codes"))
    if chains is None or indices is None:
        raise KeyError("missing residue key metadata")
    if insertions is None:
        insertions = [""] * len(indices)
    return [(str(c), int(i), str(ins).strip()) for c, i, ins in zip(chains, indices, insertions)]


def extract_chain(model, cache: dict, model_name: str) -> dict:
    names3 = [str(value).upper() for value in cache["residue_names"]]
    sequence = "".join(AA3_TO_1.get(name, "X") for name in names3)
    protein = ESMProtein(sequence=sequence)
    with torch.inference_mode():
        output = model.logits(model.encode(protein), LogitsConfig(return_embeddings=True))
    embeddings = output.embeddings[0][1:-1].float().cpu()
    if embeddings.shape[0] != len(sequence):
        raise RuntimeError(f"ESM3 length {embeddings.shape[0]} != sequence length {len(sequence)}")
    result = {
        "embeddings": embeddings,
        "chain_id": [str(v) for v in cache["chain_ids"]],
        "residue_index": [int(v) for v in cache["residue_indices"]],
        "insertion_code": [str(v).strip() for v in cache.get("insertion_codes", [""] * len(sequence))],
        "residue_name_1": list(sequence),
        "residue_name_3": names3,
        "sequence": sequence,
        "pdb_id": str(cache["pdb_id"]).lower(),
        "selected_chain": str(cache["selected_chain"]),
        "mode": "biolip_chain_sequence",
        "model_name": model_name,
    }
    if len(residue_keys(result)) != embeddings.shape[0]:
        raise RuntimeError("residue metadata and ESM3 embedding length differ")
    return result


def normalize(obj: dict, mean: torch.Tensor, std: torch.Tensor, stats_path: Path) -> dict:
    embeddings = obj["embeddings"].float()
    if embeddings.shape[-1] != mean.numel():
        raise RuntimeError(f"ESM3 dim {embeddings.shape[-1]} != stats dim {mean.numel()}")
    result = dict(obj)
    result["embeddings"] = (embeddings - mean) / std.clamp_min(1e-8)
    result["feature_normalization"] = {
        "esm3_embeddings": "zscore_all_dims",
        "stats_source": str(stats_path.resolve()),
    }
    if not torch.isfinite(result["embeddings"]).all():
        raise RuntimeError("non-finite normalized ESM3 embeddings")
    return result


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting link: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"destination exists and is not a symlink: {destination}")
    os.symlink(source.resolve(), destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="esm3-open")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    manifest = args.manifest or args.biolip_root / "pilot_manifest.tsv"
    groups = load_manifest(manifest, read_ids(args.ids))
    selected_groups = [item for index, item in enumerate(sorted(groups.items())) if index % args.num_shards == args.shard_id]

    stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    esm3_stats = stats.get("esm3")
    if not esm3_stats:
        raise KeyError(f"{args.stats} has no esm3 statistics")
    mean = torch.as_tensor(esm3_stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(esm3_stats["std"], dtype=torch.float32)

    _set_default_cache()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = _load_model(device, args.model)

    raw_root = args.output_root / "unique_raw"
    norm_root = args.output_root / "unique_norm"
    sample_root = args.biolip_root / "esm3_latents_norm"
    failures = []
    linked = 0
    for ordinal, ((pdb_id, chain), sample_ids) in enumerate(selected_groups, start=1):
        stem = f"{pdb_id}_{chain}"
        cache_path = args.biolip_root / "_chain_cache" / f"{stem}_protein.pt"
        raw_path = raw_root / f"{stem}_protein.pt"
        norm_path = norm_root / f"{stem}_protein.pt"
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if args.overwrite or not norm_path.exists():
                raw = extract_chain(model, cache, args.model)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(raw, raw_path)
                normalized = normalize(raw, mean, std, args.stats)
                norm_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(normalized, norm_path)
            else:
                normalized = torch.load(norm_path, map_location="cpu", weights_only=False)

            chain_keys = residue_keys(cache)
            if residue_keys(normalized) != chain_keys:
                raise RuntimeError("ESM3 keys differ from V4 chain-cache keys")
            for sample_id in sample_ids:
                bucket, ref_path = find_reference(args.biolip_root, sample_id)
                reference = torch.load(ref_path, map_location="cpu", weights_only=False)
                if residue_keys(reference) != chain_keys:
                    raise RuntimeError(f"{sample_id}: ESM2 keys differ from V4 chain-cache keys")
                destination = sample_root / bucket / sample_id / f"{sample_id}_protein.pt"
                link_file(norm_path, destination)
                linked += 1
            print(f"[{ordinal}/{len(selected_groups)}] {stem}: samples={len(sample_ids)} length={len(chain_keys)}")
        except Exception as exc:
            failures.append({"pdb_id": pdb_id, "chain": chain, "samples": sample_ids, "error": str(exc)})
            print(f"[{ordinal}/{len(selected_groups)}] FAIL {stem}: {exc}")

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "unique_chains": len(selected_groups),
        "linked_samples": linked,
        "failures": len(failures),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"summary_shard{args.shard_id}.json").write_text(json.dumps(summary, indent=2))
    (args.output_root / f"failures_shard{args.shard_id}.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
