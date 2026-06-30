#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-extract multi-layer ESM2 residue embeddings from existing ESM2 cache files.

This avoids needing original PDB files.  Each input cache is expected to contain
the residue-aligned `sequence` and metadata produced by esm2_extract_latents.py.
The output preserves metadata and replaces `embeddings` with [N, K, D] when
multiple layers are requested.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import torch
import esm


def _load_model(device: torch.device, model_name: str):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unknown ESM2 model: {model_name}")
    model = model.to(device)
    model.eval()
    return model, alphabet


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device_str}")
        if device_str == "cuda":
            device_str = "cuda:0"
        return torch.device(device_str)
    return torch.device(device_str)


def _parse_devices(devices_str: str) -> list[str]:
    devices = []
    for part in str(devices_str).split(","):
        p = part.strip()
        if not p:
            continue
        devices.append(f"cuda:{p}" if p.isdigit() else p)
    return devices


def _parse_repr_layers(value: str, num_layers: int) -> list[int]:
    layers = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0:
            layer = num_layers + 1 + layer
        if layer < 0 or layer > num_layers:
            raise ValueError(f"Invalid ESM repr layer {part!r}; valid range is 0..{num_layers}")
        layers.append(layer)
    return layers or [num_layers]


def _extract_sequence(model, alphabet, sequence: str, repr_layers: list[int]) -> torch.Tensor:
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(batch_tokens, repr_layers=repr_layers)
    reps = [out["representations"][layer][0, 1:-1].detach().cpu() for layer in repr_layers]
    if len(reps) == 1:
        return reps[0]
    return torch.stack(reps, dim=1)


def _iter_cache_files(input_root: Path, subdir: str | None) -> list[Path]:
    root = input_root / subdir if subdir else input_root
    return sorted(root.rglob("*_protein.pt"))


def _process_files(
    files: list[Path],
    input_root: Path,
    output_root: Path,
    model_name: str,
    repr_layers_arg: str,
    device_str: str,
    overwrite: bool,
    tag: str,
) -> int:
    device = _resolve_device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, alphabet = _load_model(device, model_name)
    repr_layers = _parse_repr_layers(repr_layers_arg, model.num_layers)
    print(f"[{tag}] files={len(files)} repr_layers={repr_layers}")

    failures = []
    for idx, src in enumerate(files, start=1):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            print(f"[{tag} {idx}/{len(files)}] skip {rel}")
            continue
        try:
            obj = torch.load(src, map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                raise TypeError("input cache must be a dict with sequence metadata")
            sequence = obj.get("sequence")
            if not sequence:
                raise KeyError("missing sequence")
            emb = _extract_sequence(model, alphabet, str(sequence), repr_layers)
            if emb.shape[0] != len(sequence):
                raise ValueError(f"embedding length mismatch: {emb.shape[0]} vs {len(sequence)}")
            out = dict(obj)
            out["embeddings"] = emb.float()
            out["repr_layers"] = list(repr_layers)
            out["embedding_layout"] = "residue_layer_dim" if len(repr_layers) > 1 else "residue_dim"
            out["source_cache_path"] = str(src)
            out["model_name"] = model_name
            torch.save(out, dst)
            print(f"[{tag} {idx}/{len(files)}] ok   {rel}")
        except Exception as exc:
            failures.append((str(src), str(exc)))
            print(f"[{tag} {idx}/{len(files)}] fail {rel}: {exc}")

    if failures:
        fail_path = output_root / f"failures_{tag.replace(':', '_')}.tsv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w", encoding="utf-8") as f:
            f.write("cache_path\terror\n")
            for path, err in failures:
                f.write(f"{path}\t{err.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
        print(f"[{tag}] failures saved to {fail_path}")
        return 1
    return 0


def _worker_entry(*args):
    raise SystemExit(_process_files(*args))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract multi-layer ESM2 embeddings from existing sequence-aligned ESM cache."
    )
    parser.add_argument("--input", default="data/esm2_latents", help="Existing ESM cache root")
    parser.add_argument("--output", default="data/esm2_multilayer_latents", help="Output cache root")
    parser.add_argument("--subdir", default=None, help="Optional subdir/year under input root")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--repr-layers", default="2,6,12,18,24,30,33")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", default=None, help="Comma-separated devices for multiprocessing")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input cache root not found: {input_root}")

    files = _iter_cache_files(input_root, args.subdir)
    if args.max is not None:
        files = files[: args.max]
    if not files:
        print("No input cache files found.")
        return 0

    if args.devices:
        devices = _parse_devices(args.devices)
        splits = [[] for _ in devices]
        for i, path in enumerate(files):
            splits[i % len(devices)].append(path)
        ctx = mp.get_context("spawn")
        procs = []
        for dev, split in zip(devices, splits):
            if not split:
                continue
            p = ctx.Process(
                target=_worker_entry,
                args=(split, input_root, output_root, args.model, args.repr_layers, dev, args.overwrite, dev),
            )
            p.start()
            procs.append(p)
        rc = 0
        for p in procs:
            p.join()
            if p.exitcode != 0:
                rc = 1
        return rc

    return _process_files(
        files,
        input_root,
        output_root,
        args.model,
        args.repr_layers,
        args.device,
        args.overwrite,
        args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
