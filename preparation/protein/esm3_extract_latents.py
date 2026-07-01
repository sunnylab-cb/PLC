#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-residue embeddings from ESM3 and align them to PDB residue indices.
"""

from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from pathlib import Path

import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils import residue_constants
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

from biotite.structure.io.pdb import PDBFile
import biotite.structure as bs


def _set_default_cache():
    base_dir = Path(__file__).resolve().parent
    os.environ.setdefault("HF_HUB_CACHE", str(base_dir / "ESM" / "hub"))


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        if device_str == "cuda":
            device_str = "cuda:0"
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
            if idx >= torch.cuda.device_count():
                raise ValueError(f"Invalid CUDA device index: {device_str}")
        return torch.device(device_str)
    return torch.device("cpu")


def _load_model(device: torch.device, model_name: str) -> ESM3:
    model = ESM3.from_pretrained(model_name, device=device)
    model.eval()
    return model


def _load_residue_name_map(pdb_path: Path) -> dict[tuple[str, int, str], str]:
    atom_array = PDBFile.read(pdb_path).get_structure(model=1, extra_fields=["b_factor"])
    atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]

    # Map (chain_id, res_id, ins_code) -> 3-letter residue name from PDB.
    name_map: dict[tuple[str, int, str], str] = {}
    for res in bs.residue_iter(atom_array):
        first = res[0]
        key = (first.chain_id, int(first.res_id), str(first.ins_code).strip())
        name_map[key] = first.res_name
    return name_map


def _fallback_res_name_3(aa_1: str) -> str:
    return residue_constants.restype_1to3.get(aa_1, "UNK")


def _extract_complex(model: ESM3, pdb_path: Path) -> dict:
    complex_ = ProteinComplex.from_pdb(pdb_path)
    prot = ESMProtein.from_protein_complex(complex_)
    name_map = _load_residue_name_map(pdb_path)

    out = model.logits(model.encode(prot), LogitsConfig(return_embeddings=True))
    emb = out.embeddings[0][1:-1]  # remove BOS/EOS

    mask = complex_.residue_index != -1  # drop chain-break positions
    mask_t = torch.from_numpy(mask).to(emb.device)
    emb = emb[mask_t]

    # Keep atom coordinates aligned to embeddings.
    atom37 = torch.from_numpy(complex_.atom37_positions).float()[mask_t.cpu()]
    atom37_mask = torch.from_numpy(complex_.atom37_mask)[mask_t.cpu()]

    seq = prot.sequence
    aa = [seq[i] for i, keep in enumerate(mask) if keep]
    chain_num = complex_.chain_id[mask]
    res_idx = complex_.residue_index[mask]
    ins_code = complex_.insertion_code[mask]
    entity_id = complex_.entity_id[mask]

    chain_lookup = complex_.metadata.chain_lookup
    chain_id = [chain_lookup[int(x)] for x in chain_num]

    res_name_3 = []
    for c, r, ins, a in zip(chain_id, res_idx, ins_code, aa):
        key = (c, int(r), str(ins).strip())
        res_name_3.append(name_map.get(key, _fallback_res_name_3(a)))

    if emb.shape[0] != len(res_idx):
        raise ValueError(
            f"Embedding length mismatch: {emb.shape[0]} vs {len(res_idx)}"
        )

    return {
        "embeddings": emb.cpu(),
        "chain_id": chain_id,
        "chain_id_numeric": chain_num.tolist(),
        "residue_index": res_idx.tolist(),
        "insertion_code": [s.strip() for s in ins_code.tolist()],
        "residue_name_1": aa,
        "residue_name_3": res_name_3,
        "sequence": "".join(aa),
        "entity_id": entity_id.tolist(),
        "atom37_positions": atom37,
        "atom37_mask": atom37_mask,
        "pdb_path": str(pdb_path),
        "mode": "complex",
    }


def _extract_chain(model: ESM3, pdb_path: Path, chain_id: str) -> dict:
    chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id)
    prot = ESMProtein.from_protein_chain(chain)
    name_map = _load_residue_name_map(pdb_path)

    out = model.logits(model.encode(prot), LogitsConfig(return_embeddings=True))
    emb = out.embeddings[0][1:-1]  # remove BOS/EOS

    if emb.shape[0] != len(chain.sequence):
        raise ValueError(
            f"Embedding length mismatch: {emb.shape[0]} vs {len(chain.sequence)}"
        )

    atom37 = torch.from_numpy(chain.atom37_positions).float()
    atom37_mask = torch.from_numpy(chain.atom37_mask)

    res_name_3 = []
    for r, ins, a in zip(chain.residue_index, chain.insertion_code, chain.sequence):
        key = (chain.chain_id, int(r), str(ins).strip())
        res_name_3.append(name_map.get(key, _fallback_res_name_3(a)))

    entity_id = (
        [int(chain.entity_id)] * len(chain.sequence)
        if chain.entity_id is not None
        else None
    )

    return {
        "embeddings": emb.cpu(),
        "chain_id": [chain.chain_id] * len(chain.sequence),
        "chain_id_numeric": None,
        "residue_index": chain.residue_index.tolist(),
        "insertion_code": [s.strip() for s in chain.insertion_code.tolist()],
        "residue_name_1": list(chain.sequence),
        "residue_name_3": res_name_3,
        "sequence": chain.sequence,
        "entity_id": entity_id,
        "atom37_positions": atom37,
        "atom37_mask": atom37_mask,
        "pdb_path": str(pdb_path),
        "mode": "chain",
    }


def _iter_pdbs(input_root: Path, pattern: str) -> list[Path]:
    return sorted(input_root.rglob(pattern))


def _parse_devices(devices_str: str) -> list[str]:
    devices = []
    for part in devices_str.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            devices.append(f"cuda:{p}")
        elif p == "cuda":
            devices.append("cuda:0")
        else:
            devices.append(p)
    return devices


def _process_files(
    pdb_files: list[Path],
    input_root: Path,
    output_root: Path,
    model_name: str,
    device_str: str,
    chain: str | None,
    overwrite: bool,
    tag: str,
) -> int:
    _set_default_cache()
    device = _resolve_device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = _load_model(device, model_name)

    total = len(pdb_files)
    failures = []

    for idx, pdb_path in enumerate(pdb_files, start=1):
        rel = pdb_path.relative_to(input_root)
        out_path = (output_root / rel).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"[{tag} {idx}/{total}] skip {rel}")
            continue

        try:
            if chain:
                data = _extract_chain(model, pdb_path, chain)
            else:
                data = _extract_complex(model, pdb_path)
            data["model_name"] = model_name
            torch.save(data, out_path)
            print(f"[{tag} {idx}/{total}] ok   {rel}")
        except Exception as exc:
            failures.append((str(pdb_path), str(exc)))
            print(f"[{tag} {idx}/{total}] fail {rel}: {exc}")

    if failures:
        print(f"\nFailures ({tag}):")
        for path, err in failures:
            print(f"- {path}: {err}")
        return 1

    return 0


def _worker_entry(
    pdb_files: list[Path],
    input_root: Path,
    output_root: Path,
    model_name: str,
    device_str: str,
    chain: str | None,
    overwrite: bool,
    tag: str,
):
    code = _process_files(
        pdb_files, input_root, output_root, model_name, device_str, chain, overwrite, tag
    )
    raise SystemExit(code)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-residue ESM3 embeddings aligned to PDB residue indices."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="New",
        help="Root directory containing PDB files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_protein.pdb",
        help="Glob pattern to search under input root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="esm3_latents",
        help="Output root directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm3-open",
        help="ESM3 model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated device list for multi-process, e.g. cuda:0,cuda:1 or 0,1.",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="If set, only extract a single chain (drops inter-chain geometry).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of PDB files to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    pdb_files = _iter_pdbs(input_root, args.pattern)
    if args.max is not None:
        pdb_files = pdb_files[: args.max]

    if not pdb_files:
        print("No PDB files found.")
        return 0

    if args.devices:
        devices = _parse_devices(args.devices)
        if not devices:
            raise ValueError("No valid devices specified.")
        if any(d.startswith("cuda") for d in devices) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for --devices.")
        for d in devices:
            if d.startswith("cuda:"):
                idx = int(d.split(":")[1])
                if idx >= torch.cuda.device_count():
                    raise ValueError(f"Invalid CUDA device index: {d}")

        splits = [[] for _ in devices]
        for i, pdb_path in enumerate(pdb_files):
            splits[i % len(devices)].append(pdb_path)

        ctx = mp.get_context("spawn")
        procs = []
        for dev, files in zip(devices, splits):
            if not files:
                continue
            tag = dev
            p = ctx.Process(
                target=_worker_entry,
                args=(
                    files,
                    input_root,
                    output_root,
                    args.model,
                    dev,
                    args.chain,
                    args.overwrite,
                    tag,
                ),
            )
            p.start()
            procs.append(p)

        rc = 0
        for p in procs:
            p.join()
            if p.exitcode != 0:
                rc = 1
        return rc

    tag = args.device
    return _process_files(
        pdb_files,
        input_root,
        output_root,
        args.model,
        args.device,
        args.chain,
        args.overwrite,
        tag,
    )


if __name__ == "__main__":
    raise SystemExit(main())
