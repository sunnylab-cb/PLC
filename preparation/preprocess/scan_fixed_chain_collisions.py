#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan fixed_receptors for OpenMM PDBxFile keepIds=False chain-collision damage.

OpenMM PDBxFile.writeFile(..., keepIds=False) regenerates chain IDs as
A, B, ..., Z, A, B, ... by chain order.  For receptors with more than 26
protein chains, this can merge chain 0 and 26 into one output chain A, chain
1 and 27 into B, etc.  This scanner detects fixed CIFs whose chains match that
concatenation pattern.

It is read-only: it writes a TSV report and a text list of PDB IDs to reprocess.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import gemmi
except ImportError as exc:  # pragma: no cover
    raise SystemExit("gemmi is required: conda install -c conda-forge gemmi") from exc


STD_AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K',
    'TPO': 'T', 'SEP': 'S', 'PTR': 'Y',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H',
    'ASH': 'D', 'GLH': 'E', 'LYN': 'K',
    'HIC': 'H', 'KCX': 'K', 'FME': 'M',
    'CSO': 'C', 'CSD': 'C', 'CME': 'C', 'OCS': 'C',
    'NEP': 'H', 'MHS': 'H', 'LLP': 'K', 'MLZ': 'K', 'M3L': 'K',
    'AGM': 'R', '2MR': 'R', 'HYP': 'P', 'P1L': 'C',
    'DYA': 'D', 'MGN': 'Q', 'SMC': 'C', 'GL3': 'G',
    'TRX': 'C', 'PCA': 'Q', '0TD': 'D', '33W': 'W',
}


@dataclass(frozen=True)
class ChainSeq:
    chain_id: str
    sequence: str
    n_residues: int


def _chain_sequences(cif_path: Path) -> List[ChainSeq]:
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError(f"no models in {cif_path}")
    chains: List[ChainSeq] = []
    for chain in st[0]:
        seq: List[str] = []
        for res in chain:
            if res.find_atom('CA', '*') is None:
                continue
            aa = STD_AA_3TO1.get(res.name.strip())
            if aa is None:
                continue
            seq.append(aa)
        if seq:
            chains.append(ChainSeq(chain.name.strip(), ''.join(seq), len(seq)))
    return chains


def _chain_text(chains: List[ChainSeq]) -> str:
    return ";".join(f"{c.chain_id}:{c.n_residues}" for c in chains)


def _generated_chain_id(chain_index: int) -> str:
    return chr(ord('A') + chain_index % 26)


def _detect_collision_groups(full: List[ChainSeq], fixed: List[ChainSeq]) -> List[str]:
    fixed_by_id = {c.chain_id: c for c in fixed}
    groups = []
    for bucket_idx in range(26):
        members = [
            full[i] for i in range(bucket_idx, len(full), 26)
        ]
        if len(members) < 2:
            continue
        gen_id = _generated_chain_id(bucket_idx)
        fixed_chain = fixed_by_id.get(gen_id)
        if fixed_chain is None:
            continue
        concat_seq = ''.join(c.sequence for c in members)
        if fixed_chain.sequence == concat_seq:
            member_text = "+".join(f"{c.chain_id}:{c.n_residues}" for c in members)
            groups.append(f"{gen_id}={member_text}->{fixed_chain.n_residues}")
    return groups


def _scan_one(args: Tuple[str, str, str]) -> dict:
    pdb_id, biolip_root_s, fixed_root_s = args
    biolip_root = Path(biolip_root_s)
    fixed_root = Path(fixed_root_s)
    full_cif = biolip_root / "shared_receptors" / pdb_id / "receptor_full.cif"
    fixed_cif = fixed_root / pdb_id / "receptor_fixed.cif"
    if not full_cif.exists() or not fixed_cif.exists():
        return {
            "pdb_id": pdb_id,
            "status": "missing",
            "full_n": 0,
            "fixed_n": 0,
            "collision_groups": "",
            "full_chains": "",
            "fixed_chains": "",
            "error": "missing full or fixed cif",
        }
    try:
        full = _chain_sequences(full_cif)
        fixed = _chain_sequences(fixed_cif)
        groups = _detect_collision_groups(full, fixed)
        status = "collision" if groups else "ok"
        return {
            "pdb_id": pdb_id,
            "status": status,
            "full_n": len(full),
            "fixed_n": len(fixed),
            "collision_groups": ";".join(groups),
            "full_chains": _chain_text(full),
            "fixed_chains": _chain_text(fixed),
            "error": "",
        }
    except Exception as exc:
        return {
            "pdb_id": pdb_id,
            "status": "error",
            "full_n": 0,
            "fixed_n": 0,
            "collision_groups": "",
            "full_chains": "",
            "fixed_chains": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def iter_pdb_ids(fixed_root: Path, only: Optional[Iterable[str]] = None) -> List[str]:
    if only is not None:
        return sorted({x.strip().lower() for x in only if x.strip()})
    return sorted(
        p.name.lower()
        for p in fixed_root.iterdir()
        if p.is_dir() and (p / "receptor_fixed.cif").exists()
    )


FIELDS = [
    "pdb_id",
    "status",
    "full_n",
    "fixed_n",
    "collision_groups",
    "full_chains",
    "fixed_chains",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan fixed receptor CIFs for chain-ID collision damage."
    )
    parser.add_argument("--biolip-root", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--delete-list", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--pdbid-list-file", default=None)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    biolip_root = Path(args.biolip_root).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    report = Path(args.report).resolve()
    delete_list = Path(args.delete_list).resolve()

    only = None
    if args.pdbid_list_file:
        only = Path(args.pdbid_list_file).read_text().splitlines()
    pdb_ids = iter_pdb_ids(fixed_root, only)
    if args.max:
        pdb_ids = pdb_ids[:args.max]

    report.parent.mkdir(parents=True, exist_ok=True)
    delete_list.parent.mkdir(parents=True, exist_ok=True)

    total = len(pdb_ids)
    print(f"Scanning {total:,} receptors with workers={args.workers}", flush=True)
    print(f"Report: {report}", flush=True)
    print(f"Delete list: {delete_list}", flush=True)

    n_collision = n_ok = n_missing = n_error = 0
    with open(report, "w") as rf, open(delete_list, "w") as df:
        rf.write("\t".join(FIELDS) + "\n")
        tasks = [(pdb_id, str(biolip_root), str(fixed_root)) for pdb_id in pdb_ids]
        with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_to_pdb = {ex.submit(_scan_one, t): t[0] for t in tasks}
            for idx, fut in enumerate(futures.as_completed(future_to_pdb), 1):
                row = fut.result()
                status = row["status"]
                if status == "collision":
                    n_collision += 1
                    df.write(f"{row['pdb_id']}\n")
                elif status == "ok":
                    n_ok += 1
                elif status == "missing":
                    n_missing += 1
                else:
                    n_error += 1
                rf.write("\t".join(str(row.get(k, "")).replace("\t", " ") for k in FIELDS) + "\n")
                if idx % 1000 == 0 or idx == total:
                    print(
                        f"[{idx:,}/{total:,}] collision={n_collision:,} ok={n_ok:,} "
                        f"missing={n_missing:,} error={n_error:,}",
                        flush=True,
                    )

    print(
        f"Done: collision={n_collision:,} ok={n_ok:,} "
        f"missing={n_missing:,} error={n_error:,}",
        flush=True,
    )
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
