#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore original BioLiP chain IDs in PDBFixer receptor_fixed.cif files.

PDBFixer/OpenMM can rewrite chain IDs to A/B/C..., while BioLiP metadata and
downstream ESM/protein-feature alignment rely on original chain IDs.  This
script compares amino-acid sequences in:

  {biolip_root}/shared_receptors/{pdb_id}/receptor_full.cif
  {fixed_root}/{pdb_id}/receptor_fixed.cif

and renames chains in receptor_fixed.cif only when sequences prove a safe map.

Mapping policy:
  1. ordered_exact: same number of protein chains and every chain sequence
     matches at the same position.  This handles duplicate homomer chains while
     preserving PDBFixer chain order.
  2. unique_exact: fallback when chain order changed, but every fixed chain has
     exactly one unused full-chain sequence match.

Ambiguous, missing, or sequence-mismatched cases are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import difflib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

ALIGN_MIN_RATIO = 0.98
ALIGN_MAX_LEN_DELTA_FRAC = 0.05
ALIGN_MAX_LEN_DELTA_ABS = 10


@dataclass(frozen=True)
class ChainSeq:
    chain_id: str
    sequence: str
    n_residues: int
    residue_keys: Tuple[Tuple[int, str], ...]


def _chain_sequences(cif_path: Path) -> List[ChainSeq]:
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError(f"no models in {cif_path}")

    chains: List[ChainSeq] = []
    for chain in st[0]:
        seq: List[str] = []
        keys: List[Tuple[int, str]] = []
        for res in chain:
            if res.find_atom('CA', '*') is None:
                continue
            aa = STD_AA_3TO1.get(res.name.strip())
            if aa is None:
                continue
            seq.append(aa)
            keys.append((int(res.seqid.num), res.seqid.icode.strip()))
        if seq:
            chains.append(
                ChainSeq(
                    chain_id=chain.name.strip(),
                    sequence=''.join(seq),
                    n_residues=len(seq),
                    residue_keys=tuple(keys),
                )
            )
    return chains


def _ordered_exact_mapping(full: List[ChainSeq], fixed: List[ChainSeq]) -> Optional[Dict[str, str]]:
    if len(full) != len(fixed):
        return None
    mapping: Dict[str, str] = {}
    for f_chain, x_chain in zip(full, fixed):
        if f_chain.sequence != x_chain.sequence:
            return None
        mapping[x_chain.chain_id] = f_chain.chain_id
    return mapping


def _seq_close_enough(full_seq: str, fixed_seq: str) -> bool:
    if not full_seq or not fixed_seq:
        return False
    if full_seq == fixed_seq:
        return True
    max_len = max(len(full_seq), len(fixed_seq))
    len_delta = abs(len(full_seq) - len(fixed_seq))
    if len_delta > max(ALIGN_MAX_LEN_DELTA_ABS, int(max_len * ALIGN_MAX_LEN_DELTA_FRAC)):
        return False
    # PDBFixer usually adds a few missing residues.  In those common cases the
    # shorter sequence remains a contiguous subsequence of the longer one.
    if full_seq in fixed_seq or fixed_seq in full_seq:
        return True
    ratio = difflib.SequenceMatcher(None, full_seq, fixed_seq, autojunk=False).ratio()
    return ratio >= ALIGN_MIN_RATIO


def _ordered_aligned_mapping(full: List[ChainSeq], fixed: List[ChainSeq]) -> Optional[Dict[str, str]]:
    if len(full) != len(fixed):
        return None
    mapping: Dict[str, str] = {}
    for f_chain, x_chain in zip(full, fixed):
        if not _seq_close_enough(f_chain.sequence, x_chain.sequence):
            return None
        mapping[x_chain.chain_id] = f_chain.chain_id
    return mapping


def _unique_exact_mapping(full: List[ChainSeq], fixed: List[ChainSeq]) -> Optional[Dict[str, str]]:
    unused = set(range(len(full)))
    mapping: Dict[str, str] = {}
    for x_chain in fixed:
        matches = [
            i for i in unused
            if full[i].sequence == x_chain.sequence
        ]
        if len(matches) != 1:
            return None
        idx = matches[0]
        unused.remove(idx)
        mapping[x_chain.chain_id] = full[idx].chain_id
    if unused:
        return None
    return mapping


def infer_mapping(full_cif: Path, fixed_cif: Path) -> Tuple[Optional[Dict[str, str]], str, List[ChainSeq], List[ChainSeq]]:
    full = _chain_sequences(full_cif)
    fixed = _chain_sequences(fixed_cif)
    if not full:
        return None, "no_full_protein_chains", full, fixed
    if not fixed:
        return None, "no_fixed_protein_chains", full, fixed

    mapping = _ordered_exact_mapping(full, fixed)
    if mapping is not None:
        return mapping, "ordered_exact", full, fixed

    mapping = _ordered_aligned_mapping(full, fixed)
    if mapping is not None:
        return mapping, "ordered_aligned", full, fixed

    mapping = _unique_exact_mapping(full, fixed)
    if mapping is not None:
        return mapping, "unique_exact", full, fixed

    return None, "sequence_mismatch_or_ambiguous", full, fixed


def _write_structure_atomic(st: "gemmi.Structure", path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        suffix=".cif.tmp",
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        doc = st.make_mmcif_document()
        doc.write_file(str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def apply_mapping(fixed_cif: Path, mapping: Dict[str, str]) -> bool:
    st = gemmi.read_structure(str(fixed_cif))
    changed = False
    for chain in st[0]:
        current = chain.name.strip()
        target = mapping.get(current)
        if target and target != current:
            chain.name = target
            changed = True
    if changed:
        _write_structure_atomic(st, fixed_cif)
    return changed


def _mapping_text(mapping: Optional[Dict[str, str]]) -> str:
    if not mapping:
        return ""
    return ";".join(f"{k}->{v}" for k, v in mapping.items())


def _chain_text(chains: List[ChainSeq]) -> str:
    return ";".join(f"{c.chain_id}:{c.n_residues}" for c in chains)


def iter_pdb_ids(fixed_root: Path, only: Optional[Iterable[str]] = None) -> List[str]:
    if only is not None:
        return sorted({x.strip().lower() for x in only if x.strip()})
    return sorted(
        p.name.lower()
        for p in fixed_root.iterdir()
        if p.is_dir() and (p / "receptor_fixed.cif").exists()
    )


def process_pdb_id(
    pdb_id: str,
    biolip_root: Path,
    fixed_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> Dict[str, object]:
    full_cif = biolip_root / "shared_receptors" / pdb_id / "receptor_full.cif"
    fixed_cif = fixed_root / pdb_id / "receptor_fixed.cif"
    if not full_cif.exists() or not fixed_cif.exists():
        return {
            "pdb_id": pdb_id, "status": "missing", "method": "",
            "changed": False, "mapping": "", "full_chains": "",
            "fixed_chains": "", "error": "missing full or fixed cif",
        }

    try:
        mapping, method, full, fixed = infer_mapping(full_cif, fixed_cif)
        if mapping is None:
            return {
                "pdb_id": pdb_id, "status": "skipped", "method": method,
                "changed": False, "mapping": "",
                "full_chains": _chain_text(full),
                "fixed_chains": _chain_text(fixed),
                "error": method,
            }

        changed_needed = any(k != v for k, v in mapping.items())
        changed = False
        if changed_needed and not dry_run:
            if not overwrite:
                raise RuntimeError("refusing to write without --overwrite")
            changed = apply_mapping(fixed_cif, mapping)

        return {
            "pdb_id": pdb_id,
            "status": "mapped" if changed_needed else "unchanged",
            "method": method,
            "changed": changed_needed if dry_run else changed,
            "mapping": _mapping_text(mapping),
            "full_chains": _chain_text(full),
            "fixed_chains": _chain_text(fixed),
            "error": "",
        }
    except Exception as exc:
        return {
            "pdb_id": pdb_id, "status": "error", "method": "",
            "changed": False, "mapping": "", "full_chains": "",
            "fixed_chains": "", "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore original chain IDs in BioLiP fixed receptor CIFs."
    )
    parser.add_argument("--biolip-root", required=True, help="BioLiP root with shared_receptors/")
    parser.add_argument("--fixed-root", required=True, help="Root with {pdb_id}/receptor_fixed.cif")
    parser.add_argument("--report", required=True, help="TSV report path")
    parser.add_argument("--pdb-id", action="append", default=None, help="Limit to one pdb_id; may repeat")
    parser.add_argument("--max", type=int, default=None, help="Limit number of structures")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify files")
    parser.add_argument("--overwrite", action="store_true", help="Allow rewriting changed receptor_fixed.cif")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    args = parser.parse_args()

    biolip_root = Path(args.biolip_root).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_ids = iter_pdb_ids(fixed_root, args.pdb_id)
    if args.max is not None:
        pdb_ids = pdb_ids[:args.max]

    workers = max(1, int(args.workers))
    counts = {"mapped": 0, "unchanged": 0, "skipped": 0, "missing": 0, "error": 0}
    with report_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pdb_id", "status", "method", "changed", "mapping",
                "full_chains", "fixed_chains", "error",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        if workers == 1:
            rows = (
                process_pdb_id(pdb_id, biolip_root, fixed_root, args.dry_run, args.overwrite)
                for pdb_id in pdb_ids
            )
            for row in rows:
                status = str(row["status"])
                counts[status] += 1
                writer.writerow(row)
        else:
            with futures.ProcessPoolExecutor(max_workers=workers) as ex:
                rows = ex.map(
                    process_pdb_id,
                    pdb_ids,
                    [biolip_root] * len(pdb_ids),
                    [fixed_root] * len(pdb_ids),
                    [args.dry_run] * len(pdb_ids),
                    [args.overwrite] * len(pdb_ids),
                    chunksize=8,
                )
                for row in rows:
                    status = str(row["status"])
                    counts[status] += 1
                    writer.writerow(row)

    print(
        "Done:",
        " ".join(f"{k}={v}" for k, v in counts.items()),
        f"report={report_path}",
    )
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
