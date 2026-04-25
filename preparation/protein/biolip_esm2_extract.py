#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip ESM2 per-residue embedding extractor.

Unlike the PDBBind version (which reads PDB via ProteinChain.from_pdb),
this script reads observed residues directly from receptor_full.cif using
gemmi, then runs ESM2 on the observed amino-acid sequence.

Residue alignment contract
──────────────────────────
Output 'embeddings'[i] corresponds to observed residue i, identified by:
    (chain_id[i], residue_index[i], insertion_code[i])

Insertion codes are preserved exactly as they appear in the CIF
(e.g. '' for ordinary residues, 'A'/'B'... for insertion residues).
This triple is the canonical key for aligning ESM2 embeddings with
any downstream coordinate array.

Usage
─────
# single GPU
python biolip_esm2_extract.py \
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \
    --output      /home/husrcf/Code/ProtBind/BioLip/DOCs/esm2_biolip \
    --device cuda:0

# multi-GPU  (each process owns one GPU)
python biolip_esm2_extract.py \
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \
    --output      /home/husrcf/Code/ProtBind/BioLip/DOCs/esm2_biolip \
    --devices 0,1,2,3

# subset only
python biolip_esm2_extract.py ... --bundle_list_file missing.txt

# smoke-test (first 10 bundles, verbose residue table)
python biolip_esm2_extract.py ... --max 10 --smoke_test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# ── ESM2 ─────────────────────────────────────────────────────────────────────
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

# ── gemmi ─────────────────────────────────────────────────────────────────────
try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────

ESM2_MAX_LEN = 1022          # ESM2 hard limit (excl. BOS/EOS)
MODEL_NAME   = "esm2_t33_650M_UR50D"

STD_AA_3TO1: dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # common modified residues → nearest standard
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K',
    'TPO': 'T', 'SEP': 'S', 'PTR': 'Y',
    'HIC': 'H', 'KCX': 'K', 'FME': 'M',
    'CSO': 'C', 'CSD': 'C', 'CME': 'C',
    'OCS': 'C', 'NEP': 'H', 'MHS': 'H',
    'LLP': 'K', 'MLZ': 'K', 'M3L': 'K',
    'AGM': 'R', '2MR': 'R',
    'HYP': 'P', 'P1L': 'C',
    'DYA': 'D', 'MGN': 'Q',
    'SMC': 'C', 'GL3': 'G',
    'TRX': 'C', 'ELY': 'K',
    '0TD': 'D', '33W': 'W',
    'GL3': 'G', 'PCA': 'Q',
}


# ── path safety ───────────────────────────────────────────────────────────────

def _validate_bundle_name(bundle_name: str, biolip_root: Path, output_root: Path) -> None:
    """
    Reject bundle names that would escape the intended filesystem roots.

    Checks:
    - No path separators or '..' components (fast lexical guard).
    - Resolved input path stays under biolip_root/data.
    - Resolved output path stays under output_root.
    """
    if os.sep in bundle_name or '/' in bundle_name or '..' in bundle_name.split('/'):
        raise ValueError(
            f'Bundle name {bundle_name!r} contains path separators or ".."; '
            f'refusing to process.'
        )

    data_root = (biolip_root / 'data').resolve()
    candidate_in  = (data_root / bundle_name).resolve()
    candidate_out = (output_root / f'{bundle_name}_esm2.pt').resolve()

    if not str(candidate_in).startswith(str(data_root) + os.sep) and \
            candidate_in != data_root:
        raise ValueError(
            f'Bundle name {bundle_name!r} resolves to {candidate_in}, '
            f'which is outside {data_root}.'
        )
    if not str(candidate_out).startswith(str(output_root.resolve()) + os.sep):
        raise ValueError(
            f'Bundle name {bundle_name!r} output path {candidate_out} '
            f'is outside {output_root}.'
        )


# ── helper: load model ────────────────────────────────────────────────────────

def _resolve_device(s: str) -> torch.device:
    if s == 'cuda':
        s = 'cuda:0'
    if s.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'Device {s!r} was requested but CUDA is not available. '
                f'Use --device cpu or --devices cpu to run on CPU explicitly.'
            )
        return torch.device(s)
    return torch.device('cpu')


def _load_model(device: torch.device):
    if not ESM_AVAILABLE:
        raise ImportError('esm not installed. pip install fair-esm')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return model, alphabet


# ── core: parse CIF chain → residue table ────────────────────────────────────

def _parse_cif_chain(cif_path: Path, chain_id: str) -> Tuple[
        List[int], List[str], List[str], List[str]]:
    """
    Return (res_index, ins_codes, res_name_3, res_name_1) for all
    polymer residues (has CA) in the given chain, in order.

    Insertion-code handling
    ───────────────────────
    gemmi stores the insertion code in res.seqid.icode (a single char).
    We normalise it to '' for the common case (space or null) and keep
    the letter as-is otherwise ('A', 'B', …).

    Example: residues 100, 100A, 100B → indices [100, 100, 100]
             ins_codes                → ['', 'A', 'B']
    The pair (res_index, ins_code) is globally unique within a chain.
    """
    if not GEMMI_AVAILABLE:
        raise ImportError('gemmi not installed. pip install gemmi')

    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    try:
        chain = model[chain_id]
    except Exception:
        available = [c.name for c in model]
        raise ValueError(
            f'Chain {chain_id!r} not found in {cif_path.name}. '
            f'Available: {available}'
        )

    res_index  : List[int] = []
    ins_codes  : List[str] = []
    res_name_3 : List[str] = []
    res_name_1 : List[str] = []

    for res in chain:
        if res.find_atom('CA', '*') is None:
            continue  # non-polymer / pure ligand residue
        seqnum = res.seqid.num
        icode  = res.seqid.icode.strip()   # '' or 'A'/'B'/...

        r3 = res.name
        r1 = STD_AA_3TO1.get(r3, 'X')

        res_index .append(seqnum)
        ins_codes .append(icode)
        res_name_3.append(r3)
        res_name_1.append(r1)

    if not res_index:
        raise ValueError(
            f'No polymer residues with CA found in chain {chain_id!r} '
            f'of {cif_path.name}'
        )

    return res_index, ins_codes, res_name_3, res_name_1


# ── core: extract ESM2 embeddings ────────────────────────────────────────────

def _run_esm2(model, alphabet, sequence: str) -> torch.Tensor:
    """
    Return per-residue embeddings [N, 1280].
    Raises ValueError for sequences longer than ESM2_MAX_LEN — callers must
    handle overlength chains explicitly rather than silently losing residues.
    """
    if len(sequence) > ESM2_MAX_LEN:
        raise ValueError(
            f'Sequence length {len(sequence)} exceeds ESM2 limit {ESM2_MAX_LEN}. '
            f'Overlength chains are not silently truncated; handle via windowed '
            f'extraction or exclude this bundle.'
        )

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([('protein', sequence)])
    tokens = tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers])

    emb = out['representations'][model.num_layers][0, 1:-1]   # strip BOS/EOS
    assert emb.shape[0] == len(sequence), (
        f'Embedding length {emb.shape[0]} ≠ sequence length {len(sequence)}'
    )
    return emb.cpu()


# ── per-bundle extraction ─────────────────────────────────────────────────────

def extract_bundle(
    bundle_name : str,
    biolip_root : Path,
    output_root : Path,
    model,
    alphabet,
    overwrite   : bool = False,
    smoke_test  : bool = False,
) -> Tuple[bool, str]:
    """
    Extract ESM2 embeddings for one bundle.
    Returns (success, message).
    """
    # ── path safety (must run before any filesystem access) ──────────────────
    try:
        _validate_bundle_name(bundle_name, biolip_root, output_root)
    except ValueError as e:
        return False, f'{bundle_name}: invalid bundle name: {e}'

    out_path = output_root / f'{bundle_name}_esm2.pt'
    if out_path.exists() and not overwrite:
        return True, f'{bundle_name}: SKIP'

    # ── read protein_meta ──────────────────────────────────────────────────
    meta_path = biolip_root / 'data' / bundle_name / 'protein_meta.json'
    if not meta_path.exists():
        return False, f'{bundle_name}: protein_meta.json missing'

    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return False, f'{bundle_name}: protein_meta.json parse error: {e}'

    pdb_id        = meta.get('pdb_id', '')
    protein_chain = meta.get('protein_chain', '')
    if not pdb_id or not protein_chain:
        return False, f'{bundle_name}: missing pdb_id or protein_chain in meta'

    # ── validate pdb_id from metadata stays within shared_receptors/ ──────
    # A malformed protein_meta.json could point at an arbitrary path and cause
    # embeddings to be computed for the wrong structure (silent data corruption).
    receptors_root    = (biolip_root / 'shared_receptors').resolve()
    candidate_cif_dir = (receptors_root / pdb_id).resolve()
    if not str(candidate_cif_dir).startswith(str(receptors_root) + os.sep):
        return False, (
            f'{bundle_name}: pdb_id {pdb_id!r} from metadata resolves to '
            f'{candidate_cif_dir}, which escapes shared_receptors root'
        )

    # ── locate CIF ────────────────────────────────────────────────────────
    cif_path = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
    if not cif_path.exists():
        return False, f'{bundle_name}: receptor_full.cif not found at {cif_path}'

    # ── parse CIF → residue table ─────────────────────────────────────────
    try:
        res_index, ins_codes, res_name_3, res_name_1 = _parse_cif_chain(
            cif_path, protein_chain
        )
    except Exception as e:
        return False, f'{bundle_name}: CIF parse error: {e}'

    sequence = ''.join(res_name_1)
    n_obs    = len(sequence)

    # ── smoke-test: print residue table ───────────────────────────────────
    if smoke_test:
        print(f'\n  [{bundle_name}] {pdb_id} chain={protein_chain}  '
              f'observed={n_obs}  seq={sequence[:40]}{"…" if n_obs>40 else ""}')
        # show first 5 residues with insertion codes highlighted
        has_icode = any(c for c in ins_codes)
        if has_icode:
            icode_rows = [(i, res_index[i], ins_codes[i], res_name_3[i])
                          for i in range(n_obs) if ins_codes[i]]
            print(f'  insertion-code residues ({len(icode_rows)}):')
            for pos, seqnum, icode, r3 in icode_rows[:10]:
                print(f'    pos={pos:4d}  seqid={seqnum}{icode}  resname={r3}')
        else:
            # show boundary residues to verify numbering
            sample = list(zip(res_index[:3], ins_codes[:3], res_name_3[:3])) + \
                     list(zip(res_index[-3:], ins_codes[-3:], res_name_3[-3:]))
            print(f'  residue boundaries: {sample}')

    # ── run ESM2 ──────────────────────────────────────────────────────────
    try:
        embeddings = _run_esm2(model, alphabet, sequence)
    except Exception as e:
        return False, f'{bundle_name}: ESM2 error: {e}'

    # ── atomic save (tmp → rename) ─────────────────────────────────────────
    # Write to a sibling .tmp file first so that an interrupted save never
    # leaves a partial artifact at the real path.  os.replace is atomic on
    # POSIX when src and dst share the same filesystem.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp file per write — prevents races when the same bundle is
    # scheduled twice or two processes share one output directory.
    tmp_path = out_path.parent / f'.{bundle_name}_{uuid.uuid4().hex}.tmp'
    result = {
        # alignment key: (chain_id, residue_index, insertion_code) → embedding row
        'embeddings'     : embeddings,                         # [N_obs, 1280]  float32
        'chain_id'       : [protein_chain] * len(res_index),   # list[str]
        'residue_index'  : res_index,                          # list[int]
        'insertion_code' : ins_codes,                          # list[str], '' or 'A'/'B'
        'residue_name_1' : res_name_1,                         # list[str]
        'residue_name_3' : res_name_3,                         # list[str]
        'sequence'       : sequence,                            # str
        'model_name'     : MODEL_NAME,
        'bundle_name'    : bundle_name,
        'pdb_id'         : pdb_id,
        'protein_chain'  : protein_chain,
    }
    try:
        torch.save(result, tmp_path)
        os.replace(tmp_path, out_path)   # atomic on POSIX
    except Exception as e:
        # Clean up orphaned temp file if possible, then report failure
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f'{bundle_name}: save error: {e}'
    return True, f'{bundle_name}: OK  ({len(sequence)} res, emb={tuple(embeddings.shape)})'


# ── worker entry (runs inside subprocess for multi-GPU) ───────────────────────

def _worker_entry(
    bundle_names : List[str],
    biolip_root  : Path,
    output_root  : Path,
    device_str   : str,
    overwrite    : bool,
    tag          : str,
    smoke_test   : bool,
    failure_tsv  : Path,
    cpu_threads  : int = 0,
):
    device = _resolve_device(device_str)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    elif cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        print(f'[{tag}] CPU threads set to {cpu_threads}', flush=True)

    model, alphabet = _load_model(device)

    failures: List[Tuple[str, str]] = []
    total = len(bundle_names)

    for idx, bundle_name in enumerate(bundle_names, 1):
        ok, msg = extract_bundle(
            bundle_name, biolip_root, output_root,
            model, alphabet, overwrite, smoke_test
        )
        print(f'[{tag} {idx}/{total}] {msg}', flush=True)
        if not ok:
            failures.append((bundle_name, msg))

    if failures:
        failure_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_tsv, 'w') as f:
            f.write('bundle_name\terror\n')
            for bn, err in failures:
                f.write(f'{bn}\t{err.replace(chr(9)," ")}\n')
        print(f'[{tag}] {len(failures)} failures saved to {failure_tsv}')

    raise SystemExit(1 if failures else 0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_devices(s: str) -> List[str]:
    out = []
    for p in s.split(','):
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            out.append(f'cuda:{p}')
        elif p == 'cuda':
            out.append('cuda:0')
        else:
            out.append(p)   # 'cpu' or explicit 'cuda:N'
    return out


def _validate_devices(devices: List[str]) -> None:
    """Raise ValueError for invalid CUDA indices; CPU is always valid."""
    cuda_devices = [d for d in devices if d.startswith('cuda')]
    if cuda_devices and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available but CUDA devices were requested.')
    for d in cuda_devices:
        if ':' in d:
            idx = int(d.split(':')[1])
            if idx >= torch.cuda.device_count():
                raise ValueError(
                    f'Invalid CUDA device index {idx} '
                    f'(only {torch.cuda.device_count()} GPU(s) detected).'
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Extract ESM2 embeddings for BioLip bundles (CIF-native).'
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip dataset root (contains data/ and shared_receptors/)')
    parser.add_argument('--output', required=True,
                        help='Output directory for .pt files')
    parser.add_argument('--device', default='cuda',
                        help='Device for single-process run: cuda, cuda:0, cpu  (default: cuda)')
    parser.add_argument('--devices', default=None,
                        help=(
                            'Comma-separated devices for multi-process, one subprocess per entry. '
                            'Examples: "0,1"  "0,1,cpu"  "cuda:0,cuda:1,cpu". '
                            'CPU process uses --cpu_threads threads.'
                        ))
    parser.add_argument('--cpu_threads', type=int, default=32,
                        help='torch CPU threads for the CPU worker subprocess (default: 32)')
    parser.add_argument('--bundle_list_file', default=None,
                        help='Text file listing bundle names (one per line); default: all')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N bundles (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-extract even if .pt already exists')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Print residue alignment table for each bundle (use with --max 5)')
    args = parser.parse_args()

    biolip_root = Path(args.biolip_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not ESM_AVAILABLE:
        print('ERROR: fair-esm not installed.  pip install fair-esm')
        return 1
    if not GEMMI_AVAILABLE:
        print('ERROR: gemmi not installed.  pip install gemmi')
        return 1

    # ── collect bundle list ────────────────────────────────────────────────
    if args.bundle_list_file:
        bundle_names = [
            l.strip() for l in Path(args.bundle_list_file).read_text().splitlines()
            if l.strip()
        ]
        before_dedup = len(bundle_names)
        bundle_names = list(dict.fromkeys(bundle_names))  # deduplicate, preserve order
        if len(bundle_names) < before_dedup:
            print(f'Bundle list from file: {before_dedup:,} entries, '
                  f'{before_dedup - len(bundle_names):,} duplicates removed → {len(bundle_names):,}')
        else:
            print(f'Bundle list from file: {len(bundle_names):,}')
    else:
        data_dir     = biolip_root / 'data'
        bundle_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        print(f'Auto-discovered bundles: {len(bundle_names):,}')

    if args.max:
        bundle_names = bundle_names[:args.max]
        print(f'Limiting to first {len(bundle_names)} bundles.')

    total = len(bundle_names)
    print(f'Total to process: {total:,}')
    print(f'Output: {output_root}')

    # ── multi-device ──────────────────────────────────────────────────────
    if args.devices:
        devices = _parse_devices(args.devices)
        if not devices:
            print('ERROR: no valid devices in --devices')
            return 1

        try:
            _validate_devices(devices)
        except (RuntimeError, ValueError) as e:
            print(f'ERROR: {e}')
            return 1

        print(f'Devices: {devices}')

        # split bundles round-robin across devices
        splits = [[] for _ in devices]
        for i, bn in enumerate(bundle_names):
            splits[i % len(devices)].append(bn)

        ctx   = mp.get_context('spawn')
        procs = []
        for dev, chunk in zip(devices, splits):
            if not chunk:
                continue
            tag         = dev
            failure_tsv = output_root / f'failures_{dev.replace(":","_")}.tsv'
            cpu_threads = args.cpu_threads if dev == 'cpu' else 0
            p = ctx.Process(
                target=_worker_entry,
                args=(chunk, biolip_root, output_root, dev,
                      args.overwrite, tag, args.smoke_test, failure_tsv,
                      cpu_threads),
            )
            p.start()
            procs.append(p)

        rc = 0
        for p in procs:
            p.join()
            if p.exitcode != 0:
                rc = 1
        return rc

    # ── single device ──────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    if device.type == 'cuda':
        try:
            _validate_devices([str(device)])
        except (RuntimeError, ValueError) as e:
            print(f'ERROR: {e}')
            return 1
        torch.cuda.set_device(device)
    else:
        if args.cpu_threads > 0:
            torch.set_num_threads(args.cpu_threads)
            print(f'CPU threads: {args.cpu_threads}')
    model, alphabet = _load_model(device)

    failures: List[Tuple[str, str]] = []
    for idx, bundle_name in enumerate(bundle_names, 1):
        ok, msg = extract_bundle(
            bundle_name, biolip_root, output_root,
            model, alphabet, args.overwrite, args.smoke_test
        )
        print(f'[{idx}/{total}] {msg}', flush=True)
        if not ok:
            failures.append((bundle_name, msg))

    if failures:
        failure_tsv = output_root / 'failures.tsv'
        with open(failure_tsv, 'w') as f:
            f.write('bundle_name\terror\n')
            for bn, err in failures:
                f.write(f'{bn}\t{err.replace(chr(9)," ")}\n')
        print(f'\n{len(failures)} failures → {failure_tsv}')
        return 1

    print(f'\nDone. {total - len(failures)}/{total} bundles extracted.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
