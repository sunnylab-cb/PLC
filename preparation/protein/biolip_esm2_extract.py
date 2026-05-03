#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip ESM2 per-residue embedding extractor.

Unlike the PDBBind version (which reads PDB via ProteinChain.from_pdb),
this script reads observed residues from a CIF file using gemmi, then runs
ESM2 on the observed amino-acid sequence.

CIF source priority
───────────────────
1. ``--fixed_root/{pdb_id}/receptor_fixed.cif``  (protonated+minimised)
2. ``--biolip_root/shared_receptors/{pdb_id}/receptor_full.cif``  (original)

If --fixed_root is omitted, only the original receptor_full.cif is used.
The ``cif_source`` field in the saved .pt records which file was used.

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

# multi-GPU with fixed_receptors (preferred) + shared_receptors fallback
python biolip_esm2_extract.py \
    --biolip_root  /home/husrcf/Code/ProtBind/BioLip \
    --fixed_root   /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \
    --output       /home/husrcf/Code/ProtBind/BioLip/esm2_features \
    --devices 0,1,cpu --cpu_threads 64

# shard by PDB/bundle prefix, useful for launching independent processes
python biolip_esm2_extract.py ... --first_id 2 --devices 0,1

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
ESM2_WINDOW_OVERLAP = 768    # high-overlap windowing for overlength chains
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
    Single-pass ESM2 call.  Callers should use _run_esm2_windowed() for
    overlength chains.
    """
    if len(sequence) > ESM2_MAX_LEN:
        raise ValueError(
            f'Sequence length {len(sequence)} exceeds ESM2 limit {ESM2_MAX_LEN}. '
            f'Use _run_esm2_windowed() for overlength chains.'
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


def _window_starts(n_res: int, window_len: int, overlap: int) -> List[int]:
    """Return starts that cover [0, n_res) with the final window end-aligned."""
    if n_res <= window_len:
        return [0]
    if overlap < 0 or overlap >= window_len:
        raise ValueError(
            f'Invalid ESM2 window overlap {overlap}; expected 0 <= overlap < {window_len}'
        )
    stride = window_len - overlap
    starts = [0]
    while starts[-1] + window_len < n_res:
        nxt = starts[-1] + stride
        if nxt + window_len >= n_res:
            nxt = n_res - window_len
        if nxt <= starts[-1]:
            break
        starts.append(nxt)
    return starts


def _run_esm2_windowed(
    model,
    alphabet,
    sequence: str,
    overlap: int = ESM2_WINDOW_OVERLAP,
) -> Tuple[torch.Tensor, dict]:
    """
    Return per-residue embeddings [N, 1280].

    Short sequences use the original single-pass path.  Overlength sequences are
    encoded in overlapping windows of length ESM2_MAX_LEN; residues covered by
    multiple windows are averaged.  No residues are truncated.
    """
    n_res = len(sequence)
    if n_res <= ESM2_MAX_LEN:
        emb = _run_esm2(model, alphabet, sequence)
        return emb, {
            'windowed': False,
            'window_len': ESM2_MAX_LEN,
            'window_overlap': 0,
            'window_stride': ESM2_MAX_LEN,
            'window_starts': [0],
            'n_windows': 1,
        }

    starts = _window_starts(n_res, ESM2_MAX_LEN, overlap)
    acc = None
    counts = torch.zeros(n_res, dtype=torch.float32)

    for start in starts:
        end = min(start + ESM2_MAX_LEN, n_res)
        emb = _run_esm2(model, alphabet, sequence[start:end])
        if acc is None:
            acc = torch.zeros(n_res, emb.shape[1], dtype=emb.dtype)
        acc[start:end] += emb
        counts[start:end] += 1.0

    if acc is None or torch.any(counts == 0):
        raise RuntimeError(
            f'Windowed ESM2 failed to cover all residues: n_res={n_res}, '
            f'windows={starts}, uncovered={(counts == 0).sum().item()}'
        )

    embeddings = acc / counts[:, None]
    assert embeddings.shape[0] == n_res, (
        f'Embedding length {embeddings.shape[0]} ≠ sequence length {n_res}'
    )
    return embeddings, {
        'windowed': True,
        'window_len': ESM2_MAX_LEN,
        'window_overlap': overlap,
        'window_stride': ESM2_MAX_LEN - overlap,
        'window_starts': starts,
        'n_windows': len(starts),
    }


# ── per-bundle extraction ─────────────────────────────────────────────────────

def extract_bundle(
    bundle_name : str,
    biolip_root : Path,
    output_root : Path,
    model,
    alphabet,
    overwrite   : bool = False,
    smoke_test  : bool = False,
    fixed_root  : Optional[Path] = None,
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

    # ── validate pdb_id from metadata stays within allowed roots ──────────
    # A malformed protein_meta.json could point at an arbitrary path and cause
    # embeddings to be computed for the wrong structure (silent data corruption).
    receptors_root    = (biolip_root / 'shared_receptors').resolve()
    candidate_cif_dir = (receptors_root / pdb_id).resolve()
    if not str(candidate_cif_dir).startswith(str(receptors_root) + os.sep):
        return False, (
            f'{bundle_name}: pdb_id {pdb_id!r} from metadata resolves to '
            f'{candidate_cif_dir}, which escapes shared_receptors root'
        )
    if fixed_root is not None:
        fixed_root_res    = fixed_root.resolve()
        candidate_fixed   = (fixed_root_res / pdb_id).resolve()
        if not str(candidate_fixed).startswith(str(fixed_root_res) + os.sep):
            return False, (
                f'{bundle_name}: pdb_id {pdb_id!r} from metadata resolves to '
                f'{candidate_fixed}, which escapes fixed_root'
            )

    # ── locate CIF: fixed_receptors preferred, shared_receptors fallback ──
    cif_path   : Optional[Path] = None
    cif_source : str            = 'unknown'
    if fixed_root is not None:
        candidate = fixed_root / pdb_id / 'receptor_fixed.cif'
        if candidate.exists():
            cif_path   = candidate
            cif_source = 'fixed'
    if cif_path is None:
        candidate = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
        if candidate.exists():
            cif_path   = candidate
            cif_source = 'full'
    if cif_path is None:
        tried = []
        if fixed_root is not None:
            tried.append(str(fixed_root / pdb_id / 'receptor_fixed.cif'))
        tried.append(str(biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'))
        return False, f'{bundle_name}: no CIF found (tried: {", ".join(tried)})'

    # ── parse CIF → residue table ─────────────────────────────────────────
    # Prefer receptor_fixed.cif, but fall back to receptor_full.cif when the
    # requested chain is missing from the fixed file.  This is needed for
    # structures whose original mmCIF contains both uppercase and lowercase
    # chain IDs: some fixer paths collapse case-distinct chains (e.g. full
    # chains A+a become fixed chain A), so a case-insensitive lookup would
    # silently extract the wrong concatenated sequence.
    cif_fallback_reason = ''
    try:
        res_index, ins_codes, res_name_3, res_name_1 = _parse_cif_chain(
            cif_path, protein_chain
        )
    except Exception as e:
        if cif_source != 'fixed':
            return False, f'{bundle_name}: CIF parse error: {e}'
        fixed_error = str(e)
        full_candidate = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
        if not full_candidate.exists():
            return False, f'{bundle_name}: CIF parse error: {e}'
        try:
            res_index, ins_codes, res_name_3, res_name_1 = _parse_cif_chain(
                full_candidate, protein_chain
            )
            cif_path = full_candidate
            cif_source = 'full'
            cif_fallback_reason = f'fixed_chain_parse_failed: {fixed_error}'
        except Exception as e_full:
            return False, (
                f'{bundle_name}: CIF parse error: fixed={fixed_error}; '
                f'full_fallback={e_full}'
            )

    sequence = ''.join(res_name_1)
    n_obs    = len(sequence)

    # ── smoke-test: print residue table ───────────────────────────────────
    if smoke_test:
        print(f'\n  [{bundle_name}] {pdb_id} chain={protein_chain}  '
              f'observed={n_obs}  src={cif_source}  '
              f'seq={sequence[:40]}{"…" if n_obs>40 else ""}')
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
        embeddings, esm2_window_info = _run_esm2_windowed(model, alphabet, sequence)
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
        'esm2_windowed'  : esm2_window_info['windowed'],
        'esm2_window_len': esm2_window_info['window_len'],
        'esm2_window_overlap': esm2_window_info['window_overlap'],
        'esm2_window_stride': esm2_window_info['window_stride'],
        'esm2_window_starts': esm2_window_info['window_starts'],
        'esm2_n_windows': esm2_window_info['n_windows'],
        'bundle_name'    : bundle_name,
        'pdb_id'         : pdb_id,
        'protein_chain'  : protein_chain,
        'cif_source'     : cif_source,      # 'fixed' | 'full'
        'cif_fallback_reason': cif_fallback_reason,
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
    window_tag = (
        f', windowed={esm2_window_info["n_windows"]}x'
        if esm2_window_info['windowed'] else ''
    )
    return True, (
        f'{bundle_name}: OK  ({len(sequence)} res, emb={tuple(embeddings.shape)}, '
        f'src={cif_source}{window_tag})'
    )


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
    fixed_root   : Optional[Path] = None,
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
            model, alphabet, overwrite, smoke_test, fixed_root,
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
    parser.add_argument('--fixed_root', default=None,
                        help=(
                            'Directory of PDBFixer-processed receptors '
                            '({fixed_root}/{pdb_id}/receptor_fixed.cif). '
                            'When provided, receptor_fixed.cif is preferred; '
                            'falls back to shared_receptors/receptor_full.cif.'
                        ))
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
    parser.add_argument('--first_id', default=None,
                        help=(
                            'Only process bundles whose name/PDB id starts with this prefix. '
                            'Example: --first_id 2 keeps 2xxx_* bundles such as 2ijb_A. '
                            'Filtering happens before --max and multi-device sharding.'
                        ))
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N bundles (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-extract even if .pt already exists')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Print residue alignment table for each bundle (use with --max 5)')
    args = parser.parse_args()

    biolip_root = Path(args.biolip_root).resolve()
    output_root = Path(args.output).resolve()
    fixed_root  = Path(args.fixed_root).resolve() if args.fixed_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    if fixed_root is not None:
        if not fixed_root.is_dir():
            print(f'WARNING: --fixed_root {fixed_root} does not exist; '
                  f'will fall back to shared_receptors for all bundles.')
        else:
            print(f'fixed_root: {fixed_root}')

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

    if args.first_id:
        first_id = args.first_id.strip().lower()
        before_filter = len(bundle_names)
        bundle_names = [
            bn for bn in bundle_names
            if bn.lower().startswith(first_id)
        ]
        print(
            f'first_id filter {first_id!r}: kept {len(bundle_names):,} / '
            f'{before_filter:,} bundles; skipped {before_filter - len(bundle_names):,}'
        )

    if args.max:
        bundle_names = bundle_names[:args.max]
        print(f'Limiting to first {len(bundle_names)} bundles.')

    total = len(bundle_names)
    print(f'Total to process: {total:,}')
    print(f'Output: {output_root}')
    if total == 0:
        print('No bundles to process after filtering.')
        return 0

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
                      cpu_threads, fixed_root),
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
            model, alphabet, args.overwrite, args.smoke_test, fixed_root,
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
