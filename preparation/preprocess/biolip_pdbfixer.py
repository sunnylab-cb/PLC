#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLip receptor preprocessing: 5-step pipeline (CIF in → CIF out)

Pipeline
────────
  1. PDBFixer      — strip heterogens, replace non-standard residues,
                     repair missing heavy atoms (optionally add missing loops)
  2. PropKa        — predict per-residue pKa at target pH → protonation states
  3. OpenMM Modeller — add hydrogens consistent with PropKa protonation
  4. Fixed heavy atoms — set all non-H particle masses to zero
  5. H-only minimization — relax hydrogens with bonded + nonbonded terms

Why CIF in / CIF out
────────────────────
PDB format is limited to 99,999 atoms, 26 chain IDs (A–Z), and residue
numbers ≤ 9999.  Large assemblies in BioLip routinely exceed these limits.
We stay in mmCIF throughout (PDBxFile for output).

PropKa note
───────────
PropKa requires a PDB file as input; we write a temporary PDB from the
PDBFixer topology (after heavy-atom repair, before adding H).  For the
vast majority of protein chains this fits within PDB format limits.
If the write fails (overflow), PropKa is skipped and OpenMM falls back
to its built-in pH-based hydrogen placement (still uses the target pH,
just without environment-specific pKa corrections).

Usage
─────
# full run, 16 workers, pH 7.4
conda run -n AIAA python3 preparation/preprocess/biolip_pdbfixer.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --workers 16 --ph 7.4

# subset
conda run -n AIAA python3 preparation/preprocess/biolip_pdbfixer.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --pdbid_list_file missing_ids.txt --workers 8

# smoke-test
conda run -n AIAA python3 preparation/preprocess/biolip_pdbfixer.py \\
    --biolip_root /home/husrcf/Code/ProtBind/BioLip \\
    --output      /home/husrcf/Code/ProtBind/BioLip/fixed_receptors \\
    --max 5 --workers 1
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import re
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# ── PDBFixer / OpenMM availability (lazy — no C-library load at module level) ──

import importlib.util as _importlib_util
_OPENMM_SPEC   = _importlib_util.find_spec('openmm')
_PDBFIXER_SPEC = _importlib_util.find_spec('pdbfixer')
OPENMM_AVAILABLE = _OPENMM_SPEC is not None and _PDBFIXER_SPEC is not None
del _importlib_util, _OPENMM_SPEC, _PDBFIXER_SPEC

# ── OpenMM thread limit (must run AFTER fork in each worker) ──────────

def _limit_openmm_threads() -> None:
    """
    Force OpenMM to use CPU-only, 1 thread per process.

    Must be called BEFORE any OpenMM import that triggers Platform init,
    and AFTER os.fork() in each worker child.

    Hides all GPU devices so OpenMM (including internal contexts created
    by addHydrogens()) cannot auto-select CUDA/HIP/OpenCL platforms.
    """
    # Hide GPUs from all backends — must happen before OpenMM Platform init
    os.environ['CUDA_VISIBLE_DEVICES'] = ''     # NVIDIA
    os.environ['HIP_VISIBLE_DEVICES'] = ''      # AMD ROCm
    os.environ['ROCR_VISIBLE_DEVICES'] = ''     # AMD ROCm (alternative)

    if not OPENMM_AVAILABLE:
        return
    try:
        from openmm import Platform
        Platform.getPlatformByName('CPU').setPropertyDefaultValue('Threads', '1')
    except Exception:
        pass

# ── PropKa ────────────────────────────────────────────────────────────────────
try:
    import propka.run as propka_run
    PROPKA_AVAILABLE = True
except ImportError:
    PROPKA_AVAILABLE = False


# ── path safety ───────────────────────────────────────────────────────────────

def _validate_pdb_id(pdb_id: str, biolip_root: Path, output_root: Path) -> None:
    """
    Reject pdb_ids that would escape the intended filesystem roots.

    Checks:
    - Empty string or OS special names ('.', '..').
    - No path separators or '..' components (fast lexical guard).
    - Resolved input dir strictly under biolip_root/shared_receptors.
    - Resolved output dir strictly under output_root.
    """
    if not pdb_id or pdb_id.strip() == '':
        raise ValueError(f'pdb_id {pdb_id!r} is empty or whitespace-only.')
    if pdb_id in ('.', '..'):
        raise ValueError(f'pdb_id {pdb_id!r} is a special directory name; refusing.')
    if os.sep in pdb_id or '/' in pdb_id or '..' in pdb_id.split('/'):
        raise ValueError(
            f'pdb_id {pdb_id!r} contains path separators or ".."; refusing to process.'
        )
    receptors_root = (biolip_root / 'shared_receptors').resolve()
    candidate_in   = (receptors_root / pdb_id).resolve()
    candidate_out  = (output_root / pdb_id).resolve()

    if not str(candidate_in).startswith(str(receptors_root) + os.sep):
        raise ValueError(
            f'pdb_id {pdb_id!r} resolves to {candidate_in}, '
            f'which is outside {receptors_root}.'
        )
    if not str(candidate_out).startswith(str(output_root.resolve()) + os.sep):
        raise ValueError(
            f'pdb_id {pdb_id!r} output path {candidate_out} '
            f'is outside {output_root}.'
        )


# ── PropKa helpers ────────────────────────────────────────────────────────────

# residue types PropKa reports for titratable sites
_TITRATABLE = {'HIS', 'ASP', 'GLU', 'LYS', 'ARG', 'CYS', 'TYR', 'C-', 'N+'}

# OpenMM variant names for addHydrogens()
#   key: (res_name, protonated_at_ph)
_VARIANT_MAP = {
    # HIS:  protonated (pKa > pH) → HIP;  deprotonated → HIE (ε, more common)
    ('HIS', True):  'HIP',
    ('HIS', False): 'HIE',
    # ASP:  protonated (pKa > pH) → ASH;  default deprotonated → None
    ('ASP', True):  'ASH',
    ('ASP', False): None,
    # GLU:  protonated (pKa > pH) → GLH;  default deprotonated → None
    ('GLU', True):  'GLH',
    ('GLU', False): None,
    # LYS:  neutral (pKa < pH) → LYN;  default protonated → None
    ('LYS', True):  None,
    ('LYS', False): 'LYN',
    # CYS:  AMBER14 addHydrogens() has no deprotonated-cysteine variant;
    #        CYX is for disulfide bridges only, not H-placement.
    #        Always emit None so OpenMM uses its default (protonated SH).
    ('CYS', True):  None,
    ('CYS', False): None,
}


def _run_propka(tmp_pdb: str, ph: float) -> Tuple[dict, Optional[str]]:
    """
    Run PropKa on a PDB file and return (pka_lookup, error).

    pka_lookup : (chain_id, res_seq_int, insertion_code, res_name) -> pka_value
    error      : None on success; error string on any failure.

    Insertion code is normalised to '' for ordinary residues, 'A'/'B'… otherwise.
    Duplicate keys (same chain/seq/icode/type — indicates PropKa lost icode) are
    removed from the lookup so they receive default OpenMM pH placement instead of
    an arbitrarily-chosen protonation state.

    Uses write_pka=False so PropKa never writes .pka files (no CWD pollution).
    """
    try:
        # Redirect stdout+stderr: PropKa prints geometry warnings (e.g. incomplete
        # C-terminal COO⁻) that --quiet does not fully suppress.  These are benign
        # for our use case; silence them to keep worker output clean.
        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            mol = propka_run.single(
                tmp_pdb, optargs=['--quiet'], write_pka=False,
            )
    except SystemExit as e:
        # PropKa uses argparse internally; on any flag/parse error it calls
        # sys.exit(2) which raises SystemExit (not caught by except Exception).
        return {}, f'propka.run.single raised SystemExit({e.code})'
    except Exception as e:
        return {}, f'propka.run.single failed: {e}'

    pka_lookup: dict = {}
    duplicate_keys: set = set()
    parse_err: Optional[str] = None
    try:
        # PropKa stores results per conformation; 'AVR' is the average
        conf = mol.conformations.get('AVR') or next(iter(mol.conformations.values()))
        for group in conf.groups:
            if group.pka_value is None:
                continue
            if group.residue_type not in _TITRATABLE:
                continue
            chain    = group.atom.chain_id.strip()
            res_seq  = int(group.atom.res_num)    # PropKa uses res_num, not res_seq
            icode    = getattr(group.atom, 'icode', '').strip()  # '' or 'A'/'B'…
            res_type = group.residue_type
            key      = (chain, res_seq, icode, res_type)
            if key in pka_lookup:
                # Collision: PropKa returned the same key twice; drop both so
                # the ambiguous site falls back to default pH placement.
                duplicate_keys.add(key)
            else:
                pka_lookup[key] = group.pka_value
    except Exception as e:
        parse_err = f'PropKa result parse error: {e}'

    for key in duplicate_keys:
        pka_lookup.pop(key, None)

    if duplicate_keys and parse_err is None:
        parse_err = (
            f'PropKa returned duplicate keys for {len(duplicate_keys)} site(s) '
            f'(insertion-code ambiguity); those sites use default pH placement.'
        )

    return pka_lookup, parse_err


def _build_variants(topology, pka_lookup: dict, ph: float) -> List[Optional[str]]:
    """
    Map PropKa pKa values to OpenMM addHydrogens() variant strings.
    Returns a list with one entry per residue (None = use default).

    Lookup key: (chain_id, res_seq_int, insertion_code, res_name)
    OpenMM residue.id encodes both seq-num and insertion code, e.g. '100' or
    '100A'.  We parse it with a regex to recover both components.
    """
    _ID_RE = re.compile(r'^(-?\d+)([A-Z]?)$')

    variants: List[Optional[str]] = []
    for residue in topology.residues():
        res_name = residue.name
        chain_id = residue.chain.id.strip()

        m = _ID_RE.match(residue.id)
        if not m:
            variants.append(None)
            continue
        res_seq = int(m.group(1))
        icode   = m.group(2)          # '' for ordinary residues, 'A'/'B'… otherwise

        # Normalise HIS variants already present in topology
        lookup_name = 'HIS' if res_name in ('HID', 'HIE', 'HIP') else res_name

        if lookup_name in ('HIS', 'ASP', 'GLU', 'LYS', 'CYS'):
            pka = pka_lookup.get((chain_id, res_seq, icode, lookup_name))
            if pka is None:
                # No PropKa data for this site (not found, ambiguous, or dropped
                # due to insertion-code collision) → let OpenMM use its pH default.
                variant = None
            else:
                variant = _VARIANT_MAP.get((lookup_name, pka > ph))
        else:
            variant = None

        variants.append(variant)
    return variants


# ── per-structure pipeline ────────────────────────────────────────────────────

_STRIP_MODEL_WARN = (
    'receptor_full.cif contains multiple NMR models; '
    'keeping only model 1.'
)


def _strip_extra_models(in_path: Path, pdb_id: str) -> Optional[Path]:
    """
    Keep only model 1 from a multi-model mmCIF (NMR ensembles).

    OpenMM's PDBxFile requires all models to have identical atom ordering;
    NMR models often violate this.  We write a single-model CIF to a system
    temp file and return its path.  Returns None when the file already has
    only one model (caller uses the original path).

    For multi-model CIFs, failure to strip is raised (not silently ignored)
    because PDBFixer will crash on atom-order mismatches between models.

    Uses a cheap text scan first: most receptors are single-model, so we
    avoid the gemmi full-parse cost in the common case.
    """
    # ── Cheap pre-check: scan for model_num > 1 in atom_site records ──
    if not _has_multiple_models(in_path):
        return None

    import gemmi
    structure = gemmi.read_structure(str(in_path))
    if len(structure) <= 1:
        return None
    # Keep only the first conformer
    del structure[1:]
    fd, tmp_str = tempfile.mkstemp(suffix='.cif', prefix=f'{pdb_id}_model1_')
    os.close(fd)
    structure.make_mmcif_document().write_file(tmp_str)
    return Path(tmp_str)


def _has_multiple_models(in_path: Path) -> bool:
    """
    Cheap text scan: does *in_path* contain atoms from model ≥2?

    Only parses the atom_site loop header to locate the
    pdbx_PDB_model_num column, then scans data lines for any value >1.
    Returns False for single-model CIFs — the common case — so we can
    skip a full gemmi parse entirely.
    """
    with open(in_path, 'r') as f:
        in_atom_site = False
        column_names: list = []
        model_idx: Optional[int] = None

        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            if line == 'loop_':
                in_atom_site = False
                column_names = []
                model_idx = None
                continue

            if line.startswith('_atom_site.'):
                in_atom_site = True
                column_names.append(line)
                continue

            if in_atom_site and column_names:
                if line.startswith('_') or line == 'loop_':
                    return False
                if model_idx is None:
                    try:
                        model_idx = column_names.index('_atom_site.pdbx_PDB_model_num')
                    except ValueError:
                        return False

                values = line.split()
                if model_idx < len(values):
                    try:
                        if int(values[model_idx]) > 1:
                            return True
                    except ValueError:
                        pass
    return False


def preprocess_receptor(
    pdb_id        : str,
    biolip_root   : Path,
    output_root   : Path,
    ph            : float = 7.0,
    overwrite     : bool  = False,
    add_residues  : bool  = False,
    min_max_iter  : int   = 100,
    forcefield    : 'Optional[ForceField]' = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Full 5-step preprocessing for one receptor.
    Returns (success, message, propka_warn).

    propka_warn is None when PropKa succeeded; otherwise an error string.
    The structure is still written on PropKa failure (falls back to pH-only H).

    Input  : {biolip_root}/shared_receptors/{pdb_id}/receptor_full.cif
    Output : {output_root}/{pdb_id}/receptor_fixed.cif
    """
    # ── Lazy imports (fork-safe: OpenMM libs loaded only in worker) ─────
    from pdbfixer import PDBFixer
    from openmm.app import (
        PDBFile, PDBxFile, ForceField, Modeller, CutoffNonPeriodic, Simulation,
    )
    from openmm import (
        LangevinMiddleIntegrator, Platform, unit,
    )

    # ── path safety (before any filesystem access) ────────────────────────
    try:
        _validate_pdb_id(pdb_id, biolip_root, output_root)
    except ValueError as e:
        return False, f'{pdb_id}: invalid pdb_id: {e}', None

    in_path  = biolip_root / 'shared_receptors' / pdb_id / 'receptor_full.cif'
    out_dir  = output_root / pdb_id
    out_path = out_dir / 'receptor_fixed.cif'

    if out_path.exists() and not overwrite:
        return True, f'{pdb_id}: SKIP', None

    if not in_path.exists():
        return False, f'{pdb_id}: receptor_full.cif not found', None

    propka_warn: Optional[str] = None   # initialised before try so except can reference it
    _stripped_cif: Optional[Path] = None

    try:
        # ── Step 0: Strip extra NMR models (keep model 1 only) ───────
        cif_path = in_path
        _stripped_cif = _strip_extra_models(in_path, pdb_id)
        if _stripped_cif is not None:
            cif_path = _stripped_cif
            propka_warn = _STRIP_MODEL_WARN
        # ── Step 1: PDBFixer — heavy-atom repair ──────────────────────────
        fixer = PDBFixer(filename=str(cif_path))

        if add_residues:
            fixer.findMissingResidues()
        else:
            fixer.missingResidues = {}   # repair side chains only, no new loops

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        try:
            fixer.addMissingAtoms()
        except Exception as e:
            # PDBFixer may NaN during its internal addMissingAtoms minimization.
            # Rebuild a clean heavy-atom-only fixer and continue without filling
            # missing atoms for this receptor.
            _repair_warn = (
                'addMissingAtoms failed; continuing without missing heavy-atom '
                f'repair. Cause: {str(e).splitlines()[0]}'
            )
            propka_warn = f'{propka_warn}; {_repair_warn}' if propka_warn else _repair_warn

            fixer = PDBFixer(filename=str(cif_path))
            fixer.missingResidues = {}
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
        # (stripped_cif temp file cleanup is now in the finally block below)

        # ── Step 2: PropKa — pKa prediction → protonation lookup ─────────
        pka_lookup: dict = {}
        if PROPKA_AVAILABLE:
            # PropKa needs a PDB file; write a temporary one.
            # For structures that overflow PDB limits, we skip PropKa gracefully.
            tmp_dir = tempfile.mkdtemp()
            tmp_pdb = os.path.join(tmp_dir, f'{pdb_id}.pdb')
            try:
                with open(tmp_pdb, 'w') as fh:
                    PDBFile.writeFile(fixer.topology, fixer.positions, fh)
                pka_lookup, _propka_warn = _run_propka(tmp_pdb, ph)
                if _propka_warn:
                    propka_warn = (
                        f'{propka_warn}; {_propka_warn}'
                        if propka_warn else _propka_warn
                    )

                # ── Chain-ID truncation check ─────────────────────────────
                # PDB format truncates mmCIF chain IDs to 1 char.
                # Only warn when topology actually has multi-char chain
                # IDs that could be affected; report pKa coverage ratio
                # instead of asserting a "mismatch" (chains without
                # titratable sites naturally won't appear in PropKa output).
                if pka_lookup:
                    _top_chains = {r.chain.id.strip()
                                   for r in fixer.topology.residues()}
                    _long_chains = {c for c in _top_chains if len(c) > 1}
                    if _long_chains:
                        _n_titratable = sum(
                            1 for r in fixer.topology.residues()
                            if r.name in _TITRATABLE
                            or r.name in ('HID', 'HIE', 'HIP')
                        )
                        _trunc_warn = (
                            f'Topology has {len(_long_chains)} chain(s) with '
                            f'ID >1 char ({sorted(_long_chains)}); '
                            f'PDB truncation may affect PropKa matching. '
                            f'pKa coverage: {len(pka_lookup)} sites / '
                            f'{_n_titratable} titratable residues.'
                        )
                        propka_warn = (
                            f'{propka_warn}; {_trunc_warn}'
                            if propka_warn else _trunc_warn
                        )
            except Exception as e:
                _propka_warn = f'PDB write for PropKa failed (overflow?): {e}'
                propka_warn = (
                    f'{propka_warn}; {_propka_warn}'
                    if propka_warn else _propka_warn
                )
            finally:
                try:
                    os.unlink(tmp_pdb)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
        else:
            _propka_warn = 'propka not installed'
            propka_warn = (
                f'{propka_warn}; {_propka_warn}' if propka_warn else _propka_warn
            )

        # ── Step 3: OpenMM Modeller — add hydrogens ───────────────────────
        # forcefield may be pre-built by the caller (worker caches it once)
        if forcefield is None:
            forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller   = Modeller(fixer.topology, fixer.positions)

        variants = _build_variants(modeller.topology, pka_lookup, ph) \
                   if pka_lookup else None   # None → OpenMM uses pH directly

        # ── addHydrogens with three-level fallback ────────────────────────
        # Level 1: PropKa-guided variants.
        # Level 2: variants=None (pure pH-based placement, no PropKa).
        # Level 3: skip H entirely — save heavy-atom-only structure, still
        #          useful for feature extraction; logged in propka_warn.
        add_h_warn: Optional[str] = None
        hydrogens_added = False
        last_addh_err = 'unknown addHydrogens failure'
        for attempt, var in enumerate(
            [variants, None] if variants is not None else [None]
        ):
            # addHydrogens may partially mutate Modeller before raising.
            # Rebuild from the clean fixer state for every attempt.
            modeller = Modeller(fixer.topology, fixer.positions)
            try:
                modeller.addHydrogens(forcefield, pH=ph, variants=var)
                hydrogens_added = True
                if attempt > 0:
                    add_h_warn = (
                        'addHydrogens with PropKa variants failed; '
                        'fell back to pH-only placement.'
                    )
                break
            except Exception as e:
                # OpenMM can also fail inside geometric H placement
                # (e.g. math domain error from degenerate coordinates).
                # Treat it as a per-structure addH failure, not a worker kill.
                err_msg = str(e).split('\n')[0]
                last_addh_err = f'{type(e).__name__}: {err_msg}'

        if not hydrogens_added:
            # Level 3: save without H; downstream heavy-atom feature extraction
            # still works.  Record the cause.
            modeller = Modeller(fixer.topology, fixer.positions)
            add_h_warn = f'addHydrogens failed; saved without H. {last_addh_err}'

        if add_h_warn:
            propka_warn = (
                f'{propka_warn}; {add_h_warn}' if propka_warn else add_h_warn
            )

        # ── Steps 4 & 5: fixed-heavy minimization (H only) ───────────────
        # Capture post-addHydrogens positions before minimization mutates
        # them.  Serves as NaN fallback and as the fixed-heavy reference.
        pre_min_positions = modeller.positions if hydrogens_added else None

        if not hydrogens_added:
            final_positions = modeller.positions
        elif min_max_iter <= 0:
            # User requested zero iterations — skip createSystem() and the
            # entire minimization stack (saves time + memory for large structures).
            final_positions = modeller.positions
        else:
            try:
                system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=CutoffNonPeriodic,
                    nonbondedCutoff=1.2 * unit.nanometer,
                    # Zero-mass heavy atoms are incompatible with H-heavy
                    # bond constraints.  Leave constraints off so H atoms can
                    # relax bond lengths/angles against fixed heavy atoms.
                    constraints=None,
                )

                # True H-only relaxation: zero-mass all non-H atoms so the
                # minimizer keeps the experimental heavy-atom scaffold fixed
                # while hydrogens respond to bonded and nonbonded terms.
                for atom in modeller.topology.atoms():
                    is_h = atom.element is not None and atom.element.symbol == 'H'
                    if not is_h:
                        system.setParticleMass(atom.index, 0.0 * unit.dalton)

                integrator = LangevinMiddleIntegrator(
                    300 * unit.kelvin,
                    1 / unit.picosecond,
                    0.002 * unit.picoseconds,
                )
                cpu_platform = Platform.getPlatformByName('CPU')
                simulation = Simulation(
                    modeller.topology, system, integrator,
                    cpu_platform, {'Threads': '1'},
                )
                simulation.context.setPositions(pre_min_positions)
                simulation.minimizeEnergy(maxIterations=min_max_iter)

                state = simulation.context.getState(getPositions=True)
                final_positions = state.getPositions()
                max_heavy_shift_nm = 0.0
                for atom in modeller.topology.atoms():
                    is_h = atom.element is not None and atom.element.symbol == 'H'
                    if is_h:
                        continue
                    p0 = pre_min_positions[atom.index]
                    p1 = final_positions[atom.index]
                    dx = p1.x - p0.x
                    dy = p1.y - p0.y
                    dz = p1.z - p0.z
                    max_heavy_shift_nm = max(
                        max_heavy_shift_nm,
                        (dx * dx + dy * dy + dz * dz) ** 0.5,
                    )
                if max_heavy_shift_nm > 1e-5:
                    drift_warn = (
                        'H-minimization moved fixed heavy atoms '
                        f'(max shift {max_heavy_shift_nm:.3g} nm); '
                        'using unrelaxed H positions.'
                    )
                    propka_warn = (
                        f'{propka_warn}; {drift_warn}' if propka_warn else drift_warn
                    )
                    final_positions = pre_min_positions

            except Exception as min_err:
                # Fallback: use addHydrogens positions (unrelaxed but valid)
                final_positions = pre_min_positions
                min_warn = f'H-minimization failed ({min_err}); using unrelaxed H positions.'
                propka_warn = (
                    f'{propka_warn}; {min_warn}' if propka_warn else min_warn
                )

        # ── NaN-safe write (multi-level fallback) ──────────────────────────
        # OpenMM PDBxFile.writeFile raises if any position is NaN.
        # We try fallback positions in order: final → pre_min → modeller → skip.
        import math as _math

        def _has_nan(positions) -> bool:
            """True when any position vector contains a NaN component."""
            try:
                for pos in positions:
                    if _math.isnan(pos.x) or _math.isnan(pos.y) or _math.isnan(pos.z):
                        return True
                return False
            except Exception:
                return True   # any access failure → treat as corrupt

        write_positions = final_positions
        for fallback_name, fallback_pos in [
            ('final_positions',        final_positions),
            ('pre_min_positions',      pre_min_positions),
            ('modeller.positions',     modeller.positions),
        ]:
            if fallback_pos is None:
                continue
            if not _has_nan(fallback_pos):
                write_positions = fallback_pos
                break
            if fallback_name != 'final_positions':
                # Log that we're falling back
                _fb_warn = f'NaN in {fallback_name}; trying next fallback'
                propka_warn = f'{propka_warn}; {_fb_warn}' if propka_warn else _fb_warn

        if _has_nan(write_positions):
            return (
                False,
                f'{pdb_id}: write error — all position sources contain NaN',
                propka_warn,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_dir / f'.{pdb_id}_{uuid.uuid4().hex}.tmp.cif'
        try:
            with open(tmp_path, 'w') as fh:
                PDBxFile.writeFile(modeller.topology, write_positions, fh)
            os.replace(tmp_path, out_path)
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False, f'{pdb_id}: write error: {e}', propka_warn

    except Exception as e:
        return False, f'{pdb_id}: error: {e}\n{traceback.format_exc()}', propka_warn
    finally:
        # Ensure temp model1.cif is deleted even on exception
        if _stripped_cif is not None:
            try:
                _stripped_cif.unlink(missing_ok=True)
            except OSError:
                pass

    n_atoms    = modeller.topology.getNumAtoms()
    n_chains   = modeller.topology.getNumChains()
    propka_tag = f'PropKa({len(pka_lookup)} sites)' if pka_lookup else 'PropKa=skip'

    # ── Explicit cleanup of large C++ objects ─────────────────────────────
    # Delete OpenMM-owned objects so their C++ destructors release GPU/heap
    # memory before the next iteration.  This is supplementary to worker
    # recycling (--max_per_worker), which is the primary OOM defence.
    del fixer, modeller
    try:
        del simulation
    except NameError:
        pass
    try:
        del system
    except NameError:
        pass
    try:
        del integrator
    except NameError:
        pass

    return True, (
        f'{pdb_id}: OK  ({n_chains} chains, {n_atoms} atoms, {propka_tag})'
    ), propka_warn


# ── worker subprocess ─────────────────────────────────────────────────────────

def _worker_entry(
    pdb_ids      : List[str],
    biolip_root  : Path,
    output_root  : Path,
    ph           : float,
    overwrite    : bool,
    add_residues : bool,
    min_max_iter : int,
    worker_id    : int,
    failure_tsv  : Path,
) -> None:
    # Initialise OpenMM Platform in this child process (AFTER fork).
    _limit_openmm_threads()

    # Lazy import (fork-safe: OpenMM libs loaded only in worker)
    from openmm.app import ForceField

    total    = len(pdb_ids)
    failures : List[Tuple[str, str]] = []

    propka_failures: List[Tuple[str, str]] = []

    # Clear stale failure TSV from previous runs
    try:
        failure_tsv.unlink(missing_ok=True)
    except OSError:
        pass
    _pk_tsv = failure_tsv.parent / failure_tsv.name.replace('failures_', 'propka_failures_')
    try:
        _pk_tsv.unlink(missing_ok=True)
    except OSError:
        pass

    # Build ForceField once per worker — loading AMBER14 XML is expensive
    ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml') \
         if OPENMM_AVAILABLE else None

    for idx, pdb_id in enumerate(pdb_ids, 1):
        ok, msg, propka_warn = preprocess_receptor(
            pdb_id, biolip_root, output_root,
            ph, overwrite, add_residues, min_max_iter,
            forcefield=ff,
        )
        print(f'[W{worker_id} {idx}/{total}] {msg}', flush=True)
        if not ok:
            failures.append((pdb_id, msg))
        if propka_warn is not None:
            propka_failures.append((pdb_id, propka_warn))
        # Force cyclic GC collection to free C++ extension cycles
        import gc as _gc; _gc.collect()

    def _write_tsv(path: Path, rows: List[Tuple[str, str]], header: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(header + '\n')
            for a, b in rows:
                f.write(f'{a}\t{b.replace(chr(9)," ").replace(chr(10)," ")}\n')

    if failures:
        _write_tsv(failure_tsv, failures, 'pdb_id\terror')
        print(f'[W{worker_id}] {len(failures)} failures → {failure_tsv}')

    if propka_failures:
        pk_tsv = failure_tsv.parent / failure_tsv.name.replace('failures_', 'propka_failures_')
        _write_tsv(pk_tsv, propka_failures, 'pdb_id\tpropka_error')
        print(f'[W{worker_id}] {len(propka_failures)} PropKa failures → {pk_tsv}')

    raise SystemExit(1 if failures else 0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'BioLip receptor preprocessing: PDBFixer → PropKa → '
            'OpenMM addH → restrained H-minimization (CIF in → CIF out).'
        )
    )
    parser.add_argument('--biolip_root', required=True,
                        help='BioLip dataset root (contains shared_receptors/)')
    parser.add_argument('--output', required=True,
                        help='Output root; one subdirectory per pdb_id')
    parser.add_argument('--ph', type=float, default=7.0,
                        help='Target pH for protonation (default: 7.0)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel worker processes (default: 8)')
    parser.add_argument('--max_per_worker', type=int, default=None,
                        help='Max structures each worker processes before '
                             'exiting; enables wave-based dispatch for worker '
                             'recycling (OS reclaims process heap). '
                             'Example: --workers 16 --max_per_worker 50 '
                             '→ 16 workers restart every 50 structures.')
    parser.add_argument('--min_max_iter', type=int, default=20,
                        help='Max iterations for H-only energy minimization '
                             '(default: 20; set 0 to skip)')
    parser.add_argument('--pdbid_list_file', default=None,
                        help='Text file of pdb_ids to process (one per line); '
                             'default: auto-discover from shared_receptors/')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to first N pdb_ids (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-process even if receptor_fixed.cif exists')
    parser.add_argument('--add_residues', action='store_true',
                        help='Model missing loop residues via PDBFixer '
                             '(coordinates are modelled, not experimental; '
                             'off by default)')
    args = parser.parse_args()

    if not OPENMM_AVAILABLE:
        print('ERROR: pdbfixer / openmm not installed.')
        print('       conda install -c conda-forge pdbfixer openmm')
        return 1
    if not PROPKA_AVAILABLE:
        print('WARNING: propka not installed — skipping pKa-based protonation.')
        print('         conda install -c conda-forge propka')
        print('         Continuing with OpenMM pH-only hydrogen placement.\n')

    biolip_root = Path(args.biolip_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # ── collect pdb_id list ────────────────────────────────────────────────
    if args.pdbid_list_file:
        pdb_ids = [
            l.strip() for l in
            Path(args.pdbid_list_file).read_text().splitlines()
            if l.strip()
        ]
        before  = len(pdb_ids)
        pdb_ids = list(dict.fromkeys(pdb_ids))
        dup_msg = f', {before - len(pdb_ids)} duplicates removed' \
                  if len(pdb_ids) < before else ''
        print(f'PDB-ID list from file: {len(pdb_ids):,}{dup_msg}')
    else:
        receptors_dir = biolip_root / 'shared_receptors'
        if not receptors_dir.is_dir():
            print(f'ERROR: {receptors_dir} not found.')
            return 1
        pdb_ids = sorted(
            p.name for p in receptors_dir.iterdir()
            if p.is_dir() and (p / 'receptor_full.cif').exists()
        )
        print(f'Auto-discovered receptors: {len(pdb_ids):,}')

    if args.max:
        pdb_ids = pdb_ids[:args.max]
        print(f'Limiting to first {len(pdb_ids)} pdb_ids.')

    total = len(pdb_ids)
    print(f'Total            : {total:,}')
    print(f'Output root      : {output_root}')
    print(f'Workers          : {args.workers}')
    print(f'pH               : {args.ph}')
    print(f'PropKa available : {PROPKA_AVAILABLE}')
    print(f'H-min iterations : {args.min_max_iter}')
    print(f'Model loops      : {args.add_residues}')
    if args.workers > 1 and args.max_per_worker:
        print(f'Max per worker   : {args.max_per_worker} (wave recycling on)')

    if total == 0:
        print('Nothing to do.')
        return 0

    kw = dict(
        ph=args.ph, overwrite=args.overwrite,
        add_residues=args.add_residues, min_max_iter=args.min_max_iter,
    )

    # ── single-worker shortcut ─────────────────────────────────────────────
    if args.workers <= 1:
        _limit_openmm_threads()

        failures        : List[Tuple[str, str]] = []
        propka_failures : List[Tuple[str, str]] = []

        # Build ForceField once for all structures
        from openmm.app import ForceField
        import gc as _gc
        ff_single = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        for idx, pdb_id in enumerate(pdb_ids, 1):
            ok, msg, propka_warn = preprocess_receptor(
                pdb_id, biolip_root, output_root, **kw, forcefield=ff_single
            )
            print(f'[{idx}/{total}] {msg}', flush=True)
            if not ok:
                failures.append((pdb_id, msg))
            if propka_warn is not None:
                propka_failures.append((pdb_id, propka_warn))
            _gc.collect()

        def _write_tsv(path: Path, rows: List[Tuple[str, str]], header: str) -> None:
            with open(path, 'w') as f:
                f.write(header + '\n')
                for a, b in rows:
                    f.write(f'{a}\t{b.replace(chr(9)," ").replace(chr(10)," ")}\n')

        if propka_failures:
            pk_path = output_root / 'propka_failures.tsv'
            _write_tsv(pk_path, propka_failures, 'pdb_id\tpropka_error')
            print(f'\n{len(propka_failures)} PropKa failures → {pk_path}')

        if failures:
            fail_path = output_root / 'failures.tsv'
            _write_tsv(fail_path, failures, 'pdb_id\terror')
            print(f'{len(failures)} preprocessing failures → {fail_path}')
            return 1

        print(f'\nDone. {total}/{total} receptors processed.')
        return 0

    # ── multi-worker dispatch ──────────────────────────────────────────
    n_workers = min(args.workers, total)
    ctx = mp.get_context('fork')

    max_per = args.max_per_worker
    if max_per is None:
        # Single wave: split all work across workers (original behaviour)
        waves = [pdb_ids]
    else:
        # Wave-based: each worker processes ≤ max_per_worker structures,
        # then exits so the OS reclaims its entire process heap.
        # This is the primary OOM defence — worker recycling.
        wave_size = n_workers * max_per
        waves = [
            pdb_ids[i:i + wave_size]
            for i in range(0, total, wave_size)
        ]
        print(f'Waves            : {len(waves)}')

    rc = 0
    for wave_num, wave_chunk in enumerate(waves):
        n_active = min(n_workers, len(wave_chunk))
        splits = [[] for _ in range(n_active)]
        for i, pid in enumerate(wave_chunk):
            splits[i % n_active].append(pid)

        if len(waves) > 1:
            print(f'\n── Wave {wave_num + 1}/{len(waves)} '
                  f'({len(wave_chunk)} structures, {n_active} workers) ──')

        procs = []
        for wid, chunk in enumerate(splits):
            if len(waves) > 1:
                fail_tsv = output_root / f'failures_w{wave_num}_w{wid}.tsv'
            else:
                fail_tsv = output_root / f'failures_w{wid}.tsv'
            p = ctx.Process(
                target=_worker_entry,
                args=(
                    chunk, biolip_root, output_root,
                    args.ph, args.overwrite, args.add_residues,
                    args.min_max_iter,
                    wid, fail_tsv,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                rc = 1

    return rc


if __name__ == '__main__':
    raise SystemExit(main())
