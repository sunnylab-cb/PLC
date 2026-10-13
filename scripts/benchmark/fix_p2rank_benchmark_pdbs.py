#!/usr/bin/env python3
"""Conservatively fix P2Rank benchmark protein PDBs with PDBFixer.

Default behavior is intentionally conservative for benchmark reproducibility:
- use *.ds manifests as the official protein list;
- remove heterogens/water to produce protein-only structures;
- replace non-standard residues and add missing heavy atoms;
- do not add missing residues or hydrogens unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FixResult:
    relpath: str
    ok: bool
    skipped: bool
    input: str
    output: str
    error: str = ""
    fallback: bool = False
    fallback_reason: str = ""


def parse_ds_manifest(path: Path) -> list[str]:
    relpaths: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("PARAM.") or line.startswith("HEADER:"):
            continue
        parts = line.split()
        protein = parts[-1]
        if protein.endswith(".pdb"):
            relpaths.append(protein)
    return relpaths


def write_pdb(topology, positions, output_path: Path) -> None:
    from openmm.app import PDBFile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        try:
            PDBFile.writeFile(topology, positions, handle, keepIds=True)
        except TypeError:
            PDBFile.writeFile(topology, positions, handle)


def run_pdbfixer(
    input_path: Path,
    output_path: Path,
    remove_heterogens: bool,
    replace_nonstandard: bool,
    add_missing_residues: bool,
    add_hydrogens: bool,
    ph: float,
) -> None:
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(input_path))
    if replace_nonstandard:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=False)

    fixer.findMissingResidues()
    if not add_missing_residues:
        fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if add_hydrogens:
        fixer.addMissingHydrogens(ph)

    write_pdb(fixer.topology, fixer.positions, output_path)


def format_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


def fix_one(task: tuple[str, str, str, bool, bool, bool, bool, bool, float, bool]) -> FixResult:
    (
        relpath,
        input_root_s,
        output_root_s,
        overwrite,
        remove_heterogens,
        replace_nonstandard,
        add_missing_residues,
        add_hydrogens,
        ph,
        fallback_no_missing_residues,
    ) = task
    input_root = Path(input_root_s)
    output_root = Path(output_root_s)
    input_path = input_root / relpath
    output_path = output_root / relpath

    if output_path.exists() and not overwrite:
        return FixResult(relpath, True, True, str(input_path), str(output_path))
    if not input_path.exists():
        return FixResult(relpath, False, False, str(input_path), str(output_path), "missing input")

    try:
        run_pdbfixer(
            input_path,
            output_path,
            remove_heterogens,
            replace_nonstandard,
            add_missing_residues,
            add_hydrogens,
            ph,
        )
        return FixResult(relpath, True, False, str(input_path), str(output_path))
    except Exception as exc:  # pragma: no cover - batch robustness path
        msg = format_exc(exc)
        if fallback_no_missing_residues and add_missing_residues:
            try:
                run_pdbfixer(
                    input_path,
                    output_path,
                    remove_heterogens,
                    replace_nonstandard,
                    False,
                    add_hydrogens,
                    ph,
                )
                return FixResult(
                    relpath,
                    True,
                    False,
                    str(input_path),
                    str(output_path),
                    fallback=True,
                    fallback_reason=msg,
                )
            except Exception as fallback_exc:  # pragma: no cover - batch robustness path
                msg = msg + "\nFallback without missing residues also failed:\n" + format_exc(fallback_exc)
        return FixResult(relpath, False, False, str(input_path), str(output_path), msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Root containing coach420/ and holo4k/")
    parser.add_argument("--output-root", required=True, help="Root for fixed protein-only PDBs")
    parser.add_argument("--manifest", action="append", required=True, help="One or more *.ds manifest files")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-heterogens", action="store_true", help="Do not remove ligands/water")
    parser.add_argument("--keep-nonstandard", action="store_true", help="Do not replace non-standard residues")
    parser.add_argument("--add-missing-residues", action="store_true", help="Also build missing residue stretches")
    parser.add_argument("--add-hydrogens", action="store_true")
    parser.add_argument(
        "--fallback-no-missing-residues-on-fail",
        action="store_true",
        help="If missing-residue insertion fails, retry the same protein with missing-residue insertion disabled.",
    )
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--status-jsonl", required=True)
    args = parser.parse_args()

    manifests = [Path(p) for p in args.manifest]
    relpaths: list[str] = []
    for manifest in manifests:
        relpaths.extend(parse_ds_manifest(manifest))

    # Preserve manifest order while removing duplicates across manifests.
    relpaths = list(dict.fromkeys(relpaths))
    if not relpaths:
        raise SystemExit("No PDB paths found in manifests")

    status_path = Path(args.status_jsonl)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            rel,
            args.input_root,
            args.output_root,
            args.overwrite,
            not args.keep_heterogens,
            not args.keep_nonstandard,
            args.add_missing_residues,
            args.add_hydrogens,
            args.ph,
            args.fallback_no_missing_residues_on_fail,
        )
        for rel in relpaths
    ]

    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Manifests: {', '.join(str(p) for p in manifests)}")
    print(f"Total proteins: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print("Policy: protein-only, no missing-residue insertion, no hydrogens by default")
    print(f"Add missing residues: {args.add_missing_residues}")
    print(f"Add hydrogens: {args.add_hydrogens}")
    print(f"Fallback without missing residues: {args.fallback_no_missing_residues_on_fail}")

    ok = fail = skipped = 0
    with status_path.open("w") as out:
        if args.workers <= 1:
            iterator = (fix_one(task) for task in tasks)
            for result in iterator:
                out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                out.flush()
                ok += int(result.ok and not result.skipped)
                skipped += int(result.skipped)
                fail += int(not result.ok)
                print(f"[{ok} ok, {skipped} skip, {fail} fail] {result.relpath}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fix_one, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    out.flush()
                    ok += int(result.ok and not result.skipped)
                    skipped += int(result.skipped)
                    fail += int(not result.ok)
                    print(f"[{ok} ok, {skipped} skip, {fail} fail] {result.relpath}")

    print("=== Summary ===")
    print(f"OK:      {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {fail}")
    print(f"Total:   {len(tasks)}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
