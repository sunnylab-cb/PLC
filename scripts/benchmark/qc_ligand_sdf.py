#!/usr/bin/env python3
"""QC extracted benchmark ligand SDF files before feature extraction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_pdb_ds(path: Path) -> set[str]:
    stems: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("HEADER") or line.startswith("PARAM."):
            continue
        parts = line.split()
        if parts:
            stems.add(Path(parts[0]).stem)
    return stems


def parse_mlig_ds(path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("HEADER") or line.startswith("PARAM."):
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[Path(parts[0]).stem] = {code.strip().upper() for code in parts[1].split(",") if code.strip()}
    return mapping


def sdf_stem_to_pdb_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def inspect_sdf(path: Path) -> dict:
    from rdkit import Chem

    row = {
        "file": path.name,
        "stem": path.stem,
        "pdb_stem": sdf_stem_to_pdb_stem(path.stem),
        "ok": False,
        "num_atoms": 0,
        "num_heavy_atoms": 0,
        "has_conformer": False,
        "ligand_resname": "",
        "extraction_method": "",
        "error": "",
    }
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False, strictParsing=False)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            row["error"] = "rdkit_load_failed"
            return row
        row["num_atoms"] = int(mol.GetNumAtoms())
        row["num_heavy_atoms"] = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
        row["has_conformer"] = bool(mol.GetNumConformers() > 0)
        row["ligand_resname"] = mol.GetProp("ligand_resname") if mol.HasProp("ligand_resname") else ""
        row["extraction_method"] = mol.GetProp("extraction_method") if mol.HasProp("extraction_method") else ""
        row["ok"] = row["num_heavy_atoms"] > 1 and row["has_conformer"]
        if not row["ok"]:
            row["error"] = "empty_or_no_conformer"
        return row
    except Exception as exc:  # pragma: no cover - QC robustness path
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sdf-dir", required=True)
    parser.add_argument("--pdb-ds", required=True)
    parser.add_argument("--mlig-ds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-missing-list", type=int, default=200)
    args = parser.parse_args()

    sdf_dir = Path(args.sdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_stems = parse_pdb_ds(Path(args.pdb_ds))
    mlig_map = parse_mlig_ds(Path(args.mlig_ds))
    sdf_files = sorted(sdf_dir.glob("*.sdf"))

    rows = [inspect_sdf(path) for path in sdf_files]
    ok_rows = [row for row in rows if row["ok"]]
    bad_rows = [row for row in rows if not row["ok"]]
    pdbs_with_sdf = {row["pdb_stem"] for row in ok_rows}
    missing_pdb_ds = sorted(pdb_stems - pdbs_with_sdf)
    missing_mlig = sorted(set(mlig_map) - pdbs_with_sdf)

    by_pdb: dict[str, int] = defaultdict(int)
    for row in ok_rows:
        by_pdb[row["pdb_stem"]] += 1

    summary = {
        "dataset": args.dataset,
        "pdb_ds_entries": len(pdb_stems),
        "mlig_entries": len(mlig_map),
        "sdf_files": len(sdf_files),
        "ok_sdf_files": len(ok_rows),
        "bad_sdf_files": len(bad_rows),
        "pdbs_with_ok_sdf": len(pdbs_with_sdf),
        "missing_pdb_ds_count": len(missing_pdb_ds),
        "missing_mlig_count": len(missing_mlig),
        "ligands_per_pdb": dict(Counter(by_pdb.values())),
        "extraction_methods": dict(Counter(row["extraction_method"] or "unknown" for row in ok_rows)),
        "ligand_resnames_top50": dict(Counter(row["ligand_resname"] or "unknown" for row in ok_rows).most_common(50)),
        "bad_errors": dict(Counter(row["error"] or "unknown" for row in bad_rows)),
        "missing_pdb_ds_sample": missing_pdb_ds[: args.max_missing_list],
        "missing_mlig_sample": missing_mlig[: args.max_missing_list],
    }

    (output_dir / f"{args.dataset}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (output_dir / f"{args.dataset}_sdf_qc.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "file", "stem", "pdb_stem", "ok", "num_atoms", "num_heavy_atoms",
            "has_conformer", "ligand_resname", "extraction_method", "error",
        ], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{args.dataset}_missing_pdb_ds.txt").write_text("\n".join(missing_pdb_ds) + ("\n" if missing_pdb_ds else ""))
    (output_dir / f"{args.dataset}_missing_mlig.txt").write_text("\n".join(missing_mlig) + ("\n" if missing_mlig else ""))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
