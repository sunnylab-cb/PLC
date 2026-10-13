#!/usr/bin/env python3
"""Count BioLiP expansion ceilings; never approve rows or modify training data.

Streams the historical full metadata ledger and refreshes file presence only.
It does not load tensors, generate features, run MMseqs or inspect model scores.
Known-clean chain membership is a reuse hint, not a new leakage certification.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time


CHEMISTRY_TYPES = {
    "regular_organic", "nucleotide_cofactor", "carbohydrate_glycan",
    "lipid_long_chain", "metal_complex",
}


def policies(row):
    heavy, contacts = int(row["heavy_atoms_meta"]), int(row["binding_residue_count"])
    identity = all(row.get(k) for k in ("pdb_id", "protein_chain", "binding_site_id", "ligand_id"))
    exact = row["mapping_status"] == "exact_match" and row["mapping_complete"] == "1"
    core = identity and exact
    regular = row["route_meta"] == "regular_organic"
    chemistry = row["primary_type_meta"] in CHEMISTRY_TYPES
    return {
        "old_metadata_policy": bool(core and regular and 6 <= heavy <= 128 and contacts >= 3),
        "route5_remove_upper_cap": bool(core and regular and heavy >= 6 and contacts >= 3),
        "chemistry_6_512_contact3": bool(core and chemistry and 6 <= heavy <= 512 and contacts >= 3),
        "chemistry_3_512_contact1": bool(core and chemistry and 3 <= heavy <= 512 and contacts >= 1),
        "all_types_exact_positive": bool(core and heavy >= 1 and contacts >= 1),
    }


def file_names(root, suffix):
    with os.scandir(root) as entries:
        return {e.name for e in entries if e.name.endswith(suffix) and e.is_file()}


def asset_index(root):
    protein = {s[:-len("_protein.pt")] for s in file_names(root / "protein_features", "_protein.pt")}
    esm = {s[:-len("_esm2.pt")] for s in file_names(root / "esm2_features", "_esm2.pt")}
    fixed, ligand = set(), set()
    with os.scandir(root / "fixed_receptors") as entries:
        for e in entries:
            if e.is_dir() and (Path(e.path) / "receptor_fixed.cif").is_file():
                fixed.add(e.name)
    with os.scandir(root / "ligand_features") as entries:
        for e in entries:
            if e.is_dir():
                ligand.update((e.name, s[:-3]) for s in file_names(Path(e.path), ".pt"))
    return dict(protein=protein, esm=esm, fixed=fixed, ligand=ligand)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Bucket:
    def __init__(self):
        self.counts = Counter()
        self.chains, self.pdbs, self.stems = set(), set(), set()
        self.types, self.heavy_bins, self.ligands = Counter(), Counter(), Counter()
        self.sites = defaultdict(set)

    def add(self, row, presence, current, known_chain, direct_eval, current_pdb_train):
        stem = row["feature_stem"]
        chain = (row["pdb_id"], row["protein_chain"])
        c = self.counts
        c["rows"] += 1
        c["duplicate_stem_rows"] += stem in self.stems
        self.stems.add(stem)
        self.chains.add(chain)
        self.pdbs.add(row["pdb_id"])
        self.types[row["primary_type_meta"]] += 1
        self.ligands[row["ligand_id"]] += 1
        heavy = int(row["heavy_atoms_meta"])
        label = next((name for limit, name in ((5, "1-5"), (50, "6-50"), (128, "51-128"),
                     (256, "129-256"), (512, "257-512")) if heavy <= limit), ">512")
        self.heavy_bins[label] += 1
        core = presence["fixed"] and presence["protein"] and presence["esm"]
        ready = core and presence["ligand"]
        c["core_assets_present"] += core
        c["all_assets_present_not_tensor_qc"] += ready
        c["core_present_ligand_missing"] += core and not presence["ligand"]
        c["protein_present_esm_missing"] += presence["protein"] and not presence["esm"]
        c["current_biolip_train"] += current
        c["direct_final_eval_or_benchmark_pdb"] += direct_eval
        c["old_pdbbind_overlap_but_current_pdbbind_train"] += bool(row["pdbbind_split"]) and current_pdb_train
        c["old_pdbbind_overlap_but_current_pdbbind_train_ready"] += bool(row["pdbbind_split"]) and current_pdb_train and ready
        c["new_all_assets_present_before_homology_qc"] += ready and not current and not direct_eval
        c["new_core_present_before_homology_qc"] += core and not current and not direct_eval
        c["new_known_clean_chain"] += known_chain and not current
        c["new_known_clean_chain_all_assets_present"] += known_chain and not current and ready
        c["new_known_clean_chain_ligand_missing"] += known_chain and not current and core and not presence["ligand"]
        if core and not direct_eval:
            residues = frozenset(row["binding_residues"].split())
            if residues:
                self.sites[chain].add((row["ligand_id"], residues))

    def report(self):
        multi, low_overlap, capped = 0, 0, 0
        for observations in self.sites.values():
            signatures = {s for _, s in observations}
            multi += len(signatures) >= 2
            ordered = sorted(observations, key=lambda x: (x[0], sorted(x[1])))
            if len(ordered) > 128:
                capped += 1
                ordered = ordered[:128]
            found = any(
                a != b and len(x & y) <= 2 and len(x & y) / len(x | y) <= .2
                for i, (a, x) in enumerate(ordered) for b, y in ordered[i + 1:]
            )
            low_overlap += found
        return dict(
            counts=dict(self.counts), unique_pdbs=len(self.pdbs), unique_chains=len(self.chains),
            unique_feature_stems=len(self.stems), primary_types=dict(self.types),
            heavy_atom_bins=dict(self.heavy_bins), top_ligand_ids=self.ligands.most_common(12),
            site_proxy=dict(scope="Core-assets-present, direct-eval-PDB excluded; NOT homology-cleared",
                groups=len(self.sites), groups_distinct_masks=multi,
                groups_different_ccd_and_low_overlap=low_overlap, groups_comparison_capped=capped,
                warning="Metadata masks, no centroid/alignment/chemical-equivalence verification"),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--biolip-root", type=Path, required=True)
    ap.add_argument("--final-split-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    root, final = args.biolip_root, args.final_split_root
    metadata = root / "qc_training_manifest_allclean_20260811/biolip_pairs_metadata.tsv.gz"
    bio_ids_file = final / "biolip_train_ids.txt"
    ids = set(bio_ids_file.read_text().split())
    known_chains = set()
    with (root / "v11_allclean_route5_20260811/pilot_manifest.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["sample_id"] in ids:
                known_chains.add((r["pdb_id"], r["protein_chain"]))
    split_file = final / "pdbbind_chain_si30_splits.txt"
    pdb_train, direct_eval = set(), set()
    for line in split_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        split, key = line.split()[:2]
        (pdb_train if split == "train" else direct_eval).add(key.lower())
    with (final / "external_chains.metadata.jsonl").open() as f:
        for line in f:
            for member in json.loads(line)["members"]:
                direct_eval.add(member["sample_id"].split(":", 1)[-1][:4].lower())
    assets = asset_index(root)
    print("ASSETS", {k: len(v) for k, v in assets.items()}, flush=True)
    buckets = defaultdict(Bucket)
    routes, seen_current = Counter(), set()
    with gzip.open(metadata, "rt", newline="") as f:
        for n, row in enumerate(csv.DictReader(f, delimiter="\t"), 1):
            pdb, chain, stem = row["pdb_id"], row["protein_chain"], row["feature_stem"]
            presence = dict(protein=pdb in assets["protein"], fixed=pdb in assets["fixed"],
                            esm=f"{pdb}_{chain}" in assets["esm"], ligand=(pdb, stem) in assets["ligand"])
            current = stem in ids
            if current:
                seen_current.add(stem)
            routes[row["route_meta"]] += 1
            selected = policies(row)
            selected["current_biolip_reference"] = current
            for policy, take in selected.items():
                if take:
                    buckets[policy].add(row, presence, current, (pdb, chain) in known_chains,
                                        pdb in direct_eval, pdb in pdb_train)
            if n % 100000 == 0:
                print("ROWS", n, "seconds", round(time.monotonic() - started, 1), flush=True)
    if seen_current != ids:
        raise RuntimeError(f"Missing current IDs: {sorted(ids-seen_current)[:10]}")
    report = dict(
        created_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        raw_rows=n, asset_index_counts={k:len(v) for k,v in assets.items()}, raw_routes=dict(routes),
        current_biolip_ids=len(ids), known_current_biolip_chains=len(known_chains),
        policies={name:b.report() for name,b in buckets.items()},
        provenance={str(p): dict(sha256=sha256(p), size=p.stat().st_size,
            mtime_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat())
            for p in (metadata, bio_ids_file, split_file, Path(__file__))},
        limitations=["Historical metadata, refreshed file presence only; no tensor QC",
            "No new MMseqs search; expansion counts are upper bounds before homology filtering",
            "Known-clean chain keys must still be verified against exact cached sequence/residue axes",
            "No new train allowlist; no features extracted; no split/config/model changed"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "policies"}, indent=2), flush=True)
    print("AUDIT_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
