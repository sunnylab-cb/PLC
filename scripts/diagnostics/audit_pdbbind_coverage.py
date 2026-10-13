#!/usr/bin/env python3
"""Reconstruct PDBbind coverage without approving new training samples.

Only inventories filenames and previously computed homology alignments. No
sequence search, tensor loading, feature extraction or split modification.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


YEARS = ("1981-2000", "2001-2010", "2011-2019")


def split_ids(path):
    groups = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split row in {path}: {line}")
        groups[parts[0]].add(parts[1].lower())
    return dict(groups)


def all_ids(groups):
    return set().union(*groups.values())


def sample_dirs(root):
    result = {}
    if root is None:
        return result
    for year in YEARS:
        parent = root / year
        if parent.is_dir():
            for p in parent.iterdir():
                if p.is_dir() and len(p.name) == 4 and p.name.isalnum():
                    key = p.name.lower()
                    if key in result:
                        raise ValueError(f"Duplicate PDB directory {key}: {p}")
                    result[key] = p
    return result


def feature_inventory(root):
    specs = {
        "esm2": ("esm2_latents", "_protein.pt"),
        "protein_v4_conf0": ("complete_features_v4_nma", "_protein_conf0.pt"),
        "ligand_v5": ("ligand_features_v5", "_ligand.pt"),
    }
    return {name: {key for key, p in sample_dirs(root / folder).items()
                   if (p / (key + suffix)).is_file()}
            for name, (folder, suffix) in specs.items()}


def metadata(path):
    with path.open() as f:
        return {r["record_id"]: r for r in map(json.loads, f)}


def qualifying_hit(values):
    identity = float(values[4])
    if identity > 1:
        identity /= 100
    qcov, tcov = float(values[7]), float(values[8])
    return identity, qcov, tcov, identity >= .3 and max(qcov, tcov) >= .8


def homology_breakdown(root):
    queries = metadata(root / "pdbbind_chains.metadata.jsonl")
    targets = metadata(root / "external_chains.metadata.jsonl")
    target_sources = {k: {m["sample_id"].split(":", 1)[0] for m in r["members"]}
                      for k, r in targets.items()}
    external_ids = {m["sample_id"].split(":", 1)[-1][:4].lower()
                    for r in targets.values() for m in r["members"]}
    by_source = defaultdict(set)
    max_identity, longest_shorter_chain = {}, {}
    both_coverage, passing, n = set(), set(), 0
    with (root / "pdbbind_vs_external.tsv").open() as f:
        for line in f:
            v = line.split()
            ident, qc, tc, ok = qualifying_hit(v)
            if not ok:
                continue
            n += 1
            samples = {m["sample_id"] for m in queries[v[0]]["members"]}
            passing.update(samples)
            for source in target_sources[v[1]]:
                by_source[source].update(samples)
            if min(qc, tc) >= .8:
                both_coverage.update(samples)
            short = min(int(v[5]), int(v[6]))
            for sample in samples:
                max_identity[sample] = max(max_identity.get(sample, 0), ident)
                longest_shorter_chain[sample] = max(longest_shorter_chain.get(sample, 0), short)
    expected = set((root / "pdbbind_external_excluded_ids.txt").read_text().split())
    if passing != expected:
        raise ValueError("Reparsed external exclusions differ from saved IDs")
    bins = Counter()
    for ident in max_identity.values():
        bins[next(label for bound, label in ((.4, "[.3,.4)"), (.5, "[.4,.5)"),
                  (.9, "[.5,.9)"), (.999999, "[.9,1)"), (1.01, "1.0")) if ident < bound)] += 1
    coach, holo = by_source["coach"], by_source["holo"]
    report = dict(
        qualifying_alignments=n, excluded_complexes=len(passing),
        by_source={s: len(x) for s, x in by_source.items()},
        coach_only=len(coach-holo), holo_only=len(holo-coach), both=len(coach & holo),
        maximum_hit_identity_bins=dict(bins),
        excluded_even_with_both_coverage=len(both_coverage),
        excluded_only_by_asymmetric_coverage=len(passing-both_coverage),
        all_qualifying_hits_have_shorter_chain_below={str(k): sum(v < k for v in longest_shorter_chain.values())
                                                    for k in (20, 30, 50, 100)},
        caveat="Sensitivity counts only; no changed leakage policy or recovery approval",
    )
    return report, external_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-split", type=Path, required=True)
    ap.add_argument("--legacy-split", type=Path, required=True)
    ap.add_argument("--legacy-exclude", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path)
    ap.add_argument("--chain-audit-root", type=Path)
    ap.add_argument("--final-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    full, legacy = split_ids(args.full_split), split_ids(args.legacy_split)
    full_ids, old_ids = all_ids(full), all_ids(legacy)
    exclusions = {r.strip().lower() for r in args.legacy_exclude.read_text().splitlines()
                  if r.strip() and not r.startswith("#")}
    raw = sample_dirs(args.raw_root)
    assets = feature_inventory(args.data_root)
    common = set.intersection(*assets.values())
    removed = full_ids-old_ids
    report = dict(
        full_split=dict(counts={k: len(v) for k,v in full.items()}, total=len(full_ids)),
        legacy_split=dict(counts={k: len(v) for k,v in legacy.items()}, total=len(old_ids)),
        full_minus_legacy=len(removed), legacy_minus_full=sorted(old_ids-full_ids),
        legacy_exclusion_list=len(exclusions),
        removed_in_legacy_exclusion_list=len(removed & exclusions),
        removed_not_in_legacy_exclusion_list=sorted(removed-exclusions),
        feature_counts={k: len(v) for k,v in assets.items()}, all_three_files=len(common),
        full_missing_each_file={k: len(full_ids-v) for k,v in assets.items()},
        outside_legacy_all_three_files=len(common-old_ids),
        outside_legacy_in_full_all_three_files=len((common-old_ids) & full_ids),
        raw_directory_count=len(raw) if args.raw_root else None,
        raw_not_in_full=sorted(set(raw)-full_ids) if args.raw_root else None,
        full_not_in_raw=sorted(full_ids-set(raw)) if args.raw_root else None,
        raw_pair_files_missing={key: [suffix for suffix in ("_protein.pdb", "_ligand.sdf")
                                     if not (p/(key+suffix)).is_file()]
                                for key,p in raw.items()
                                if any(not (p/(key+s)).is_file() for s in ("_protein.pdb", "_ligand.sdf"))},
        warnings=["File presence is not tensor/geometry QC; no new homology search", "No approved training expansion"],
    )
    external = set()
    if args.chain_audit_root:
        report["chain_split_summary"] = json.loads((args.chain_audit_root/"summary.json").read_text())
        report["external_homology"], external = homology_breakdown(args.chain_audit_root)
        report["removed_by_legacy_not_current_direct_external"] = len(removed-external)
        report["outside_legacy_all_three_files_not_current_direct_external"] = len(common-old_ids-external)
    if args.final_root:
        report["final_repair"] = json.loads((args.final_root/"repair_summary.json").read_text())
        report["final_counts"] = {k: len(v) for k,v in split_ids(args.final_root/"pdbbind_chain_si30_splits.txt").items()}
    report["provenance"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.full_split, args.legacy_split, args.legacy_exclude, Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    with args.output.with_suffix(".candidates.tsv").open("w") as f:
        f.write("pdb_id\tin_full_split\tlegacy_direct_excluded\tcurrent_direct_external\tall_three_files\n")
        for key in sorted((full_ids | set(raw) | set.union(*assets.values()))-old_ids):
            direct = str(int(key in external)) if args.chain_audit_root else "not_checked"
            f.write(f"{key}\t{int(key in full_ids)}\t{int(key in exclusions)}\t{direct}\t{int(key in common)}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
