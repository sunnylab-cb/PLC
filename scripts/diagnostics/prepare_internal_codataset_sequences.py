#!/usr/bin/env python3
"""Prepare an internal-only PDBbind/BioLiP sequence pool, NOT training approval.

No old split, benchmark-ID exclusion, or external-homology exclusion is applied.
BioLiP regular-organic, exact-mapped positive observations are the first cohort;
actual native-coordinate contacts and feature/axis QC precede SI splitting. All chains
of PDBbind inputs are retained. BioLiP uses its selected input chain; all rows
sharing a PDB ID will nevertheless be indivisible during splitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time


YEARS = ("1981-2000", "2001-2010", "2011-2019")
AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))
AA3.update(MSE="M", HID="H", HIE="H", HIP="H", ASH="D", GLH="E", CYX="C",
           CYM="C", LYN="K", ASX="B", GLX="Z", SEC="U", PYL="O", UNK="X")
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUOJ")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def feature_path(root, folder, sample, suffix):
    paths = [root / folder / year / sample / (sample + suffix) for year in YEARS]
    matches = [p for p in paths if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"{sample} {folder}: expected one file, got {matches}")
    return matches[0]


def chain_sequences(data):
    """Require explicit residue axes; never silently concatenate missing chains."""
    names = data.get("residue_name_1")
    if names is None:
        names = data.get("residue_names")
        if names is not None:
            names = [AA3.get(str(x).upper(), "X") for x in names]
    chains = data.get("chain_id", data.get("chain_ids"))
    indices = data.get("residue_index", data.get("residue_indices"))
    icodes = data.get("insertion_code", data.get("insertion_codes"))
    if names is None or chains is None or indices is None or not len(names):
        raise ValueError("missing residue names/chain/index metadata")
    if icodes is None:
        icodes = [""] * len(names)
    if len({len(x) for x in (names, chains, indices, icodes)}) != 1:
        raise ValueError("residue-axis lengths disagree")
    result, seen, keys = defaultdict(list), set(), []
    for aa, chain, index, icode in zip(names, chains, indices, icodes):
        aa, chain = str(aa).upper(), str(chain)
        key = (chain, int(index), str(icode or "").strip())
        if key in seen or len(aa) != 1 or aa not in ALPHABET:
            raise ValueError(f"duplicate residue key or invalid residue: {key}, {aa}")
        seen.add(key)
        keys.append((*key, aa))
        result[chain].append(aa)
    if data.get("embeddings") is not None and len(data["embeddings"]) != len(names):
        raise ValueError("ESM length does not match residue metadata")
    return {c: "".join(a) for c, a in result.items()}, hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def bio_eligible(row):
    # Chemistry/QC scope is explicit and independent of any benchmark or split.
    return (row["route_meta"] == "regular_organic"
            and row["mapping_status"] == "exact_match" and row["mapping_complete"] == "1"
            and 6 <= int(row["heavy_atoms_meta"]) <= 128 and int(row["binding_residue_count"]) >= 3
            and all(row[k] for k in ("pdb_id", "protein_chain", "ligand_id", "binding_site_id")))


def main():
    from internal_cohort_qc import qc_sample
    from src.contact_labels import contact_protocol
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdbbind-root", type=Path, required=True)
    ap.add_argument("--pdbbind-ids", type=Path, required=True)
    ap.add_argument("--biolip-root", type=Path, help="Omit for the independent PDBbind-only pool")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-atom-contacts", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "pool_summary.json"
    metadata = (args.biolip_root / "qc_training_manifest_allclean_20260811/biolip_pairs_metadata.tsv.gz"
                if args.biolip_root else None)
    input_hashes = {str(p.resolve()): digest(p) for p in (
        args.pdbbind_ids, metadata, Path(__file__), Path(__file__).with_name("internal_cohort_qc.py"),
        Path(__file__).with_name("internal_contact_geometry.py"),
        Path(__file__).resolve().parents[2]/"src/contact_labels.py") if p}
    input_hashes["pdbbind_root"] = str(args.pdbbind_root.resolve())
    input_hashes["biolip_root"] = str(args.biolip_root.resolve()) if args.biolip_root else None
    input_hashes["contact_label_protocol"] = contact_protocol(args.min_atom_contacts)
    contract_path = out/"pool_input_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != input_hashes:
        raise ValueError("pool extraction contract changed; use a new output directory")
    if not contract_path.exists() and (out/"sequence_sources.jsonl").exists():
        raise ValueError("cannot reuse unversioned partial QC results")
    save_json(contract_path, input_hashes)
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["input_sha256"] != input_hashes:
            raise ValueError("pool input hash changed; use a new output directory")
        for name, sha in saved["output_sha256"].items():
            if digest(out / name) != sha:
                raise ValueError(f"pool artifact changed: {name}")
        print("POOL_ALREADY_PREPARED", json.dumps(saved), flush=True)
        return
    started = time.monotonic()
    ids = []
    for line in args.pdbbind_ids.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            ids.append(line.split()[1].lower())
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("duplicate or empty PDBbind IDs")
    samples, early_excluded = {}, []
    for sid in sorted(ids):
        try:
            esm = feature_path(args.pdbbind_root, "esm2_latents", sid, "_protein.pt")
            v4 = feature_path(args.pdbbind_root, "complete_features_v4_nma", sid, "_protein_conf0.pt")
            ligand = feature_path(args.pdbbind_root, "ligand_features_v5_ori", sid, "_ligand.pt")
        except ValueError as exc:
            early_excluded.append(dict(sample_id=sid, source="pdbbind", reason="missing_or_ambiguous_raw_features", detail=str(exc)))
            continue
        samples[f"pdbbind:{sid}"] = dict(source="pdbbind", sample_id=sid, pdb_id=sid,
            selected_chain=None, ligand_path=str(ligand),
            esm_path=str(esm), v4_path=str(v4), feature_files_present=True,
            min_atom_contacts=args.min_atom_contacts, training_approved=False)
    counts, excluded, aliases, seen_obs, seen_ids = Counter(), early_excluded, [], {}, set()
    counts["raw_pdbbind_rows"] = len(ids)
    with gzip.open(metadata, "rt") if metadata else nullcontext(None) as f:
        for row in csv.DictReader(f, delimiter="\t") if f else ():
            counts["raw_biolip_rows"] += 1
            if not bio_eligible(row):
                counts["outside_regular_organic_metadata_policy"] += 1
                continue
            counts["eligible_biolip_rows"] += 1
            sid, pdb, chain = row["feature_stem"], row["pdb_id"].lower(), row["protein_chain"]
            obs = (pdb, chain, row["ligand_id"], row["ligand_resseq"],
                   tuple(sorted(row["binding_residues"].split())))
            if obs in seen_obs:
                aliases.append(dict(sample_id=sid, kept=seen_obs[obs], reason="same_metadata_observation"))
                continue
            if sid in seen_ids:
                raise ValueError(f"conflicting BioLiP sample ID: {sid}")
            seen_obs[obs] = sid
            seen_ids.add(sid)
            v4 = args.biolip_root / "protein_features" / f"{pdb}_protein.pt"
            if not v4.is_file():
                excluded.append(dict(sample_id=sid, reason="missing_input_sequence_v4", path=str(v4)))
                continue
            ligand = args.biolip_root / "ligand_features" / pdb / (sid + ".pt")
            esm = args.biolip_root / "esm2_features" / f"{pdb}_{chain}_esm2.pt"
            if not ligand.is_file() or not esm.is_file():
                excluded.append(dict(sample_id=sid, source="biolip", reason="missing_ligand_or_esm",
                                     missing_ligand=not ligand.is_file(), missing_esm=not esm.is_file()))
                continue
            samples[f"biolip:{sid}"] = dict(source="biolip", sample_id=sid, pdb_id=pdb,
                selected_chain=chain, biolip_root=str(args.biolip_root), ligand_metadata=row,
                min_atom_contacts=args.min_atom_contacts,
                ligand_path=str(ligand), esm_path=str(esm), v4_path=str(v4),
                feature_files_present=ligand.is_file() and esm.is_file(), training_approved=False,
                ligand_id=row["ligand_id"], ligand_resseq=row["ligand_resseq"],
                binding_residues=row["binding_residues"], heavy_atoms=int(row["heavy_atoms_meta"]))
    print("POOL_INVENTORY", dict(counts), "candidate_rows", len(samples), flush=True)
    cache_path = out / "sequence_sources.jsonl"
    extracted = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                record = json.loads(line)
                key = record["key"]
                if key in extracted:
                    raise ValueError(f"duplicate sequence cache key: {key}")
                if key not in samples:
                    raise ValueError(f"sequence cache input mismatch: {key}")
                for path, sig in record["sources"].items():
                    st = Path(path).stat()
                    if st.st_size != sig["size"] or st.st_mtime_ns != sig["mtime_ns"]:
                        raise ValueError(f"source changed since partial extraction: {path}")
                if not record["error"] and digest(record["labels_path"]) != record["labels_sha256"]:
                    raise ValueError(f"cached labels corrupt: {key}")
                extracted[key] = record
    pending = [(key, sample, str(out.resolve())) for key, sample in sorted(samples.items()) if key not in extracted]
    with cache_path.open("a") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(qc_sample, pending, chunksize=8):
            extracted[record["key"]] = record
            log.write(json.dumps(record, sort_keys=True) + "\n")
            if len(extracted) % 500 == 0:
                log.flush()
                print("FEATURE_GEOMETRY_QC", len(extracted), "/", len(samples), "seconds", round(time.monotonic()-started), flush=True)
    unique, accepted = {}, {}
    for key, sample in sorted(samples.items()):
        record = extracted[key]
        if record["error"]:
            excluded.append(dict(sample_id=sample["sample_id"], source=sample["source"], reason=record["error"]))
            continue
        seqs = set(record["chains"].values())
        seq_ids = []
        for seq in sorted(seqs):
            seq_id = "seq_" + hashlib.sha256(seq.encode()).hexdigest()
            if seq_id in unique and unique[seq_id]["sequence"] != seq:
                raise ValueError("SHA256 sequence collision")
            seq_record = unique.setdefault(seq_id, dict(sequence=seq, members=[]))
            seq_record["members"].append(key)
            seq_ids.append(seq_id)
        accepted[key] = dict(sample, sequence_ids=seq_ids, qc=record)
    with (out / "samples.jsonl").open("w") as f:
        for key, sample in sorted(accepted.items()):
            f.write(json.dumps(dict(key=key, **sample), sort_keys=True) + "\n")
    with (out / "sequences.jsonl").open("w") as f, (out / "sequences.fasta").open("w") as fa:
        for seq_id, value in sorted(unique.items()):
            f.write(json.dumps(dict(sequence_id=seq_id, **value), sort_keys=True) + "\n")
            fa.write(f">{seq_id}\n{value['sequence']}\n")
    save_json(out / "excluded_rows.json", excluded)
    save_json(out / "observation_aliases.json", aliases)
    report = dict(input_sha256=input_hashes, counts=dict(counts),
        cohort="pdbbind_biolip_joint" if args.biolip_root else "pdbbind_only",
        accepted_by_source=dict(Counter(s["source"] for s in accepted.values())),
        files_present_not_training_approved=dict(Counter(s["source"] for s in accepted.values() if s["feature_files_present"])),
        unique_sequences=len(unique), all_pairs=len(unique)*(len(unique)-1)//2,
        excluded_rows=len(excluded), exclusion_counts=dict(Counter(r["reason"] for r in excluded)),
        duplicate_observations=len(aliases),
        external_benchmarks_used=False, old_split_assignments_used=False,
        contact_label_protocol=contact_protocol(args.min_atom_contacts),
        contact_counts_by_source={source: {
            field: sum(extracted[k].get(field, 0) for k, s in accepted.items() if s["source"] == source)
            for field in ("n_contact", "n_contact_casp1", "n_contact_biolip2", "n_contact_lt5", "n_contact_le5p5")}
            for source in sorted({s["source"] for s in accepted.values()})},
        policy="native-heavy-atom d<=vdW_sum+0.5A; minimum atom contacts from versioned protocol; "
               "binding residues >=3 (also after residue cap1800); finite/axis QC; "
               "BioLiP legacy regular_organic route5, exact complete mapping, 6-128 heavy atoms, completeness>=0.8; "
               "PDBbind existing V5 native-coordinate ligands >=6 heavy atoms, no new upper atom exclusion",
        limitations=["Geometry/QC-passed pool; training still requires SI audit and separate TRAIN-only normalization",
            "Missing/invalid features excluded with ledger, not silently zero-filled",
            "Broader cross-source chemical/site deduplication not certified by protein SI",
            "Existing features are not certified train-only normalized for this new split"],
        elapsed_seconds=time.monotonic()-started)
    report["output_sha256"] = {name: digest(out/name) for name in (
        "samples.jsonl", "sequences.jsonl", "sequences.fasta", "sequence_sources.jsonl",
        "excluded_rows.json", "observation_aliases.json")}
    save_json(complete, report)
    print("POOL_COMPLETE", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
