"""Small real-data contact-label preflight before submitting long CPU jobs."""

import csv
import gzip
import json
from pathlib import Path

from internal_cohort_qc import qc_sample, load_tensor
from prepare_internal_codataset_sequences import bio_eligible, feature_path, save_json


def run_smoke(pdbbind_root, biolip_root, output, contact_rule="fixed5"):
    pdbbind_root, biolip_root, output = map(Path, (pdbbind_root, biolip_root, output))
    output.mkdir(parents=True, exist_ok=True)
    sid = "3ddq"
    sample = dict(source="pdbbind", sample_id=sid, pdb_id=sid, selected_chain=None,
                  contact_rule=contact_rule, min_atom_contacts=2)
    for key, folder, suffix in (
        ("v4_path", "complete_features_v4_nma", "_protein_conf0.pt"),
        ("esm_path", "esm2_latents", "_protein.pt"),
        ("ligand_path", "ligand_features_v5_ori", "_ligand.pt"),
    ):
        sample[key] = str(feature_path(pdbbind_root, folder, sid, suffix))
    pdb = qc_sample(("pdbbind:"+sid, sample, str(output)))
    if pdb["error"]:
        save_json(output/"failed.json", pdb)
        raise ValueError(f"PDBbind contact preflight failed: {pdb['error']}")
    attempts, bio, legacy_skipped = [], None, 0
    metadata = biolip_root/"qc_training_manifest_allclean_20260811/biolip_pairs_metadata.tsv.gz"
    with gzip.open(metadata, "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if not bio_eligible(row):
                continue
            sid, pdb_id, chain = row["feature_stem"], row["pdb_id"].lower(), row["protein_chain"]
            paths = dict(v4_path=biolip_root/"protein_features"/(pdb_id+"_protein.pt"),
                         esm_path=biolip_root/"esm2_features"/(pdb_id+"_"+chain+"_esm2.pt"),
                         ligand_path=biolip_root/"ligand_features"/pdb_id/(sid+".pt"))
            if not all(p.is_file() for p in paths.values()):
                continue
            # This directory mixes older V5 tensors and verified V6 tensors.
            # Full cohort QC records these rejections; smoke needs a V6 example.
            ligand = load_tensor(str(paths["ligand_path"]))
            if int(ligand.get("ligand_route", -1)) != 5:
                legacy_skipped += 1
                continue
            sample = dict(source="biolip", sample_id=sid, pdb_id=pdb_id, selected_chain=chain,
                          contact_rule=contact_rule, min_atom_contacts=2,
                          biolip_root=str(biolip_root), ligand_metadata=row,
                          **{k: str(p) for k, p in paths.items()})
            result = qc_sample(("biolip:"+sid, sample, str(output)))
            attempts.append(dict(key=result["key"], error=result["error"]))
            if not result["error"]:
                bio = result
                break
            if len(attempts) >= 20:
                break
    report = dict(passed=bio is not None, pdbbind=pdb, biolip=bio, biolip_attempts=attempts,
                  legacy_or_other_route_skipped=legacy_skipped,
                  scope="two real observations only; not full-cohort QC or training approval")
    save_json(output/"report.json", report)
    if bio is None:
        raise ValueError("No passing BioLiP smoke sample: "+json.dumps(attempts))
    return {source: {k: r[k] for k in (
        "key", "contact_label_protocol", "n_contact", "n_contact_casp1", "n_contact_biolip2",
        "n_contact_lt5", "n_contact_le5p5") if k in r}
        for source, r in (("pdbbind", pdb), ("biolip", bio))}
