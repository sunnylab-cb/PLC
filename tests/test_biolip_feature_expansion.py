import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics import biolip_feature_expansion as expand


def row(p="1abc", c="A", ligand="LIG"):
    return dict(pdb_id=p, protein_chain=c, ligand_id=ligand,
                feature_stem=f"{p}_{c}_BS01_{ligand}", binding_site_id="BS01",
                ligand_resseq="100", binding_residues="1 2 3", route_meta="regular_organic",
                mapping_status="exact_match", mapping_complete="1", heavy_atoms_meta="20",
                binding_residue_count="3")


def fixture_plan(tmp_path):
    root, exp, qc = (tmp_path / k for k in ("bio", "experiment", "qc"))
    for p in (root / "protein_features", root / "esm2_features", root / "ligand_features",
              root / "fixed_receptors", root / "shared_receptors", exp, qc):
        p.mkdir(parents=True)
    rows = [row(), row("2abc"), row("3abc"), row("4abc"), row("5abc")]
    rows[-1]["route_meta"] = "unexpected"
    # One accepted existing sample; one missing ligand; one missing all features;
    # one legacy tensor to regenerate without overwriting it.
    for p in ("1abc", "2abc", "4abc"):
        (root / "protein_features" / f"{p}_protein.pt").touch()
        (root / "esm2_features" / f"{p}_A_esm2.pt").touch()
    for r in (rows[0], rows[3]):
        p = root / "ligand_features" / r["pdb_id"]
        p.mkdir()
        (p / f"{r['feature_stem']}.pt").touch()
    metadata = root / expand.METADATA
    metadata.parent.mkdir()
    with gzip.open(metadata, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows + [dict(rows[0])])
    (qc / "samples.jsonl").write_text(json.dumps(dict(sample_id=rows[0]["feature_stem"], source="biolip")) + "\n")
    (qc / "pool_summary.json").write_text(json.dumps({"output_sha256": {
        "samples.jsonl": hashlib.sha256((qc / "samples.jsonl").read_bytes()).hexdigest()}}))
    return argparse.Namespace(biolip_root=root, experiment=exp, qc_pool=qc)


def test_full_plan_keeps_missing_features_and_deduplicates(tmp_path):
    args = fixture_plan(tmp_path)
    expand.plan(args)
    summary = expand.verified_plan(args)
    assert summary["candidates"] == 4
    assert summary["ligand_ready"] == 2
    assert summary["ligand_remaining"] == 1
    assert summary["missing_unique_proteins"] == 1
    assert summary["chains_to_extract"] == 1
    assert summary["counts"]["previously_qc_passed"] == 1
    assert summary["counts"]["ligand_reextract_existing"] == 1
    assert summary["training_approved"] is False
    assert len(expand.read_tsv(args.experiment / "plan/observation_aliases.tsv")) == 1
    expand.plan(args)  # Same contract may resume.
    candidate = args.experiment / "plan/candidates.tsv"
    candidate.write_text(candidate.read_text() + "corruption\n")
    with pytest.raises(ValueError, match="changed plan artifact"):
        expand.verified_plan(args)


def test_reject_modified_qc_ledger(tmp_path):
    args = fixture_plan(tmp_path)
    (args.qc_pool / "samples.jsonl").write_text("")
    with pytest.raises(ValueError, match="ledger hash"):
        expand.plan(args)


@pytest.mark.parametrize("name", ["", "..", "../evil", "x/y", "x\\y", "x\n"])
def test_path_guard(name):
    with pytest.raises(ValueError):
        expand.safe_name(name)


def test_chain_axis_handles_insertion_codes_and_aliases():
    p = dict(residue_names=["ALA", "MSE", "GLY"], chain_ids=["A", "A", "B"],
             residue_indices=[10, 10, 10], insertion_codes=["", "A", ""])
    keys, letters = expand.chain_axis(p, "A")
    assert keys == [("A", 10, ""), ("A", 10, "A")]
    assert letters == ["A", "M"]
    with pytest.raises(ValueError, match="chain missing"):
        expand.chain_axis(p, "C")
    p["insertion_codes"][1] = ""
    with pytest.raises(ValueError, match="duplicate"):
        expand.chain_axis(p, "A")


def test_embedding_axis_is_exact_not_prefix():
    keys = [("A", 1, ""), ("A", 2, "A")]
    letters = ["A", "G"]
    data = dict(embeddings=torch.zeros(2, 1280), chain_id=["A", "A"],
                residue_index=[1, 2], insertion_code=["", "A"], residue_name_1=letters)
    expand.validate_embedding(data, keys, letters)
    with pytest.raises(ValueError, match="identity"):
        expand.validate_embedding(data, keys, ["A", "A"])
    data["residue_index"].reverse()
    with pytest.raises(ValueError, match="axis"):
        expand.validate_embedding(data, keys, letters)


def test_overlay_isolated_precedence_and_no_link_replacement(tmp_path):
    root, prep, view = (tmp_path / n for n in ("old", "new", "view"))
    r = row()
    rel = Path("ligand_features/1abc/1abc_A_BS01_LIG.pt")
    for base in (root, prep):
        (base / rel).parent.mkdir(parents=True)
        (base / rel).touch()
    assert expand.paths_for(root, prep, r)["ligand"] == prep / rel
    expand.link_checked(prep / rel, view / rel)
    expand.link_checked(prep / rel, view / rel)
    with pytest.raises(ValueError, match="refusing"):
        expand.link_checked(root / rel, view / rel)


def test_collect_not_training_admission(tmp_path):
    args = fixture_plan(tmp_path)
    expand.plan(args)
    expand.collect(args)
    report = json.loads((args.experiment / "coverage_summary.json").read_text())
    assert report["training_approved"] is False
    assert report["counts"]["all_files_present"] == 2
    assert set(report["stages"].values()) == {"unfinished"}
    assert (args.experiment / "view/protein_features/1abc_protein.pt").is_symlink()
