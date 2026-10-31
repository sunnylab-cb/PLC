from argparse import Namespace
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from preparation.protein.esm2_chainwise_complex import (
    RECIPE, VARIANT, chain_parts, residue_keys, seq_hash, stitch_embeddings,
    validate_cache, window_starts, prepare, finalize, verify,
)
from scripts.diagnostics.audit_chain_background import atom_element, geometry_background, bucket_metrics
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json


def record(chains=("A", "B", "A")):
    return dict(sequence="ACD", embeddings=torch.zeros(3, 1280), chain_id=list(chains),
                residue_index=[1, 2, 3], insertion_code=["", "A", ""],
                atom37_positions=torch.randn(3, 37, 3), mode="complex")


def test_reassemble_full_complex_not_chain_samples():
    d = record()
    parts = chain_parts(d)
    assert [p["indices"] for p in parts] == [[0, 2], [1]]
    assert [p["sequence"] for p in parts] == ["AD", "C"]
    def cache(p):
        return torch.full((len(p["indices"]), 1280), 1. if p["chain"] == "A" else 2.)
    result = stitch_embeddings(d, parts, cache)
    assert result["embeddings"][:, 0].tolist() == [1., 2., 1.]
    assert residue_keys(result) == residue_keys(d)
    assert result["mode"] == "complex"
    assert result["atom37_positions"] is d["atom37_positions"]
    assert d["embeddings"].count_nonzero() == 0


def test_duplicate_missing_and_changed_axes_fail():
    d = record()
    p = chain_parts(d)
    f = lambda x: torch.ones(len(x["indices"]), 1280)
    with pytest.raises(ValueError):
        stitch_embeddings(d, p[:1], f)
    with pytest.raises(ValueError):
        stitch_embeddings(d, p + [p[0]], f)
    p[0]["sequence"] = "AA"
    with pytest.raises(ValueError):
        stitch_embeddings(d, p, f)


@pytest.mark.parametrize("n", [1, 512, 1022, 1023, 1500, 2044, 4096, 12000])
def test_windows_no_truncation(n):
    starts = window_starts(n)
    count = np.zeros(n)
    for start in starts:
        count[start:start+1022] += 1
    assert np.all(count > 0)
    assert starts[0] == 0
    assert starts[-1] == max(n - 1022, 0)


def test_bad_axes_and_nonfinite_fail():
    d = record()
    d["residue_index"][2] = 1
    with pytest.raises(ValueError, match="duplicate"):
        chain_parts(d)
    d = record()
    d["embeddings"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        chain_parts(d)


def test_cache_recipe_and_weights_required():
    d = dict(sequence="A", embeddings=torch.zeros(1, 1280), recipe=RECIPE, weights_sha256="abc")
    validate_cache(d, "A", {"weights_sha256": "abc"})
    for key, bad in (("recipe", {}), ("weights_sha256", "wrong"), ("sequence", "B")):
        wrong = dict(d, **{key: bad})
        with pytest.raises(ValueError):
            validate_cache(wrong, "A", {"weights_sha256": "abc"})


def atom_record(chains=("A", "B")):
    return dict(chain_ids=list(chains), residue_indices=[1, 2], insertion_codes=["", ""],
                residue_names=["ALA", "ALA"], all_atom_names=[["CA"], ["CA"]],
                all_atom_coords=[[[0., 0., 0.]], [[3., 0., 0.]]])


def test_background_occlusion_is_not_model_inference():
    a = geometry_background(atom_record())
    assert (a["delta_sasa"] > 0).all()
    np.testing.assert_allclose(a["nearest_other_chain"], [3., 3.])
    b = geometry_background(atom_record(("A", "A")))
    np.testing.assert_allclose(b["delta_sasa"], 0)
    assert np.isinf(b["nearest_other_chain"]).all()


def test_heavy_atom_names_and_metrics():
    assert atom_element("1HG") is None
    assert atom_element("CA") == "C"
    assert atom_element("SE") == "SE"
    with pytest.raises(ValueError):
        atom_element("ZZ")
    a = dict(nearest_other_chain=np.array([3., 7.]), delta_sasa=np.array([20., 0.]))
    m = bucket_metrics(np.array([.7, .8]), np.array([2, 0]), a, .75)
    assert m["other_chain_within_6A"]["at_0_5"]["fp"] == 1
    assert m["other_chain_within_6A"]["at_val_gate"]["fp"] == 0


def test_strata_are_logged_without_changing_original_metrics(tmp_path, monkeypatch):
    import scripts.train.train_chainwise_esm_complex as wrapper
    (tmp_path / "prepared").mkdir()
    write_json(tmp_path / "prepared/manifest.json", {"training": str(tmp_path / "old")})
    monkeypatch.setattr(wrapper.sys, "path", list(wrapper.sys.path))

    class Writer:
        def __init__(self):
            self.log_dir = str(tmp_path)
            self.scalars = []
        def add_scalar(self, *args):
            self.scalars.append(args)
        def flush(self):
            pass

    def original(records, gate):
        return {f"{source}_n_pairs": sum(r[0]["source"] == source for r in records)
                for source in ("pdbbind", "biolip")}
    trainer = SimpleNamespace(source_metrics=original, base=SimpleNamespace(SummaryWriter=Writer))
    monkeypatch.setitem(wrapper.sys.modules, "train_local_spatial_scan", trainer)
    wrapper.install(tmp_path)
    writer = trainer.base.SummaryWriter()
    rows = [(dict(source=s, n_chains=n), None, None)
            for s, n in (("pdbbind", 2), ("pdbbind", 1), ("biolip", 1))]
    result = trainer.source_metrics(rows, .53)
    assert result["pdbbind_n_pairs"] == 2
    assert result["pdbbind_multichain_n_pairs"] == 1
    assert result["biolip_single_chain_n_pairs"] == 1
    assert ("ValStrata/pdbbind_multichain_n_pairs", 1, 0) in writer.scalars
    saved = json.loads((tmp_path / "fullcomplex_val_strata.jsonl").read_text())
    assert saved["joint_val_gate"] == .53 and saved["metrics"] == result


def test_toy_end_to_end_preserves_cohort_and_configuration(tmp_path):
    root, training, upstream = [tmp_path / s for s in ("new", "g0", "joint")]
    (training / "prepared").mkdir(parents=True)
    (upstream / "split").mkdir(parents=True)
    (root / "code_v1").mkdir(parents=True)
    data = upstream / "dataset"
    rows = [dict(split="train", sample_id="multi", source="pdbbind", pdb_id="m", component_id="1"),
            dict(split="val", sample_id="single", source="pdbbind", pdb_id="s", component_id="2"),
            dict(split="test", sample_id="bio", source="biolip", pdb_id="b", component_id="3")]
    originals = {}
    for r in rows:
        sid = r["sample_id"]
        p = data / "esm2_latents/2001-2010" / sid / f"{sid}_protein.pt"
        p.parent.mkdir(parents=True)
        d = record() if sid == "multi" else record(("A", "A", "A"))
        torch.save(d, p)
        originals[sid] = p
    split = data / "splits.txt"
    split.write_text("".join(f"{r['split']}\t{r['sample_id']}\n" for r in rows))
    with (upstream / "split/candidate_split.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    write_json(upstream / "split/summary.json", dict(passed=True, cross_set_audit=dict(edges=0,max_global_si=.29)))
    cfg = dict(model=dict(scan_variant="G0_small_global"), experiment=dict(seed=44),
               training=dict(init_from_checkpoint=None, init_from_v11m3=None),
               data=dict(label_contact_thresh=5,max_seq_len=1800,data_root=str(data),
                         esm_features_dir="esm2_latents",split_file=str(split)))
    cfg_path = training / "prepared/G0_small_global.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    write_json(training / "prepared/provenance.json", dict(asset_hashes={str(split):sha(split)},
               source_hashes={}, config_hashes={"G0_small_global":sha(cfg_path)}))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test weights")
    args = Namespace(experiment=root,training=training,upstream=upstream,weights=weights,
                     libre_cache=tmp_path / "unused",workers=1)
    prepare(args)
    m = json.loads((root / "prepared/manifest.json").read_text())
    assert list(m["changed"]) == ["multi"]
    assert len(m["sequences"]) == 2
    for sid in ("single", "bio"):
        target = root / "esm2_latents/2001-2010" / sid / f"{sid}_protein.pt"
        assert target.resolve() == originals[sid].resolve()
    new = yaml.safe_load((root / "prepared" / f"{VARIANT}.yaml").read_text())
    assert new["training"] == cfg["training"] and new["model"] == cfg["model"]
    assert {k:v for k,v in new["data"].items() if k != "esm_features_dir"} == {
        k:v for k,v in cfg["data"].items() if k != "esm_features_dir"}
    (root / "chain_cache").mkdir()
    for key, rec in m["sequences"].items():
        torch.save(dict(sequence=rec["sequence"], embeddings=torch.ones(len(rec["sequence"]),1280),
                        recipe=RECIPE,weights_sha256=sha(weights)),root / "chain_cache" / f"{key}.pt")
    write_json(root / "prepared/extraction_complete.json", dict(complete=True,
                manifest_sha256=sha(root / "prepared/manifest.json")))
    finalize(args)
    assert verify(root)["no_chain_only_samples"]
    rewritten = torch.load(root / "esm2_latents/2001-2010/multi/multi_protein.pt",weights_only=False)
    assert residue_keys(rewritten) == residue_keys(record())
    assert rewritten["embeddings"].shape == (3,1280)
