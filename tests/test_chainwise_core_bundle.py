import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts.sync.sync_chainwise_core_to_star import (
    FAMILIES, VARIANT, check_inventory, finish_on_star, paths_for,
    relocate_config, verify_core, write_json,
)


def config():
    return dict(model=dict(scan_variant="G0_small_global", use_unimol_ligand_input=False),
        data=dict(nma_max_conf=0, label_contact_thresh=5.0, max_seq_len=1800),
        training=dict(batch_size=16, init_from_checkpoint=None, init_from_v11m3=None,
                      freeze_v11_base=False, freeze_v11m3_base=False, early_stopping_metric="f1_c0"),
        experiment=dict(seed=44, name="unchanged"))


def test_relocation_preserves_science(tmp_path):
    cfg = config()
    original = copy.deepcopy(cfg)
    new = relocate_config(cfg, tmp_path)
    assert cfg == original
    assert new["model"] == cfg["model"] and new["training"] == cfg["training"]
    assert new["data"]["label_contact_thresh"] == 5.0
    assert new["data"]["esm_features_dir"] == "esm2_latents"
    cfg["training"]["init_from_checkpoint"] = "old_checkpoint.pt"
    with pytest.raises(ValueError, match="scratch"):
        relocate_config(cfg, tmp_path)


def build_toy(root):
    for folder in ("prepared", "source", "data", "code_v1", "model_code", "deps"):
        (root / folder).mkdir(parents=True)
    cfg = config()
    (root / "source/hpc_config.yaml").write_text(yaml.safe_dump(cfg))
    (root / "prepared" / f"{VARIANT}.yaml").write_text(yaml.safe_dump(relocate_config(cfg, root)))
    # Deliberately missing weights. Historical provenance is descriptive only.
    write_json(root / "source/chainwise_manifest.json", {"weights": "/missing/esm2.pt", "weights_sha256": "historical"})
    (root / "prepared/metadata.tsv").write_text("sample_id\tsource\tsplit\tn_chains\np1\tpdbbind\ttrain\t2\n")
    rows = []
    for name, suffix in FAMILIES.items():
        rel = f"{name}/2001-2010/p1/p1{suffix}"
        p = root / "data" / rel
        p.parent.mkdir(parents=True)
        p.write_bytes(b"cached feature, not a model checkpoint")
        rows.append(dict(path=rel, bytes=p.stat().st_size))
    (root / "data/residue_ligand_distances_dcenter.pt").write_bytes(b"frozen labels")
    with (root / "prepared/features.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=("path", "bytes"), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    write_json(root / "prepared/transfer_receipt.json", dict(checksum_verified=True, destination=str(root),
        total=1, counts={"train": 1}, source_completion_sha256="verified-source", weights_transferred=False))


def test_core_verification_requires_no_esm_weights(tmp_path):
    build_toy(tmp_path)
    finish_on_star(tmp_path)
    p = verify_core(tmp_path)
    assert p["complete"] and not p["weights_transferred"]
    assert p["checkpoint_initialization"] is None
    assert p["model_context"] == "FULL COMPLEX"
    assert not p["training_launched"]
    with pytest.raises(FileNotFoundError):
        verify_core(tmp_path, smoke=True)
    (tmp_path / "data/residue_ligand_distances_dcenter.pt").write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="Changed core asset"):
        verify_core(tmp_path)


def test_missing_feature_or_absolute_symlink_is_not_ready(tmp_path):
    build_toy(tmp_path)
    path = tmp_path / "data/esm2_latents/2001-2010/p1/p1_protein.pt"
    source = tmp_path / "original_feature.pt"
    path.rename(source)
    with pytest.raises(ValueError, match="Missing/changed"):
        check_inventory(tmp_path, tmp_path / "prepared/features.tsv", 1)
    path.symlink_to(source)
    with pytest.raises(ValueError, match="linked feature"):
        check_inventory(tmp_path, tmp_path / "prepared/features.tsv", 1)


def test_changed_config_and_wrong_receipt_rejected(tmp_path):
    build_toy(tmp_path)
    p = tmp_path / "prepared/transfer_receipt.json"
    d = json.loads(p.read_text()); d["checksum_verified"] = False
    write_json(p, d)
    with pytest.raises(ValueError, match="receipt"):
        finish_on_star(tmp_path)
    d["checksum_verified"] = True
    write_json(p, d)
    path = tmp_path / "prepared" / f"{VARIANT}.yaml"
    cfg = yaml.safe_load(path.read_text()); cfg["training"]["batch_size"] = 4
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="configuration changed"):
        finish_on_star(tmp_path)


def test_source_paths_are_exact(tmp_path):
    suffix = "_protein_conf0.pt"
    p = tmp_path / "2001-2010/p1" / ("p1" + suffix)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"feature")
    assert paths_for(tmp_path, suffix, {"p1"}) == {"p1": "2001-2010/p1/p1_protein_conf0.pt"}
    with pytest.raises(ValueError, match="Missing"):
        paths_for(tmp_path, suffix, {"p1", "p2"})
