#!/usr/bin/env python3
"""Freeze G0/S0/S1/T1 scratch pilots against the audited expanded joint split."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_libre_transfer import sha256, source_hashes

VARIANTS = ("G0_small_global", "S0_local_gvp", "S1_cond_gvp", "T1_cond_triangle")
BASE_SHA = "1dee23e167615c253233c02ca2376fe80fa61868373b44725b9fd305ee3caa60"


def make_configs(base, experiment, source_root):
    if base["experiment"]["seed"] != 44 or base["model"]["protein_hybrid_pattern"] != "GFGF":
        raise ValueError("Expected seed44 GFGF reference")
    if base["data"]["label_contact_thresh"] != 5 or base["data"]["max_seq_len"] != 1800:
        raise ValueError("Unexpected label/crop contract")
    result = {}
    for name in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["experiment"].update(name=f"joint_si30_{name}_scratch_seed44", version_suffix="localscan",
                                 output_dir=str(experiment / "outputs" / name))
        cfg["scan_metadata_file"] = str(source_root / "split/candidate_split.tsv")
        cfg["training"].update(init_from_checkpoint=None, init_from_v11m3=None, freeze_v11_base=False,
            freeze_v11m3_base=False, early_stopping_metric="f1_c0", early_stopping_patience=30,
            save_every=0, save_latest_checkpoint=False, batch_size=16, accumulation_steps=1)
        if name != "G0_small_global":
            keep = ("d_esm", "dropout", "classifier_dropout", "coord_noise_std", "esm_dropout_prob",
                    "physchem_dropout_prob", "gvp_dropout_prob", "ligand_source_dropout_p",
                    "ligand_dropout_hidden_p", "ligand_dropout_local54_p", "ligand_edge_rbf_bins")
            cfg["model"] = {k: base["model"][k] for k in keep}
            cfg["model"].update(scan_radius=12.0, scan_k=32, scan_chunk=128, use_unimol_ligand_input=False)
        else:
            cfg["model"]["d_model"] = 256
        cfg["model"].update(scan_variant=name, use_gradient_checkpointing=True, use_compile=False)
        result[name] = cfg
    return result


def verify(root, require_smoke=False):
    manifest = json.loads((root / "prepared/provenance.json").read_text())
    for path, expected in manifest["asset_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    if source_hashes(root / "code_v1") != manifest["source_hashes"]:
        raise RuntimeError("Immutable code snapshot changed")
    for name, expected in manifest["config_hashes"].items():
        if sha256(root / "prepared" / f"{name}.yaml") != expected:
            raise RuntimeError(f"Config changed: {name}")
        if require_smoke:
            smoke = json.loads((root / "smoke" / f"{name}.json").read_text())
            if not smoke["passed"] or smoke["provenance_sha256"] != sha256(root / "prepared/provenance.json"):
                raise RuntimeError(f"No matching successful smoke: {name}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--require-smoke", action="store_true")
    args = p.parse_args()
    root = args.experiment
    if args.verify:
        verify(root, args.require_smoke)
        print("LOCAL_SCAN_FROZEN_INPUTS_VERIFIED", flush=True)
        return
    source = args.source_root
    base_path = source / "prepared/GFGF.yaml"
    if sha256(base_path) != BASE_SHA:
        raise RuntimeError("Not the audited expanded-joint reference configuration")
    old = json.loads((source / "prepared/provenance.json").read_text())
    # Reuse labels/stats/membership; never copy, rebuild or normalize the data.
    assets = {k: v for k, v in old["asset_hashes"].items()
              if k.startswith(str(source / "dataset")) or k.startswith(str(source / "split"))}
    assets[str(base_path)] = BASE_SHA
    assets[str(source / "prepared/provenance.json")] = sha256(source / "prepared/provenance.json")
    for path, expected in assets.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Upstream input changed: {path}")
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    configs = make_configs(yaml.safe_load(base_path.read_text()), root, source)
    for name, cfg in configs.items():
        (prepared / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    manifest = dict(asset_hashes=assets, source_hashes=source_hashes(root / "code_v1"),
        config_hashes={name: sha256(prepared / f"{name}.yaml") for name in VARIANTS},
        seed=44, start="scratch", test_forward_used=False, cohort="expanded_joint_global_si30",
        counts=old["loader_qc"]["counts"], sources=old["loader_qc"]["by_source"],
        checkpoint_policy="best_f1_c0_model.pt and best finite loss best_model.pt only",
        note="Local arms replace the global trunk. G0 reduces d_model only. All use F1 stopping/checkpointing.")
    (prepared / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(root)
    print(json.dumps({"prepared": str(prepared), "variants": VARIANTS}), flush=True)


if __name__ == "__main__":
    main()
