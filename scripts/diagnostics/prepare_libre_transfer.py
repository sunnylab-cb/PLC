#!/usr/bin/env python3
"""Freeze four one-variable, seed44 scratch probes from the audited GFGF recipe."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

BASE_SHA = "c58d7a8d4dd1a1840232e5cda9b648b62085a93da7dc14c00753e3ac5944fa69"
SPLIT_SHA = "170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc"
LABEL_SHA = "c1b56f25dc15f80c81c3321ec48ad2c67fd87ec9aca02eca76d09f240ff980b6"
VARIANTS = {
    "plc_product_readout": {"model.conditional_readout_mode": "product", "model.conditional_readout_dim": 64},
    "plc_additive_readout_control": {"model.conditional_readout_mode": "additive", "model.conditional_readout_dim": 64},
    "plc_cnc_off": {"training.cnc.enabled": False},
    "plc_residue_classmass": {"model.residue_classmass_weight": 0.02,
                              "model.residue_classmass_temperature": 0.2,
                              "model.residue_classmass_max_per_class": 32,
                              "model.residue_classmass_seed": 44},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for folder in ("src", "scripts", "tests")
            for p in sorted((root / folder).rglob("*")) if p.suffix in (".py", ".slurm", ".sh")}


def make_configs(base, experiment):
    t, m = base["training"], base["model"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "freeze_v11_base", "freeze_v11m3_base", "resume")):
        raise ValueError("This experiment must start from scratch with an unfrozen trunk")
    if m["protein_hybrid_pattern"] != "GFGF" or m["use_compile"] or base["experiment"]["seed"] != 44:
        raise ValueError("Expected the frozen seed44 GFGF recipe, without torch.compile")
    configs = {}
    for name, changes in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = name + "_GFGF_scratch_seed44"
        cfg["experiment"]["output_dir"] = str(Path(experiment) / "outputs")
        for dotted, value in changes.items():
            parts, target = dotted.split("."), cfg
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        configs[name] = cfg
    return configs


def verify_prepared(experiment):
    experiment = Path(experiment)
    manifest = json.loads((experiment / "prepared/provenance.json").read_text())
    for path, expected in manifest["asset_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen asset changed: {path}")
    if source_hashes(experiment / "code_v1") != manifest["source_hashes"]:
        raise RuntimeError("Source snapshot changed after preparation/smoke")
    for name, expected in manifest["config_hashes"].items():
        if sha256(experiment / "prepared" / f"{name}.yaml") != expected:
            raise RuntimeError(f"Prepared config changed: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_prepared(args.experiment)
        print("FROZEN_INPUTS_VERIFIED")
        return
    if args.base is None or sha256(args.base) != BASE_SHA:
        raise RuntimeError("Use the exact audited GFGF output config, not an edited recipe")
    base = yaml.safe_load(args.base.read_text())
    assets = {str(args.base): BASE_SHA, base["data"]["split_file"]: SPLIT_SHA,
              base["data"]["distances_file"]: LABEL_SHA}
    for path, expected in assets.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Asset hash mismatch: {path}")
    prepared = args.experiment / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    configs = make_configs(base, args.experiment)
    for name, cfg in configs.items():
        (prepared / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    manifest = dict(asset_hashes=assets, source_hashes=source_hashes(args.experiment / "code_v1"),
                    config_hashes={n: sha256(prepared / f"{n}.yaml") for n in configs},
                    scientific_changes=VARIANTS, seed=44, start="scratch", test_used=False,
                    note="Experiment name/output directory also differ. No change to scheduler/stopping.")
    (prepared / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(prepared), "variants": list(configs)}))


if __name__ == "__main__":
    main()
