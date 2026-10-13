#!/usr/bin/env python3
"""Freeze six original-model noise probes, with no self-distillation assets."""

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.prepare_libre_transfer import BASE_SHA, SPLIT_SHA, LABEL_SHA, sha256, source_hashes

VARIANTS = {f"gfgf_{mode}_noise{int(sigma * 100):02d}": (mode, sigma)
            for sigma in (.05, .10, .20) for mode in ("random", "background")}


def make_configs(base, experiment):
    t, m = base["training"], base["model"]
    if "self_distillation" in base or any(t.get(k) for k in (
            "init_from_checkpoint", "init_from_v11m3", "resume", "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Use the original scratch baseline, not a KD or warm-start config")
    if (m.get("use_compile") or m.get("use_gradient_checkpointing")
            or m.get("protein_hybrid_pattern") != "GFGF" or base["experiment"]["seed"] != 44):
        raise ValueError("Expected original GFGF seed44 without compile")
    configs = {}
    for name, (mode, sigma) in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = name + "_scratch_seed44"
        cfg["experiment"]["output_dir"] = str(Path(experiment) / "outputs")
        cfg["context_noise"] = dict(mode=mode, sigma=sigma, foreground_floor=.5,
            batch_probability=.25, seed=190765, num_candidates=8,
            min_center_distance=10., membership_sigma=8.)
        configs[name] = cfg
    return configs


def verify_prepared(experiment, check_smoke=False):
    experiment = Path(experiment)
    manifest = json.loads((experiment / "prepared/provenance.json").read_text())
    for path, expected in manifest["asset_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    if source_hashes(experiment / "code_v1") != manifest["source_hashes"]:
        raise RuntimeError("Source snapshot changed after preparation")
    for name, expected in manifest["config_hashes"].items():
        if sha256(experiment / "prepared" / f"{name}.yaml") != expected:
            raise RuntimeError(f"Prepared config changed: {name}")
    if check_smoke:
        smoke = json.loads((experiment / "smoke_pass.json").read_text())
        if (smoke.get("passed") is not True
                or smoke.get("provenance_sha256") != sha256(experiment / "prepared/provenance.json")
                or {r["name"] for r in smoke.get("runs", [])} != set(VARIANTS)
                or not smoke.get("zero_noise_equivalence")
                or not smoke.get("clean_eval_equivalence")
                or not smoke.get("mask_qc", {}).get("passed")):
            raise RuntimeError("Missing, failed or stale context-noise smoke")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check-smoke", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify_prepared(args.experiment, args.check_smoke)
        print("CONTEXT_NOISE_INPUTS_VERIFIED", flush=True)
        return
    if args.base is None or sha256(args.base) != BASE_SHA:
        raise RuntimeError("Use the exact original GFGF output config")
    base = yaml.safe_load(args.base.read_text())
    assets = {str(args.base): BASE_SHA, base["data"]["split_file"]: SPLIT_SHA,
              base["data"]["distances_file"]: LABEL_SHA}
    for path, expected in assets.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Asset mismatch: {path}")
    prepared = args.experiment / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    configs = make_configs(base, args.experiment)
    for name, cfg in configs.items():
        (prepared / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    manifest = dict(asset_hashes=assets, source_hashes=source_hashes(args.experiment / "code_v1"),
        config_hashes={n: sha256(prepared / f"{n}.yaml") for n in configs},
        variants={n: list(v) for n, v in VARIANTS.items()}, seed=44, start="scratch",
        self_distillation=False, teacher_used=False, test_used=False,
        scientific_change="Protein-only soft-candidate noise before protein context encoder; matched random amplitudes",
        invariant="Model parameters, losses, data, normalization, optimizer, stopping and clean VAL unchanged")
    (prepared / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(prepared), "variants": list(configs)}), flush=True)


if __name__ == "__main__":
    main()
