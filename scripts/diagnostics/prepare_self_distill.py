#!/usr/bin/env python3
"""Freeze two scratch KD configs from the audited GFGF seed44 baseline."""

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.prepare_libre_transfer import BASE_SHA, SPLIT_SHA, LABEL_SHA, sha256, source_hashes

TEACHER_SHA = "1186945af7a3d603031eb031597afb734e2106bd596aa4959a4de148f11aeb7e"
VARIANTS = {"gfgf_sd_t2_lam005": 0.05, "gfgf_sd_t2_lam010": 0.10}


def make_configs(base, experiment, teacher_cache):
    t, m = base["training"], base["model"]
    if any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "resume", "freeze_v11_base", "freeze_v11m3_base")):
        raise ValueError("Expected scratch baseline")
    if m["use_compile"] or m["protein_hybrid_pattern"] != "GFGF" or base["experiment"]["seed"] != 44:
        raise ValueError("Expected seed44 GFGF, no compile")
    configs = {}
    for name, weight in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = name + "_scratch_seed44"
        cfg["experiment"]["output_dir"] = str(Path(experiment) / "outputs")
        cfg["self_distillation"] = dict(
            temperature=2.0, weight=weight, warmup_epochs=5, ramp_epochs=5,
            teacher_cache=str(teacher_cache), teacher_checkpoint_sha256=TEACHER_SHA,
            split_sha256=SPLIT_SHA, label_sha256=LABEL_SHA,
            reduction="residue_mean_per_row_then_all_batch_rows", skip_source_dropped_rows=True,
        )
        configs[name] = cfg
    return configs


def verify_prepared(experiment, check_cache=False, check_smoke=False):
    experiment = Path(experiment)
    manifest = json.loads((experiment / "prepared/provenance.json").read_text())
    for path, expected in manifest["asset_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    if source_hashes(experiment / "code_v1") != manifest["source_hashes"]:
        raise RuntimeError("Source snapshot changed after preparation")
    for name, expected in manifest["config_hashes"].items():
        if sha256(experiment / "prepared" / f"{name}.yaml") != expected:
            raise RuntimeError(f"Config changed: {name}")
    if check_cache:
        from scripts.diagnostics.strict_assignment_common import read_strict_splits
        from src.losses.self_distillation import TeacherCache
        cfg = yaml.safe_load((experiment / "prepared" / f"{next(iter(VARIANTS))}.yaml").read_text())
        TeacherCache.load(manifest["teacher_cache"], cfg,
                          read_strict_splits(cfg["data"]["split_file"]), cfg["self_distillation"])
    if check_smoke:
        report = json.loads((experiment / "smoke_pass.json").read_text())
        if (report["passed"] is not True
                or report["provenance_sha256"] != sha256(experiment / "prepared/provenance.json")
                or report["teacher_cache_sha256"] != sha256(manifest["teacher_cache"])
                or {r["name"] for r in report["runs"]} != set(VARIANTS)
                or not report["lambda_zero_equivalence"]["passed"]
                or not report["dropout_arithmetic"]["passed"]):
            raise RuntimeError("Missing/stale/failed smoke or changed teacher cache")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--teacher-cache", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check-cache", action="store_true")
    ap.add_argument("--check-smoke", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify_prepared(args.experiment, args.check_cache, args.check_smoke)
        print("SELF_DISTILL_INPUTS_VERIFIED", flush=True)
        return
    if args.base is None or args.teacher_cache is None or sha256(args.base) != BASE_SHA:
        raise RuntimeError("Specify the exact audited baseline config and TRAIN cache path")
    cfg = yaml.safe_load(args.base.read_text())
    teacher = args.base.parent / "best_f1_c0_model.pt"
    assets = {str(args.base): BASE_SHA, str(teacher): TEACHER_SHA,
              cfg["data"]["split_file"]: SPLIT_SHA, cfg["data"]["distances_file"]: LABEL_SHA}
    for path, expected in assets.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Asset mismatch: {path}")
    prepared = args.experiment / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    configs = make_configs(cfg, args.experiment, args.teacher_cache)
    for name, value in configs.items():
        (prepared / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=True))
    manifest = dict(asset_hashes=assets, source_hashes=source_hashes(args.experiment / "code_v1"),
                    config_hashes={n: sha256(prepared / f"{n}.yaml") for n in configs},
                    teacher_cache=str(args.teacher_cache), seed=44, start="scratch", test_used=False,
                    scientific_changes="Add TRAIN-only offline KL, T=2, lambda=.05/.10; 5-epoch wait + 5-epoch ramp",
                    note="Cache checked at smoke/train time. Hard loss, data, model, optimizer, stopping and VAL unchanged.")
    (prepared / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(prepared), "variants": list(configs)}), flush=True)


if __name__ == "__main__":
    main()
